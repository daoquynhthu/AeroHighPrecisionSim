#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include "aero/cfd/lusgs.hpp"
#include "aero/cfd/real.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

constexpr int kBlockSize = 128;
constexpr int kMaxCellColors = 64;

int greedy_color_cells(
    const std::vector<int>& h_left,
    const std::vector<int>& h_right,
    int n_cells,
    std::vector<int>& color_of_cell) {
    std::vector<std::vector<int>> adj(static_cast<std::size_t>(n_cells));
    for (std::size_t f = 0; f < h_left.size(); ++f) {
        int l = h_left[f];
        int r = h_right[f];
        if (l >= 0 && l < n_cells && r >= 0 && r < n_cells) {
            adj[static_cast<std::size_t>(l)].push_back(r);
            adj[static_cast<std::size_t>(r)].push_back(l);
        }
    }

    color_of_cell.assign(static_cast<std::size_t>(n_cells), -1);
    int n_colors = 0;
    for (int i = 0; i < n_cells; ++i) {
        std::uint64_t used = 0;
        for (int nb : adj[static_cast<std::size_t>(i)]) {
            if (color_of_cell[static_cast<std::size_t>(nb)] >= 0) {
                int c = color_of_cell[static_cast<std::size_t>(nb)];
                if (c < 64) used |= (static_cast<std::uint64_t>(1) << c);
            }
        }
        int c = 0;
        while (c < kMaxCellColors && (used & (static_cast<std::uint64_t>(1) << c))) ++c;
        if (c >= kMaxCellColors) return 0;
        color_of_cell[static_cast<std::size_t>(i)] = c;
        if (c + 1 > n_colors) n_colors = c + 1;
    }
    return n_colors;
}

__global__ void __launch_bounds__(256) compute_diag_kernel(
    Real* d_D, Real* d_srad, int n_cells, int nvar, Real gamma,
    const Real* d_q, const Real* d_volume,
    const Real* d_nx, const Real* d_ny, const Real* d_nz,
    const Real* d_area,
    const int* d_cell_face_start, const int* d_cell_faces, const int* d_cell_face_nbr,
    const Real* d_dt_cell, Real* d_inv_vol,
    const Real* d_mu, Real Re) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;

    Real vol = d_volume[cell];
    Real inv_vol = 1.0f / vol;
    if (d_inv_vol) d_inv_vol[cell] = inv_vol;

    float4 q4 = reinterpret_cast<const float4*>(d_q + cell * nvar)[0];
    Real rho = q4.x;
    if (!real_isfinite(rho) || rho <= 0) { d_D[cell] = 1e10f; if (d_srad) d_srad[cell] = 0; return; }
    Real inv_rho = 1.0f / rho;
    Real u = q4.y * inv_rho;
    Real v = q4.z * inv_rho;
    Real w = q4.w * inv_rho;
    float2 q2 = reinterpret_cast<const float2*>(d_q + cell * nvar + 4)[0];
    Real E = q2.x;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (E - rho * kinetic);
    if (!real_isfinite(p) || p <= 0) { d_D[cell] = 1e10f; if (d_srad) d_srad[cell] = 0; return; }
    Real a = real_sqrt(gamma * p / rho);
    Real mu_local = (d_mu) ? d_mu[cell] : 0;

    Real diag = 0;
    int face_start = d_cell_face_start[cell];
    int face_end = d_cell_face_start[cell + 1];
    for (int fi = face_start; fi < face_end; ++fi) {
        int f = d_cell_faces[fi];
        int nbr = d_cell_face_nbr[fi];
        Real area = d_area[f];
        if (nbr < 0 || nbr >= n_cells) {
            diag += (real_fabs(u * d_nx[f] + v * d_ny[f] + w * d_nz[f]) + a) * area;
        } else {
            Real vn_self = u * d_nx[f] + v * d_ny[f] + w * d_nz[f];
            float4 qn4 = reinterpret_cast<const float4*>(d_q + nbr * nvar)[0];
            Real rho_n = qn4.x;
            Real lambda_nbr = 0;
            if (rho_n > 0) {
                Real inv_rho_n = 1.0f / rho_n;
                Real u_n = qn4.y * inv_rho_n;
                Real v_n = qn4.z * inv_rho_n;
                Real w_n = qn4.w * inv_rho_n;
                float2 qn2 = reinterpret_cast<const float2*>(d_q + nbr * nvar + 4)[0];
                Real E_n = qn2.x;
                Real kin_n = 0.5f * (u_n*u_n + v_n*v_n + w_n*w_n);
                Real p_n = (gamma - 1.0f) * (E_n - rho_n * kin_n);
                Real a_n = p_n > 0 ? real_sqrt(gamma * p_n / rho_n) : a;
                Real vn_nbr = u_n * d_nx[f] + v_n * d_ny[f] + w_n * d_nz[f];
                lambda_nbr = 0.5f * (real_fabs(vn_self) + real_fabs(vn_nbr) + a + a_n);
            } else {
                lambda_nbr = real_fabs(vn_self) + a;
            }
            diag += lambda_nbr * area;
        }

        if (d_mu && Re > 0) {
            Real mu_nbr = d_mu[(nbr >= 0 && nbr < n_cells) ? nbr : cell];
            Real mu_avg = 0.5f * (mu_local + mu_nbr);
            Real visc_srad = (4.0f / 3.0f) * mu_avg * area * area / (Re * vol);
            diag += visc_srad;
        }
    }

    diag = 0.5f * diag * inv_vol;
    Real cell_dt = 1e20f;
    if (d_dt_cell) {
        cell_dt = d_dt_cell[cell];
    }
    Real unsteady = (cell_dt > 0 && cell_dt < 1e20f) ? (1.0f / cell_dt) : 0;
    d_D[cell] = unsteady + diag;
    if (d_srad) d_srad[cell] = diag;
}

__global__ void __launch_bounds__(256) forward_sweep_kernel(
    Real* d_dz, const Real* d_D, const Real* d_r,
    const Real* d_q, int n_cells, int nvar, Real gamma,
    const Real* d_nx, const Real* d_ny, const Real* d_nz,
    const Real* d_area,
    const int* d_cell_face_start, const int* d_cell_faces, const int* d_cell_face_nbr,
    const Real* d_inv_vol,
    int color_num, const int* d_cell_color) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;
    if (d_cell_color[cell] != color_num) return;

    float4 q4 = reinterpret_cast<const float4*>(d_q + cell * nvar)[0];
    Real rho = q4.x;
    if (!real_isfinite(rho) || rho <= 0) {
        for (int iv = 0; iv < nvar; ++iv) d_dz[cell * nvar + iv] = 0;
        return;
    }
    Real inv_rho = 1.0f / rho;
    Real u = q4.y * inv_rho;
    Real v = q4.z * inv_rho;
    Real w = q4.w * inv_rho;
    float2 q2 = reinterpret_cast<const float2*>(d_q + cell * nvar + 4)[0];
    Real E = q2.x;
    Real kin = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (E - rho * kin);
    Real a = p > 0 ? real_sqrt(gamma * p / rho) : 1e-8f;

    Real res[6] = {0};
    for (int iv = 0; iv < nvar; ++iv) res[iv] = d_r[cell * nvar + iv];

    int face_start = d_cell_face_start[cell];
    int face_end = d_cell_face_start[cell + 1];
    for (int fi = face_start; fi < face_end; ++fi) {
        int nbr = d_cell_face_nbr[fi];
        if (nbr < 0 || nbr >= n_cells) continue;

        int nbr_color = d_cell_color[nbr];
        if (nbr_color >= color_num) continue;

        int f = d_cell_faces[fi];
        Real vn_self = u * d_nx[f] + v * d_ny[f] + w * d_nz[f];
        float4 qn4 = reinterpret_cast<const float4*>(d_q + nbr * nvar)[0];
        Real rho_n = qn4.x;
        Real lambda;
        if (rho_n > 0) {
            Real inv_rho_n = 1.0f / rho_n;
            Real u_n = qn4.y * inv_rho_n;
            Real v_n = qn4.z * inv_rho_n;
            Real w_n = qn4.w * inv_rho_n;
            float2 qn2 = reinterpret_cast<const float2*>(d_q + nbr * nvar + 4)[0];
            Real E_n = qn2.x;
            Real kin_n = 0.5f * (u_n*u_n + v_n*v_n + w_n*w_n);
            Real p_n = (gamma - 1.0f) * (E_n - rho_n * kin_n);
            Real a_n = p_n > 0 ? real_sqrt(gamma * p_n / rho_n) : a;
            Real vn_nbr = u_n * d_nx[f] + v_n * d_ny[f] + w_n * d_nz[f];
            lambda = 0.5f * (real_fabs(vn_self) + real_fabs(vn_nbr) + a + a_n);
        } else {
            lambda = real_fabs(vn_self) + a;
        }
        Real contrib = 0.5f * lambda * d_area[f] * d_inv_vol[cell];
        for (int iv = 0; iv < nvar; ++iv) {
            res[iv] -= contrib * d_dz[nbr * nvar + iv];
        }
    }

    Real inv_D = 1.0f / d_D[cell];
    for (int iv = 0; iv < nvar; ++iv) d_dz[cell * nvar + iv] = res[iv] * inv_D;
}

__global__ void __launch_bounds__(256) backward_sweep_kernel(
    Real* d_z, const Real* d_dz, const Real* d_D,
    const Real* d_q, int n_cells, int nvar, Real gamma,
    const Real* d_nx, const Real* d_ny, const Real* d_nz,
    const Real* d_area,
    const int* d_cell_face_start, const int* d_cell_faces, const int* d_cell_face_nbr,
    const Real* d_inv_vol,
    int color_num, const int* d_cell_color) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;
    if (d_cell_color[cell] != color_num) return;

    float4 q4 = reinterpret_cast<const float4*>(d_q + cell * nvar)[0];
    Real rho = q4.x;
    if (!real_isfinite(rho) || rho <= 0) {
        for (int iv = 0; iv < nvar; ++iv) d_z[cell * nvar + iv] = 0;
        return;
    }
    Real inv_rho = 1.0f / rho;
    Real u = q4.y * inv_rho;
    Real v = q4.z * inv_rho;
    Real w = q4.w * inv_rho;
    float2 q2 = reinterpret_cast<const float2*>(d_q + cell * nvar + 4)[0];
    Real E = q2.x;
    Real kin = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (E - rho * kin);
    Real a = p > 0 ? real_sqrt(gamma * p / rho) : 1e-8f;

    Real correction[6] = {0};
    int face_start = d_cell_face_start[cell];
    int face_end = d_cell_face_start[cell + 1];
    for (int fi = face_start; fi < face_end; ++fi) {
        int nbr = d_cell_face_nbr[fi];
        if (nbr < 0 || nbr >= n_cells) continue;

        int nbr_color = d_cell_color[nbr];
        if (nbr_color <= color_num) continue;

        int f = d_cell_faces[fi];
        Real vn_self = u * d_nx[f] + v * d_ny[f] + w * d_nz[f];
        float4 qn4 = reinterpret_cast<const float4*>(d_q + nbr * nvar)[0];
        Real rho_n = qn4.x;
        Real lambda;
        if (rho_n > 0) {
            Real inv_rho_n = 1.0f / rho_n;
            Real u_n = qn4.y * inv_rho_n;
            Real v_n = qn4.z * inv_rho_n;
            Real w_n = qn4.w * inv_rho_n;
            float2 qn2 = reinterpret_cast<const float2*>(d_q + nbr * nvar + 4)[0];
            Real E_n = qn2.x;
            Real kin_n = 0.5f * (u_n*u_n + v_n*v_n + w_n*w_n);
            Real p_n = (gamma - 1.0f) * (E_n - rho_n * kin_n);
            Real a_n = p_n > 0 ? real_sqrt(gamma * p_n / rho_n) : a;
            Real vn_nbr = u_n * d_nx[f] + v_n * d_ny[f] + w_n * d_nz[f];
            lambda = 0.5f * (real_fabs(vn_self) + real_fabs(vn_nbr) + a + a_n);
        } else {
            lambda = real_fabs(vn_self) + a;
        }
        Real contrib = 0.5f * lambda * d_area[f] * d_inv_vol[cell];
        for (int iv = 0; iv < nvar; ++iv) {
            correction[iv] += contrib * d_z[nbr * nvar + iv];
        }
    }

    Real inv_D = 1.0f / d_D[cell];
    for (int iv = 0; iv < nvar; ++iv) {
        d_z[cell * nvar + iv] = d_dz[cell * nvar + iv] - correction[iv] * inv_D;
    }
}

// Count face incidences per cell (parallel over faces)
__global__ void __launch_bounds__(256) count_incidences_kernel(
    const int* d_left, const int* d_right, int nf, int n_cells,
    int* d_incidence) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nf) return;
    int l = d_left[f];
    int r = d_right[f];
    if (l >= 0 && l < n_cells) atomicAdd(&d_incidence[l], 1);
    if (r >= 0 && r < n_cells) atomicAdd(&d_incidence[r], 1);
}

// Fill CSR cell-face adjacency from face arrays (parallel over faces)
__global__ void __launch_bounds__(256) fill_csr_kernel(
    const int* d_left, const int* d_right, int nf, int n_cells,
    int* d_cursor,
    int* d_cell_faces, int* d_cell_face_nbr) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nf) return;
    int l = d_left[f];
    int r = d_right[f];
    if (l >= 0 && l < n_cells) {
        int pos = atomicAdd(&d_cursor[l], 1);
        d_cell_faces[pos] = f;
        d_cell_face_nbr[pos] = r;
    }
    if (r >= 0 && r < n_cells) {
        int pos = atomicAdd(&d_cursor[r], 1);
        d_cell_faces[pos] = f;
        d_cell_face_nbr[pos] = l;
    }
}

// Multi-pass greedy coloring: assign target_color to any uncolored cell
// whose neighbors don't already have target_color.
__global__ void __launch_bounds__(256) greedy_color_kernel(
    const int* d_cell_face_start,
    const int* d_cell_face_nbr,
    int n_cells,
    int* d_colors,
    int target_color,
    int* d_colored_this_pass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells || d_colors[i] >= 0) return;

    int start = d_cell_face_start[i];
    int end = d_cell_face_start[i + 1];
    for (int j = start; j < end; ++j) {
        if (d_colors[d_cell_face_nbr[j]] == target_color) return;
    }
    d_colors[i] = target_color;
    atomicAdd(d_colored_this_pass, 1);
}

__global__ void __launch_bounds__(256) init_colors_kernel(int* d_colors, int n_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cells) d_colors[i] = -1;
}

} // namespace

LusgsPreconditioner::~LusgsPreconditioner() { release(); }

bool LusgsPreconditioner::allocate(DeviceMesh& mesh, std::string* error) {
    n_cells_ = static_cast<int>(mesh.cell_count());
    nvar_ = DeviceMesh::NVAR;
    if (n_cells_ <= 0) { if (error) *error = "LusgsPreconditioner: empty mesh"; return false; }

    if (!cuda_check(cudaMalloc(&d_D_, n_cells_ * sizeof(Real)), "cudaMalloc d_D", error)) return false;
    if (!cuda_check(cudaMalloc(&d_dz_, n_cells_ * nvar_ * sizeof(Real)), "cudaMalloc d_dz", error)) return false;
    if (!cuda_check(cudaMalloc(&d_inv_vol_, n_cells_ * sizeof(Real)), "cudaMalloc d_inv_vol", error)) return false;
    if (!cuda_check(cudaMalloc(&d_spectral_radius_, n_cells_ * sizeof(Real)), "cudaMalloc d_srad", error)) return false;

    int nf = static_cast<int>(mesh.face_count());
    std::vector<int> h_left(nf), h_right(nf);
    DeviceFaceData fd = mesh.face_data();
    if (!cuda_check(cudaMemcpyAsync(h_left.data(), fd.left_cell, nf * sizeof(int), cudaMemcpyDeviceToHost, 0), "copy left", error)) return false;
    if (!cuda_check(cudaMemcpyAsync(h_right.data(), fd.right_cell, nf * sizeof(int), cudaMemcpyDeviceToHost, 0), "copy right", error)) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "sync left/right", error)) return false;

    std::vector<int> cell_color;
    n_cell_colors_ = greedy_color_cells(h_left, h_right, n_cells_, cell_color);
    if (n_cell_colors_ <= 0) {
        if (error) *error = "LusgsPreconditioner: cell coloring failed";
        return false;
    }

    if (!cuda_check(cudaMalloc(&d_cell_color_, n_cells_ * sizeof(int)), "cudaMalloc d_cell_color", error)) return false;
    if (!cuda_check(cudaMemcpy(d_cell_color_, cell_color.data(), n_cells_ * sizeof(int), cudaMemcpyHostToDevice), "copy d_cell_color", error)) return false;

    // Build CSR cell-face adjacency
    // Count incidences (each non-boundary face appears for both cells)
    std::vector<int> incidence_count(n_cells_, 0);
    for (int f = 0; f < nf; ++f) {
        int l = h_left[f];
        int r = h_right[f];
        if (l >= 0 && l < n_cells_) incidence_count[l]++;
        if (r >= 0 && r < n_cells_) incidence_count[r]++;
    }

    std::vector<int> h_cell_face_start(n_cells_ + 1, 0);
    int total = 0;
    for (int i = 0; i < n_cells_; ++i) {
        h_cell_face_start[i] = total;
        total += incidence_count[i];
    }
    h_cell_face_start[n_cells_] = total;
    n_incidence_ = total;

    std::vector<int> h_cell_faces(total);
    std::vector<int> h_cell_face_nbr(total);
    std::vector<int> cursor = h_cell_face_start;
    for (int f = 0; f < nf; ++f) {
        int l = h_left[f];
        int r = h_right[f];
        if (l >= 0 && l < n_cells_) { int pos = cursor[l]++; h_cell_faces[pos] = f; h_cell_face_nbr[pos] = r; }
        if (r >= 0 && r < n_cells_) { int pos = cursor[r]++; h_cell_faces[pos] = f; h_cell_face_nbr[pos] = l; }
    }

    if (!cuda_check(cudaMalloc(&d_cell_face_start_, (n_cells_ + 1) * sizeof(int)), "cudaMalloc d_cell_face_start", error)) return false;
    if (!cuda_check(cudaMalloc(&d_cell_faces_, total * sizeof(int)), "cudaMalloc d_cell_faces", error)) return false;
    if (!cuda_check(cudaMalloc(&d_cell_face_nbr_, total * sizeof(int)), "cudaMalloc d_cell_face_nbr", error)) return false;
    if (!cuda_check(cudaMemcpy(d_cell_face_start_, h_cell_face_start.data(), (n_cells_ + 1) * sizeof(int), cudaMemcpyHostToDevice), "copy cell_face_start", error)) return false;
    if (!cuda_check(cudaMemcpy(d_cell_faces_, h_cell_faces.data(), total * sizeof(int), cudaMemcpyHostToDevice), "copy cell_faces", error)) return false;
    if (!cuda_check(cudaMemcpy(d_cell_face_nbr_, h_cell_face_nbr.data(), total * sizeof(int), cudaMemcpyHostToDevice), "copy cell_face_nbr", error)) return false;

    return true;
}

void LusgsPreconditioner::release() {
    cuda_free_safe(d_D_);
    cuda_free_safe(d_dz_);
    cuda_free_safe(d_inv_vol_);
    cuda_free_safe(d_spectral_radius_);
    cuda_free_safe(d_cell_color_);
    cuda_free_safe(d_cell_face_start_);
    cuda_free_safe(d_cell_faces_);
    cuda_free_safe(d_cell_face_nbr_);
    n_cells_ = 0;
    n_cell_colors_ = 0;
    n_incidence_ = 0;
}

bool LusgsPreconditioner::rebuild_coloring(DeviceMesh& mesh, std::string* error) {
    int n_cells_new = static_cast<int>(mesh.cell_count());
    if (n_cells_new <= 0) { if (error) *error = "rebuild_coloring: empty mesh"; return false; }
    int nf = static_cast<int>(mesh.face_count());
    DeviceFaceData fd = mesh.face_data();

    // --- Phase 1: Count face incidences per cell on GPU ---
    int* d_incidence = nullptr;
    if (!cuda_check(cudaMalloc(&d_incidence, n_cells_new * sizeof(int)), "cudaMalloc d_incidence", error)) return false;
    if (!cuda_check(cudaMemset(d_incidence, 0, n_cells_new * sizeof(int)), "zero d_incidence", error)) { cuda_free_safe(d_incidence); return false; }

    {
        int block = 256;
        int grid = (nf + block - 1) / block;
        count_incidences_kernel<<<grid, block>>>(fd.left_cell, fd.right_cell, nf, n_cells_new, d_incidence);
        if (!cuda_check(cudaGetLastError(), "count_incidences_kernel", error)) { cuda_free_safe(d_incidence); return false; }
    }

    // --- Phase 2: Download incidences (n_cells ints vs 2*nf ints previously) and compute prefix sum on CPU ---
    std::vector<int> h_incidence(n_cells_new);
    if (!cuda_check(cudaMemcpy(h_incidence.data(), d_incidence, n_cells_new * sizeof(int), cudaMemcpyDeviceToHost), "copy incidences", error)) { cuda_free_safe(d_incidence); return false; }
    cuda_free_safe(d_incidence);

    std::vector<int> h_cell_face_start(n_cells_new + 1, 0);
    int total = 0;
    for (int i = 0; i < n_cells_new; ++i) {
        h_cell_face_start[i] = total;
        total += h_incidence[i];
    }
    h_cell_face_start[n_cells_new] = total;

    // --- Phase 3: (Re)allocate device arrays ---
    // Always free CSR arrays first (adjacency may have changed even if count is same)
    cuda_free_safe(d_cell_face_start_);
    cuda_free_safe(d_cell_faces_);
    cuda_free_safe(d_cell_face_nbr_);

    if (n_cells_new != n_cells_) {
        cuda_free_safe(d_cell_color_);
        if (!cuda_check(cudaMalloc(&d_cell_color_, n_cells_new * sizeof(int)), "cudaMalloc d_cell_color", error)) return false;
        n_cells_ = n_cells_new;
    }

    if (!cuda_check(cudaMalloc(&d_cell_face_start_, (n_cells_ + 1) * sizeof(int)), "cudaMalloc d_cell_face_start", error)) return false;
    if (total > 0) {
        if (!cuda_check(cudaMalloc(&d_cell_faces_, total * sizeof(int)), "cudaMalloc d_cell_faces", error)) return false;
        if (!cuda_check(cudaMalloc(&d_cell_face_nbr_, total * sizeof(int)), "cudaMalloc d_cell_face_nbr", error)) return false;
    } else {
        d_cell_faces_ = nullptr;
        d_cell_face_nbr_ = nullptr;
    }
    n_incidence_ = total;

    // Upload cell_face_start
    if (!cuda_check(cudaMemcpy(d_cell_face_start_, h_cell_face_start.data(), (n_cells_ + 1) * sizeof(int), cudaMemcpyHostToDevice), "copy cell_face_start", error)) return false;

    // --- Phase 4: Fill CSR from face arrays on GPU ---
    if (total > 0) {
        int* d_cursor = nullptr;
        if (!cuda_check(cudaMalloc(&d_cursor, n_cells_ * sizeof(int)), "cudaMalloc d_cursor", error)) return false;
        if (!cuda_check(cudaMemcpy(d_cursor, d_cell_face_start_, n_cells_ * sizeof(int), cudaMemcpyDeviceToDevice), "copy cursor", error)) { cuda_free_safe(d_cursor); return false; }

        {
            int block = 256;
            int grid = (nf + block - 1) / block;
            fill_csr_kernel<<<grid, block>>>(fd.left_cell, fd.right_cell, nf, n_cells_, d_cursor, d_cell_faces_, d_cell_face_nbr_);
            if (!cuda_check(cudaGetLastError(), "fill_csr_kernel", error)) { cuda_free_safe(d_cursor); return false; }
        }
        cuda_free_safe(d_cursor);
    }

    // --- Phase 5: Multi-pass greedy coloring on GPU ---
    {
        int block = 256;
        int grid = (n_cells_ + block - 1) / block;
        init_colors_kernel<<<grid, block>>>(d_cell_color_, n_cells_);
        if (!cuda_check(cudaGetLastError(), "init_colors_kernel", error)) return false;
    }

    int* d_colored = nullptr;
    if (!cuda_check(cudaMalloc(&d_colored, sizeof(int)), "cudaMalloc d_colored", error)) return false;

    int n_colors = 0;
    int block = 256;
    int grid = (n_cells_ + block - 1) / block;
    for (int target = 0; target < kMaxCellColors; ++target) {
        if (!cuda_check(cudaMemset(d_colored, 0, sizeof(int)), "zero d_colored", error)) { cuda_free_safe(d_colored); return false; }
        greedy_color_kernel<<<grid, block>>>(d_cell_face_start_, d_cell_face_nbr_, n_cells_, d_cell_color_, target, d_colored);
        if (!cuda_check(cudaGetLastError(), "greedy_color_kernel", error)) { cuda_free_safe(d_colored); return false; }
        int colored = 0;
        if (!cuda_check(cudaMemcpy(&colored, d_colored, sizeof(int), cudaMemcpyDeviceToHost), "read colored", error)) { cuda_free_safe(d_colored); return false; }
        if (colored == 0) break;
        n_colors = target + 1;
    }
    cuda_free_safe(d_colored);

    if (n_colors <= 0) {
        if (error) *error = "rebuild_coloring: GPU greedy coloring failed";
        return false;
    }
    n_cell_colors_ = n_colors;

    return true;
}

bool LusgsPreconditioner::compute_diagonal(DeviceMesh& mesh, const Real* d_dt_cell,
    Real gamma, bool viscous, const Real* d_mu, Real Re,
    std::string* error) {
    int block = kBlockSize;
    int grid = (n_cells_ + block - 1) / block;
    if (grid < 1) grid = 1;

    DeviceFaceData fd = mesh.face_data();
    DeviceCellData cd = mesh.cell_data();

    compute_diag_kernel<<<grid, block>>>(
        d_D_, d_spectral_radius_, n_cells_, nvar_, gamma,
        mesh.state_device(), cd.volume,
        fd.nx, fd.ny, fd.nz, fd.area,
        d_cell_face_start_, d_cell_faces_, d_cell_face_nbr_,
        d_dt_cell, d_inv_vol_,
        viscous ? d_mu : nullptr, Re);
    if (!cuda_check(cudaGetLastError(), "compute_diag_kernel", error)) return false;
    return true;
}

bool LusgsPreconditioner::apply(DeviceMesh& mesh, const Real* d_r, Real* d_z,
    Real gamma, std::string* error) {
    int block = kBlockSize;
    int grid = (n_cells_ + block - 1) / block;
    if (grid < 1) grid = 1;

    DeviceFaceData fd = mesh.face_data();

    for (int c = 0; c < n_cell_colors_; ++c) {
        forward_sweep_kernel<<<grid, block>>>(
            d_dz_, d_D_, d_r, mesh.state_device(),
            n_cells_, nvar_, gamma,
            fd.nx, fd.ny, fd.nz, fd.area,
            d_cell_face_start_, d_cell_faces_, d_cell_face_nbr_,
            d_inv_vol_, c, d_cell_color_);
        if (!cuda_check(cudaGetLastError(), "forward_sweep_kernel", error)) return false;
    }

    for (int c = n_cell_colors_ - 1; c >= 0; --c) {
        backward_sweep_kernel<<<grid, block>>>(
            d_z, d_dz_, d_D_, mesh.state_device(),
            n_cells_, nvar_, gamma,
            fd.nx, fd.ny, fd.nz, fd.area,
            d_cell_face_start_, d_cell_faces_, d_cell_face_nbr_,
            d_inv_vol_, c, d_cell_color_);
        if (!cuda_check(cudaGetLastError(), "backward_sweep_kernel", error)) return false;
    }

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
