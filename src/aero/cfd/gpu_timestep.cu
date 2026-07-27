#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include <cuda_runtime.h>
#include <limits>
namespace aerosp {
namespace aero {
namespace cfd {

namespace {

constexpr int kTimeStepBlockSize = 128;
constexpr Real kSaCv1 = 7.1f;
constexpr Real kSaSigma = 2.0f / 3.0f;
constexpr Real kSaCw1 = 0.1355f / (0.41f * 0.41f) + (1.0f + 0.622f) / kSaSigma;

static Real* d_partial_buf = nullptr;
static int   d_partial_cap = 0;

static bool ensure_partial_buf(int min_blocks) {
    if (d_partial_cap >= min_blocks) return true;
    cuda_free_safe(d_partial_buf);
    d_partial_cap = 0;
    if (!cuda_check(cudaMalloc(&d_partial_buf, (size_t)min_blocks * sizeof(Real)),
                    "ensure_partial_buf timestep"))
        return false;
    d_partial_cap = min_blocks;
    return true;
}

__device__ Real d_sa_eddy_visc(Real nu_tilde, Real rho, Real mu, Real Re) {
    if (!(nu_tilde > 0) || !(rho > 0) || !(mu > 0) || !(Re > 0)) return 0.0f;
    Real chi = Re * rho * nu_tilde / (mu + 1e-30f);
    if (chi > Real(1.0e6)) chi = Real(1.0e6);
    Real chi3 = chi * chi * chi;
    Real cv13 = kSaCv1 * kSaCv1 * kSaCv1;
    Real fv1 = chi3 / (chi3 + cv13 + 1e-30f);
    return nu_tilde * fv1;
}

__device__ Real d_cell_timestep(
    Real rho, Real u, Real v, Real w, Real p,
    Real h, Real cfl, Real gamma,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re,
    bool sa, Real nu_tilde, Real wall_distance)
{
    Real vmag = real_sqrt(u*u + v*v + w*w);
    Real a = real_sqrt(gamma * p / rho);
    Real denom = vmag + a;
    Real dt = cfl * h / (denom > Real(1e-30) ? denom : Real(1e-30));

    Real mu_cell = 0.0f;
    Real nu_mol = 0.0f;
    if (viscous) {
        Real T = p / rho;
        if (T > 0.0f) {
            Real t_ratio = T / T_ref;
            mu_cell = mu_ref * t_ratio * real_sqrt(t_ratio) * (T_ref + sutherland_T) / (T + sutherland_T);
            if (mu_cell > 0.0f) {
                nu_mol = mu_cell / (rho * Re + 1e-30f);
                Real nu_t = sa ? d_sa_eddy_visc(nu_tilde, rho, mu_cell, Re) : 0.0f;
                Real nu_eff = nu_mol + real_fmax(nu_t, Real(0));
                Real dt_visc = cfl * h * h / (nu_eff + 1e-30f);
                if (dt_visc < dt) dt = dt_visc;
            }
        }
    }
    if (sa && mu_cell > 0.0f) {
        Real d = wall_distance;
        if (!(d > 0) || !real_isfinite(d) || d > Real(1.0e10)) d = Real(1.0e10);
        Real d2 = d * d + 1e-30f;
        Real nu_ref = real_fmax(real_fabs(nu_tilde), real_fmax(nu_mol, Real(1e-30)));
        Real g_dest = 2.0f * kSaCw1 * nu_ref / d2;
        Real h_eff = h;
        if (d < h_eff) h_eff = d;
        Real nu_eff = real_fmax(nu_mol, Real(0)) + real_fmax(nu_tilde, Real(0));
        Real g_diff = nu_eff / (kSaSigma * h_eff * h_eff + 1e-30f);
        Real g_sa = real_fmax(g_dest, g_diff);
        if (g_sa > 0.0f) {
            Real dt_sa = cfl / g_sa;
            if (dt_sa < dt) dt = dt_sa;
        }
    }
    if (!(dt > 0) || !real_isfinite(dt)) dt = Real(1e-30);
    return dt;
}

__global__ void __launch_bounds__(256) timestep_kernel(
    const Real* d_q, int n_cells, int nvar, Real gamma, Real cfl,
    const Real* d_h_min, const Real* d_wall_distance,
    Real* d_partial,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re,
    bool sa) {
    __shared__ Real sdata[kTimeStepBlockSize];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    Real dt = std::numeric_limits<Real>::max();
    if (idx < n_cells) {
        Real rho = d_q[idx * nvar + 0];
        if (real_isfinite(rho) && rho > 0.0f) {
            Real inv_rho = 1.0f / rho;
            Real u = d_q[idx * nvar + 1] * inv_rho;
            Real v = d_q[idx * nvar + 2] * inv_rho;
            Real w = d_q[idx * nvar + 3] * inv_rho;
            Real E = d_q[idx * nvar + 4];
            Real kinetic = 0.5f * (u*u + v*v + w*w);
            Real p = (gamma - 1.0f) * (E - rho * kinetic);
            if (real_isfinite(p) && p > 0.0f) {
                Real h = d_h_min[idx];
                Real nu_tilde = (nvar > 5) ? d_q[idx * nvar + 5] * inv_rho : 0.0f;
                Real wd = d_wall_distance ? d_wall_distance[idx] : Real(1e10);
                dt = d_cell_timestep(rho, u, v, w, p, h, cfl, gamma,
                    viscous, mu_ref, T_ref, sutherland_T, Re,
                    sa, nu_tilde, wd);
            }
        }
    }
    sdata[tid] = dt;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}

__global__ void __launch_bounds__(256) reduce_min_kernel(const Real* d_partial, Real* d_result, int n_blocks) {
    Real m = std::numeric_limits<Real>::max();
    for (int i = 0; i < n_blocks; ++i) {
        if (d_partial[i] < m) m = d_partial[i];
    }
    *d_result = m;
}

__global__ void __launch_bounds__(256) local_timestep_kernel(
    const Real* d_q, int n_cells, int nvar, Real gamma, Real cfl,
    const Real* d_h_min, const Real* d_wall_distance,
    Real* d_dt_cell,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re,
    bool sa, Real lts_ratio_max, Real min_dt_hint) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real rho = d_q[idx * nvar + 0];
    if (!real_isfinite(rho) || rho <= 0.0f) { d_dt_cell[idx] = 0; return; }
    Real inv_rho = 1.0f / rho;
    Real u = d_q[idx * nvar + 1] * inv_rho;
    Real v = d_q[idx * nvar + 2] * inv_rho;
    Real w = d_q[idx * nvar + 3] * inv_rho;
    Real E = d_q[idx * nvar + 4];
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (E - rho * kinetic);
    if (!real_isfinite(p) || p <= 0.0f) { d_dt_cell[idx] = 0; return; }

    Real h = d_h_min[idx];
    Real nu_tilde = (nvar > 5) ? d_q[idx * nvar + 5] * inv_rho : 0.0f;
    Real wd = d_wall_distance ? d_wall_distance[idx] : Real(1e10);
    Real dt = d_cell_timestep(rho, u, v, w, p, h, cfl, gamma,
        viscous, mu_ref, T_ref, sutherland_T, Re,
        sa, nu_tilde, wd);
    if (lts_ratio_max > 1.0f && min_dt_hint > 0.0f) {
        Real cap = min_dt_hint * lts_ratio_max;
        if (dt > cap) dt = cap;
    }
    d_dt_cell[idx] = dt;
}

} // namespace

__global__ void __launch_bounds__(256) cap_local_dt_kernel(
    Real* d_dt_cell, const Real* d_min_dt, Real ratio_max, int n_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    Real min_dt = __ldg(d_min_dt);
    if (!(min_dt > 0) || !real_isfinite(min_dt)) min_dt = Real(1e-30);
    // ratio_max <= 1: force every cell to global min_dt (broadcast fill)
    if (!(ratio_max > 1.0f)) {
        d_dt_cell[idx] = min_dt;
        return;
    }
    Real dt = d_dt_cell[idx];
    Real cap = min_dt * ratio_max;
    if (dt > cap) dt = cap;
    if (!(dt > 0) || !real_isfinite(dt)) dt = min_dt;
    d_dt_cell[idx] = dt;
}

bool compute_local_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_dt_cell,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re, std::string* error,
    cudaStream_t stream, bool sa) {
    if (d_dt_cell == nullptr) { if (error) *error = "compute_local_timestep_gpu: null d_dt_cell"; return false; }
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();
    local_timestep_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), nc, mesh.nvar(), gamma, cfl,
        cd.h_min, cd.wall_distance, d_dt_cell,
        viscous, mu_ref, T_ref, sutherland_T, Re,
        sa, Real(0), Real(0));
    if (!cuda_check(cudaGetLastError(), "local timestep kernel launch", error)) return false;
    return true;
}

bool cap_local_timestep_gpu(Real* d_dt_cell, const Real* d_min_dt, Real ratio_max,
    int n_cells, std::string* error, cudaStream_t stream) {
    if (!d_dt_cell || !d_min_dt || n_cells <= 0) return true;
    int block = 128;
    int grid = (n_cells + block - 1) / block;
    cap_local_dt_kernel<<<grid, block, 0, stream>>>(d_dt_cell, d_min_dt, ratio_max, n_cells);
    if (!cuda_check(cudaGetLastError(), "cap_local_dt kernel launch", error)) return false;
    return true;
}

bool compute_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_min_dt,
    cudaStream_t stream) {
    return compute_timestep_gpu(mesh, gamma, cfl, d_min_dt, false, 1.0f, 1.0f, 1.0f, 1e6f, stream, false);
}

bool compute_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_min_dt,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re,
    cudaStream_t stream, bool sa) {
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + kTimeStepBlockSize - 1) / kTimeStepBlockSize;
    if (!ensure_partial_buf(grid)) return false;

    DeviceCellData cd = mesh.cell_data();

    timestep_kernel<<<grid, kTimeStepBlockSize, 0, stream>>>(
        mesh.state_device(), nc, mesh.nvar(), gamma, cfl,
        cd.h_min, cd.wall_distance, d_partial_buf,
        viscous, mu_ref, T_ref, sutherland_T, Re, sa);
    if (!cuda_check(cudaGetLastError(), "timestep kernel launch")) return false;

    reduce_min_kernel<<<1, 1, 0, stream>>>(d_partial_buf, d_min_dt, grid);
    if (!cuda_check(cudaGetLastError(), "reduce_min kernel launch")) return false;

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
