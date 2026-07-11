#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/partition.hpp"
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

namespace {

__device__ bool d_conservative_to_primitive(const Real* q, int cell, int nvar, Real gamma,
    Real& rho, Real& u, Real& v, Real& w, Real& p, Real& nu_tilde) {
    rho = q[cell * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) return false;
    Real inv_rho = 1.0f / rho;
    u = q[cell * nvar + 1] * inv_rho;
    v = q[cell * nvar + 2] * inv_rho;
    w = q[cell * nvar + 3] * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    p = (gamma - 1.0f) * (q[cell * nvar + 4] - rho * kinetic);
    nu_tilde = q[cell * nvar + 5] * inv_rho;
    return real_isfinite(u) && real_isfinite(v) && real_isfinite(w) && real_isfinite(p) && real_isfinite(nu_tilde) && p > 0.0f;
}

__device__ Real d_speed_of_sound(Real rho, Real p, Real gamma) {
    return real_sqrt(gamma * p / rho);
}

__device__ void d_physical_flux(Real rho, Real u, Real v, Real w, Real p, Real nu_tilde, Real gamma,
    Real nx, Real ny, Real nz, Real& mass, Real& mom_x, Real& mom_y, Real& mom_z, Real& energy, Real& turbulence) {
    Real vn = u*nx + v*ny + w*nz;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real rho_E = p / (gamma - 1.0f) + rho * kinetic;
    mass = rho * vn;
    mom_x = rho * u * vn + p * nx;
    mom_y = rho * v * vn + p * ny;
    mom_z = rho * w * vn + p * nz;
    energy = (rho_E + p) * vn;
    turbulence = rho * nu_tilde * vn;
}

__device__ void d_slip_wall_flux(Real p, Real nx, Real ny, Real nz,
    Real& mass, Real& mom_x, Real& mom_y, Real& mom_z, Real& energy, Real& turbulence) {
    mass = 0.0f;
    mom_x = p * nx;
    mom_y = p * ny;
    mom_z = p * nz;
    energy = 0.0f;
    turbulence = 0.0f;
}

__device__ void d_farfield_ghost_state(Real left_rho, Real left_u, Real left_v, Real left_w, Real left_p,
    Real left_nu_tilde,
    Real inf_rho, Real inf_p, Real inf_u, Real inf_v, Real inf_w, Real inf_nu_tilde, Real inf_a,
    Real nx, Real ny, Real nz,
    Real& ghost_rho, Real& ghost_p, Real& ghost_u, Real& ghost_v, Real& ghost_w, Real& ghost_nu_tilde) {
    Real vn_inf = inf_u*nx + inf_v*ny + inf_w*nz;
    if (vn_inf >= inf_a) {
        ghost_rho = left_rho; ghost_p = left_p;
        ghost_u = left_u; ghost_v = left_v; ghost_w = left_w;
        ghost_nu_tilde = left_nu_tilde;
    } else {
        ghost_rho = inf_rho; ghost_p = inf_p;
        ghost_u = inf_u; ghost_v = inf_v; ghost_w = inf_w;
        ghost_nu_tilde = inf_nu_tilde;
    }
}

__device__ void d_hllc_flux(
    Real rhoL, Real uL, Real vL, Real wL, Real pL, Real nu_tildeL,
    Real rhoR, Real uR, Real vR, Real wR, Real pR, Real nu_tildeR,
    Real gamma, Real nx, Real ny, Real nz,
    Real& mass, Real& mom_x, Real& mom_y, Real& mom_z, Real& energy, Real& turbulence) {
    Real vn_l = uL*nx + vL*ny + wL*nz;
    Real vn_r = uR*nx + vR*ny + wR*nz;
    Real a_l = d_speed_of_sound(rhoL, pL, gamma);
    Real a_r = d_speed_of_sound(rhoR, pR, gamma);
    Real s_l = real_fmin(vn_l - a_l, vn_r - a_r);
    Real s_r = real_fmax(vn_l + a_l, vn_r + a_r);

    Real fL_mass, fL_mx, fL_my, fL_mz, fL_en, fL_turb;
    Real fR_mass, fR_mx, fR_my, fR_mz, fR_en, fR_turb;
    d_physical_flux(rhoL, uL, vL, wL, pL, nu_tildeL, gamma, nx, ny, nz, fL_mass, fL_mx, fL_my, fL_mz, fL_en, fL_turb);
    d_physical_flux(rhoR, uR, vR, wR, pR, nu_tildeR, gamma, nx, ny, nz, fR_mass, fR_mx, fR_my, fR_mz, fR_en, fR_turb);

    if (s_l >= 0.0f) { mass = fL_mass; mom_x = fL_mx; mom_y = fL_my; mom_z = fL_mz; energy = fL_en; turbulence = fL_turb; return; }
    if (s_r <= 0.0f) { mass = fR_mass; mom_x = fR_mx; mom_y = fR_my; mom_z = fR_mz; energy = fR_en; turbulence = fR_turb; return; }

    Real denom = rhoL * (s_l - vn_l) - rhoR * (s_r - vn_r);
    if (real_fabs(denom) < 1e-30f) denom = real_copysign(1e-30f, denom);
    Real s_m = (pR - pL + rhoL*vn_l*(s_l - vn_l) - rhoR*vn_r*(s_r - vn_r)) / denom;

    if (s_m >= 0.0f) {
        Real s_l_minus_sm = s_l - s_m;
        if (real_fabs(s_l_minus_sm) < 1e-30f) s_l_minus_sm = real_copysign(1e-30f, s_l_minus_sm);
        Real rho_star = rhoL * (s_l - vn_l) / s_l_minus_sm;
        Real kineticL = 0.5f * (uL*uL + vL*vL + wL*wL);
        Real e_l = pL / ((gamma - 1.0f) * rhoL) + kineticL;
        Real sld_l = s_l - vn_l;
        if (real_fabs(sld_l) < 1e-30f) sld_l = real_copysign(1e-30f, sld_l);
        Real e_star = e_l + (s_m - vn_l) * (s_m + pL / (rhoL * sld_l));
        Real qL_rho = rhoL;
        Real qL_rhou = rhoL * uL;
        Real qL_rhov = rhoL * vL;
        Real qL_rhow = rhoL * wL;
        Real qL_rhoE = pL / (gamma - 1.0f) + rhoL * kineticL;
        Real qL_nu = rhoL * nu_tildeL;

        Real qs_rho = rho_star;
        Real qs_rhou = rho_star * (uL + (s_m - vn_l) * nx);
        Real qs_rhov = rho_star * (vL + (s_m - vn_l) * ny);
        Real qs_rhow = rho_star * (wL + (s_m - vn_l) * nz);
        Real qs_rhoE = rho_star * e_star;
        Real qs_nu = qL_nu * (s_l - vn_l) / s_l_minus_sm;

        mass = fL_mass + s_l * (qs_rho - qL_rho);
        mom_x = fL_mx + s_l * (qs_rhou - qL_rhou);
        mom_y = fL_my + s_l * (qs_rhov - qL_rhov);
        mom_z = fL_mz + s_l * (qs_rhow - qL_rhow);
        energy = fL_en + s_l * (qs_rhoE - qL_rhoE);
        turbulence = fL_turb + s_l * (qs_nu - qL_nu);
    } else {
        Real s_r_minus_sm = s_r - s_m;
        if (real_fabs(s_r_minus_sm) < 1e-30f) s_r_minus_sm = real_copysign(1e-30f, s_r_minus_sm);
        Real rho_star = rhoR * (s_r - vn_r) / s_r_minus_sm;
        Real kineticR = 0.5f * (uR*uR + vR*vR + wR*wR);
        Real e_r = pR / ((gamma - 1.0f) * rhoR) + kineticR;
        Real sld_r = s_r - vn_r;
        if (real_fabs(sld_r) < 1e-30f) sld_r = real_copysign(1e-30f, sld_r);
        Real e_star = e_r + (s_m - vn_r) * (s_m + pR / (rhoR * sld_r));
        Real qR_rho = rhoR;
        Real qR_rhou = rhoR * uR;
        Real qR_rhov = rhoR * vR;
        Real qR_rhow = rhoR * wR;
        Real qR_rhoE = pR / (gamma - 1.0f) + rhoR * kineticR;
        Real qR_nu = rhoR * nu_tildeR;

        Real qs_rho = rho_star;
        Real qs_rhou = rho_star * (uR + (s_m - vn_r) * nx);
        Real qs_rhov = rho_star * (vR + (s_m - vn_r) * ny);
        Real qs_rhow = rho_star * (wR + (s_m - vn_r) * nz);
        Real qs_rhoE = rho_star * e_star;
        Real qs_nu = qR_nu * (s_r - vn_r) / s_r_minus_sm;

        mass = fR_mass + s_r * (qs_rho - qR_rho);
        mom_x = fR_mx + s_r * (qs_rhou - qR_rhou);
        mom_y = fR_my + s_r * (qs_rhov - qR_rhov);
        mom_z = fR_mz + s_r * (qs_rhow - qR_rhow);
        energy = fR_en + s_r * (qs_rhoE - qR_rhoE);
        turbulence = fR_turb + s_r * (qs_nu - qR_nu);
    }
}

__device__ void d_reconstruct_primitive(
    const Real* gradients, int cell,
    Real dx, Real dy, Real dz,
    Real& rho, Real& u, Real& v, Real& w, Real& p, Real& nu_tilde) {
    const Real* g = gradients + cell * 18;
    rho = rho + g[0]*dx + g[1]*dy + g[2]*dz;
    u = u + g[3]*dx + g[4]*dy + g[5]*dz;
    v = v + g[6]*dx + g[7]*dy + g[8]*dz;
    w = w + g[9]*dx + g[10]*dy + g[11]*dz;
    p = p + g[12]*dx + g[13]*dy + g[14]*dz;
    nu_tilde = nu_tilde + g[15]*dx + g[16]*dy + g[17]*dz;
}

template <bool COLORED>
__global__ void __launch_bounds__(128) euler_residual_kernel(
    const Real* d_nx, const Real* d_ny, const Real* d_nz,
    const Real* d_area,
    const int* d_left_cell, const int* d_right_cell,
    const int* d_boundary,
    const Real* d_q,
    int face_start, int face_end, int nvar, int n_cells,
    Real gamma,
    Real inf_rho, Real inf_p,
    Real inf_u, Real inf_v, Real inf_w, Real inf_a, Real inf_nu_tilde,
    Real* d_residual,
    int* d_failed,
    const Real* d_gradients,
    const Real* d_face_cx, const Real* d_face_cy, const Real* d_face_cz,
    const Real* d_cx, const Real* d_cy, const Real* d_cz,
    const int* d_partition_owner, int my_rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + face_start;
    if (idx >= face_end) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) { atomicExch(d_failed, 1); return; }
    if (d_partition_owner && d_partition_owner[left] != my_rank) return;
    int bnd = d_boundary[idx];
    Real nx = d_nx[idx];
    Real ny = d_ny[idx];
    Real nz = d_nz[idx];
    Real area = d_area[idx];

    Real rhoL, uL, vL, wL, pL, nu_tildeL;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL, nu_tildeL)) {
        atomicExch(d_failed, 1);
        return;
    }

    if (d_gradients != nullptr) {
        Real dx = d_face_cx[idx] - d_cx[left];
        Real dy = d_face_cy[idx] - d_cy[left];
        Real dz = d_face_cz[idx] - d_cz[left];
        d_reconstruct_primitive(d_gradients, left, dx, dy, dz, rhoL, uL, vL, wL, pL, nu_tildeL);
        if (!real_isfinite(rhoL) || rhoL <= 0.0f || !real_isfinite(pL) || pL <= 0.0f) {
            atomicExch(d_failed, 1);
            return;
        }
    }

    Real mass, mom_x, mom_y, mom_z, energy, turbulence;

    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) { atomicExch(d_failed, 1); return; }
        Real rhoR, uR, vR, wR, pR, nu_tildeR;
        if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR, nu_tildeR)) {
            atomicExch(d_failed, 1);
            return;
        }
        if (d_gradients != nullptr) {
            Real dx = d_face_cx[idx] - d_cx[right];
            Real dy = d_face_cy[idx] - d_cy[right];
            Real dz = d_face_cz[idx] - d_cz[right];
            d_reconstruct_primitive(d_gradients, right, dx, dy, dz, rhoR, uR, vR, wR, pR, nu_tildeR);
            if (!real_isfinite(rhoR) || rhoR <= 0.0f || !real_isfinite(pR) || pR <= 0.0f) {
                atomicExch(d_failed, 1);
                return;
            }
        }
        d_hllc_flux(rhoL, uL, vL, wL, pL, nu_tildeL, rhoR, uR, vR, wR, pR, nu_tildeR, gamma, nx, ny, nz,
            mass, mom_x, mom_y, mom_z, energy, turbulence);
    } else if (bnd == static_cast<int>(BoundaryKind::SlipWall) || bnd == static_cast<int>(BoundaryKind::NoSlipWall) || bnd == static_cast<int>(BoundaryKind::Symmetry)) {
        d_slip_wall_flux(pL, nx, ny, nz, mass, mom_x, mom_y, mom_z, energy, turbulence);
    } else {
        Real ghrho, ghp, ghu, ghv, ghw, ghnu;
        d_farfield_ghost_state(rhoL, uL, vL, wL, pL, nu_tildeL,
            inf_rho, inf_p, inf_u, inf_v, inf_w, inf_nu_tilde, inf_a,
            nx, ny, nz, ghrho, ghp, ghu, ghv, ghw, ghnu);
        d_hllc_flux(rhoL, uL, vL, wL, pL, nu_tildeL, ghrho, ghu, ghv, ghw, ghp, ghnu, gamma, nx, ny, nz,
            mass, mom_x, mom_y, mom_z, energy, turbulence);
    }

    Real fmass = mass * area;
    Real fmx = mom_x * area;
    Real fmy = mom_y * area;
    Real fmz = mom_z * area;
    Real fen = energy * area;
    Real fturb = turbulence * area;

    if constexpr (COLORED) {
        d_residual[left * nvar + 0] += -fmass;
        d_residual[left * nvar + 1] += -fmx;
        d_residual[left * nvar + 2] += -fmy;
        d_residual[left * nvar + 3] += -fmz;
        d_residual[left * nvar + 4] += -fen;
        d_residual[left * nvar + 5] += -fturb;

        if (bnd == static_cast<int>(BoundaryKind::Interior)) {
            int right = d_right_cell[idx];
            if (right >= 0 && right < n_cells) {
                d_residual[right * nvar + 0] += fmass;
                d_residual[right * nvar + 1] += fmx;
                d_residual[right * nvar + 2] += fmy;
                d_residual[right * nvar + 3] += fmz;
                d_residual[right * nvar + 4] += fen;
                d_residual[right * nvar + 5] += fturb;
            }
        }
    } else {
        real_atomic_add(&d_residual[left * nvar + 0], -fmass);
        real_atomic_add(&d_residual[left * nvar + 1], -fmx);
        real_atomic_add(&d_residual[left * nvar + 2], -fmy);
        real_atomic_add(&d_residual[left * nvar + 3], -fmz);
        real_atomic_add(&d_residual[left * nvar + 4], -fen);
        real_atomic_add(&d_residual[left * nvar + 5], -fturb);

        if (bnd == static_cast<int>(BoundaryKind::Interior)) {
            int right = d_right_cell[idx];
            if (right >= 0 && right < n_cells) {
                real_atomic_add(&d_residual[right * nvar + 0], fmass);
                real_atomic_add(&d_residual[right * nvar + 1], fmx);
                real_atomic_add(&d_residual[right * nvar + 2], fmy);
                real_atomic_add(&d_residual[right * nvar + 3], fmz);
                real_atomic_add(&d_residual[right * nvar + 4], fen);
                real_atomic_add(&d_residual[right * nvar + 5], fturb);
            }
        }
    }
}

} // namespace

bool launch_euler_residual_kernel(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    int* d_failed,
    cudaEvent_t start_event,
    std::string* error,
    int reconstruction_order,
    cudaStream_t stream) {
    if (!mesh.clear_residual(error)) return false;
    if (!cuda_check(cudaMemsetAsync(d_failed, 0, sizeof(int), stream), "cudaMemset failed", error)) return false;
    if (start_event && !cuda_check(cudaEventRecord(start_event, stream), "cudaEventRecord start", error)) return false;

    if (reconstruction_order == 2 && !mesh.gradients_device()) {
        if (error) *error = "reconstruction_order=2 but gradients not allocated";
        return false;
    }

    DeviceFaceData fd = mesh.face_data();
    DeviceCellData cd = mesh.cell_data();
    Real a_inf = speed_of_sound(freestream, gamma);

    int block = 128;
    int nf = static_cast<int>(mesh.face_count());
    int nc = static_cast<int>(mesh.cell_count());
    bool second_order = (reconstruction_order == 2 && mesh.gradients_device() != nullptr);
    int n_colors = mesh.color_count();
    const GpuPartition* gpu_part = mesh.get_partition();
    const int* d_partition_owner = gpu_part ? gpu_part->d_partition_owner : nullptr;
    int my_rank = gpu_part ? gpu_part->my_rank : 0;

    if (n_colors > 0) {
        for (int c = 0; c < n_colors; ++c) {
            int start = mesh.host_color_offsets()[c];
            int end   = mesh.host_color_offsets()[c + 1];
            int nf_c  = end - start;
            int grid_c = (nf_c + block - 1) / block;
            euler_residual_kernel<true><<<grid_c, block, 0, stream>>>(
                fd.nx, fd.ny, fd.nz, fd.area,
                fd.left_cell, fd.right_cell, fd.boundary,
                mesh.state_device(),
                start, end, DeviceMesh::NVAR, nc,
                gamma,
                freestream.rho, freestream.p,
                freestream.u, freestream.v, freestream.w, a_inf, freestream.nu_tilde,
                mesh.residual_device(),
                d_failed,
                second_order ? mesh.gradients_device() : nullptr,
                fd.cx, fd.cy, fd.cz,
                cd.cx, cd.cy, cd.cz,
                d_partition_owner, my_rank);
            CUDA_KERNEL_CHECK("euler_residual_kernel<true>", error);
        }
    } else {
        int grid = (nf + block - 1) / block;
        euler_residual_kernel<false><<<grid, block, 0, stream>>>(
            fd.nx, fd.ny, fd.nz, fd.area,
            fd.left_cell, fd.right_cell, fd.boundary,
            mesh.state_device(),
            0, nf, DeviceMesh::NVAR, nc,
            gamma,
            freestream.rho, freestream.p,
            freestream.u, freestream.v, freestream.w, a_inf, freestream.nu_tilde,
            mesh.residual_device(),
            d_failed,
            second_order ? mesh.gradients_device() : nullptr,
            fd.cx, fd.cy, fd.cz,
            cd.cx, cd.cy, cd.cz,
            d_partition_owner, my_rank);
        CUDA_KERNEL_CHECK("euler_residual_kernel<false>", error);
    }
    return true;
}

namespace {

bool read_kernel_failed_flag(int* d_failed, std::string* error) {
    int failed = 0;
    if (!cuda_check(cudaMemcpy(&failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed", error)) return false;
    if (failed != 0) {
        if (error) *error = "GPU residual encountered invalid state";
        return false;
    }
    return true;
}

} // namespace

bool compute_euler_residual_gpu(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    int* d_failed,
    std::string* error,
    int reconstruction_order) {
    if (mesh.cell_count() == 0 || mesh.face_count() == 0) {
        if (error) *error = "DeviceMesh is not ready";
        return false;
    }
    if (!launch_euler_residual_kernel(mesh, freestream, gamma, d_failed, nullptr, error, reconstruction_order)) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "euler_residual_kernel synchronize", error)) return false;
    return read_kernel_failed_flag(d_failed, error);
}

bool compute_euler_residual_gpu(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    std::string* error,
    int reconstruction_order) {
    int* d_failed = nullptr;
    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc d_failed", error)) return false;
    bool ok = compute_euler_residual_gpu(mesh, freestream, gamma, d_failed, error, reconstruction_order);
    cuda_free_safe(d_failed);
    return ok;
}

bool compute_euler_residual_gpu_timed(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    Real* elapsed_ms,
    std::string* error,
    int reconstruction_order) {
    if (elapsed_ms) *elapsed_ms = 0.0f;
    if (mesh.cell_count() == 0 || mesh.face_count() == 0) {
        if (error) *error = "DeviceMesh is not ready";
        return false;
    }

    // PERF-G9: cache events across calls
    static cudaEvent_t s_start = nullptr;
    static cudaEvent_t s_stop = nullptr;
    if (!s_start) {
        if (!cuda_check(cudaEventCreate(&s_start), "cudaEventCreate start", error)) return false;
    }
    if (!s_stop) {
        if (!cuda_check(cudaEventCreate(&s_stop), "cudaEventCreate stop", error)) return false;
    }

    int* d_failed = nullptr;
    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc failed", error)) { return false; }
    if (!launch_euler_residual_kernel(mesh, freestream, gamma, d_failed, s_start, error, reconstruction_order)) {
        cuda_free_safe(d_failed);
        return false;
    }
    if (!cuda_check(cudaEventRecord(s_stop), "cudaEventRecord stop", error)) {
        cuda_free_safe(d_failed);
        return false;
    }
    if (!cuda_check(cudaEventSynchronize(s_stop), "cudaEventSynchronize stop", error)) {
        cuda_free_safe(d_failed);
        return false;
    }
    if (elapsed_ms) {
        if (!cuda_check(cudaEventElapsedTime(elapsed_ms, s_start, s_stop), "cudaEventElapsedTime", error)) {
            cuda_free_safe(d_failed);
            return false;
        }
    }
    bool ok = read_kernel_failed_flag(d_failed, error);
    cuda_free_safe(d_failed);
    return ok;
}

bool compute_euler_residual_gpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual,
    std::string* error) {
    DeviceMesh device_mesh;
    if (!device_mesh.upload_mesh(mesh, error)) return false;
    if (!device_mesh.upload_state(q, error)) return false;
    if (!compute_euler_residual_gpu(device_mesh, freestream, gamma, error)) return false;
    return device_mesh.download_residual(residual, error);
}

std::size_t estimate_euler_residual_gpu_bytes(const CfdMesh& mesh) {
    std::size_t face_bytes = mesh.faces.size() * 7 * sizeof(Real);
    std::size_t state_reads = 0;
    for (const auto& face : mesh.faces) {
        state_reads += DeviceMesh::NVAR * sizeof(Real);
        if (face.boundary == BoundaryKind::Interior) {
            state_reads += DeviceMesh::NVAR * sizeof(Real);
        }
    }
    std::size_t residual_writes = state_reads;
    return face_bytes + state_reads + residual_writes;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp




