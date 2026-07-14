#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

namespace {

__device__ Real d_sa_vorticity(const PrimitiveGradient& g) {
    Real vx = g.dw_dy - g.dv_dz;
    Real vy = g.du_dz - g.dw_dx;
    Real vz = g.dv_dx - g.du_dy;
    return real_sqrt(vx*vx + vy*vy + vz*vz);
}

__device__ Real d_sutherland_mu(Real T, Real T_ref, Real S) {
    if (!real_isfinite(T) || T <= 0.0f) return 0.0f;
    Real t_ratio = T / T_ref;
    return t_ratio * real_sqrt(t_ratio) * (T_ref + S) / (T + S);
}

__global__ void __launch_bounds__(256) rans_source_kernel(
    Real* d_q,
    Real* d_residual,
    const Real* d_gradients,
    const Real* d_volume,
    const Real* d_wall_distance,
    const Real* d_delta_ddes,
    int n_cells, int nvar,
    Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real rho = d_q[idx * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    Real inv_rho = 1.0f / rho;
    Real u = d_q[idx * nvar + 1] * inv_rho;
    Real v = d_q[idx * nvar + 2] * inv_rho;
    Real w = d_q[idx * nvar + 3] * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (d_q[idx * nvar + 4] - rho * kinetic);
    if (!real_isfinite(p) || p <= 0.0f) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    Real T = p * inv_rho;
    Real mu = d_sutherland_mu(T, T_ref, sutherland_T);
    if (mu <= 0.0f) mu = 1.0f;

    Real wall_distance = d_wall_distance[idx];
    if (wall_distance <= 0.0f || !real_isfinite(wall_distance)) {
        wall_distance = 1e30f;
    }

    Real dest_len_scale = (d_delta_ddes != nullptr) ? d_delta_ddes[idx] : wall_distance;
    if (!real_isfinite(dest_len_scale) || dest_len_scale <= 0.0f) {
        dest_len_scale = wall_distance;
    }

    const PrimitiveGradient* g = reinterpret_cast<const PrimitiveGradient*>(d_gradients) + idx;
    Real nu_tilde = d_q[idx * nvar + 5] * inv_rho;

    if (!real_isfinite(nu_tilde)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    constexpr Real karman = 0.41f;
    constexpr Real cb1 = 0.1355f;
    constexpr Real cb2 = 0.622f;
    constexpr Real sigma = 2.0f / 3.0f;
    constexpr Real cw2 = 0.3f;
    constexpr Real cw3 = 2.0f;
    constexpr Real cv1 = 7.1f;
    constexpr Real ct3 = 1.2f;
    constexpr Real ct4 = 0.5f;
    constexpr Real cv13 = cv1 * cv1 * cv1;
    constexpr Real cw1_val = cb1 / (karman * karman) + (1.0f + cb2) / sigma;
    constexpr Real cw3_6 = cw3 * cw3 * cw3 * cw3 * cw3 * cw3;

    Real grad_nu2 = g->dnu_tilde_dx * g->dnu_tilde_dx
                 + g->dnu_tilde_dy * g->dnu_tilde_dy
                 + g->dnu_tilde_dz * g->dnu_tilde_dz
                 + 1e-30f;
    Real diffusion = (cb2 / sigma) * grad_nu2;

    Real chi = Re * rho * nu_tilde / (mu + 1e-30f);

    Real source;
    if (chi >= 0.0f) {
        Real chi3 = chi*chi*chi;
        Real fv1 = chi3 / (chi3 + cv13 + 1e-30f);

        Real vort = d_sa_vorticity(*g);
        Real fv2 = 1.0f - chi / (1.0f + chi * fv1 + 1e-30f);
        Real chi_fv2_nu = nu_tilde * fv2;
        Real inv_kd2 = 1.0f / (karman * karman * wall_distance * wall_distance + 1e-30f);
        Real omega_tilde = vort + chi_fv2_nu * inv_kd2;

        Real production = cb1 * omega_tilde * nu_tilde;

        Real r = nu_tilde / (omega_tilde * karman * karman * dest_len_scale * dest_len_scale + 1e-30f);
        if (r > 10.0f) r = 10.0f;
        Real r6 = r*r*r*r*r*r;
        Real fw_g = r + cw2 * (r6 - r);
        Real fw_num = 1.0f + cw3_6;
        Real fw_den = fw_g*fw_g*fw_g*fw_g*fw_g*fw_g + cw3_6 + 1e-30f;
        Real fw = fw_g * real_pow(fw_num / fw_den, Real(1.0 / 6.0));
        Real destruction = cw1_val * fw * (nu_tilde / dest_len_scale) * (nu_tilde / dest_len_scale);

        source = production - destruction + diffusion;
    } else {
        Real vort = d_sa_vorticity(*g);
        source = cb1 * (1.0f - ct3) * vort * nu_tilde
               + cw1_val * (nu_tilde / dest_len_scale) * (nu_tilde / dest_len_scale)
               + diffusion;
    }

    Real vol_source = rho * source;

    if (!real_isfinite(vol_source)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    d_residual[idx * nvar + 5] += vol_source;
}

// Point-implicit correction for SA destruction term.
// Modifies d_residual[5] so the explicit update produces an implicit result:
//   q_new[5] = (q_old[5] + dtv * residual[5]) / (1 + dtv * d_dest)
// Applied as: residual[5] = (q_old[5] * (implicit - 1)) / dtv + residual[5] * implicit
__global__ void __launch_bounds__(256) apply_rans_implicit_kernel(
    const Real* d_q,
    Real* d_residual,
    const Real* d_volume,
    const Real* d_wall_distance,
    const Real* d_min_dt,
    int n_cells, int nvar, Real Re) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real min_dt = __ldg(d_min_dt);
    Real rho = d_q[idx * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) return;

    Real wall_distance = d_wall_distance[idx];
    if (wall_distance <= 0.0f || !real_isfinite(wall_distance)) {
        wall_distance = 1e30f;
    }

    Real nu_tilde = d_q[idx * nvar + 5] / rho;
    if (!real_isfinite(nu_tilde)) return;

    constexpr Real cw1 = 0.1355f / (0.41f * 0.41f) + (1.0f + 0.622f) / (2.0f / 3.0f); // ~3.239

    Real d_dest = 2.0f * cw1 * nu_tilde / (wall_distance * wall_distance + 1e-30f);
    Real dt_over_V = min_dt / (d_volume[idx] + 1e-30f);
    Real implicit_factor = 1.0f / (1.0f + dt_over_V * d_dest + 1e-30f);

    Real old_rhont = d_q[idx * nvar + 5];
    Real old_residual = d_residual[idx * nvar + 5];
    d_residual[idx * nvar + 5] = (old_rhont * (implicit_factor - 1.0f)) / (dt_over_V + 1e-30f)
                               + old_residual * implicit_factor;
}

__global__ void __launch_bounds__(256) apply_rans_implicit_per_cell_kernel(
    const Real* d_q,
    Real* d_residual,
    const Real* d_volume,
    const Real* d_wall_distance,
    const Real* d_dt_cell,
    int n_cells, int nvar, Real Re) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real rho = d_q[idx * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) return;

    Real wall_distance = d_wall_distance[idx];
    if (wall_distance <= 0.0f || !real_isfinite(wall_distance)) {
        wall_distance = 1e30f;
    }

    Real nu_tilde = d_q[idx * nvar + 5] / rho;
    if (!real_isfinite(nu_tilde)) return;

    constexpr Real cw1 = 0.1355f / (0.41f * 0.41f) + (1.0f + 0.622f) / (2.0f / 3.0f); // ~3.239

    Real d_dest = 2.0f * cw1 * nu_tilde / (wall_distance * wall_distance + 1e-30f);
    Real dt_over_V = d_dt_cell[idx] / (d_volume[idx] + 1e-30f);
    Real implicit_factor = 1.0f / (1.0f + dt_over_V * d_dest + 1e-30f);

    Real old_rhont = d_q[idx * nvar + 5];
    Real old_residual = d_residual[idx * nvar + 5];
    d_residual[idx * nvar + 5] = (old_rhont * (implicit_factor - 1.0f)) / (dt_over_V + 1e-30f)
                               + old_residual * implicit_factor;
}

} // namespace

bool apply_rans_implicit_gpu(DeviceMesh& mesh, Real Re,
    const Real* d_min_dt, std::string* error, cudaStream_t stream) {
    if (mesh.cell_count() == 0) return true;
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    apply_rans_implicit_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), mesh.residual_device(), cd.volume,
        cd.wall_distance, d_min_dt,
        nc, DeviceMesh::NVAR, Re);
    if (!cuda_check(cudaGetLastError(), "apply_rans_implicit_kernel launch", error)) return false;
    return true;
}

bool apply_rans_implicit_per_cell_gpu(DeviceMesh& mesh, Real Re,
    const Real* d_dt_cell, std::string* error, cudaStream_t stream) {
    if (mesh.cell_count() == 0) return true;
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    apply_rans_implicit_per_cell_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), mesh.residual_device(), cd.volume,
        cd.wall_distance, d_dt_cell,
        nc, DeviceMesh::NVAR, Re);
    if (!cuda_check(cudaGetLastError(), "apply_rans_implicit_per_cell_kernel launch", error)) return false;
    return true;
}

bool compute_rans_source_gpu(DeviceMesh& mesh, Real gamma, Real Re, Real mu_ref, Real T_ref, Real sutherland_T, const Real* d_delta_ddes, int* d_failed, std::string* error, cudaStream_t stream) {
    if (mesh.cell_count() == 0) return true;
    if (!mesh.gradients_device()) {
        if (error) *error = "gradients not allocated for RANS source";
        return false;
    }

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    rans_source_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(),
        mesh.residual_device(),
        mesh.gradients_device(),
        cd.volume, cd.wall_distance,
        d_delta_ddes,
        nc, DeviceMesh::NVAR, gamma, Re,
        mu_ref, T_ref, sutherland_T,
        d_failed);
    if (!cuda_check(cudaGetLastError(), "rans_source_kernel launch")) return false;
    return true;
}

bool compute_turbulence_source_gpu(DeviceMesh& mesh, const CfdConfig& config,
    int* d_failed, std::string* error, cudaStream_t stream,
    Real inf_k, Real inf_omega, Real inf_rho,
    const Real* d_min_dt) {
    if (config.turbulence_model == TurbulenceModel::LAMINAR) return true;

    if (config.turbulence_model == TurbulenceModel::SST) {
        if (!mesh.has_sst()) {
            if (error) *error = "SST buffers not allocated";
            return false;
        }
        if (!clear_sst_residual_gpu(mesh, error, stream)) return false;
        if (!compute_sst_gradients_gpu(mesh, d_failed, error, stream)) return false;
        if (!compute_sst_advection_gpu(mesh, inf_k, inf_omega, inf_rho, d_failed, error, stream)) return false;
        if (!compute_sst_diffusion_gpu(mesh, config.gamma,
                config.mu_ref, config.T_ref, config.sutherland_T,
                d_failed, error, stream)) return false;
        if (!compute_sst_source_gpu(mesh, config.gamma, config.Re,
                config.mu_ref, config.T_ref, config.sutherland_T,
                d_failed, error, stream)) return false;
        if (d_min_dt) {
            if (!compute_sst_update_gpu(mesh, d_min_dt, error, stream)) return false;
        }
        return true;
    }

    const Real* d_delta = nullptr;
    if (config.turbulence_model == TurbulenceModel::SA_DDES) {
        if (!mesh.has_delta_ddes() && !mesh.allocate_ddes()) {
            if (error) *error = "allocate_ddes failed in turbulence source dispatch";
            return false;
        }
        if (!compute_ddes_length_scale_gpu(mesh, config.gamma, config.Re,
                config.mu_ref, config.T_ref, config.sutherland_T, d_failed, error, stream)) {
            return false;
        }
        d_delta = mesh.delta_ddes_device();
    }

    if (!compute_rans_source_gpu(mesh, config.gamma, config.Re,
            config.mu_ref, config.T_ref, config.sutherland_T,
            d_delta, d_failed, error, stream)) {
        return false;
    }

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp