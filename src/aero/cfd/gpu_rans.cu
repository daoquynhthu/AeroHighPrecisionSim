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
    constexpr Real kWdFar = 1.0e10f;
    if (wall_distance <= 0.0f || !real_isfinite(wall_distance) || wall_distance > kWdFar) {
        wall_distance = kWdFar;
    }

    Real dest_len_scale = (d_delta_ddes != nullptr) ? d_delta_ddes[idx] : wall_distance;
    if (!real_isfinite(dest_len_scale) || dest_len_scale <= 0.0f || dest_len_scale > kWdFar) {
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
    if (chi > Real(1.0e6)) chi = Real(1.0e6);
    if (chi < Real(-1.0e6)) chi = Real(-1.0e6);
    Real d = dest_len_scale;
    Real d2 = d * d + 1e-30f;

    constexpr Real sa_r_max = 200.0f;
    Real r_nd = real_fabs(nu_tilde) / (d + 1e-30f);
    Real stiff_scale = 1.0f;
    if (r_nd > sa_r_max && r_nd > 0.0f) {
        Real r_cap = sa_r_max / r_nd;
        stiff_scale = r_cap * r_cap;
    }
    diffusion *= stiff_scale;

    Real source;
    Real dS_dnu;
    if (chi >= 0.0f) {
        Real chi3 = chi*chi*chi;
        Real fv1 = chi3 / (chi3 + cv13 + 1e-30f);

        Real vort = d_sa_vorticity(*g);
        Real fv2 = 1.0f - chi / (1.0f + chi * fv1 + 1e-30f);
        Real chi_fv2_nu = nu_tilde * fv2;
        Real inv_kd2 = 1.0f / (karman * karman * wall_distance * wall_distance + 1e-30f);
        Real omega_tilde = vort + chi_fv2_nu * inv_kd2;
        Real s_floor = Real(0.3) * vort;
        if (omega_tilde < s_floor) omega_tilde = s_floor;

        Real production = cb1 * omega_tilde * nu_tilde;

        Real r = nu_tilde / (omega_tilde * karman * karman * dest_len_scale * dest_len_scale + 1e-30f);
        if (r < 0.0f) r = 0.0f;
        if (r > 10.0f) r = 10.0f;
        Real r6 = r*r*r*r*r*r;
        Real fw_g = r + cw2 * (r6 - r);
        Real fw_num = 1.0f + cw3_6;
        Real fw_den = fw_g*fw_g*fw_g*fw_g*fw_g*fw_g + cw3_6 + 1e-30f;
        Real fw = fw_g * real_pow(fw_num / fw_den, Real(1.0 / 6.0));
        Real destruction = cw1_val * fw * (nu_tilde / dest_len_scale) * (nu_tilde / dest_len_scale) * stiff_scale;

        source = production - destruction + diffusion;
        dS_dnu = cb1 * omega_tilde - 2.0f * cw1_val * fw * nu_tilde / d2 * stiff_scale;
    } else {
        Real vort = d_sa_vorticity(*g);
        source = cb1 * (1.0f - ct3) * vort * nu_tilde
               + cw1_val * (nu_tilde / dest_len_scale) * (nu_tilde / dest_len_scale) * stiff_scale
               + diffusion;
        dS_dnu = cb1 * (1.0f - ct3) * vort + 2.0f * cw1_val * nu_tilde / d2 * stiff_scale;
    }

    {
        Real nu_mol = mu / (rho * Re + 1e-30f);
        Real nu_scale = real_fmax(real_fabs(nu_tilde), real_fmax(nu_mol, Real(1e-12)));
        Real vort = d_sa_vorticity(*g);
        Real S_cap = 10.0f * nu_scale * real_fmax(vort, Real(1e-6))
                   + 10.0f * cw1_val * nu_scale * nu_scale / d2;
        if (source > S_cap) source = S_cap;
        if (source < -S_cap) source = -S_cap;
    }

    Real vol_source = rho * source;

    if (!real_isfinite(vol_source)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    d_residual[idx * nvar + 5] += vol_source;
    (void)dS_dnu;
}

// Device helper: g = max(0,-∂S/∂nu) + destruction floor + viscous spectral radius.
__device__ Real d_sa_damping_rate(
    Real nu_tilde, Real rho, Real mu, Real Re,
    Real wall_distance, Real dest_len_scale,
    const PrimitiveGradient& g,
    Real h_min = 0.0f)
{
    constexpr Real karman = 0.41f;
    constexpr Real cb1 = 0.1355f;
    constexpr Real cb2 = 0.622f;
    constexpr Real sigma = 2.0f / 3.0f;
    constexpr Real cw2 = 0.3f;
    constexpr Real cw3 = 2.0f;
    constexpr Real cv1 = 7.1f;
    constexpr Real ct3 = 1.2f;
    constexpr Real cv13 = cv1 * cv1 * cv1;
    constexpr Real cw1_val = cb1 / (karman * karman) + (1.0f + cb2) / sigma;
    constexpr Real cw3_6 = cw3 * cw3 * cw3 * cw3 * cw3 * cw3;
    constexpr Real sa_r_max = 200.0f;

    constexpr Real kWdFar = 1.0e10f;
    Real d = dest_len_scale;
    if (!(d > 0) || !real_isfinite(d)) d = wall_distance;
    if (!(d > 0) || !real_isfinite(d) || d > kWdFar) d = kWdFar;
    Real d2 = d * d + 1e-30f;
    Real r_nd = real_fabs(nu_tilde) / (d + 1e-30f);
    Real stiff_scale = 1.0f;
    if (r_nd > sa_r_max && r_nd > 0.0f) {
        Real r_cap = sa_r_max / r_nd;
        stiff_scale = r_cap * r_cap;
    }

    Real chi = Re * rho * nu_tilde / (mu + 1e-30f);
    if (chi > Real(1.0e6)) chi = Real(1.0e6);
    if (chi < Real(-1.0e6)) chi = Real(-1.0e6);
    Real dS_dnu;
    if (chi >= 0.0f) {
        Real chi3 = chi * chi * chi;
        Real fv1 = chi3 / (chi3 + cv13 + 1e-30f);
        Real vort = d_sa_vorticity(g);
        Real fv2 = 1.0f - chi / (1.0f + chi * fv1 + 1e-30f);
        Real inv_kd2 = 1.0f / (karman * karman * wall_distance * wall_distance + 1e-30f);
        Real omega_tilde = vort + nu_tilde * fv2 * inv_kd2;
        Real s_floor = Real(0.3) * vort;
        if (omega_tilde < s_floor) omega_tilde = s_floor;
        Real r = nu_tilde / (omega_tilde * karman * karman * d * d + 1e-30f);
        if (r < 0.0f) r = 0.0f;
        if (r > 10.0f) r = 10.0f;
        Real r6 = r*r*r*r*r*r;
        Real fw_g = r + cw2 * (r6 - r);
        Real fw_num = 1.0f + cw3_6;
        Real fw_den = fw_g*fw_g*fw_g*fw_g*fw_g*fw_g + cw3_6 + 1e-30f;
        Real fw = fw_g * real_pow(fw_num / fw_den, Real(1.0 / 6.0));
        dS_dnu = cb1 * omega_tilde - 2.0f * cw1_val * fw * nu_tilde / d2 * stiff_scale;
    } else {
        Real vort = d_sa_vorticity(g);
        dS_dnu = cb1 * (1.0f - ct3) * vort + 2.0f * cw1_val * nu_tilde / d2 * stiff_scale;
    }

    Real g_damp = (dS_dnu < 0) ? -dS_dnu : 0.0f;
    Real nu_mol = mu / (rho * Re + 1e-30f);
    Real nu_ref = real_fmax(real_fabs(nu_tilde), real_fmax(nu_mol, Real(1e-30)));
    Real g_dest = 2.0f * cw1_val * nu_ref / d2;
    if (g_damp < g_dest) g_damp = g_dest;

    Real h = d;
    if (h_min > 0.0f && real_isfinite(h_min)) h = real_fmin(h, h_min);
    if (!(h > 0) || !real_isfinite(h) || h > kWdFar) h = kWdFar;
    Real nu_eff = real_fmax(nu_mol, Real(0)) + real_fmax(nu_tilde, Real(0));
    Real g_diff = nu_eff / (sigma * h * h + 1e-30f);
    if (g_damp < g_diff) g_damp = g_diff;
    return g_damp;
}

// Point-implicit SA: g = max(0,-∂S/∂nu) [1/time], beta = V*g [volume/time].
// q_new = (q + dtv*R)/(1 + dtv*beta)  with  dtv*beta = dt*g.
// Prior bug: beta omitted volume → wall cells (small V) almost undamped.
__global__ void __launch_bounds__(256) apply_rans_implicit_kernel(
    const Real* d_q,
    Real* d_residual,
    const Real* d_volume,
    const Real* d_wall_distance,
    const Real* d_h_min,
    const Real* d_gradients,
    const Real* d_min_dt,
    int n_cells, int nvar,
    Real gamma, Real Re, Real mu_ref, Real T_ref, Real sutherland_T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real min_dt = __ldg(d_min_dt);
    Real rho = d_q[idx * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) return;

    Real inv_rho = 1.0f / rho;
    Real u = d_q[idx * nvar + 1] * inv_rho;
    Real v = d_q[idx * nvar + 2] * inv_rho;
    Real w = d_q[idx * nvar + 3] * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (d_q[idx * nvar + 4] - rho * kinetic);
    if (!(p > 0) || !real_isfinite(p)) return;
    Real T = p * inv_rho;
    Real mu = d_sutherland_mu(T, T_ref, sutherland_T);
    if (mu <= 0.0f) mu = 1.0f;

    Real wall_distance = d_wall_distance[idx];
    constexpr Real kWdFar = 1.0e10f;
    if (wall_distance <= 0.0f || !real_isfinite(wall_distance) || wall_distance > kWdFar)
        wall_distance = kWdFar;
    Real h_min = d_h_min ? d_h_min[idx] : 0.0f;

    Real nu_tilde = d_q[idx * nvar + 5] * inv_rho;
    if (!real_isfinite(nu_tilde)) return;

    const PrimitiveGradient* pg =
        reinterpret_cast<const PrimitiveGradient*>(d_gradients) + idx;
    Real g_damp = d_sa_damping_rate(nu_tilde, rho, mu, Re, wall_distance,
                                    wall_distance, *pg, h_min);

    Real V = d_volume[idx] + 1e-30f;
    Real beta = V * g_damp;
    Real dtv = min_dt / V;
    Real f = 1.0f / (1.0f + dtv * beta + 1e-30f);

    Real old_rhont = d_q[idx * nvar + 5];
    Real old_residual = d_residual[idx * nvar + 5];
    d_residual[idx * nvar + 5] = (old_rhont * (f - 1.0f)) / (dtv + 1e-30f)
                               + old_residual * f;
}

__global__ void __launch_bounds__(256) apply_rans_implicit_per_cell_kernel(
    const Real* d_q,
    Real* d_residual,
    const Real* d_volume,
    const Real* d_wall_distance,
    const Real* d_h_min,
    const Real* d_gradients,
    const Real* d_dt_cell,
    int n_cells, int nvar,
    Real gamma, Real Re, Real mu_ref, Real T_ref, Real sutherland_T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real dt = d_dt_cell[idx];
    Real rho = d_q[idx * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) return;

    Real inv_rho = 1.0f / rho;
    Real u = d_q[idx * nvar + 1] * inv_rho;
    Real v = d_q[idx * nvar + 2] * inv_rho;
    Real w = d_q[idx * nvar + 3] * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (d_q[idx * nvar + 4] - rho * kinetic);
    if (!(p > 0) || !real_isfinite(p)) return;
    Real T = p * inv_rho;
    Real mu = d_sutherland_mu(T, T_ref, sutherland_T);
    if (mu <= 0.0f) mu = 1.0f;

    Real wall_distance = d_wall_distance[idx];
    constexpr Real kWdFar = 1.0e10f;
    if (wall_distance <= 0.0f || !real_isfinite(wall_distance) || wall_distance > kWdFar)
        wall_distance = kWdFar;
    Real h_min = d_h_min ? d_h_min[idx] : 0.0f;

    Real nu_tilde = d_q[idx * nvar + 5] * inv_rho;
    if (!real_isfinite(nu_tilde)) return;

    const PrimitiveGradient* pg =
        reinterpret_cast<const PrimitiveGradient*>(d_gradients) + idx;
    Real g_damp = d_sa_damping_rate(nu_tilde, rho, mu, Re, wall_distance,
                                    wall_distance, *pg, h_min);

    Real V = d_volume[idx] + 1e-30f;
    Real beta = V * g_damp;
    Real dtv = dt / V;
    Real f = 1.0f / (1.0f + dtv * beta + 1e-30f);

    Real old_rhont = d_q[idx * nvar + 5];
    Real old_residual = d_residual[idx * nvar + 5];
    d_residual[idx * nvar + 5] = (old_rhont * (f - 1.0f)) / (dtv + 1e-30f)
                               + old_residual * f;
}

} // namespace

bool apply_rans_implicit_gpu(DeviceMesh& mesh, Real Re,
    const Real* d_min_dt, std::string* error, cudaStream_t stream) {
    if (mesh.cell_count() == 0) return true;
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    // Defaults match CfdConfig when caller uses apply_rans_implicit_gpu(mesh, Re, dt)
    // without thermo args — gamma etc. from typical freestream.
    Real gamma = 1.4f, mu_ref = 1.0f, T_ref = 288.15f, sutherland_T = 110.4f;
    apply_rans_implicit_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), mesh.residual_device(), cd.volume,
        cd.wall_distance, cd.h_min, mesh.gradients_device(), d_min_dt,
        nc, mesh.nvar(), gamma, Re, mu_ref, T_ref, sutherland_T);
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

    Real gamma = 1.4f, mu_ref = 1.0f, T_ref = 288.15f, sutherland_T = 110.4f;
    apply_rans_implicit_per_cell_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), mesh.residual_device(), cd.volume,
        cd.wall_distance, cd.h_min, mesh.gradients_device(), d_dt_cell,
        nc, mesh.nvar(), gamma, Re, mu_ref, T_ref, sutherland_T);
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
        nc, mesh.nvar(), gamma, Re,
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
        if (!compute_sst_gradients_gpu(mesh, d_failed, error, stream)) return false;
        if (!compute_sst_advection_gpu(mesh, inf_k, inf_omega, inf_rho, d_failed, error, stream)) return false;
        if (!compute_sst_diffusion_gpu(mesh, config.gamma,
                config.mu_ref, config.T_ref, config.sutherland_T,
                d_failed, error, stream)) return false;
        if (!compute_sst_source_gpu(mesh, config.gamma,
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