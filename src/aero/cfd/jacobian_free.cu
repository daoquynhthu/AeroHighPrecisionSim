#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include "aero/cfd/real.hpp"

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

__global__ void __launch_bounds__(256) perturb_kernel(Real* d_q_pert, const Real* d_q,
    const Real* d_v, Real eps, int n, int nvar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * nvar) return;
    d_q_pert[idx] = d_q[idx] + eps * d_v[idx];
}

__global__ void __launch_bounds__(256) jfv_result_kernel(Real* d_result, const Real* d_residual_pert,
    const Real* d_residual, Real inv_eps, int n, int nvar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * nvar) return;
    d_result[idx] = (d_residual_pert[idx] - d_residual[idx]) * inv_eps;
}

__global__ void __launch_bounds__(256) perturb_sst_kernel(Real* d_q_k, Real* d_q_omega,
    const Real* d_v_sst, Real eps, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_q_k[idx] += eps * d_v_sst[idx];
    d_q_omega[idx] += eps * d_v_sst[n + idx];
}

__global__ void __launch_bounds__(256) sst_jfv_result_kernel(Real* d_result_sst,
    const Real* d_residual_k_pert, const Real* d_residual_k_saved,
    const Real* d_residual_omega_pert, const Real* d_residual_omega_saved,
    Real inv_eps, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_result_sst[idx] = (d_residual_k_pert[idx] - d_residual_k_saved[idx]) * inv_eps;
    d_result_sst[n + idx] = (d_residual_omega_pert[idx] - d_residual_omega_saved[idx]) * inv_eps;
}

} // namespace

bool compute_jfv_product(DeviceMesh& mesh, const Real* d_v, Real* d_result,
    const Real* d_residual, Real epsilon, const CfdConfig& config,
    const PrimitiveState& w_inf, Real* d_scratch, int* d_failed,
    std::string* error, cudaStream_t stream,
    const Real* d_v_sst, Real* d_result_sst) {
    int n = static_cast<int>(mesh.cell_count());
    int nvar = mesh.nvar();
    int nvar_cells = n * nvar;

    bool sst_active = (d_v_sst != nullptr && d_result_sst != nullptr
        && config.turbulence_model == TurbulenceModel::SST && mesh.has_sst());

    int block = 128;
    int grid = (nvar_cells + block - 1) / block;
    if (grid < 1) grid = 1;

    Real* d_q_pert = d_scratch;

    perturb_kernel<<<grid, block, 0, stream>>>(d_q_pert, mesh.state_device(), d_v, epsilon, n, nvar);
    if (!cuda_check(cudaGetLastError(), "perturb kernel", error)) return false;

    Real* d_q_orig = mesh.state_device();
    mesh.set_state_device(d_q_pert);

    // Save SST baseline state and residuals if SST perturbation is active
    Real* d_q_k_saved = nullptr;
    Real* d_q_omega_saved = nullptr;
    Real* d_residual_k_saved = nullptr;
    Real* d_residual_omega_saved = nullptr;
    if (sst_active) {
        d_q_k_saved = d_scratch + nvar_cells;
        d_q_omega_saved = d_scratch + nvar_cells + n;
        d_residual_k_saved = d_scratch + nvar_cells + 2 * n;
        d_residual_omega_saved = d_scratch + nvar_cells + 3 * n;

        if (cudaMemcpy(d_q_k_saved, mesh.q_k_device(), n * sizeof(Real), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            mesh.set_state_device(d_q_orig);
            if (error) *error = "JFV SST save k failed";
            return false;
        }
        if (cudaMemcpy(d_q_omega_saved, mesh.q_omega_device(), n * sizeof(Real), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            mesh.set_state_device(d_q_orig);
            if (error) *error = "JFV SST save omega failed";
            return false;
        }
        if (cudaMemcpy(d_residual_k_saved, mesh.residual_k_device(), n * sizeof(Real), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            mesh.set_state_device(d_q_orig);
            if (error) *error = "JFV SST save residual_k failed";
            return false;
        }
        if (cudaMemcpy(d_residual_omega_saved, mesh.residual_omega_device(), n * sizeof(Real), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            mesh.set_state_device(d_q_orig);
            if (error) *error = "JFV SST save residual_omega failed";
            return false;
        }

        int sst_grid = (n + block - 1) / block;
        if (sst_grid < 1) sst_grid = 1;
        perturb_sst_kernel<<<sst_grid, block, 0, stream>>>(
            mesh.q_k_device(), mesh.q_omega_device(), d_v_sst, epsilon, n);
        if (!cuda_check(cudaGetLastError(), "perturb SST kernel", error)) {
            mesh.set_state_device(d_q_orig);
            return false;
        }
    }

    if (!launch_euler_residual_kernel(mesh, w_inf, config.gamma, d_failed, nullptr, error,
            config.reconstruction_order, stream)) {
        if (sst_active) {
            cudaMemcpy(mesh.q_k_device(), d_q_k_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(mesh.q_omega_device(), d_q_omega_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(mesh.residual_k_device(), d_residual_k_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(mesh.residual_omega_device(), d_residual_omega_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
        }
        mesh.set_state_device(d_q_orig);
        return false;
    }
    {   int jfv_fail = 0;
        cudaMemcpy(&jfv_fail, d_failed, sizeof(int), cudaMemcpyDeviceToHost);
        if (jfv_fail) {
            // Perturbed state had invalid cells; this Krylov direction is unusable.
            // Zero the result (w = 0) so FGMRES can continue with other directions.
            cudaMemsetAsync(d_result, 0, nvar_cells * sizeof(Real), stream);
            mesh.set_state_device(d_q_orig);
            return true;
        }
    }

    if (config.viscous) {
        if (!compute_viscous_flux_gpu(mesh, config.gamma, config.prandtl,
                config.mu_ref, config.T_ref, config.sutherland_T,
                config.Re, config.wall_temperature, static_cast<int>(config.turbulence_model), d_failed, stream)) {
            if (sst_active) {
                cudaMemcpy(mesh.q_k_device(), d_q_k_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
                cudaMemcpy(mesh.q_omega_device(), d_q_omega_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
                cudaMemcpy(mesh.residual_k_device(), d_residual_k_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
                cudaMemcpy(mesh.residual_omega_device(), d_residual_omega_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
            }
            mesh.set_state_device(d_q_orig);
            if (error) *error = "JFV viscous flux failed";
            return false;
        }
    }

    Real jfv_inf_k = 0.0f, jfv_inf_omega = 0.0f;
    if (config.turbulence_model == TurbulenceModel::SST) {
        Real U_inf = real_sqrt(w_inf.u * w_inf.u + w_inf.v * w_inf.v + w_inf.w * w_inf.w);
        Real tu = 0.001f;
        jfv_inf_k = 1.5f * (tu * tu) * U_inf * U_inf;
        Real T_inf = w_inf.p / w_inf.rho;
        Real t_ratio = T_inf / config.T_ref;
        Real mu_inf = config.mu_ref * t_ratio * real_sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T_inf + config.sutherland_T);
        Real nu_inf = mu_inf / w_inf.rho;
        Real mu_t_mu_ratio = 0.1f;
        Real nu_t_inf = mu_t_mu_ratio * nu_inf;
        jfv_inf_omega = jfv_inf_k / (0.09f * nu_t_inf + 1e-30f);
    }
    if (config.turbulence_model != TurbulenceModel::LAMINAR) {
        if (!compute_turbulence_source_gpu(mesh, config, d_failed, error, stream, jfv_inf_k, jfv_inf_omega)) {
            if (sst_active) {
                cudaMemcpy(mesh.q_k_device(), d_q_k_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
                cudaMemcpy(mesh.q_omega_device(), d_q_omega_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
                cudaMemcpy(mesh.residual_k_device(), d_residual_k_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
                cudaMemcpy(mesh.residual_omega_device(), d_residual_omega_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice);
            }
            mesh.set_state_device(d_q_orig);
            if (error && error->empty()) *error = "JFV turbulence source failed";
            return false;
        }
    }

    Real* d_residual_pert = mesh.residual_device();
    mesh.set_state_device(d_q_orig);

    Real inv_eps = 1.0f / epsilon;
    jfv_result_kernel<<<grid, block, 0, stream>>>(d_result, d_residual_pert, d_residual, inv_eps, n, nvar);
    if (!cuda_check(cudaGetLastError(), "jfv result kernel", error)) return false;

    if (sst_active) {
        int sst_grid = (n + block - 1) / block;
        if (sst_grid < 1) sst_grid = 1;
        sst_jfv_result_kernel<<<sst_grid, block, 0, stream>>>(
            d_result_sst, mesh.residual_k_device(), d_residual_k_saved,
            mesh.residual_omega_device(), d_residual_omega_saved, inv_eps, n);
        if (!cuda_check(cudaGetLastError(), "sst jfv result kernel", error)) return false;

        if (cudaMemcpy(mesh.q_k_device(), d_q_k_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            if (error) *error = "JFV SST restore k failed";
            return false;
        }
        if (cudaMemcpy(mesh.q_omega_device(), d_q_omega_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            if (error) *error = "JFV SST restore omega failed";
            return false;
        }
        if (cudaMemcpy(mesh.residual_k_device(), d_residual_k_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            if (error) *error = "JFV SST restore residual_k failed";
            return false;
        }
        if (cudaMemcpy(mesh.residual_omega_device(), d_residual_omega_saved, n * sizeof(Real), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            if (error) *error = "JFV SST restore residual_omega failed";
            return false;
        }
    }

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
