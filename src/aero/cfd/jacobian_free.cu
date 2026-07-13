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

} // namespace

bool compute_jfv_product(DeviceMesh& mesh, const Real* d_v, Real* d_result,
    const Real* d_residual, Real epsilon, const CfdConfig& config,
    const PrimitiveState& w_inf, Real* d_scratch, int* d_failed,
    std::string* error, cudaStream_t stream) {
    int n = static_cast<int>(mesh.cell_count());
    int nvar = DeviceMesh::NVAR;
    int nvar_cells = n * nvar;

    int block = 128;
    int grid = (nvar_cells + block - 1) / block;
    if (grid < 1) grid = 1;

    Real* d_q_pert = d_scratch;

    perturb_kernel<<<grid, block, 0, stream>>>(d_q_pert, mesh.state_device(), d_v, epsilon, n, nvar);
    if (!cuda_check(cudaGetLastError(), "perturb kernel", error)) return false;

    Real* d_q_orig = mesh.state_device();
    mesh.set_state_device(d_q_pert);

    if (!launch_euler_residual_kernel(mesh, w_inf, config.gamma, d_failed, nullptr, error,
            config.reconstruction_order, stream)) {
        mesh.set_state_device(d_q_orig);
        return false;
    }

    if (config.viscous) {
        if (!compute_viscous_flux_gpu(mesh, config.gamma, config.prandtl,
                config.mu_ref, config.T_ref, config.sutherland_T,
                config.Re, config.wall_temperature, static_cast<int>(config.turbulence_model), d_failed, stream)) {
            mesh.set_state_device(d_q_orig);
            if (error) *error = "JFV viscous flux failed";
            return false;
        }
    }

    if (config.turbulence_model != TurbulenceModel::LAMINAR) {
        if (!compute_turbulence_source_gpu(mesh, config, d_failed, error, stream)) {
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

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
