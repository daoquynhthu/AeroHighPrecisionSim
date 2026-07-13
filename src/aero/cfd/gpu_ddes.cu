#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

namespace {

__device__ Real ddes_sutherland_mu(Real T, Real T_ref, Real S) {
    if (!real_isfinite(T) || T <= 0.0f) return 0.0f;
    Real t_ratio = T / T_ref;
    return t_ratio * real_sqrt(t_ratio) * (T_ref + S) / (T + S);
}

__global__ void __launch_bounds__(256) ddes_length_scale_kernel(
    const Real* d_q,
    const Real* d_gradients,
    const Real* d_wall_distance,
    const Real* d_volume,
    Real* d_delta_ddes,
    int n_cells, int nvar,
    Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real rho = d_q[idx * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) {
        d_delta_ddes[idx] = d_wall_distance[idx];
        return;
    }
    Real inv_rho = 1.0f / rho;
    Real u = d_q[idx * nvar + 1] * inv_rho;
    Real v = d_q[idx * nvar + 2] * inv_rho;
    Real w = d_q[idx * nvar + 3] * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (d_q[idx * nvar + 4] - rho * kinetic);
    if (!real_isfinite(p) || p <= 0.0f) {
        d_delta_ddes[idx] = d_wall_distance[idx];
        return;
    }

    Real T = p * inv_rho;
    Real mu = mu_ref * ddes_sutherland_mu(T, T_ref, sutherland_T);
    if (mu <= 0.0f) mu = mu_ref;

    Real wall_distance = d_wall_distance[idx];
    if (wall_distance <= 0.0f || !real_isfinite(wall_distance)) {
        wall_distance = 1e30f;
    }

    const PrimitiveGradient* g = reinterpret_cast<const PrimitiveGradient*>(d_gradients) + idx;
    Real nu_tilde = d_q[idx * nvar + 5] * inv_rho;
    if (!real_isfinite(nu_tilde)) {
        d_delta_ddes[idx] = wall_distance;
        return;
    }

    constexpr Real karman = 0.41f;
    Real nu = mu / (Re * rho + 1e-30f);

    Real uij_sq = g->du_dx * g->du_dx + g->du_dy * g->du_dy + g->du_dz * g->du_dz
                + g->dv_dx * g->dv_dx + g->dv_dy * g->dv_dy + g->dv_dz * g->dv_dz
                + g->dw_dx * g->dw_dx + g->dw_dy * g->dw_dy + g->dw_dz * g->dw_dz;

    Real rd = (nu_tilde + nu) / (real_sqrt(uij_sq) * karman * karman * wall_distance * wall_distance + 1e-30f);
    Real rd3 = rd * rd * rd;
    Real fd_arg = 8.0f * rd3;
    fd_arg = fminf(fd_arg, 80.0f);
    Real fd = tanh(fd_arg);

    constexpr Real C_DES = 0.65f;
    Real volume = d_volume[idx];
    if (volume <= 0.0f || !real_isfinite(volume)) volume = 1e-30f;
    Real h_max = powf(volume, 1.0f / 3.0f);

    Real delta_ddes = wall_distance - fd * fmaxf(0.0f, wall_distance - C_DES * h_max);

    if (!real_isfinite(delta_ddes)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        d_delta_ddes[idx] = wall_distance;
        return;
    }

    d_delta_ddes[idx] = delta_ddes;
}

} // namespace

bool compute_ddes_length_scale_gpu(DeviceMesh& mesh, Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T, int* d_failed,
    std::string* error, cudaStream_t stream) {
    if (mesh.cell_count() == 0) return true;
    if (!mesh.gradients_device()) {
        if (error) *error = "gradients not allocated for DDES length scale";
        return false;
    }

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    ddes_length_scale_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(),
        mesh.gradients_device(),
        cd.wall_distance,
        cd.volume,
        mesh.delta_ddes_device(),
        nc, DeviceMesh::NVAR, gamma, Re,
        mu_ref, T_ref, sutherland_T,
        d_failed);
    if (!cuda_check(cudaGetLastError(), "ddes_length_scale_kernel launch")) return false;
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
