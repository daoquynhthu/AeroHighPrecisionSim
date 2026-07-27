#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include <cmath>
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

namespace {

__global__ void init_update_zero_kernel(Real* d_l2_sum, int* d_failed) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_l2_sum = 0.0f;
        *d_failed = 0;
    }
}

// Persistent scratch buffer for two-stage L2 reduction.
static Real* d_partial_buf = nullptr;
static int   d_partial_cap = 0;

static bool ensure_partial_buf(int min_blocks) {
    if (d_partial_cap >= min_blocks) return true;
    cuda_free_safe(d_partial_buf);
    d_partial_cap = 0;
    if (!cuda_check(cudaMalloc(&d_partial_buf, (size_t)min_blocks * sizeof(Real)),
                    "ensure_partial_buf"))
        return false;
    d_partial_cap = min_blocks;
    return true;
}

__global__ void __launch_bounds__(256) reduce_sum_kernel(const Real* d_partial, int num_blocks, Real* result) {
    Real sum = 0;
    for (int i = 0; i < num_blocks; ++i) {
        sum += d_partial[i];
    }
    *result = sum;
}

__global__ void __launch_bounds__(256) update_and_l2_kernel(
    Real* d_q,
    const Real* d_residual,
    const Real* d_volume,
    int n_cells, int nvar, const Real* d_min_dt, const Real* d_dt_cell, Real gamma,
    Real* /* d_l2_sum -- written by reduce_sum_kernel */,
    int* d_failed,
    int* d_failure_cell, Real* d_failure_state,
    Real* d_partial) {
    extern __shared__ Real sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Real my_l2 = 0;

    if (idx < n_cells) {
        Real dt = d_dt_cell ? d_dt_cell[idx] : __ldg(d_min_dt);

        Real old_rho = d_q[idx * nvar + 0];
        Real old_rhou = d_q[idx * nvar + 1];
        Real old_rhov = d_q[idx * nvar + 2];
        Real old_rhow = d_q[idx * nvar + 3];
        Real old_rhoE = d_q[idx * nvar + 4];
        Real old_rhont = d_q[idx * nvar + 5];

        Real scale = dt / d_volume[idx];

        Real new_rho = old_rho + scale * d_residual[idx * nvar + 0];
        Real new_rhou = old_rhou + scale * d_residual[idx * nvar + 1];
        Real new_rhov = old_rhov + scale * d_residual[idx * nvar + 2];
        Real new_rhow = old_rhow + scale * d_residual[idx * nvar + 3];
        Real new_rhoE = old_rhoE + scale * d_residual[idx * nvar + 4];
        Real new_rhont = old_rhont + scale * d_residual[idx * nvar + 5];

        bool fail = !real_isfinite(new_rho) || new_rho <= 0.0f;
        if (!fail) {
            Real inv_rho = 1.0f / new_rho;
            Real u = new_rhou * inv_rho;
            Real v = new_rhov * inv_rho;
            Real w = new_rhow * inv_rho;
            Real kinetic = 0.5f * (u*u + v*v + w*w);
            Real p = (gamma - 1.0f) * (new_rhoE - new_rho * kinetic);
            fail = !real_isfinite(p) || p <= 0.0f;
        }

        if (fail) {
            int old = atomicCAS(d_failed, 0, 1);
            if (old == 0 && d_failure_cell) {
                *d_failure_cell = idx;
                d_failure_state[0] = new_rho;
                d_failure_state[1] = new_rhou;
                d_failure_state[2] = new_rhov;
                d_failure_state[3] = new_rhow;
                d_failure_state[4] = new_rhoE;
            }
        } else {
            Real dr = new_rho - old_rho;
            Real d1 = new_rhou - old_rhou;
            Real d2 = new_rhov - old_rhov;
            Real d3 = new_rhow - old_rhow;
            Real d4 = new_rhoE - old_rhoE;
            Real d5 = new_rhont - old_rhont;
            my_l2 = dr*dr + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5;

            d_q[idx * nvar + 0] = new_rho;
            d_q[idx * nvar + 1] = new_rhou;
            d_q[idx * nvar + 2] = new_rhov;
            d_q[idx * nvar + 3] = new_rhow;
            d_q[idx * nvar + 4] = new_rhoE;
            d_q[idx * nvar + 5] = new_rhont;
        }
    }

    sdata[threadIdx.x] = my_l2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}

} // namespace

__global__ void __launch_bounds__(256) mean_flow_point_implicit_kernel(
    Real* d_q, Real* d_residual, const Real* d_h_min, const Real* d_dt,
    int n_cells, int nvar, Real gamma,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re, bool sa) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    Real rho = d_q[idx * nvar + 0];
    if (!(rho > 0) || !real_isfinite(rho)) return;
    Real inv = 1.0f / rho;
    Real u = d_q[idx * nvar + 1] * inv;
    Real v = d_q[idx * nvar + 2] * inv;
    Real w = d_q[idx * nvar + 3] * inv;
    Real E = d_q[idx * nvar + 4];
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (E - rho * kinetic);
    if (!(p > 0) || !real_isfinite(p)) return;
    Real h = d_h_min[idx];
    if (!(h > 0) || !real_isfinite(h)) h = 1e-10f;
    Real vmag = real_sqrt(u*u + v*v + w*w);
    Real a = real_sqrt(gamma * p / rho);
    Real lambda = (vmag + a) / h;
    if (viscous) {
        Real T = p * inv;
        Real t_ratio = T / T_ref;
        Real mu = mu_ref * t_ratio * real_sqrt(t_ratio) * (T_ref + sutherland_T) / (T + sutherland_T);
        if (mu > 0) {
            Real nu_mol = mu / (rho * Re + 1e-30f);
            Real nu_t = 0;
            if (sa && nvar > 5) {
                Real nu = d_q[idx * nvar + 5] * inv;
                if (nu > 0) {
                    Real chi = Re * rho * nu / (mu + 1e-30f);
                    if (chi > Real(1e6)) chi = Real(1e6);
                    Real chi3 = chi*chi*chi;
                    Real cv13 = 7.1f*7.1f*7.1f;
                    Real fv1 = chi3 / (chi3 + cv13 + 1e-30f);
                    nu_t = nu * fv1;
                }
            }
            lambda += (nu_mol + real_fmax(nu_t, Real(0))) / (h * h + 1e-30f);
        }
    }
    Real dt = d_dt ? d_dt[idx] : 0;
    if (!(dt > 0)) return;
    Real f = 1.0f / (1.0f + dt * lambda + 1e-30f);
    d_residual[idx * nvar + 0] *= f;
    d_residual[idx * nvar + 1] *= f;
    d_residual[idx * nvar + 2] *= f;
    d_residual[idx * nvar + 3] *= f;
    d_residual[idx * nvar + 4] *= f;
}

bool apply_mean_flow_point_implicit_gpu(DeviceMesh& mesh, const Real* d_dt,
    Real gamma, bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re,
    bool sa, std::string* error, cudaStream_t stream) {
    int nc = static_cast<int>(mesh.cell_count());
    if (nc <= 0 || !d_dt) return true;
    int block = 128;
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();
    mean_flow_point_implicit_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), mesh.residual_device(), cd.h_min, d_dt,
        nc, mesh.nvar(), gamma, viscous, mu_ref, T_ref, sutherland_T, Re, sa);
    if (!cuda_check(cudaGetLastError(), "mean_flow_point_implicit launch", error)) return false;
    return true;
}

bool compute_update_gpu(DeviceMesh& mesh, const Real* d_min_dt, Real gamma,
    Real* d_l2_sum, int* d_failed,
    int* d_failure_cell, Real* d_failure_state,
    cudaStream_t stream, const Real* d_dt_cell) {
    init_update_zero_kernel<<<1, 1, 0, stream>>>(d_l2_sum, d_failed);
    if (!cuda_check(cudaGetLastError(), "init_update_zero kernel launch")) return false;
    if (d_failure_cell) {
        if (!cuda_check(cudaMemsetAsync(d_failure_cell, 0xFF, sizeof(int), stream), "init_failure_cell memset")) return false;
    }

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;

    if (!ensure_partial_buf(grid)) return false;

    DeviceCellData cd = mesh.cell_data();

    update_and_l2_kernel<<<grid, block, block * sizeof(Real), stream>>>(
        mesh.state_device(), mesh.residual_device(), cd.volume,
        nc, mesh.nvar(), d_min_dt, d_dt_cell, gamma,
        d_l2_sum, d_failed,
        d_failure_cell, d_failure_state,
        d_partial_buf);
    if (!cuda_check(cudaGetLastError(), "update kernel launch")) return false;
    reduce_sum_kernel<<<1, 1, 0, stream>>>(d_partial_buf, grid, d_l2_sum);
    if (!cuda_check(cudaGetLastError(), "reduce_sum kernel launch")) return false;
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp




