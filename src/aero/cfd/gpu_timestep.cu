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

__global__ void __launch_bounds__(256) timestep_kernel(
    const Real* d_q, int n_cells, int nvar, Real gamma, Real cfl,
    const Real* d_volume, const Real* d_h_min,
    Real* d_partial,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re) {
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
                Real vmag = real_sqrt(u*u + v*v + w*w);
                Real a = real_sqrt(gamma * p / rho);
                Real denom = vmag + a;
                dt = cfl * h / (denom > Real(1e-30) ? denom : Real(1e-30));
                if (viscous) {
                    Real T = p / rho;
                    if (T > 0.0f) {
                        Real t_ratio = T / T_ref;
                        Real mu_cell = mu_ref * t_ratio * real_sqrt(t_ratio) * (T_ref + sutherland_T) / (T + sutherland_T);
                        if (mu_cell > 0.0f) {
                            Real dt_visc = cfl * rho * h * h * Re / mu_cell;
                            if (dt_visc < dt) dt = dt_visc;
                        }
                    }
                }
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
    const Real* d_volume, const Real* d_h_min,
    Real* d_dt_cell,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re) {
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
    Real vmag = real_sqrt(u*u + v*v + w*w);
    Real a = real_sqrt(gamma * p / rho);
    Real denom = vmag + a;
    Real dt = cfl * h / (denom > std::numeric_limits<Real>::min() ? denom : std::numeric_limits<Real>::min());

    if (viscous) {
        Real T = p / rho;
        if (T > 0.0f) {
            Real t_ratio = T / T_ref;
            Real mu_cell = mu_ref * t_ratio * real_sqrt(t_ratio) * (T_ref + sutherland_T) / (T + sutherland_T);
            if (mu_cell > 0.0f) {
                Real dt_visc = cfl * rho * h * h * Re / mu_cell;
                if (dt_visc < dt) dt = dt_visc;
            }
        }
    }
    d_dt_cell[idx] = dt;
}

} // namespace

bool compute_local_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_dt_cell,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re, std::string* error,
    cudaStream_t stream) {
    if (d_dt_cell == nullptr) { if (error) *error = "compute_local_timestep_gpu: null d_dt_cell"; return false; }
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();
    local_timestep_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), nc, mesh.nvar(), gamma, cfl,
        cd.volume, cd.h_min, d_dt_cell,
        viscous, mu_ref, T_ref, sutherland_T, Re);
    if (!cuda_check(cudaGetLastError(), "local timestep kernel launch", error)) return false;
    return true;
}

bool compute_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_min_dt,
    cudaStream_t stream) {
    return compute_timestep_gpu(mesh, gamma, cfl, d_min_dt, false, 1.0f, 1.0f, 1.0f, 1e6f, stream);
}

bool compute_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_min_dt,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re,
    cudaStream_t stream) {
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + kTimeStepBlockSize - 1) / kTimeStepBlockSize;
    if (!ensure_partial_buf(grid)) return false;

    DeviceCellData cd = mesh.cell_data();

    timestep_kernel<<<grid, kTimeStepBlockSize, 0, stream>>>(
        mesh.state_device(), nc, mesh.nvar(), gamma, cfl,
        cd.volume, cd.h_min, d_partial_buf,
        viscous, mu_ref, T_ref, sutherland_T, Re);
    if (!cuda_check(cudaGetLastError(), "timestep kernel launch")) return false;

    reduce_min_kernel<<<1, 1, 0, stream>>>(d_partial_buf, d_min_dt, grid);
    if (!cuda_check(cudaGetLastError(), "reduce_min kernel launch")) return false;

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp




