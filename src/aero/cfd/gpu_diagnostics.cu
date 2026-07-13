#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <limits>
namespace aerosp {
namespace aero {
namespace cfd {

namespace {



__global__ void init_bounds_slot_kernel(Real* slot) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        slot[0] = std::numeric_limits<Real>::max();
        slot[1] = std::numeric_limits<Real>::lowest();
        slot[2] = std::numeric_limits<Real>::max();
        slot[3] = std::numeric_limits<Real>::lowest();
        slot[4] = std::numeric_limits<Real>::max();
        slot[5] = std::numeric_limits<Real>::lowest();
    }
}

__global__ void __launch_bounds__(256) state_bounds_kernel(
    const Real* d_q, int n_cells, int nvar, Real gamma,
    Real* d_min_rho, Real* d_max_rho,
    Real* d_min_p, Real* d_max_p,
    Real* d_min_mach, Real* d_max_mach) {
    __shared__ Real s_min_rho[128], s_max_rho[128];
    __shared__ Real s_min_p[128], s_max_p[128];
    __shared__ Real s_min_mach[128], s_max_mach[128];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    Real l_min_rho = std::numeric_limits<Real>::max(), l_max_rho = std::numeric_limits<Real>::lowest();
    Real l_min_p = std::numeric_limits<Real>::max(), l_max_p = std::numeric_limits<Real>::lowest();
    Real l_min_mach = std::numeric_limits<Real>::max(), l_max_mach = std::numeric_limits<Real>::lowest();

    if (idx < n_cells) {
        Real rho = d_q[idx * nvar + 0];
        Real rhou = d_q[idx * nvar + 1];
        Real rhov = d_q[idx * nvar + 2];
        Real rhow = d_q[idx * nvar + 3];
        Real rhoE = d_q[idx * nvar + 4];

        if (real_isfinite(rho) && rho > 0.0f) {
            l_min_rho = rho; l_max_rho = rho;

            Real inv_rho = 1.0f / rho;
            Real u = rhou * inv_rho;
            Real v = rhov * inv_rho;
            Real w = rhow * inv_rho;
            Real ke = 0.5f * (u*u + v*v + w*w);
            Real p = (gamma - 1.0f) * (rhoE - rho * ke);

            if (real_isfinite(p) && p > 0.0f) {
                l_min_p = p; l_max_p = p;
                Real a = real_sqrt(gamma * p * inv_rho);
                Real vm = real_sqrt(u*u + v*v + w*w);
                Real mach = vm / real_fmax(a, 1e-30f);
                l_min_mach = mach; l_max_mach = mach;
            }
        }
    }

    s_min_rho[tid] = l_min_rho; s_max_rho[tid] = l_max_rho;
    s_min_p[tid] = l_min_p; s_max_p[tid] = l_max_p;
    s_min_mach[tid] = l_min_mach; s_max_mach[tid] = l_max_mach;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_rho[tid] = real_fmin(s_min_rho[tid], s_min_rho[tid + s]);
            s_max_rho[tid] = real_fmax(s_max_rho[tid], s_max_rho[tid + s]);
            s_min_p[tid] = real_fmin(s_min_p[tid], s_min_p[tid + s]);
            s_max_p[tid] = real_fmax(s_max_p[tid], s_max_p[tid + s]);
            s_min_mach[tid] = real_fmin(s_min_mach[tid], s_min_mach[tid + s]);
            s_max_mach[tid] = real_fmax(s_max_mach[tid], s_max_mach[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        real_atomic_min(d_min_rho, s_min_rho[0]);
        real_atomic_max(d_max_rho, s_max_rho[0]);
        real_atomic_min(d_min_p, s_min_p[0]);
        real_atomic_max(d_max_p, s_max_p[0]);
        real_atomic_min(d_min_mach, s_min_mach[0]);
        real_atomic_max(d_max_mach, s_max_mach[0]);
    }
}

__global__ void __launch_bounds__(256) failure_snapshot_kernel(
    const Real* d_q, int n_cells, int nvar, Real gamma,
    int* d_failure_cell, Real* d_failure_state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real rho = d_q[idx * nvar + 0];
    Real rhou = d_q[idx * nvar + 1];
    Real rhov = d_q[idx * nvar + 2];
    Real rhow = d_q[idx * nvar + 3];
    Real rhoE = d_q[idx * nvar + 4];

    if (!real_isfinite(rho) || rho <= 0.0f) {
        int old = atomicCAS(d_failure_cell, -1, idx);
        if (old == -1) {
            d_failure_state[0] = rho;
            d_failure_state[1] = rhou;
            d_failure_state[2] = rhov;
            d_failure_state[3] = rhow;
            d_failure_state[4] = rhoE;
        }
        return;
    }
    Real inv_rho = 1.0f / rho;
    Real u = rhou * inv_rho;
    Real v = rhov * inv_rho;
    Real w = rhow * inv_rho;
    Real ke = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (rhoE - rho * ke);
    if (!real_isfinite(p) || p <= 0.0f) {
        int old = atomicCAS(d_failure_cell, -1, idx);
        if (old == -1) {
            d_failure_state[0] = rho;
            d_failure_state[1] = rhou;
            d_failure_state[2] = rhov;
            d_failure_state[3] = rhow;
            d_failure_state[4] = rhoE;
        }
    }
}

} // namespace

bool compute_state_bounds_gpu(DeviceMesh& mesh, Real gamma, Real* d_bounds_slot,
    cudaStream_t stream) {
    init_bounds_slot_kernel<<<1, 1, 0, stream>>>(d_bounds_slot);
    if (!cuda_check(cudaGetLastError(), "init_bounds_slot launch")) return false;

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;

    state_bounds_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), nc, DeviceMesh::NVAR, gamma,
        d_bounds_slot + 0, d_bounds_slot + 1,
        d_bounds_slot + 2, d_bounds_slot + 3,
        d_bounds_slot + 4, d_bounds_slot + 5);
    if (!cuda_check(cudaGetLastError(), "state_bounds_kernel launch")) return false;

    return true;
}

bool compute_failure_snapshot_gpu(DeviceMesh& mesh, Real gamma,
    int* d_failure_cell, Real* d_failure_state,
    cudaStream_t stream) {
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;

    failure_snapshot_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), nc, DeviceMesh::NVAR, gamma,
        d_failure_cell, d_failure_state);
    if (!cuda_check(cudaGetLastError(), "failure_snapshot_kernel launch")) return false;

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp




