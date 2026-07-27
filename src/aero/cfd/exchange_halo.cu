#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/partition.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/gpu_communicator.hpp"
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

#ifdef MPI_ENABLED

namespace {

__global__ void __launch_bounds__(256) pack_halo_kernel(
    const Real* d_q, int nvar,
    const int* d_ghost_indices, int n_ghost,
    Real* d_send_buf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ghost) return;
    int cell = d_ghost_indices[idx];
    for (int v = 0; v < nvar; ++v) {
        d_send_buf[idx * nvar + v] = d_q[cell * nvar + v];
    }
}

__global__ void __launch_bounds__(256) unpack_halo_kernel(
    Real* d_q, int nvar,
    const int* d_ghost_indices, int n_ghost,
    const Real* d_recv_buf) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ghost) return;
    int cell = d_ghost_indices[idx];
    for (int v = 0; v < nvar; ++v) {
        d_q[cell * nvar + v] = d_recv_buf[idx * nvar + v];
    }
}

} // namespace

bool exchange_halo_gpu(DeviceMesh& mesh, const GpuPartition& gpu_part,
    GpuCommunicator& comm, cudaStream_t stream) {
    if (!mesh.has_halo()) return true;
    if (comm.size() <= 1) return true;

    int nvar = mesh.nvar();
    int n_ghost = gpu_part.n_ghost;

    if (n_ghost <= 0) return true;

    int block = 128;
    int grid = (n_ghost + block - 1) / block;

    pack_halo_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), nvar,
        gpu_part.d_ghost_indices, n_ghost,
        mesh.halo_send_device());

    if (!cuda_check(cudaGetLastError(), "pack_halo_kernel", nullptr)) return false;
    if (!cuda_check(cudaStreamSynchronize(stream), "pack_halo sync", nullptr)) return false;

    Real* d_send = mesh.halo_send_device();
    Real* d_recv = mesh.halo_recv_device();

    for (int peer = 0; peer < comm.size(); ++peer) {
        if (peer == comm.rank()) continue;
        int peer_tag = peer;

        comm.send_recv_exchange(d_send, d_recv, n_ghost * nvar, peer, peer_tag, nullptr);
    }

    unpack_halo_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(), nvar,
        gpu_part.d_ghost_indices, n_ghost,
        mesh.halo_recv_device());

    if (!cuda_check(cudaGetLastError(), "unpack_halo_kernel", nullptr)) return false;
    return true;
}

#else // !MPI_ENABLED

bool exchange_halo_gpu(DeviceMesh& mesh, const GpuPartition& gpu_part,
    GpuCommunicator& comm, cudaStream_t stream) {
    (void)mesh;
    (void)gpu_part;
    (void)comm;
    (void)stream;
    return true;
}

#endif // MPI_ENABLED

} // namespace cfd
} // namespace aero
} // namespace aerosp
