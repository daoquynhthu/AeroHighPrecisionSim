#pragma once

#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/device_mesh.hpp"

#include "aero/cfd/reconstruction.hpp"

#include <cstddef>
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
struct CUevent_st;
using cudaEvent_t = CUevent_st*;
#endif
#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// Forward decl for MMS-compatible farfield BC.
struct MmsSolutionEulerBC;

bool compute_euler_residual_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual,
    const std::vector<PrimitiveState>* primitive_override = nullptr,
    const MmsSolutionEulerBC* mms_solution = nullptr);

bool compute_euler_residual_cpu_order2(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual,
    const std::vector<PrimitiveState>* primitive_override = nullptr,
    const MmsSolutionEulerBC* mms_solution = nullptr);

bool compute_euler_residual_cpu_order2(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    const std::vector<PrimitiveGradient>& limited_gradients,
    std::vector<EulerFlux>& residual,
    const std::vector<PrimitiveState>* primitive_override = nullptr,
    const MmsSolutionEulerBC* mms_solution = nullptr);

bool launch_euler_residual_kernel(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    int* d_failed,
    cudaEvent_t start_event = nullptr,
    std::string* error = nullptr,
    int reconstruction_order = 1,
    cudaStream_t stream = nullptr);

bool compute_euler_residual_gpu(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    int* d_failed,
    std::string* error = nullptr,
    int reconstruction_order = 1);

bool compute_euler_residual_gpu(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    std::string* error = nullptr,
    int reconstruction_order = 1);

bool compute_euler_residual_gpu_timed(
    DeviceMesh& mesh,
    const PrimitiveState& freestream,
    Real gamma,
    Real* elapsed_ms,
    std::string* error = nullptr,
    int reconstruction_order = 1);

std::size_t estimate_euler_residual_gpu_bytes(const CfdMesh& mesh);

bool compute_euler_residual_gpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual,
    std::string* error = nullptr);

bool compute_viscous_flux_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    Real gamma,
    Real prandtl,
    Real mu_ref,
    Real T_ref,
    Real sutherland_T,
    Real Re,
    Real wall_T,
    int turbulence,
    std::vector<EulerFlux>& residual,
    const std::vector<PrimitiveState>* primitive_override = nullptr,
    const std::vector<Real>* sst_k = nullptr,
    const std::vector<Real>* sst_omega = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp


