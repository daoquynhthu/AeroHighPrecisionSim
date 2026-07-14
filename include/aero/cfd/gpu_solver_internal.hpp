#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/partition.hpp"

#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

class GpuCommunicator;
struct CfdConfig;

bool compute_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_min_dt,
    cudaStream_t stream = nullptr);
bool compute_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_min_dt,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re,
    cudaStream_t stream = nullptr);

bool compute_gradients_gpu(DeviceMesh& mesh, Real gamma, std::string* error = nullptr,
    int* d_failed = nullptr, cudaStream_t stream = nullptr);

bool compute_limiters_gpu(DeviceMesh& mesh, Real gamma, std::string* error = nullptr,
    int* d_failed = nullptr, cudaStream_t stream = nullptr);

bool compute_update_gpu(DeviceMesh& mesh, const Real* d_min_dt, Real gamma,
    Real* d_l2_sum, int* d_failed,
    int* d_failure_cell = nullptr, Real* d_failure_state = nullptr,
    cudaStream_t stream = nullptr);

bool compute_wall_forces_gpu(DeviceMesh& mesh, Real gamma, Real* d_forces);
bool compute_wall_forces_gpu(DeviceMesh& mesh, Real gamma, Real* d_forces,
    bool viscous, Real prandtl, Real mu_ref, Real T_ref, Real sutherland_T,
    Real Re, Real wall_T);

bool compute_state_bounds_gpu(DeviceMesh& mesh, Real gamma, Real* d_bounds_slot,
    cudaStream_t stream = nullptr);

bool compute_failure_snapshot_gpu(DeviceMesh& mesh, Real gamma,
    int* d_failure_cell, Real* d_failure_state,
    cudaStream_t stream = nullptr);

bool compute_viscous_flux_gpu(DeviceMesh& mesh, Real gamma, Real prandtl,
    Real mu_ref, Real T_ref, Real sutherland_T, Real Re, Real wall_T,
    int turbulence, int* d_failed, cudaStream_t stream = nullptr);

bool compute_ddes_length_scale_gpu(DeviceMesh& mesh, Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T, int* d_failed,
    std::string* error = nullptr, cudaStream_t stream = nullptr);

bool compute_rans_source_gpu(DeviceMesh& mesh, Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    const Real* d_delta_ddes, int* d_failed, std::string* error = nullptr,
    cudaStream_t stream = nullptr);

bool compute_sst_gradients_gpu(DeviceMesh& mesh, int* d_failed,
    std::string* error = nullptr, cudaStream_t stream = nullptr);
bool compute_sst_advection_gpu(DeviceMesh& mesh,
    Real inf_k, Real inf_omega,
    int* d_failed, std::string* error = nullptr, cudaStream_t stream = nullptr);
bool compute_sst_diffusion_gpu(DeviceMesh& mesh, Real gamma,
    Real mu_ref, Real T_ref, Real sutherland_T,
    int* d_failed, std::string* error = nullptr, cudaStream_t stream = nullptr);
bool compute_sst_source_gpu(DeviceMesh& mesh, Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    int* d_failed, std::string* error = nullptr, cudaStream_t stream = nullptr);
bool compute_sst_update_gpu(DeviceMesh& mesh, const Real* d_min_dt,
    std::string* error = nullptr, cudaStream_t stream = nullptr);
bool compute_sst_init_gpu(DeviceMesh& mesh,
    Real k_val, Real omega_val,
    std::string* error = nullptr, cudaStream_t stream = nullptr);
bool clear_sst_residual_gpu(DeviceMesh& mesh,
    std::string* error = nullptr, cudaStream_t stream = nullptr);

bool compute_turbulence_source_gpu(DeviceMesh& mesh, const CfdConfig& config,
    int* d_failed, std::string* error = nullptr, cudaStream_t stream = nullptr,
    Real inf_k = 0.0f, Real inf_omega = 0.0f,
    const Real* d_min_dt = nullptr);

bool apply_rans_implicit_gpu(DeviceMesh& mesh, Real Re,
    const Real* d_min_dt, std::string* error = nullptr,
    cudaStream_t stream = nullptr);
bool apply_rans_implicit_per_cell_gpu(DeviceMesh& mesh, Real Re,
    const Real* d_dt_cell, std::string* error = nullptr,
    cudaStream_t stream = nullptr);

bool compute_local_timestep_gpu(DeviceMesh& mesh, Real gamma, Real cfl, Real* d_dt_cell,
    bool viscous, Real mu_ref, Real T_ref, Real sutherland_T, Real Re, std::string* error = nullptr,
    cudaStream_t stream = nullptr);

// Jacobian-free matrix-vector product for implicit solver
struct PrimitiveState;
bool compute_jfv_product(DeviceMesh& mesh, const Real* d_v, Real* d_result,
    const Real* d_residual, Real epsilon, const CfdConfig& config,
    const PrimitiveState& w_inf, Real* d_scratch, int* d_failed,
    std::string* error = nullptr, cudaStream_t stream = nullptr);

// Multi-GPU halo exchange
bool exchange_halo_gpu(DeviceMesh& mesh, const GpuPartition& gpu_part,
    GpuCommunicator& comm, cudaStream_t stream);

} // namespace cfd
} // namespace aero
} // namespace aerosp
