#pragma once

#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_result.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/device_mesh.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// Callback type for host-side AMR cycle.
// Called from solve_gpu when mesh refinement is triggered.
// Receives the current host mesh and state (mutable), must update both
// in-place (refine, prolongate, rebuild faces, compute metrics).
// Returns false on failure with error set.
using AmrCycleCallback = bool (*)(CfdMesh& mesh, std::vector<ConservativeState>& q,
    int iter, const CfdConfig& config, std::string* error);

CfdSolveSummary solve_gpu(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    std::string* error = nullptr,
    CfdMesh* host_mesh = nullptr,
    std::vector<ConservativeState>* host_state = nullptr,
    AmrCycleCallback amr_callback = nullptr);

CfdSolveSummary solve_gpu(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    int* d_failed,
    Real* d_min_dt,
    Real* d_l2_sum,
    Real* d_forces,
    std::string* error = nullptr,
    CfdMesh* host_mesh = nullptr,
    std::vector<ConservativeState>* host_state = nullptr,
    AmrCycleCallback amr_callback = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp


