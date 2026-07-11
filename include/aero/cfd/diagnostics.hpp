#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

enum class DiagnosticLevel : int {
    Off = 0,
    Basic = 1,
    Detailed = 2,
    Verbose = 3
};

struct StateBounds {
    Real min_rho = 0.0f;
    Real max_rho = 0.0f;
    Real min_p = 0.0f;
    Real max_p = 0.0f;
    Real min_mach = 0.0f;
    Real max_mach = 0.0f;
    int bad_cell = -1;
    bool valid = false;
};

struct DtLimiterSnapshot {
    int iteration = -1;
    int cell = -1;
    Real dt = 0.0f;
    Real h_min = 0.0f;
    Real signal_speed = 0.0f;
};

struct FailureSnapshot {
    bool valid = false;
    int iteration = -1;
    int cell = -1;
    std::string reason;
    ConservativeState state;
    PrimitiveState primitive;
};

struct CfdDiagnostics {
    std::vector<StateBounds> state_bounds_history;
    std::vector<DtLimiterSnapshot> dt_limiter_history;
    FailureSnapshot failure;
};

StateBounds compute_state_bounds(const std::vector<ConservativeState>& q, Real gamma, const std::vector<PrimitiveState>* primitive_override = nullptr);

FailureSnapshot make_failure_snapshot(
    int iteration,
    int cell,
    const char* reason,
    const ConservativeState& q,
    Real gamma);

bool write_vtk_cells(
    const char* path,
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    Real gamma,
    std::string* error = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp
