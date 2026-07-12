#pragma once

#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_result.hpp"
#include "aero/cfd/cfd_state.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

struct CfdSolveSummary {
    CfdForceResult forces;
    CfdDiagnostics diagnostics;
    std::vector<Real> residual_history;
    std::vector<ConservativeState> final_state;
    bool converged = false;
    bool failed = false;
};

class CfdSolver {
public:
    bool load_mesh(const CfdMesh& mesh);

    CfdSolveSummary solve(const FreestreamCondition& condition, const CfdConfig& config);

    CfdSolveSummary solve_from_state(
        const FreestreamCondition& condition,
        const CfdConfig& config,
        const std::vector<ConservativeState>& initial_state);

    const CfdMesh& mesh() const { return mesh_; }

private:
    CfdMesh mesh_;
    std::vector<int> wall_face_indices_;
};

bool assert_oracle_equivalent(
    const CfdSolveSummary& gpu,
    const CfdSolveSummary& cpu,
    Real tol_residual,
    Real tol_forces,
    std::string* error);

} // namespace cfd
} // namespace aero
} // namespace aerosp


