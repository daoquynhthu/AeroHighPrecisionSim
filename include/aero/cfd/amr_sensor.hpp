#pragma once

#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"

#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// Compute refinement requests from density-gradient sensor.
// Each interior face contributes |density jump|/|avg density| to both adjacent cells.
// Cells with max error > refine_tol get Refine; cells with max error < coarsen_tol
// AND refinement_level>0 get Coarsen.
std::vector<RefinementRequest> compute_gradient_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma = 1.4f);

} // namespace cfd
} // namespace aero
} // namespace aerosp
