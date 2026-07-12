#pragma once

#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"

#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// Build index mapping from old cell indices to new cell indices after refinement.
// Unrefined cells map 1:1. Refined cells map to -1 (no single target).
std::vector<int> build_old_to_new_map(
    const CfdMesh& mesh_old,
    const std::vector<RefinementRecord>& records);

// Prolongation: parent cell state → child cells (injection: all children get parent's state).
// q_new is resized to mesh_new.cells.size().
void prolongate_solution(
    const std::vector<ConservativeState>& q_old,
    const CfdMesh& mesh_old,
    const CfdMesh& mesh_new,
    const std::vector<RefinementRecord>& records,
    std::vector<ConservativeState>& q_new);

// Restriction: child cells → parent cell (volume-weighted average children → parent).
// q_out is resized to mesh_after_coarsen.cells.size().
void restrict_solution(
    const std::vector<ConservativeState>& q_children,
    const CfdMesh& mesh_children,
    const std::vector<RefinementRecord>& records,
    std::vector<ConservativeState>& q_parent);

} // namespace cfd
} // namespace aero
} // namespace aerosp
