#pragma once

#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/reconstruction.hpp"

#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// Build index mapping from old cell indices to new cell indices after refinement.
// Unrefined cells map 1:1. Refined cells map to -1 (no single target).
std::vector<int> build_old_to_new_map(
    const CfdMesh& mesh_old,
    const std::vector<RefinementRecord>& records,
    int n_coarsened_parents = 0);

// Prolongation: parent cell state → child cells (injection: all children get parent's state).
// q_new is resized to mesh_new.cells.size().
void prolongate_solution(
    const std::vector<ConservativeState>& q_old,
    const CfdMesh& mesh_old,
    const CfdMesh& mesh_new,
    const std::vector<RefinementRecord>& records,
    std::vector<ConservativeState>& q_new);

// Second-order prolongation: child = parent + grad_parent · (child_center - parent_center).
// For each child, reconstructs primitive at child center using parent's limited gradient,
// then converts to conservative. Falls back to injection for any child where the
// reconstructed primitive has negative rho or p.
void prolongate_solution_order2(
    const std::vector<ConservativeState>& q_old,
    const std::vector<PrimitiveState>& w_old,
    const std::vector<PrimitiveGradient>& grad_old,
    const CfdMesh& mesh_old,
    const CfdMesh& mesh_new,
    const std::vector<RefinementRecord>& records,
    Real gamma,
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
