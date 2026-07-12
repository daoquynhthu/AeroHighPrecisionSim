#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/amr_interpolate.hpp"

#include <algorithm>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

std::vector<int> build_old_to_new_map(
    const CfdMesh& mesh_old,
    const std::vector<RefinementRecord>& records) {

    int n_old = static_cast<int>(mesh_old.cells.size());
    std::vector<int> old_to_new(n_old, -1);

    // Mark replaced old cells
    std::vector<bool> replaced(n_old, false);
    for (const auto& rec : records) {
        if (rec.parent_cell_id >= 0 && rec.parent_cell_id < n_old)
            replaced[rec.parent_cell_id] = true;
    }

    // Compute total child count to know where children start in new array
    int n_children = 0;
    for (const auto& rec : records) n_children += rec.n_children;

    // Map unchanged old cells to their new positions
    int idx = n_children;  // children come first, then unchanged cells in order
    for (int ci = 0; ci < n_old; ++ci) {
        if (!replaced[ci]) {
            old_to_new[ci] = idx;
            ++idx;
        }
    }

    return old_to_new;
}

void prolongate_solution(
    const std::vector<ConservativeState>& q_old,
    const CfdMesh& mesh_old,
    const CfdMesh& mesh_new,
    const std::vector<RefinementRecord>& records,
    std::vector<ConservativeState>& q_new) {

    int n_new = static_cast<int>(mesh_new.cells.size());
    q_new.resize(n_new);

    // Build old-to-new map for unchanged cells
    auto old_to_new = build_old_to_new_map(mesh_old, records);

    // Unchanged cells: copy directly
    for (int old_i = 0; old_i < static_cast<int>(mesh_old.cells.size()); ++old_i) {
        int new_i = old_to_new[old_i];
        if (new_i >= 0) {
            q_new[new_i] = q_old[old_i];
        }
    }

    // Child cells: injection prolongation (parent's state)
    for (const auto& rec : records) {
        int parent_id = rec.parent_cell_id;
        if (parent_id < 0 || parent_id >= static_cast<int>(q_old.size())) continue;
        for (int c = 0; c < rec.n_children; ++c) {
            int child_id = rec.child_cell_ids[c];
            if (child_id >= 0 && child_id < n_new) {
                q_new[child_id] = q_old[parent_id];
            }
        }
    }
}

void restrict_solution(
    const std::vector<ConservativeState>& q_children,
    const CfdMesh& mesh_children,
    const std::vector<RefinementRecord>& records,
    std::vector<ConservativeState>& q_parent) {

    // Size: n_old = n_new - total_children + n_records
    int n_records = static_cast<int>(records.size());
    q_parent.resize(0);
    q_parent.reserve(n_records);

    // Volume-weighted average each group of children
    for (const auto& rec : records) {
        Real vol_sum = 0.0f;
        ConservativeState avg;
        for (int c = 0; c < rec.n_children; ++c) {
            int child_id = rec.child_cell_ids[c];
            if (child_id < 0) continue;
            Real vol = mesh_children.cells[child_id].volume;
            vol_sum += vol;
            avg.rho += q_children[child_id].rho * vol;
            avg.rho_u += q_children[child_id].rho_u * vol;
            avg.rho_v += q_children[child_id].rho_v * vol;
            avg.rho_w += q_children[child_id].rho_w * vol;
            avg.rho_E += q_children[child_id].rho_E * vol;
            avg.rho_nu_tilde += q_children[child_id].rho_nu_tilde * vol;
        }
        if (vol_sum > 0.0f) {
            Real inv = 1.0f / vol_sum;
            avg.rho *= inv; avg.rho_u *= inv; avg.rho_v *= inv;
            avg.rho_w *= inv; avg.rho_E *= inv; avg.rho_nu_tilde *= inv;
        }
        q_parent.push_back(avg);
    }
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
