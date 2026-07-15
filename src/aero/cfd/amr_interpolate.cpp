#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/amr_interpolate.hpp"
#include "aero/cfd/reconstruction.hpp"

#include <algorithm>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

std::vector<int> build_old_to_new_map(
    const CfdMesh& mesh_old,
    const std::vector<RefinementRecord>& records,
    int /*n_coarsened_parents*/) {

    int n_old = static_cast<int>(mesh_old.cells.size());
    std::vector<int> old_to_new(n_old, -1);

    // Which old cells are replaced by their children?
    std::vector<bool> replaced(n_old, false);
    for (const auto& rec : records) {
        if (rec.parent_cell_id >= 0 && rec.parent_cell_id < n_old)
            replaced[rec.parent_cell_id] = true;
    }

    // Match the actual layout produced by refine_cells:
    //   For each old cell in order:
    //     If replaced → 8 children occupy indices cur..cur+7
    //     If unchanged → 1 cell occupies index cur
    //   Then coarsened parents are appended (not mapped here)
    int cur = 0;
    for (int ci = 0; ci < n_old; ++ci) {
        if (replaced[ci]) {
            cur += 8;  // children occupy positions, skip them
        } else {
            old_to_new[ci] = cur;
            ++cur;
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

    auto old_to_new = build_old_to_new_map(mesh_old, records);

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

void prolongate_solution_order2(
    const std::vector<ConservativeState>& q_old,
    const std::vector<PrimitiveState>& w_old,
    const std::vector<PrimitiveGradient>& grad_old,
    const CfdMesh& mesh_old,
    const CfdMesh& mesh_new,
    const std::vector<RefinementRecord>& records,
    Real gamma,
    std::vector<ConservativeState>& q_new) {

    int n_new = static_cast<int>(mesh_new.cells.size());
    q_new.resize(n_new);

    auto old_to_new = build_old_to_new_map(mesh_old, records);

    for (int old_i = 0; old_i < static_cast<int>(mesh_old.cells.size()); ++old_i) {
        int new_i = old_to_new[old_i];
        if (new_i >= 0) {
            q_new[new_i] = q_old[old_i];
        }
    }

    // Child cells: gradient-based reconstruction
    for (const auto& rec : records) {
        int parent_id = rec.parent_cell_id;
        if (parent_id < 0 || parent_id >= static_cast<int>(q_old.size())) continue;
        const CfdCell& parent_cell = mesh_old.cells[parent_id];
        const PrimitiveState& wp = w_old[parent_id];
        const PrimitiveGradient& gp = grad_old[parent_id];

        for (int c = 0; c < rec.n_children; ++c) {
            int child_id = rec.child_cell_ids[c];
            if (child_id < 0 || child_id >= n_new) continue;

            const CfdCell& child_cell = mesh_new.cells[child_id];
            Real dx = child_cell.cx - parent_cell.cx;
            Real dy = child_cell.cy - parent_cell.cy;
            Real dz = child_cell.cz - parent_cell.cz;

            PrimitiveState w_child = reconstruct_primitive(wp, gp, dx, dy, dz);

            // Fall back to injection if reconstruction produces invalid state.
            // NaN checks use !real_isfinite because IEEE 754 NaN <= 0 is false.
            if (w_child.rho <= 0.0f || w_child.p <= 0.0f ||
                !real_isfinite(w_child.rho) || !real_isfinite(w_child.p) ||
                !real_isfinite(w_child.nu_tilde) ||
                !real_isfinite(w_child.u) || !real_isfinite(w_child.v) || !real_isfinite(w_child.w)) {
                // Use injection — parent state is always valid
                q_new[child_id] = q_old[parent_id];
                // Ensure nu_tilde >= 1e-8 on SA child cells
                if (w_child.rho > 0.0f) {
                    PrimitiveState wc;
                    if (conservative_to_primitive(q_new[child_id], gamma, wc) &&
                        wc.nu_tilde < Real(1e-8))
                        q_new[child_id].rho_nu_tilde = wc.rho * Real(1e-8);
                }
            } else {
                q_new[child_id] = primitive_to_conservative(w_child, gamma);
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
