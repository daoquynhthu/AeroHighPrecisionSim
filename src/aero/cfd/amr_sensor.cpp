#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/amr_sensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

std::vector<RefinementRequest> compute_gradient_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma) {

    int n_cells = static_cast<int>(mesh.cells.size());
    std::vector<RefinementRequest> requests(n_cells);
    for (int i = 0; i < n_cells; ++i)
        requests[i] = {i, RefinementFlag::Unchanged};

    if (n_cells == 0 || config.coarsen_tol >= config.refine_tol)
        return requests;

    // Per-cell max density gradient indicator
    std::vector<Real> cell_error(n_cells, 0.0f);

    // Loop over faces, compute max across density/pressure/velocity-magnitude jumps
    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::Interior) continue;
        int L = face.left_cell;
        int R = face.right_cell;
        if (L < 0 || R < 0 || L >= n_cells || R >= n_cells) continue;

        PrimitiveState wL, wR;
        if (!conservative_to_primitive(q[L], gamma, wL)) continue;
        if (!conservative_to_primitive(q[R], gamma, wR)) continue;

        Real rho_L = q[L].rho;
        Real rho_R = q[R].rho;
        Real avg_rho = 0.5f * (std::fabs(rho_L) + std::fabs(rho_R));
        Real rho_jump = 0.0f;
        if (avg_rho >= 1e-10f)
            rho_jump = std::fabs(rho_L - rho_R) / avg_rho;

        Real p_L = wL.p;
        Real p_R = wR.p;
        Real avg_p = 0.5f * (std::fabs(p_L) + std::fabs(p_R));
        Real p_jump = 0.0f;
        if (avg_p >= 1e-10f)
            p_jump = std::fabs(p_L - p_R) / avg_p;

        Real vel_L = std::sqrt(wL.u*wL.u + wL.v*wL.v + wL.w*wL.w);
        Real vel_R = std::sqrt(wR.u*wR.u + wR.v*wR.v + wR.w*wR.w);
        Real avg_vel = 0.5f * (vel_L + vel_R);
        Real vel_jump = 0.0f;
        if (avg_vel >= 1e-10f)
            vel_jump = std::fabs(vel_L - vel_R) / avg_vel;

        Real max_jump = std::max({rho_jump, p_jump, vel_jump});
        if (max_jump > cell_error[L]) cell_error[L] = max_jump;
        if (max_jump > cell_error[R]) cell_error[R] = max_jump;
    }

    // Boundary faces are skipped: ghost density ≈ interior cell density, so
    // no refinement trigger from boundaries.

    // Generate requests based on cell error vs thresholds
    for (int i = 0; i < n_cells; ++i) {
        if (cell_error[i] > config.refine_tol) {
            requests[i].flag = RefinementFlag::Refine;
        } else if (cell_error[i] <= config.coarsen_tol && mesh.cells[i].refinement_level > 0) {
            requests[i].flag = RefinementFlag::Coarsen;
        }
    }

    return requests;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
