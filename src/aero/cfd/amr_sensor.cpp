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
    const AmrConfig& config) {

    int n_cells = static_cast<int>(mesh.cells.size());
    std::vector<RefinementRequest> requests(n_cells);
    for (int i = 0; i < n_cells; ++i)
        requests[i] = {i, RefinementFlag::Unchanged};

    if (n_cells == 0 || config.coarsen_tol >= config.refine_tol)
        return requests;

    // Per-cell max density gradient indicator
    std::vector<Real> cell_error(n_cells, 0.0f);

    // Loop over faces, compute density jump for interior faces
    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::Interior) continue;
        int L = face.left_cell;
        int R = face.right_cell;
        if (L < 0 || R < 0 || L >= n_cells || R >= n_cells) continue;

        Real rho_L = q[L].rho;
        Real rho_R = q[R].rho;
        Real avg_rho = 0.5f * (std::fabs(rho_L) + std::fabs(rho_R));
        if (avg_rho < 1e-10f) continue;

        Real jump = std::fabs(rho_L - rho_R) / avg_rho;
        if (jump > cell_error[L]) cell_error[L] = jump;
        if (jump > cell_error[R]) cell_error[R] = jump;
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
