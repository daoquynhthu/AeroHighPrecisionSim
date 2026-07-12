#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/amr_hanging.hpp"

#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

std::vector<HangingFaceInfo> detect_hanging_faces(const CfdMesh& mesh) {
    std::vector<HangingFaceInfo> result;
    int n_faces = static_cast<int>(mesh.faces.size());
    for (int fi = 0; fi < n_faces; ++fi) {
        const CfdFace& f = mesh.faces[fi];
        if (f.boundary != BoundaryKind::Interior) continue;
        int l = f.left_cell;
        int r = f.right_cell;
        if (l < 0 || r < 0) continue;
        int ll = mesh.cells[l].refinement_level;
        int lr = mesh.cells[r].refinement_level;
        if (ll == lr) continue;
        HangingFaceInfo info;
        info.face_id = fi;
        if (ll < lr) {
            info.coarse_cell_id = l;
            info.fine_cell_id = r;
            info.coarse_level = ll;
            info.fine_level = lr;
        } else {
            info.coarse_cell_id = r;
            info.fine_cell_id = l;
            info.coarse_level = lr;
            info.fine_level = ll;
        }
        result.push_back(info);
    }
    return result;
}

void apply_hanging_interpolation(
    const CfdMesh& mesh,
    const std::vector<HangingFaceInfo>& hanging_faces,
    const std::vector<ConservativeState>& q_cell,
    const std::vector<CellGradient3>& grad_rho,
    const std::vector<CellGradient3>& grad_rho_u,
    const std::vector<CellGradient3>& grad_rho_v,
    const std::vector<CellGradient3>& grad_rho_w,
    const std::vector<CellGradient3>& grad_rho_E,
    const std::vector<CellGradient3>& grad_rho_nu,
    std::vector<ConservativeState>& q_face_left,
    std::vector<ConservativeState>& q_face_right) {

    int n_faces = static_cast<int>(mesh.faces.size());
    if (static_cast<int>(q_face_left.size()) != n_faces) return;
    if (static_cast<int>(q_face_right.size()) != n_faces) return;

    for (const auto& hf : hanging_faces) {
        const CfdFace& face = mesh.faces[hf.face_id];
        int coarse_id = hf.coarse_cell_id;
        const CfdCell& coarse_cell = mesh.cells[coarse_id];

        Real dx = face.cx - coarse_cell.cx;
        Real dy = face.cy - coarse_cell.cy;
        Real dz = face.cz - coarse_cell.cz;

        // Linear extrapolation: q_face = q_cell + grad_q . d
        ConservativeState q_interp = q_cell[coarse_id];
        q_interp.rho   += grad_rho[coarse_id].gx   * dx + grad_rho[coarse_id].gy   * dy + grad_rho[coarse_id].gz   * dz;
        q_interp.rho_u += grad_rho_u[coarse_id].gx * dx + grad_rho_u[coarse_id].gy * dy + grad_rho_u[coarse_id].gz * dz;
        q_interp.rho_v += grad_rho_v[coarse_id].gx * dx + grad_rho_v[coarse_id].gy * dy + grad_rho_v[coarse_id].gz * dz;
        q_interp.rho_w += grad_rho_w[coarse_id].gx * dx + grad_rho_w[coarse_id].gy * dy + grad_rho_w[coarse_id].gz * dz;
        q_interp.rho_E += grad_rho_E[coarse_id].gx * dx + grad_rho_E[coarse_id].gy * dy + grad_rho_E[coarse_id].gz * dz;
        q_interp.rho_nu_tilde += grad_rho_nu[coarse_id].gx * dx + grad_rho_nu[coarse_id].gy * dy + grad_rho_nu[coarse_id].gz * dz;

        // Determine which side the coarse cell is on and update the corresponding face state
        if (face.left_cell == coarse_id) {
            q_face_left[hf.face_id] = q_interp;
        } else {
            q_face_right[hf.face_id] = q_interp;
        }
    }
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
