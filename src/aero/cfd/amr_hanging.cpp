#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/amr_hanging.hpp"
#include "aero/cfd/reconstruction.hpp"

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

        ConservativeState q_interp = q_cell[coarse_id];
        q_interp.rho   += grad_rho[coarse_id].gx   * dx + grad_rho[coarse_id].gy   * dy + grad_rho[coarse_id].gz   * dz;
        q_interp.rho_u += grad_rho_u[coarse_id].gx * dx + grad_rho_u[coarse_id].gy * dy + grad_rho_u[coarse_id].gz * dz;
        q_interp.rho_v += grad_rho_v[coarse_id].gx * dx + grad_rho_v[coarse_id].gy * dy + grad_rho_v[coarse_id].gz * dz;
        q_interp.rho_w += grad_rho_w[coarse_id].gx * dx + grad_rho_w[coarse_id].gy * dy + grad_rho_w[coarse_id].gz * dz;
        q_interp.rho_E += grad_rho_E[coarse_id].gx * dx + grad_rho_E[coarse_id].gy * dy + grad_rho_E[coarse_id].gz * dz;
        q_interp.rho_nu_tilde += grad_rho_nu[coarse_id].gx * dx + grad_rho_nu[coarse_id].gy * dy + grad_rho_nu[coarse_id].gz * dz;

        if (face.left_cell == coarse_id) {
            q_face_left[hf.face_id] = q_interp;
        } else {
            q_face_right[hf.face_id] = q_interp;
        }
    }
}

void apply_hanging_flux_correction_primitive(
    const CfdMesh& mesh,
    const std::vector<HangingFaceInfo>& hanging_faces,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveState>& w,
    const std::vector<PrimitiveGradient>& grads,
    Real gamma,
    std::vector<EulerFlux>& residual) {

    for (const auto& hf : hanging_faces) {
        const CfdFace& face = mesh.faces[hf.face_id];
        int coarse_id = hf.coarse_cell_id;
        int fine_id = hf.fine_cell_id;
        const CfdCell& coarse_cell = mesh.cells[coarse_id];
        const CfdCell& fine_cell = mesh.cells[fine_id];

        Real dx_c = face.cx - coarse_cell.cx;
        Real dy_c = face.cy - coarse_cell.cy;
        Real dz_c = face.cz - coarse_cell.cz;
        Real dx_f = face.cx - fine_cell.cx;
        Real dy_f = face.cy - fine_cell.cy;
        Real dz_f = face.cz - fine_cell.cz;

        // Old flux (what the main residual already used):
        PrimitiveState wl = reconstruct_primitive(w[coarse_id], grads[coarse_id], dx_c, dy_c, dz_c);
        PrimitiveState wr = reconstruct_primitive(w[fine_id], grads[fine_id], dx_f, dy_f, dz_f);
        if (wl.rho <= 0.0f || wl.p <= 0.0f) continue;
        if (wr.rho <= 0.0f || wr.p <= 0.0f) continue;
        EulerFlux flux_old = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);

        // New flux: reconstruct coarse side with positive-preserving limiter
        Real theta = 1.0f;
        wl = reconstruct_primitive_positive(w[coarse_id], grads[coarse_id], dx_c, dy_c, dz_c,
            static_cast<Real>(1e-8f), static_cast<Real>(1e-10f), &theta);
        if (wl.rho <= 0.0f || wl.p <= 0.0f) continue;
        EulerFlux flux_new = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);

        Real area = face.area;
        residual[face.left_cell].mass   += (flux_old.mass   - flux_new.mass)   * area;
        residual[face.left_cell].mom_x  += (flux_old.mom_x  - flux_new.mom_x)  * area;
        residual[face.left_cell].mom_y  += (flux_old.mom_y  - flux_new.mom_y)  * area;
        residual[face.left_cell].mom_z  += (flux_old.mom_z  - flux_new.mom_z)  * area;
        residual[face.left_cell].energy += (flux_old.energy - flux_new.energy) * area;
        residual[face.left_cell].turbulence += (flux_old.turbulence - flux_new.turbulence) * area;
        residual[face.right_cell].mass   -= (flux_old.mass   - flux_new.mass)   * area;
        residual[face.right_cell].mom_x  -= (flux_old.mom_x  - flux_new.mom_x)  * area;
        residual[face.right_cell].mom_y  -= (flux_old.mom_y  - flux_new.mom_y)  * area;
        residual[face.right_cell].mom_z  -= (flux_old.mom_z  - flux_new.mom_z)  * area;
        residual[face.right_cell].energy -= (flux_old.energy - flux_new.energy) * area;
        residual[face.right_cell].turbulence -= (flux_old.turbulence - flux_new.turbulence) * area;
    }
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
