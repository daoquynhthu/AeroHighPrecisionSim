#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/mms.hpp"
#include "aero/cfd/reconstruction.hpp"
#include "aero/cfd/viscous.hpp"

#include <algorithm>

namespace aerosp {
namespace aero {
namespace cfd {

bool compute_euler_residual_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual,
    const std::vector<PrimitiveState>* primitive_override,
    const MmsSolutionEulerBC* mms_solution) {
    if (q.size() != mesh.cells.size()) return false;
    residual.resize(q.size());
    std::fill(residual.begin(), residual.end(), EulerFlux{});

    for (const auto& face : mesh.faces) {
        PrimitiveState wl;
        if (primitive_override && primitive_override->size() == q.size()) {
            wl = (*primitive_override)[face.left_cell];
        } else if (!conservative_to_primitive(q[face.left_cell], gamma, wl)) {
            return false;
        }

        EulerFlux flux;
        if (face.boundary == BoundaryKind::Interior) {
            PrimitiveState wr;
            if (primitive_override && primitive_override->size() == q.size()) {
                wr = (*primitive_override)[face.right_cell];
            } else if (!conservative_to_primitive(q[face.right_cell], gamma, wr)) {
                return false;
            }
            flux = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
        } else if (face.boundary == BoundaryKind::SlipWall || face.boundary == BoundaryKind::NoSlipWall || face.boundary == BoundaryKind::Symmetry) {
            flux = slip_wall_flux(wl, face.nx, face.ny, face.nz);
        } else if (face.boundary == BoundaryKind::Farfield) {
            PrimitiveState wr;
            if (mms_solution) {
                wr = mms_solution->eval(face.cx, face.cy, face.cz);
            } else {
                wr = farfield_ghost_state(wl, freestream, gamma, face.nx, face.ny, face.nz);
            }
            flux = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
        } else {
            return false;
        }

        Real area = face.area;
        residual[face.left_cell].mass -= flux.mass * area;
        residual[face.left_cell].mom_x -= flux.mom_x * area;
        residual[face.left_cell].mom_y -= flux.mom_y * area;
        residual[face.left_cell].mom_z -= flux.mom_z * area;
        residual[face.left_cell].energy -= flux.energy * area;
        residual[face.left_cell].turbulence -= flux.turbulence * area;

        if (face.boundary == BoundaryKind::Interior) {
            residual[face.right_cell].mass += flux.mass * area;
            residual[face.right_cell].mom_x += flux.mom_x * area;
            residual[face.right_cell].mom_y += flux.mom_y * area;
            residual[face.right_cell].mom_z += flux.mom_z * area;
            residual[face.right_cell].energy += flux.energy * area;
            residual[face.right_cell].turbulence += flux.turbulence * area;
        }
    }

    return true;
}

bool compute_euler_residual_cpu_order2(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    std::vector<EulerFlux>& residual,
    const std::vector<PrimitiveState>* primitive_override,
    const MmsSolutionEulerBC* mms_solution) {
    if (q.size() != mesh.cells.size()) return false;
    residual.resize(q.size());
    std::fill(residual.begin(), residual.end(), EulerFlux{});

    std::vector<PrimitiveGradient> gradients = compute_green_gauss_gradients(mesh, q, gamma, primitive_override);
    if (gradients.size() != mesh.cells.size()) return false;

    std::vector<PrimitiveLimiter> limiters = compute_barth_jespersen_limiters(mesh, q, gradients, gamma, primitive_override);
    if (limiters.size() != mesh.cells.size()) return false;

    std::vector<PrimitiveGradient> limited(gradients.size());
    for (std::size_t i = 0; i < gradients.size(); ++i) {
        limited[i] = apply_limiter(gradients[i], limiters[i]);
    }

    for (const auto& face : mesh.faces) {
        PrimitiveState w_center;
        if (primitive_override && primitive_override->size() == q.size()) {
            w_center = (*primitive_override)[face.left_cell];
        } else if (!conservative_to_primitive(q[face.left_cell], gamma, w_center)) {
            return false;
        }

        PrimitiveState wl = reconstruct_primitive(w_center, limited[face.left_cell],
            face.cx - mesh.cells[face.left_cell].cx,
            face.cy - mesh.cells[face.left_cell].cy,
            face.cz - mesh.cells[face.left_cell].cz);
        if (wl.rho <= 0.0f || wl.p <= 0.0f) return false;

        EulerFlux flux;
        if (face.boundary == BoundaryKind::Interior) {
            PrimitiveState w_center_r;
            if (primitive_override && primitive_override->size() == q.size()) {
                w_center_r = (*primitive_override)[face.right_cell];
            } else if (!conservative_to_primitive(q[face.right_cell], gamma, w_center_r)) {
                return false;
            }

            PrimitiveState wr = reconstruct_primitive(w_center_r, limited[face.right_cell],
                face.cx - mesh.cells[face.right_cell].cx,
                face.cy - mesh.cells[face.right_cell].cy,
                face.cz - mesh.cells[face.right_cell].cz);
            if (wr.rho <= 0.0f || wr.p <= 0.0f) return false;

            flux = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
        } else if (face.boundary == BoundaryKind::SlipWall || face.boundary == BoundaryKind::NoSlipWall || face.boundary == BoundaryKind::Symmetry) {
            flux = slip_wall_flux(wl, face.nx, face.ny, face.nz);
        } else if (face.boundary == BoundaryKind::Farfield) {
            PrimitiveState wr;
            if (mms_solution) {
                wr = mms_solution->eval(face.cx, face.cy, face.cz);
            } else {
                wr = farfield_ghost_state(wl, freestream, gamma, face.nx, face.ny, face.nz);
            }
            flux = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
        } else {
            return false;
        }

        Real area = face.area;
        residual[face.left_cell].mass -= flux.mass * area;
        residual[face.left_cell].mom_x -= flux.mom_x * area;
        residual[face.left_cell].mom_y -= flux.mom_y * area;
        residual[face.left_cell].mom_z -= flux.mom_z * area;
        residual[face.left_cell].energy -= flux.energy * area;
        residual[face.left_cell].turbulence -= flux.turbulence * area;

        if (face.boundary == BoundaryKind::Interior) {
            residual[face.right_cell].mass += flux.mass * area;
            residual[face.right_cell].mom_x += flux.mom_x * area;
            residual[face.right_cell].mom_y += flux.mom_y * area;
            residual[face.right_cell].mom_z += flux.mom_z * area;
            residual[face.right_cell].energy += flux.energy * area;
            residual[face.right_cell].turbulence += flux.turbulence * area;
        }
    }

    return true;
}

bool compute_euler_residual_cpu_order2(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const PrimitiveState& freestream,
    Real gamma,
    const std::vector<PrimitiveGradient>& limited_gradients,
    std::vector<EulerFlux>& residual,
    const std::vector<PrimitiveState>* primitive_override,
    const MmsSolutionEulerBC* mms_solution) {
    if (q.size() != mesh.cells.size()) return false;
    if (limited_gradients.size() != mesh.cells.size()) return false;
    residual.assign(q.size(), EulerFlux{});

    for (const auto& face : mesh.faces) {
        PrimitiveState w_center;
        if (primitive_override && primitive_override->size() == q.size()) {
            w_center = (*primitive_override)[face.left_cell];
        } else if (!conservative_to_primitive(q[face.left_cell], gamma, w_center)) {
            return false;
        }

        PrimitiveState wl = reconstruct_primitive(w_center, limited_gradients[face.left_cell],
            face.cx - mesh.cells[face.left_cell].cx,
            face.cy - mesh.cells[face.left_cell].cy,
            face.cz - mesh.cells[face.left_cell].cz);
        if (wl.rho <= 0.0f || wl.p <= 0.0f) return false;

        EulerFlux flux;
        if (face.boundary == BoundaryKind::Interior) {
            PrimitiveState w_center_r;
            if (primitive_override && primitive_override->size() == q.size()) {
                w_center_r = (*primitive_override)[face.right_cell];
            } else if (!conservative_to_primitive(q[face.right_cell], gamma, w_center_r)) {
                return false;
            }

            PrimitiveState wr = reconstruct_primitive(w_center_r, limited_gradients[face.right_cell],
                face.cx - mesh.cells[face.right_cell].cx,
                face.cy - mesh.cells[face.right_cell].cy,
                face.cz - mesh.cells[face.right_cell].cz);
            if (wr.rho <= 0.0f || wr.p <= 0.0f) return false;

            flux = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
        } else if (face.boundary == BoundaryKind::SlipWall || face.boundary == BoundaryKind::NoSlipWall || face.boundary == BoundaryKind::Symmetry) {
            flux = slip_wall_flux(wl, face.nx, face.ny, face.nz);
        } else if (face.boundary == BoundaryKind::Farfield) {
            PrimitiveState wr;
            if (mms_solution) {
                wr = mms_solution->eval(face.cx, face.cy, face.cz);
            } else {
                wr = farfield_ghost_state(wl, freestream, gamma, face.nx, face.ny, face.nz);
            }
            flux = hllc_flux(wl, wr, gamma, face.nx, face.ny, face.nz);
        } else {
            return false;
        }

        Real area = face.area;
        residual[face.left_cell].mass -= flux.mass * area;
        residual[face.left_cell].mom_x -= flux.mom_x * area;
        residual[face.left_cell].mom_y -= flux.mom_y * area;
        residual[face.left_cell].mom_z -= flux.mom_z * area;
        residual[face.left_cell].energy -= flux.energy * area;
        residual[face.left_cell].turbulence -= flux.turbulence * area;

        if (face.boundary == BoundaryKind::Interior) {
            residual[face.right_cell].mass += flux.mass * area;
            residual[face.right_cell].mom_x += flux.mom_x * area;
            residual[face.right_cell].mom_y += flux.mom_y * area;
            residual[face.right_cell].mom_z += flux.mom_z * area;
            residual[face.right_cell].energy += flux.energy * area;
            residual[face.right_cell].turbulence += flux.turbulence * area;
        }
    }

    return true;
}

bool compute_viscous_flux_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    Real gamma,
    Real prandtl,
    Real mu_ref,
    Real T_ref,
    Real sutherland_T,
    Real Re,
    Real wall_T,
    int turbulence,
    std::vector<EulerFlux>& residual,
    const std::vector<PrimitiveState>* primitive_override) {
    if (q.size() != mesh.cells.size()) return false;
    if (gradients.size() != mesh.cells.size()) return false;

    Real inv_Re = 1.0f / (Re > 0.0f ? Re : 1e6f);
    constexpr Real sigma_sa = 2.0f / 3.0f;
    constexpr Real cv1 = 7.1f;
    constexpr Real cv13 = cv1 * cv1 * cv1;

    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::Interior &&
            face.boundary != BoundaryKind::NoSlipWall) continue;

        PrimitiveState wL;
        if (primitive_override && primitive_override->size() == q.size()) {
            wL = (*primitive_override)[face.left_cell];
        } else if (!conservative_to_primitive(q[face.left_cell], gamma, wL)) {
            return false;
        }
        const PrimitiveGradient& gL = gradients[face.left_cell];

        Real T_L = primitive_temperature(wL);
        Real inv_rho_L = 1.0f / wL.rho;
        Real inv_rho_L2 = inv_rho_L * inv_rho_L;
        Real dT_dx_L = (wL.rho * gL.dp_dx - wL.p * gL.drho_dx) * inv_rho_L2;
        Real dT_dy_L = (wL.rho * gL.dp_dy - wL.p * gL.drho_dy) * inv_rho_L2;
        Real dT_dz_L = (wL.rho * gL.dp_dz - wL.p * gL.drho_dz) * inv_rho_L2;

        Real du_dx, du_dy, du_dz;
        Real dv_dx, dv_dy, dv_dz;
        Real dw_dx, dw_dy, dw_dz;
        Real dT_dx, dT_dy, dT_dz;
        Real face_u, face_v, face_w, face_T;

        if (face.boundary == BoundaryKind::Interior) {
            PrimitiveState wR;
            if (primitive_override && primitive_override->size() == q.size()) {
                wR = (*primitive_override)[face.right_cell];
            } else if (!conservative_to_primitive(q[face.right_cell], gamma, wR)) {
                return false;
            }
            const PrimitiveGradient& gR = gradients[face.right_cell];

            Real T_R = primitive_temperature(wR);
            Real inv_rho_R = 1.0f / wR.rho;
            Real inv_rho_R2 = inv_rho_R * inv_rho_R;
            Real dT_dx_R = (wR.rho * gR.dp_dx - wR.p * gR.drho_dx) * inv_rho_R2;
            Real dT_dy_R = (wR.rho * gR.dp_dy - wR.p * gR.drho_dy) * inv_rho_R2;
            Real dT_dz_R = (wR.rho * gR.dp_dz - wR.p * gR.drho_dz) * inv_rho_R2;

            du_dx = 0.5f * (gL.du_dx + gR.du_dx);
            du_dy = 0.5f * (gL.du_dy + gR.du_dy);
            du_dz = 0.5f * (gL.du_dz + gR.du_dz);
            dv_dx = 0.5f * (gL.dv_dx + gR.dv_dx);
            dv_dy = 0.5f * (gL.dv_dy + gR.dv_dy);
            dv_dz = 0.5f * (gL.dv_dz + gR.dv_dz);
            dw_dx = 0.5f * (gL.dw_dx + gR.dw_dx);
            dw_dy = 0.5f * (gL.dw_dy + gR.dw_dy);
            dw_dz = 0.5f * (gL.dw_dz + gR.dw_dz);
            dT_dx = 0.5f * (dT_dx_L + dT_dx_R);
            dT_dy = 0.5f * (dT_dy_L + dT_dy_R);
            dT_dz = 0.5f * (dT_dz_L + dT_dz_R);

            Real dr_x = mesh.cells[face.right_cell].cx - mesh.cells[face.left_cell].cx;
            Real dr_y = mesh.cells[face.right_cell].cy - mesh.cells[face.left_cell].cy;
            Real dr_z = mesh.cells[face.right_cell].cz - mesh.cells[face.left_cell].cz;
            Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
            if (d2 > 1e-30f) {
                Real inv_d2 = 1.0f / d2;
                Real proj_du = du_dx*dr_x + du_dy*dr_y + du_dz*dr_z;
                Real proj_dv = dv_dx*dr_x + dv_dy*dr_y + dv_dz*dr_z;
                Real proj_dw = dw_dx*dr_x + dw_dy*dr_y + dw_dz*dr_z;
                Real proj_dT = dT_dx*dr_x + dT_dy*dr_y + dT_dz*dr_z;
                Real du_corr = ((wR.u - wL.u) - proj_du) * inv_d2;
                Real dv_corr = ((wR.v - wL.v) - proj_dv) * inv_d2;
                Real dw_corr = ((wR.w - wL.w) - proj_dw) * inv_d2;
                Real dT_corr = ((T_R - T_L) - proj_dT) * inv_d2;
                du_dx += du_corr * dr_x; du_dy += du_corr * dr_y; du_dz += du_corr * dr_z;
                dv_dx += dv_corr * dr_x; dv_dy += dv_corr * dr_y; dv_dz += dv_corr * dr_z;
                dw_dx += dw_corr * dr_x; dw_dy += dw_corr * dr_y; dw_dz += dw_corr * dr_z;
                dT_dx += dT_corr * dr_x; dT_dy += dT_corr * dr_y; dT_dz += dT_corr * dr_z;
            }

            face_u = 0.5f * (wL.u + wR.u);
            face_v = 0.5f * (wL.v + wR.v);
            face_w = 0.5f * (wL.w + wR.w);
            face_T = 0.5f * (T_L + T_R);
        } else {
            du_dx = gL.du_dx; du_dy = gL.du_dy; du_dz = gL.du_dz;
            dv_dx = gL.dv_dx; dv_dy = gL.dv_dy; dv_dz = gL.dv_dz;
            dw_dx = gL.dw_dx; dw_dy = gL.dw_dy; dw_dz = gL.dw_dz;
            dT_dx = dT_dx_L;  dT_dy = dT_dy_L;  dT_dz = dT_dz_L;

            Real dr_x = face.cx - mesh.cells[face.left_cell].cx;
            Real dr_y = face.cy - mesh.cells[face.left_cell].cy;
            Real dr_z = face.cz - mesh.cells[face.left_cell].cz;
            Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
            if (d2 > 1e-30f) {
                Real inv_d2 = 1.0f / d2;
                Real proj_du = du_dx*dr_x + du_dy*dr_y + du_dz*dr_z;
                Real proj_dv = dv_dx*dr_x + dv_dy*dr_y + dv_dz*dr_z;
                Real proj_dw = dw_dx*dr_x + dw_dy*dr_y + dw_dz*dr_z;
                Real proj_dT = dT_dx*dr_x + dT_dy*dr_y + dT_dz*dr_z;
                Real du_corr = ((0.0f - wL.u) - proj_du) * inv_d2;
                Real dv_corr = ((0.0f - wL.v) - proj_dv) * inv_d2;
                Real dw_corr = ((0.0f - wL.w) - proj_dw) * inv_d2;
                Real dT_corr = ((wall_T - T_L) - proj_dT) * inv_d2;
                du_dx += du_corr * dr_x; du_dy += du_corr * dr_y; du_dz += du_corr * dr_z;
                dv_dx += dv_corr * dr_x; dv_dy += dv_corr * dr_y; dv_dz += dv_corr * dr_z;
                dw_dx += dw_corr * dr_x; dw_dy += dw_corr * dr_y; dw_dz += dw_corr * dr_z;
                dT_dx += dT_corr * dr_x; dT_dy += dT_corr * dr_y; dT_dz += dT_corr * dr_z;
            }

            face_u = 0.0f;
            face_v = 0.0f;
            face_w = 0.0f;
            face_T = wall_T;
        }

        if (face_T <= 0.0f) continue;
        Real mu_face = sutherland_viscosity(face_T, T_ref, sutherland_T) * mu_ref;
        if (mu_face <= 0.0f) continue;

        Real nx = face.nx, ny = face.ny, nz = face.nz;
        Real area = face.area;

        Real div_u = du_dx + dv_dy + dw_dz;
        Real tau_xx = 2.0f * (du_dx - div_u / 3.0f);
        Real tau_yy = 2.0f * (dv_dy - div_u / 3.0f);
        Real tau_zz = 2.0f * (dw_dz - div_u / 3.0f);
        Real tau_xy = (du_dy + dv_dx);
        Real tau_xz = (du_dz + dw_dx);
        Real tau_yz = (dv_dz + dw_dy);

        Real mu_invRe = mu_face * inv_Re;
        Real visc_mom_x = (tau_xx*nx + tau_xy*ny + tau_xz*nz) * mu_invRe * area;
        Real visc_mom_y = (tau_xy*nx + tau_yy*ny + tau_yz*nz) * mu_invRe * area;
        Real visc_mom_z = (tau_xz*nx + tau_yz*ny + tau_zz*nz) * mu_invRe * area;

        Real dT_dn = dT_dx*nx + dT_dy*ny + dT_dz*nz;
        Real kappa_over_Re = mu_face * gamma / ((gamma - 1.0f) * prandtl) * inv_Re;
        Real visc_energy = (face_u * visc_mom_x + face_v * visc_mom_y + face_w * visc_mom_z
            + kappa_over_Re * dT_dn * area);

        residual[face.left_cell].mom_x += visc_mom_x;
        residual[face.left_cell].mom_y += visc_mom_y;
        residual[face.left_cell].mom_z += visc_mom_z;
        residual[face.left_cell].energy += visc_energy;

        if (face.boundary == BoundaryKind::Interior) {
            residual[face.right_cell].mom_x -= visc_mom_x;
            residual[face.right_cell].mom_y -= visc_mom_y;
            residual[face.right_cell].mom_z -= visc_mom_z;
            residual[face.right_cell].energy -= visc_energy;
        }

        if (!turbulence) continue;

        // SA conservative diffusion: (1/sigma) * div((mu/Re + rho*nu_tilde*fv1) * grad(nu_tilde))
        Real nu_tilde_L = q[face.left_cell].rho_nu_tilde / wL.rho;
        if (!std::isfinite(nu_tilde_L)) continue;

        Real dnu_dn, grad_dnu_dx, grad_dnu_dy, grad_dnu_dz;
        Real face_nu_tilde, face_rho;

        if (face.boundary == BoundaryKind::Interior) {
            PrimitiveState wR;
            if (primitive_override && primitive_override->size() == q.size()) {
                wR = (*primitive_override)[face.right_cell];
            } else if (!conservative_to_primitive(q[face.right_cell], gamma, wR)) {
                return false;
            }
            const PrimitiveGradient& gR = gradients[face.right_cell];
            Real nu_tilde_R = q[face.right_cell].rho_nu_tilde / wR.rho;
            if (!std::isfinite(nu_tilde_R)) continue;

            grad_dnu_dx = 0.5f * (gL.dnu_tilde_dx + gR.dnu_tilde_dx);
            grad_dnu_dy = 0.5f * (gL.dnu_tilde_dy + gR.dnu_tilde_dy);
            grad_dnu_dz = 0.5f * (gL.dnu_tilde_dz + gR.dnu_tilde_dz);

            Real dr_x = mesh.cells[face.right_cell].cx - mesh.cells[face.left_cell].cx;
            Real dr_y = mesh.cells[face.right_cell].cy - mesh.cells[face.left_cell].cy;
            Real dr_z = mesh.cells[face.right_cell].cz - mesh.cells[face.left_cell].cz;
            Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
            if (d2 > 1e-30f) {
                Real inv_d2 = 1.0f / d2;
                Real proj_dnu = grad_dnu_dx*dr_x + grad_dnu_dy*dr_y + grad_dnu_dz*dr_z;
                Real dnu_corr = ((nu_tilde_R - nu_tilde_L) - proj_dnu) * inv_d2;
                grad_dnu_dx += dnu_corr * dr_x;
                grad_dnu_dy += dnu_corr * dr_y;
                grad_dnu_dz += dnu_corr * dr_z;
            }

            face_nu_tilde = 0.5f * (nu_tilde_L + nu_tilde_R);
            face_rho = 0.5f * (wL.rho + wR.rho);
        } else {
            grad_dnu_dx = gL.dnu_tilde_dx;
            grad_dnu_dy = gL.dnu_tilde_dy;
            grad_dnu_dz = gL.dnu_tilde_dz;

            Real dr_x = face.cx - mesh.cells[face.left_cell].cx;
            Real dr_y = face.cy - mesh.cells[face.left_cell].cy;
            Real dr_z = face.cz - mesh.cells[face.left_cell].cz;
            Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
            if (d2 > 1e-30f) {
                Real inv_d2 = 1.0f / d2;
                Real proj_dnu = grad_dnu_dx*dr_x + grad_dnu_dy*dr_y + grad_dnu_dz*dr_z;
                Real dnu_corr = ((0.0f - nu_tilde_L) - proj_dnu) * inv_d2;
                grad_dnu_dx += dnu_corr * dr_x;
                grad_dnu_dy += dnu_corr * dr_y;
                grad_dnu_dz += dnu_corr * dr_z;
            }

            face_nu_tilde = 0.5f * nu_tilde_L;
            face_rho = wL.rho;
        }

        dnu_dn = grad_dnu_dx*nx + grad_dnu_dy*ny + grad_dnu_dz*nz;
        Real chi_face = Re * face_rho * face_nu_tilde / (mu_face + 1e-30f);
        Real chi3 = chi_face * chi_face * chi_face;
        Real fv1_face = chi3 / (chi3 + cv13 + 1e-30f);
        Real mu_tilde = face_rho * face_nu_tilde * fv1_face / sigma_sa;
        Real mu_total = (mu_face * inv_Re) / sigma_sa + mu_tilde;
        Real visc_nu = mu_total * dnu_dn * area;

        residual[face.left_cell].turbulence += visc_nu;
        if (face.boundary == BoundaryKind::Interior) {
            residual[face.right_cell].turbulence -= visc_nu;
        }
    }

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp

