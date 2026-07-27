#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/rans.hpp"
#include "aero/cfd/reconstruction.hpp"
#include "aero/cfd/viscous.hpp"
#include "aero/cfd/amr_sensor.hpp"
#include "aero/cfd/amr_interpolate.hpp"
#include "aero/cfd/amr_hanging.hpp"
#include "aero/cfd/rans_sst.hpp"

#include <algorithm>
#include <cmath>

#include <limits>

namespace aerosp {
namespace aero {
namespace cfd {

// ========== CPU SST pipeline functions ==========

bool compute_sst_gradients_cpu(
    const CfdMesh& mesh,
    const std::vector<Real>& k,
    const std::vector<Real>& omega,
    std::vector<Real>& grad_k,
    std::vector<Real>& grad_omega)
{
    std::size_t nc = mesh.cells.size();
    grad_k.assign(nc * 3, Real(0));
    grad_omega.assign(nc * 3, Real(0));

    for (const auto& face : mesh.faces) {
        int L = face.left_cell;
        int R = face.right_cell;
        if (L < 0 || static_cast<std::size_t>(L) >= nc) return false;
        Real k_face = k[L];
        Real w_face = omega[L];
        if (R >= 0 && static_cast<std::size_t>(R) < nc) {
            k_face = Real(0.5) * (k[L] + k[R]);
            w_face = Real(0.5) * (omega[L] + omega[R]);
        }
        Real ax = face.nx * face.area;
        Real ay = face.ny * face.area;
        Real az = face.nz * face.area;
        grad_k[L * 3 + 0] += k_face * ax;
        grad_k[L * 3 + 1] += k_face * ay;
        grad_k[L * 3 + 2] += k_face * az;
        grad_omega[L * 3 + 0] += w_face * ax;
        grad_omega[L * 3 + 1] += w_face * ay;
        grad_omega[L * 3 + 2] += w_face * az;
        if (R >= 0 && static_cast<std::size_t>(R) < nc) {
            grad_k[R * 3 + 0] -= k_face * ax;
            grad_k[R * 3 + 1] -= k_face * ay;
            grad_k[R * 3 + 2] -= k_face * az;
            grad_omega[R * 3 + 0] -= w_face * ax;
            grad_omega[R * 3 + 1] -= w_face * ay;
            grad_omega[R * 3 + 2] -= w_face * az;
        }
    }

    for (std::size_t i = 0; i < nc; ++i) {
        Real inv_vol = Real(1) / (mesh.cells[i].volume + Real(1e-30));
        grad_k[i * 3 + 0] *= inv_vol;
        grad_k[i * 3 + 1] *= inv_vol;
        grad_k[i * 3 + 2] *= inv_vol;
        grad_omega[i * 3 + 0] *= inv_vol;
        grad_omega[i * 3 + 1] *= inv_vol;
        grad_omega[i * 3 + 2] *= inv_vol;
    }
    return true;
}

bool compute_sst_advection_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<Real>& k,
    const std::vector<Real>& omega,
    Real inf_k, Real inf_omega,
    Real gamma,
    std::vector<Real>& res_k,
    std::vector<Real>& res_omega)
{
    std::size_t nc = mesh.cells.size();
    for (const auto& face : mesh.faces) {
        int L = face.left_cell;
        if (L < 0 || static_cast<std::size_t>(L) >= nc) return false;

        PrimitiveState wL;
        if (!conservative_to_primitive(q[L], gamma, wL)) return false;
        Real vn = wL.u * face.nx + wL.v * face.ny + wL.w * face.nz;

        Real k_donor = k[L], w_donor = omega[L];

        if (face.boundary == BoundaryKind::Interior) {
            int R = face.right_cell;
            if (R < 0 || static_cast<std::size_t>(R) >= nc) return false;
            PrimitiveState wR;
            if (!conservative_to_primitive(q[R], gamma, wR)) return false;
            Real vnR = wR.u * face.nx + wR.v * face.ny + wR.w * face.nz;
            vn = Real(0.5) * (vn + vnR);
            // Upwind: donor = upwind cell based on vn sign
            k_donor = (vn >= Real(0)) ? k[L] : k[R];
            w_donor = (vn >= Real(0)) ? omega[L] : omega[R];
        } else if (face.boundary == BoundaryKind::Farfield) {
            // Farfield: freestream on inflow, cell value on outflow
            if (vn < Real(0)) {
                k_donor = inf_k;
                w_donor = inf_omega;
            }
        } else if (face.boundary == BoundaryKind::SlipWall ||
                   face.boundary == BoundaryKind::NoSlipWall ||
                   face.boundary == BoundaryKind::Symmetry) {
            // Wall: zero flux (k and omega = 0 at wall)
            continue;
        } else {
            return false;
        }

        Real area = face.area;
        Real flux_k = vn * wL.rho * k_donor * area;
        Real flux_w = vn * wL.rho * w_donor * area;

        res_k[L] -= flux_k;
        res_omega[L] -= flux_w;
        if (face.boundary == BoundaryKind::Interior) {
            int R = face.right_cell;
            res_k[R] += flux_k;
            res_omega[R] += flux_w;
        }
    }
    return true;
}

bool compute_sst_diffusion_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<Real>& k,
    const std::vector<Real>& omega,
    const std::vector<Real>& grad_k,
    const std::vector<Real>& grad_omega,
    const std::vector<Real>& f1,
    Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    std::vector<Real>& res_k,
    std::vector<Real>& res_omega)
{
    std::size_t nc = mesh.cells.size();
    Real inv_Re = Real(1) / (Re > Real(0) ? Re : Real(1e6));

    for (const auto& face : mesh.faces) {
        int L = face.left_cell;
        if (L < 0 || static_cast<std::size_t>(L) >= nc) return false;
        int R = face.right_cell;

        PrimitiveState wL;
        if (!conservative_to_primitive(q[L], gamma, wL)) return false;
        Real T_L = wL.p / std::max(wL.rho, Real(1e-30));
        Real mu_L = sutherland_viscosity(T_L, T_ref, sutherland_T) * mu_ref;
        if (mu_L <= Real(0)) mu_L = Real(1);

        Real face_k, face_w, face_rho, face_mu, face_f1;
        Real dk_dn, dw_dn;

        if (R >= 0 && static_cast<std::size_t>(R) < nc &&
            face.boundary == BoundaryKind::Interior) {
            PrimitiveState wR;
            if (!conservative_to_primitive(q[R], gamma, wR)) return false;
            Real T_R = wR.p / std::max(wR.rho, Real(1e-30));
            Real mu_R = sutherland_viscosity(T_R, T_ref, sutherland_T) * mu_ref;
            if (mu_R <= Real(0)) mu_R = Real(1);

            face_k = Real(0.5) * (k[L] + k[R]);
            face_w = Real(0.5) * (omega[L] + omega[R]);
            face_rho = Real(0.5) * (wL.rho + wR.rho);
            face_mu = Real(0.5) * (mu_L + mu_R);
            Real f1_face = Real(0.5) * (f1[L] + f1[R]);

            // Orthogonal correction for gradients
            Real dr_x = mesh.cells[R].cx - mesh.cells[L].cx;
            Real dr_y = mesh.cells[R].cy - mesh.cells[L].cy;
            Real dr_z = mesh.cells[R].cz - mesh.cells[L].cz;
            Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;

            Real gk_dot_dr = Real(0.5) * (grad_k[L*3+0] + grad_k[R*3+0]) * dr_x
                           + Real(0.5) * (grad_k[L*3+1] + grad_k[R*3+1]) * dr_y
                           + Real(0.5) * (grad_k[L*3+2] + grad_k[R*3+2]) * dr_z;
            Real gw_dot_dr = Real(0.5) * (grad_omega[L*3+0] + grad_omega[R*3+0]) * dr_x
                           + Real(0.5) * (grad_omega[L*3+1] + grad_omega[R*3+1]) * dr_y
                           + Real(0.5) * (grad_omega[L*3+2] + grad_omega[R*3+2]) * dr_z;
            Real dk_corr = d2 > Real(1e-30) ? ((k[R] - k[L]) - gk_dot_dr) / d2 : Real(0);
            Real dw_corr = d2 > Real(1e-30) ? ((omega[R] - omega[L]) - gw_dot_dr) / d2 : Real(0);

            dk_dn = (Real(0.5)*(grad_k[L*3+0] + grad_k[R*3+0]) + dk_corr * dr_x) * face.nx
                  + (Real(0.5)*(grad_k[L*3+1] + grad_k[R*3+1]) + dk_corr * dr_y) * face.ny
                  + (Real(0.5)*(grad_k[L*3+2] + grad_k[R*3+2]) + dk_corr * dr_z) * face.nz;
            dw_dn = (Real(0.5)*(grad_omega[L*3+0] + grad_omega[R*3+0]) + dw_corr * dr_x) * face.nx
                  + (Real(0.5)*(grad_omega[L*3+1] + grad_omega[R*3+1]) + dw_corr * dr_y) * face.ny
                  + (Real(0.5)*(grad_omega[L*3+2] + grad_omega[R*3+2]) + dw_corr * dr_z) * face.nz;

            // Blended sigma
            Real sigma_k = f1_face * sst_coeff::sigma_k1 + (Real(1) - f1_face) * sst_coeff::sigma_k2;
            Real sigma_w = f1_face * sst_coeff::sigma_w1 + (Real(1) - f1_face) * sst_coeff::sigma_w2;

            Real mu_t = face_rho * face_k / (face_w + Real(1e-30));
            Real mu_eff_k = (face_mu * inv_Re) + sigma_k * mu_t;
            Real mu_eff_w = (face_mu * inv_Re) + sigma_w * mu_t;

            Real area = face.area;
            Real flux_k = mu_eff_k * dk_dn * area;
            Real flux_w = mu_eff_w * dw_dn * area;

            res_k[L] += flux_k;
            res_omega[L] += flux_w;
            res_k[R] -= flux_k;
            res_omega[R] -= flux_w;
        } else if (face.boundary == BoundaryKind::NoSlipWall) {
            // Wall: k=0, omega=6*nu/(beta1*y^2). Diffusion handled naturally.
            continue;
        }
    }
    return true;
}

bool compute_sst_source_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<Real>& k,
    const std::vector<Real>& omega,
    const std::vector<PrimitiveGradient>& flow_grads,
    const std::vector<Real>& grad_k,
    const std::vector<Real>& grad_omega,
    Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    std::vector<Real>& res_k,
    std::vector<Real>& res_omega,
    std::vector<Real>& f1)
{
    std::size_t nc = mesh.cells.size();
    f1.resize(nc);

    for (std::size_t i = 0; i < nc; ++i) {
        PrimitiveState wc;
        if (!conservative_to_primitive(q[i], gamma, wc)) return false;
        Real T = wc.p / std::max(wc.rho, Real(1e-30));
        Real mu = sutherland_viscosity(T, T_ref, sutherland_T) * mu_ref;
        if (mu <= Real(0)) mu = Real(1);

        Real d = mesh.cells[i].wall_distance;
        if (d <= Real(0) || !std::isfinite(d)) d = Real(1e30);

        Real k_val = std::max(Real(0), k[i]);
        Real omega_val = std::max(Real(1e-10), omega[i]);

        Real gk_x = grad_k[i * 3 + 0];
        Real gk_y = grad_k[i * 3 + 1];
        Real gk_z = grad_k[i * 3 + 2];
        Real gw_x = grad_omega[i * 3 + 0];
        Real gw_y = grad_omega[i * 3 + 1];
        Real gw_z = grad_omega[i * 3 + 2];

        SstBlending b = compute_sst_blending(k_val, omega_val, d,
            wc.rho, mu, gk_x, gk_y, gk_z, gw_x, gw_y, gw_z);
        f1[i] = b.F1;

        Real mu_t = wc.rho * k_val / (omega_val + Real(1e-30));

        // Strain magnitude from velocity gradients
        const auto& pg = flow_grads[i];
        Real S11 = pg.du_dx, S22 = pg.dv_dy, S33 = pg.dw_dz;
        Real S12 = Real(0.5) * (pg.du_dy + pg.dv_dx);
        Real S13 = Real(0.5) * (pg.du_dz + pg.dw_dx);
        Real S23 = Real(0.5) * (pg.dv_dz + pg.dw_dy);
        Real S_mag = std::sqrt(Real(2) * (S11*S11 + S22*S22 + S33*S33
                    + Real(2) * (S12*S12 + S13*S13 + S23*S23)) + Real(1e-30));

        SstSource s = compute_sst_source(k_val, omega_val, wc.rho, mu_t, b, S_mag);

        Real source_k = s.source_k;
        Real source_w = s.source_w;

        Real vol = mesh.cells[i].volume;
        res_k[i] += source_k * vol;
        res_omega[i] += source_w * vol;
    }
    return true;
}

void compact_mesh_nodes(CfdMesh& mesh) {
    int n_cells = static_cast<int>(mesh.cells.size());
    int n_faces = static_cast<int>(mesh.faces.size());
    if (n_cells == 0) return;
    std::vector<int> old_to_new(mesh.nodes.size(), -1);
    int new_count = 0;
    for (int ci = 0; ci < n_cells; ++ci) {
        const auto& cell = mesh.cells[ci];
        int nn = (cell.type == ElementType::TET4) ? 4 : 8;
        for (int i = 0; i < nn; ++i) {
            int nid = cell.node[i];
            if (nid >= 0 && nid < static_cast<int>(old_to_new.size()) && old_to_new[nid] < 0)
                old_to_new[nid] = new_count++;
        }
    }
    if (new_count == static_cast<int>(mesh.nodes.size())) return;
    std::vector<CfdNode> compact(new_count);
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i)
        if (old_to_new[i] >= 0) compact[old_to_new[i]] = mesh.nodes[i];
    for (int ci = 0; ci < n_cells; ++ci) {
        auto& cell = mesh.cells[ci];
        int nn = (cell.type == ElementType::TET4) ? 4 : 8;
        for (int i = 0; i < nn; ++i)
            if (cell.node[i] >= 0) cell.node[i] = old_to_new[cell.node[i]];
    }
    for (int fi = 0; fi < n_faces; ++fi) {
        auto& face = mesh.faces[fi];
        for (int i = 0; i < face.node_count; ++i)
            if (face.node[i] >= 0) face.node[i] = old_to_new[face.node[i]];
    }
    mesh.nodes.swap(compact);
}

namespace {

ConservativeState add_scaled(ConservativeState q, EulerFlux f, Real scale) {
    q.rho += scale * f.mass;
    q.rho_u += scale * f.mom_x;
    q.rho_v += scale * f.mom_y;
    q.rho_w += scale * f.mom_z;
    q.rho_E += scale * f.energy;
    q.rho_nu_tilde += scale * f.turbulence;
    return q;
}


} // namespace

void integrate_wall_forces(const CfdMesh& mesh, const std::vector<int>& wall_face_indices,
    const std::vector<ConservativeState>& q, const FreestreamCondition& condition,
    const CfdConfig& config, CfdForceResult& result,
    const std::vector<PrimitiveGradient>* grads,
    int body_id) {
    Real fx = 0.0f;
    Real fy = 0.0f;
    Real fz = 0.0f;
    Real mx = 0.0f;
    Real my = 0.0f;
    Real mz = 0.0f;
    Real qw = 0.0f;

    for (int idx : wall_face_indices) {
        const auto& face = mesh.faces[idx];
        if (face.left_cell < 0 || static_cast<std::size_t>(face.left_cell) >= q.size()) continue;
        if (body_id >= 0 && face.body_id != body_id) continue;
        PrimitiveState w;
        if (!conservative_to_primitive(q[face.left_cell], config.gamma, w)) continue;

        Real px = -w.p * face.nx * face.area;
        Real py = -w.p * face.ny * face.area;
        Real pz = -w.p * face.nz * face.area;
        fx += px;
        fy += py;
        fz += pz;
        mx += face.cy * pz - face.cz * py;
        my += face.cz * px - face.cx * pz;
        mz += face.cx * py - face.cy * px;

        if (config.viscous && face.boundary == BoundaryKind::NoSlipWall) {
            Real du_dx = 0.0f, du_dy = 0.0f, du_dz = 0.0f;
            Real dv_dx = 0.0f, dv_dy = 0.0f, dv_dz = 0.0f;
            Real dw_dx = 0.0f, dw_dy = 0.0f, dw_dz = 0.0f;
            Real dT_dx = 0.0f, dT_dy = 0.0f, dT_dz = 0.0f;

            Real dr_x = face.cx - mesh.cells[face.left_cell].cx;
            Real dr_y = face.cy - mesh.cells[face.left_cell].cy;
            Real dr_z = face.cz - mesh.cells[face.left_cell].cz;
            Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
            Real inv_d2 = 1.0f / (d2 + 1e-30f);

            int left_idx = face.left_cell;
            if (grads && static_cast<std::size_t>(left_idx) < grads->size()) {
                const PrimitiveGradient& g = (*grads)[left_idx];
                du_dx += g.du_dx; du_dy += g.du_dy; du_dz += g.du_dz;
                dv_dx += g.dv_dx; dv_dy += g.dv_dy; dv_dz += g.dv_dz;
                dw_dx += g.dw_dx; dw_dy += g.dw_dy; dw_dz += g.dw_dz;
                Real inv_rho2 = 1.0f / (w.rho * w.rho + 1e-30f);
                dT_dx = (w.rho * g.dp_dx - w.p * g.drho_dx) * inv_rho2;
                dT_dy = (w.rho * g.dp_dy - w.p * g.drho_dy) * inv_rho2;
                dT_dz = (w.rho * g.dp_dz - w.p * g.drho_dz) * inv_rho2;
                if (d2 > 1e-30f) {
                    Real proj_du = du_dx*dr_x + du_dy*dr_y + du_dz*dr_z;
                    Real proj_dv = dv_dx*dr_x + dv_dy*dr_y + dv_dz*dr_z;
                    Real proj_dw = dw_dx*dr_x + dw_dy*dr_y + dw_dz*dr_z;
                    Real proj_dT = dT_dx*dr_x + dT_dy*dr_y + dT_dz*dr_z;
                    Real du_corr = ((0.0f - w.u) - proj_du) * inv_d2;
                    Real dv_corr = ((0.0f - w.v) - proj_dv) * inv_d2;
                    Real dw_corr = ((0.0f - w.w) - proj_dw) * inv_d2;
                    Real T_cell = w.p / w.rho;
                    Real dT_corr = ((config.wall_temperature - T_cell) - proj_dT) * inv_d2;
                    du_dx += du_corr * dr_x; du_dy += du_corr * dr_y; du_dz += du_corr * dr_z;
                    dv_dx += dv_corr * dr_x; dv_dy += dv_corr * dr_y; dv_dz += dv_corr * dr_z;
                    dw_dx += dw_corr * dr_x; dw_dy += dw_corr * dr_y; dw_dz += dw_corr * dr_z;
                    dT_dx += dT_corr * dr_x; dT_dy += dT_corr * dr_y; dT_dz += dT_corr * dr_z;
                }
            } else {
                Real du_corr = (-w.u) * inv_d2;
                Real dv_corr = (-w.v) * inv_d2;
                Real dw_corr = (-w.w) * inv_d2;
                du_dx += du_corr * dr_x; du_dy += du_corr * dr_y; du_dz += du_corr * dr_z;
                dv_dx += dv_corr * dr_x; dv_dy += dv_corr * dr_y; dv_dz += dv_corr * dr_z;
                dw_dx += dw_corr * dr_x; dw_dy += dw_corr * dr_y; dw_dz += dw_corr * dr_z;
                Real T_cell = w.p / w.rho;
                Real dT_corr = (config.wall_temperature - T_cell) * inv_d2;
                dT_dx = dT_corr * dr_x; dT_dy = dT_corr * dr_y; dT_dz = dT_corr * dr_z;
            }

            Real div_u = du_dx + dv_dy + dw_dz;
            Real tau_xx = 2.0f * (du_dx - div_u / 3.0f);
            Real tau_yy = 2.0f * (dv_dy - div_u / 3.0f);
            Real tau_zz = 2.0f * (dw_dz - div_u / 3.0f);
            Real tau_xy = (du_dy + dv_dx);
            Real tau_xz = (du_dz + dw_dx);
            Real tau_yz = (dv_dz + dw_dy);

            Real mu_face = sutherland_viscosity(config.wall_temperature, config.T_ref, config.sutherland_T) * config.mu_ref;
            if (mu_face <= 0.0f) mu_face = config.mu_ref;

            Real inv_Re = 1.0f / (config.Re > 0.0f ? config.Re : 1e6f);
            Real mu_invRe = mu_face * inv_Re;

            Real tx = (tau_xx*face.nx + tau_xy*face.ny + tau_xz*face.nz) * mu_invRe * face.area;
            Real ty = (tau_xy*face.nx + tau_yy*face.ny + tau_yz*face.nz) * mu_invRe * face.area;
            Real tz = (tau_xz*face.nx + tau_yz*face.ny + tau_zz*face.nz) * mu_invRe * face.area;

            fx += tx;
            fy += ty;
            fz += tz;
            mx += face.cy * tz - face.cz * ty;
            my += face.cz * tx - face.cx * tz;
            mz += face.cx * ty - face.cy * tx;

            Real dT_dn = dT_dx * face.nx + dT_dy * face.ny + dT_dz * face.nz;
            Real conductivity = mu_face * config.gamma / ((config.gamma - 1.0f) * config.prandtl + 1e-30f);
            qw += -conductivity * dT_dn * inv_Re * face.area;
        }
    }

    Real q_inf = 0.5f * condition.mach * condition.mach;
    Real inv_force_ref = 1.0f / std::max(q_inf * config.ref_area, Real(1e-30));
    result.CX = fx * inv_force_ref;
    result.CY = fy * inv_force_ref;
    result.CZ = fz * inv_force_ref;
    result.Q_wall = qw * inv_force_ref;
    result.Cl = mx / std::max(q_inf * config.ref_area * config.ref_span, Real(1e-30));
    result.Cm = my / std::max(q_inf * config.ref_area * config.ref_length, Real(1e-30));
    result.Cn = mz / std::max(q_inf * config.ref_area * config.ref_span, Real(1e-30));

    Real alpha = condition.alpha_deg * 3.14159265358979323846 / 180.0;
    Real beta = condition.beta_deg * 3.14159265358979323846 / 180.0;
    Real ca = std::cos(alpha);
    Real sa = std::sin(alpha);
    Real cb = std::cos(beta);
    Real sb = std::sin(beta);
    Real fsx = result.CX * ca * cb + result.CY * sb + result.CZ * sa * cb;
    Real fsz = -result.CX * sa + result.CZ * ca;
    result.CD = -fsx;
    result.CL = -fsz;
}

PrimitiveState make_freestream(Real mach, Real alpha_deg, Real beta_deg, Real gamma) {
    Real alpha = alpha_deg * 3.14159265358979323846 / 180.0;
    Real beta = beta_deg * 3.14159265358979323846 / 180.0;

    PrimitiveState w;
    w.rho = 1.0f;
    w.p = 1.0f / gamma;
    w.u = -mach * std::cos(alpha) * std::cos(beta);
    w.v = -mach * std::sin(beta);
    w.w = -mach * std::sin(alpha) * std::cos(beta);
    return w;
}

PrimitiveState farfield_ghost_state(const PrimitiveState& left, const PrimitiveState& freestream, Real gamma,
    Real nx, Real ny, Real nz) {
    Real vn_inf = freestream.u*nx + freestream.v*ny + freestream.w*nz;
    Real a_inf = speed_of_sound(freestream, gamma);
    if (vn_inf >= a_inf) return left;
    return freestream;
}

EulerFlux hllc_flux(const PrimitiveState& left, const PrimitiveState& right, Real gamma, Real nx, Real ny, Real nz) {
    Real vn_l = left.u*nx + left.v*ny + left.w*nz;
    Real vn_r = right.u*nx + right.v*ny + right.w*nz;
    Real a_l = speed_of_sound(left, gamma);
    Real a_r = speed_of_sound(right, gamma);
    Real s_l = std::min(Real(0), std::min(vn_l - a_l, vn_r - a_r));
    Real s_r = std::max(Real(0), std::max(vn_l + a_l, vn_r + a_r));

    EulerFlux f_l = physical_flux(left, gamma, nx, ny, nz);
    EulerFlux f_r = physical_flux(right, gamma, nx, ny, nz);
    ConservativeState q_l = primitive_to_conservative(left, gamma);
    ConservativeState q_r = primitive_to_conservative(right, gamma);

    if (s_l >= 0.0f) return f_l;
    if (s_r <= 0.0f) return f_r;

    Real denom = left.rho * (s_l - vn_l) - right.rho * (s_r - vn_r);
    if (std::fabs(denom) < 1e-30f) denom = std::copysign(1e-30f, denom);
    Real s_m = (right.p - left.p + left.rho*vn_l*(s_l - vn_l) - right.rho*vn_r*(s_r - vn_r)) / denom;

    if (s_m >= 0.0f) {
        Real s_l_minus_sm = s_l - s_m;
        if (std::fabs(s_l_minus_sm) < 1e-30f) s_l_minus_sm = std::copysign(1e-30f, s_l_minus_sm);
        Real rho_star = left.rho * (s_l - vn_l) / s_l_minus_sm;
        Real e_l = q_l.rho_E / left.rho;
        Real p_star = left.p + left.rho * (s_l - vn_l) * (s_m - vn_l);
        Real e_star = e_l + (s_m - vn_l) * (s_m + left.p / (left.rho * (s_l - vn_l)));
        ConservativeState q_star;
        q_star.rho = rho_star;
        q_star.rho_u = rho_star * (left.u + (s_m - vn_l) * nx);
        q_star.rho_v = rho_star * (left.v + (s_m - vn_l) * ny);
        q_star.rho_w = rho_star * (left.w + (s_m - vn_l) * nz);
        q_star.rho_E = rho_star * e_star;
        q_star.rho_nu_tilde = q_l.rho_nu_tilde * (s_l - vn_l) / s_l_minus_sm;

        EulerFlux f = f_l;
        f.mass += s_l * (q_star.rho - q_l.rho);
        f.mom_x += s_l * (q_star.rho_u - q_l.rho_u);
        f.mom_y += s_l * (q_star.rho_v - q_l.rho_v);
        f.mom_z += s_l * (q_star.rho_w - q_l.rho_w);
        f.energy += s_l * (q_star.rho_E - q_l.rho_E);
        f.turbulence += s_l * (q_star.rho_nu_tilde - q_l.rho_nu_tilde);
        (void)p_star;
        return f;
    }

    Real s_r_minus_sm = s_r - s_m;
    if (std::fabs(s_r_minus_sm) < 1e-30f) s_r_minus_sm = std::copysign(1e-30f, s_r_minus_sm);
    Real rho_star = right.rho * (s_r - vn_r) / s_r_minus_sm;
    Real e_r = q_r.rho_E / right.rho;
    Real e_star = e_r + (s_m - vn_r) * (s_m + right.p / (right.rho * (s_r - vn_r)));
    ConservativeState q_star;
    q_star.rho = rho_star;
    q_star.rho_u = rho_star * (right.u + (s_m - vn_r) * nx);
    q_star.rho_v = rho_star * (right.v + (s_m - vn_r) * ny);
    q_star.rho_w = rho_star * (right.w + (s_m - vn_r) * nz);
    q_star.rho_E = rho_star * e_star;
    q_star.rho_nu_tilde = q_r.rho_nu_tilde * (s_r - vn_r) / s_r_minus_sm;

    EulerFlux f = f_r;
    f.mass += s_r * (q_star.rho - q_r.rho);
    f.mom_x += s_r * (q_star.rho_u - q_r.rho_u);
    f.mom_y += s_r * (q_star.rho_v - q_r.rho_v);
    f.mom_z += s_r * (q_star.rho_w - q_r.rho_w);
    f.energy += s_r * (q_star.rho_E - q_r.rho_E);
    f.turbulence += s_r * (q_star.rho_nu_tilde - q_r.rho_nu_tilde);
    return f;
}

bool CfdSolver::load_mesh(const CfdMesh& mesh) {
    mesh_ = mesh;
    wall_face_indices_.clear();

    double sx = 0, sy = 0, sz = 0;
    double total_area = 0;
    for (std::size_t i = 0; i < mesh_.faces.size(); ++i) {
        const auto& face = mesh_.faces[i];
        if (face.boundary == BoundaryKind::SlipWall || face.boundary == BoundaryKind::NoSlipWall)
            wall_face_indices_.push_back(static_cast<int>(i));
        if (face.boundary != BoundaryKind::Interior) {
            sx += static_cast<double>(face.area) * static_cast<double>(face.nx);
            sy += static_cast<double>(face.area) * static_cast<double>(face.ny);
            sz += static_cast<double>(face.area) * static_cast<double>(face.nz);
            total_area += static_cast<double>(face.area);
        }
    }

    double closure_error = std::sqrt(sx * sx + sy * sy + sz * sz);
    if (closure_error > 1e-4 * (total_area + 1e-30)) return false;

    return true;
}

CfdSolveSummary CfdSolver::solve(const FreestreamCondition& condition, const CfdConfig& config) {
    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    w_inf.nu_tilde = condition.nu_tilde;
    if (condition.nu_tilde_ratio > 0.0f && config.viscous) {
        Real T_inf = w_inf.p / w_inf.rho;
        Real t_ratio = T_inf / config.T_ref;
        Real mu_inf = config.mu_ref * t_ratio * std::sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T_inf + config.sutherland_T);
        w_inf.nu_tilde = sa_freestream_nu_tilde(condition.nu_tilde_ratio, mu_inf, w_inf.rho, config.Re);
    }
    ConservativeState q_inf = primitive_to_conservative(w_inf, config.gamma);
    std::vector<ConservativeState> q(mesh_.cells.size(), q_inf);
    return solve_from_state(condition, config, q);
}

CfdSolveSummary CfdSolver::solve_from_state(
    const FreestreamCondition& condition,
    const CfdConfig& config,
    const std::vector<ConservativeState>& initial_state) {
    CfdSolveSummary summary;
    if (mesh_.cells.empty() || mesh_.faces.empty()) {
        summary.failed = true;
        return summary;
    }
    if (initial_state.size() != mesh_.cells.size()) {
        summary.failed = true;
        return summary;
    }

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    w_inf.nu_tilde = condition.nu_tilde;
    std::vector<ConservativeState> q = initial_state;

    Real inf_k = 0, inf_omega = 0;
    if (config.turbulence_model == TurbulenceModel::SST) {
        Real U_inf = std::sqrt(w_inf.u*w_inf.u + w_inf.v*w_inf.v + w_inf.w*w_inf.w);
        Real tu = 0.001f;
        inf_k = 1.5f * (tu*tu) * U_inf*U_inf;
        Real t_ratio = w_inf.p / w_inf.rho / config.T_ref;
        Real mu_inf = config.mu_ref * t_ratio * std::sqrt(t_ratio)
                    * (config.T_ref + config.sutherland_T) / (w_inf.p / w_inf.rho + config.sutherland_T);
        Real nu_inf = mu_inf / w_inf.rho;
        inf_omega = inf_k / (0.09f * 0.1f * nu_inf + 1e-30f);
        sst_k_.assign(mesh_.cells.size(), inf_k);
        sst_omega_.assign(mesh_.cells.size(), inf_omega);
        sst_residual_k_.assign(mesh_.cells.size(), Real(0));
        sst_residual_omega_.assign(mesh_.cells.size(), Real(0));
        sst_grad_k_.assign(mesh_.cells.size() * 3, Real(0));
        sst_grad_omega_.assign(mesh_.cells.size() * 3, Real(0));
        sst_f1_.assign(mesh_.cells.size(), Real(0));
    }
    std::vector<ConservativeState> q_next;
    q_next.resize(mesh_.cells.size());
    std::vector<EulerFlux> residual;
    std::vector<PrimitiveGradient> grads, limited;
    std::vector<RansSource> sources;
    std::vector<PrimitiveState> w(mesh_.cells.size());
    std::vector<RefinementRecord> amr_records;
    bool diagnostics_enabled = config.diagnostic_level != DiagnosticLevel::Off;

    std::vector<Real> dt_cell(mesh_.cells.size(), Real(0));
    const bool use_lts = config.local_time_stepping;
    const bool use_sa = (config.turbulence_model == TurbulenceModel::SA ||
                         config.turbulence_model == TurbulenceModel::SA_DDES);
    Real cfl_adapt = Real(1); // residual-driven CFL multiplier (never increases past 1 without ramp)

    for (int iter = 0; iter < config.max_iter; ++iter) {
        Real cfl_now = config.cfl;
        if (config.implicit || config.cfl_ramp) {
            Real c0 = config.cfl_start > 0 ? config.cfl_start : config.cfl;
            Real c1 = config.cfl_end > c0 ? config.cfl_end : c0;
            int n_ramp = config.cfl_ramp_steps > 0 ? config.cfl_ramp_steps : 1;
            Real t = static_cast<Real>(iter) / static_cast<Real>(n_ramp);
            if (t > Real(1)) t = Real(1);
            cfl_now = c0 * std::pow(c1 / c0, t);
        }
        cfl_now *= cfl_adapt;
        if (cfl_now < Real(1e-6)) cfl_now = Real(1e-6);

        Real min_dt = std::numeric_limits<Real>::max();
        DtLimiterSnapshot dt_limiter;
        dt_limiter.iteration = iter;
        StateBounds iter_bounds;
        iter_bounds.min_rho = std::numeric_limits<Real>::max();
        iter_bounds.min_p = std::numeric_limits<Real>::max();
        iter_bounds.min_mach = std::numeric_limits<Real>::max();
        iter_bounds.max_rho = -std::numeric_limits<Real>::max();
        iter_bounds.max_p = -std::numeric_limits<Real>::max();
        iter_bounds.max_mach = -std::numeric_limits<Real>::max();
        iter_bounds.valid = true;
        for (std::size_t i = 0; i < q.size(); ++i) {
            if (!conservative_to_primitive(q[i], config.gamma, w[i])) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    const char* reason = (iter == 0) ? "invalid initial state" : "invalid state before timestep";
                    summary.diagnostics.failure = make_failure_snapshot(iter, static_cast<int>(i), reason, q[i], config.gamma);
                }
                return summary;
            }
            Real vmag = std::sqrt(w[i].u*w[i].u + w[i].v*w[i].v + w[i].w*w[i].w);
            Real a = speed_of_sound(w[i], config.gamma);
            Real signal_speed = vmag + a;
            Real h_min_val = mesh_.cells[i].h_min;
            if (h_min_val <= 0.0f) h_min_val = 1e-10f;
            Real dt = cfl_now * h_min_val / (signal_speed + Real(1e-30));
            Real mu = Real(0);
            Real nu_mol = Real(0);
            if (config.viscous) {
                Real T = w[i].p / w[i].rho;
                if (T > 0.0f) {
                    Real t_ratio = T / config.T_ref;
                    mu = config.mu_ref * t_ratio * std::sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T + config.sutherland_T);
                    if (mu > 0.0f) {
                        nu_mol = mu / (w[i].rho * config.Re + 1e-30f);
                        Real nu_t = Real(0);
                        if (use_sa)
                            nu_t = sa_eddy_viscosity(w[i].nu_tilde, w[i].rho, mu, config.Re);
                        Real nu_eff = nu_mol + std::max(nu_t, Real(0));
                        Real dt_visc = cfl_now * h_min_val * h_min_val / (nu_eff + 1e-30f);
                        if (dt_visc < dt) dt = dt_visc;
                    }
                }
            }
            if (use_sa && mu > 0) {
                Real d_wall = mesh_.cells[i].wall_distance;
                Real g_sa = sa_damping_rate(Real(0), w[i].nu_tilde, nu_mol, d_wall, h_min_val);
                g_sa = std::max(g_sa, sa_diffusive_rate(w[i].nu_tilde, nu_mol, h_min_val));
                if (g_sa > 0) {
                    Real dt_sa = cfl_now / g_sa;
                    if (dt_sa < dt) dt = dt_sa;
                }
            }
            if (!(dt > 0) || !std::isfinite(dt)) dt = Real(1e-30);
            dt_cell[i] = dt;
            if (dt < min_dt) {
                min_dt = dt;
                dt_limiter.cell = static_cast<int>(i);
                dt_limiter.dt = dt;
                dt_limiter.h_min = mesh_.cells[i].h_min;
                dt_limiter.signal_speed = signal_speed;
            }
            if (diagnostics_enabled) {
                Real mach = vmag / std::max(a, Real(1e-30));
                if (w[i].rho < iter_bounds.min_rho) iter_bounds.min_rho = w[i].rho;
                if (w[i].rho > iter_bounds.max_rho) iter_bounds.max_rho = w[i].rho;
                if (w[i].p < iter_bounds.min_p) iter_bounds.min_p = w[i].p;
                if (w[i].p > iter_bounds.max_p) iter_bounds.max_p = w[i].p;
                if (mach < iter_bounds.min_mach) iter_bounds.min_mach = mach;
                if (mach > iter_bounds.max_mach) iter_bounds.max_mach = mach;
            }
        }
        if (!(min_dt > 0) || !std::isfinite(min_dt)) min_dt = Real(1e-30);
        if (use_lts && config.lts_dt_ratio_max > 1) {
            Real dt_cap = min_dt * config.lts_dt_ratio_max;
            for (std::size_t i = 0; i < dt_cell.size(); ++i)
                if (dt_cell[i] > dt_cap) dt_cell[i] = dt_cap;
        } else {
            for (std::size_t i = 0; i < dt_cell.size(); ++i)
                dt_cell[i] = min_dt;
        }
        if (diagnostics_enabled) {
            summary.diagnostics.dt_limiter_history.push_back(dt_limiter);
            summary.diagnostics.state_bounds_history.push_back(iter_bounds);
        }

        bool need_gradients = (config.reconstruction_order == 2) || config.viscous || (config.turbulence_model != TurbulenceModel::LAMINAR);
        std::vector<PrimitiveLimiter> limiters_vec;
        bool apply_limiting = config.reconstruction_order == 2 || (config.turbulence_model != TurbulenceModel::LAMINAR);

        if (need_gradients) {
            grads = compute_green_gauss_gradients(mesh_, q, config.gamma, &w);
            if (grads.size() != mesh_.cells.size()) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "gradient computation failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
            if (apply_limiting) {
                limiters_vec = compute_barth_jespersen_limiters(mesh_, q, grads, config.gamma, &w);
                if (limiters_vec.size() != mesh_.cells.size()) {
                    summary.failed = true;
                    if (diagnostics_enabled) {
                        summary.diagnostics.failure.reason = "limiter computation failed";
                        summary.diagnostics.failure.valid = true;
                        summary.diagnostics.failure.iteration = iter;
                    }
                    return summary;
                }
                limited.resize(grads.size());
                for (std::size_t i = 0; i < grads.size(); ++i)
                    limited[i] = apply_limiter(grads[i], limiters_vec[i]);
            }
        }

        if (config.reconstruction_order == 2) {
            if (!compute_euler_residual_cpu_order2(mesh_, q, w_inf, config.gamma, limited, residual, &w, config.mms_solution)) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "order2 residual assembly failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
        } else {
            if (!compute_euler_residual_cpu(mesh_, q, w_inf, config.gamma, residual, &w, config.mms_solution)) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "residual assembly failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
        }

        // ----- Hanging face flux correction (AMR refinement boundaries) -----
        // Reconstructs the coarse-side state in primitive space using the limited
        // gradient (positive-preserving), then replaces the residual flux at each
        // hanging face with the corrected value.
        if (!amr_records.empty() && need_gradients && !grads.empty()) {
            auto hanging_faces = detect_hanging_faces(mesh_);
            if (!hanging_faces.empty()) {
                const auto& gsrc = apply_limiting ? limited : grads;
                apply_hanging_flux_correction_primitive(mesh_, hanging_faces, q, w, gsrc, config.gamma, residual);
            }
        }

        if (config.viscous) {
            const auto& visc_grads = apply_limiting ? limited : grads;
            const std::vector<Real>* sst_k_ptr = nullptr;
            const std::vector<Real>* sst_omega_ptr = nullptr;
            if (config.turbulence_model == TurbulenceModel::SST) {
                sst_k_ptr = &sst_k_;
                sst_omega_ptr = &sst_omega_;
            }
            if (!compute_viscous_flux_cpu(mesh_, q, visc_grads, config.gamma,
                    config.prandtl, config.mu_ref, config.T_ref,
                    config.sutherland_T, config.Re, config.wall_temperature,
                    static_cast<int>(config.turbulence_model), residual, &w,
                    sst_k_ptr, sst_omega_ptr)) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "viscous flux failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
        }

        if (config.turbulence_model == TurbulenceModel::SST) {
            // SST pipeline: gradients → advection → diffusion → source
            sst_residual_k_.assign(mesh_.cells.size(), Real(0));
            sst_residual_omega_.assign(mesh_.cells.size(), Real(0));

            if (!compute_sst_gradients_cpu(mesh_, sst_k_, sst_omega_, sst_grad_k_, sst_grad_omega_)) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "SST gradient computation failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
            if (!compute_sst_advection_cpu(mesh_, q, sst_k_, sst_omega_,
                    inf_k, inf_omega, config.gamma,
                    sst_residual_k_, sst_residual_omega_)) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "SST advection failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
            if (config.viscous) {
                if (!compute_sst_diffusion_cpu(mesh_, q, sst_k_, sst_omega_,
                        sst_grad_k_, sst_grad_omega_, sst_f1_,
                        config.gamma, config.Re, config.mu_ref, config.T_ref,
                        config.sutherland_T,
                        sst_residual_k_, sst_residual_omega_)) {
                    summary.failed = true;
                    if (diagnostics_enabled) {
                        summary.diagnostics.failure.reason = "SST diffusion failed";
                        summary.diagnostics.failure.valid = true;
                        summary.diagnostics.failure.iteration = iter;
                    }
                    return summary;
                }
            }
            if (!compute_sst_source_cpu(mesh_, q, sst_k_, sst_omega_,
                    limited, sst_grad_k_, sst_grad_omega_,
                    config.gamma, config.Re, config.mu_ref, config.T_ref,
                    config.sutherland_T,
                    sst_residual_k_, sst_residual_omega_, sst_f1_)) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "SST source failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
        } else if (config.turbulence_model != TurbulenceModel::LAMINAR) {
            sources = compute_rans_sources(mesh_, q, limited, config.gamma, config.Re, &w);
            if (sources.size() != mesh_.cells.size()) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "RANS source computation failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
            for (std::size_t i = 0; i < q.size(); ++i)
                residual[i].turbulence += sources[i].total_source * mesh_.cells[i].volume;
        }

        // MMS source injection (all components, incl. turbulence).
        // Must come before semi-implicit correction: the MMS source S_mms = R(q_exact)
        // should cancel the explicit residual so that q=q_exact gives residual=0.
        if (!config.mms_source.empty()) {
            for (std::size_t i = 0; i < q.size(); ++i) {
                residual[i].mass      -= config.mms_source[i].mass;
                residual[i].mom_x     -= config.mms_source[i].mom_x;
                residual[i].mom_y     -= config.mms_source[i].mom_y;
                residual[i].mom_z     -= config.mms_source[i].mom_z;
                residual[i].energy    -= config.mms_source[i].energy;
                residual[i].turbulence -= config.mms_source[i].turbulence;
            }
        }

        // Spatial residual L2 (before any point-implicit transforms) — true
        // steady-state convergence metric. PI only changes the update map.
        Real l2_spatial = 0.0f;
        for (std::size_t i = 0; i < q.size(); ++i) {
            l2_spatial += residual[i].mass * residual[i].mass
                + residual[i].mom_x * residual[i].mom_x
                + residual[i].mom_y * residual[i].mom_y
                + residual[i].mom_z * residual[i].mom_z
                + residual[i].energy * residual[i].energy
                + residual[i].turbulence * residual[i].turbulence;
        }

        // Mean-flow spectral point-implicit: R_eff = R / (1 + dt*lambda)
        // lambda ≈ |u|/h + a/h + nu_eff/h^2  (no zero-state pull on q).
        if (config.mean_flow_point_implicit && config.mms_source.empty()) {
            for (std::size_t i = 0; i < q.size(); ++i) {
                Real h = mesh_.cells[i].h_min;
                if (!(h > 0) || !std::isfinite(h)) h = 1e-10f;
                Real vmag = std::sqrt(w[i].u*w[i].u + w[i].v*w[i].v + w[i].w*w[i].w);
                Real a = speed_of_sound(w[i], config.gamma);
                Real lambda = (vmag + a) / h;
                if (config.viscous) {
                    Real T = w[i].p / std::max(w[i].rho, Real(1e-30));
                    Real mu = sutherland_viscosity(T, config.T_ref, config.sutherland_T);
                    if (!(mu > 0)) mu = config.mu_ref;
                    Real nu_mol = mu / (w[i].rho * config.Re + 1e-30f);
                    Real nu_t = use_sa
                        ? sa_eddy_viscosity(w[i].nu_tilde, w[i].rho, mu, config.Re)
                        : Real(0);
                    lambda += (nu_mol + std::max(nu_t, Real(0))) / (h * h + 1e-30f);
                }
                Real f = 1.0f / (1.0f + dt_cell[i] * lambda + 1e-30f);
                residual[i].mass *= f;
                residual[i].mom_x *= f;
                residual[i].mom_y *= f;
                residual[i].mom_z *= f;
                residual[i].energy *= f;
            }
        }

        // SA full-residual point-implicit (sources + inviscid/viscous turb fluxes):
        //   U_new = (U + dt_i/V * R) / (1 + dt_i*g)
        // Sub-iters re-linearize volume source against frozen face fluxes (R_flux).
        if (use_sa &&
            config.mms_source.empty() &&
            sources.size() == q.size()) {
            std::vector<Real> R_flux(q.size(), Real(0));
            for (std::size_t i = 0; i < q.size(); ++i) {
                Real V = mesh_.cells[i].volume;
                Real R_src = sources[i].total_source * V;
                if (!std::isfinite(R_src)) R_src = 0;
                R_flux[i] = residual[i].turbulence - R_src;
            }
            const int n_sa_pass = 1 + std::max(0, config.sa_sub_iters);
            for (int sa_pass = 0; sa_pass < n_sa_pass; ++sa_pass) {
                if (sa_pass > 0) {
                    for (std::size_t i = 0; i < q.size(); ++i) {
                        Real rho = q[i].rho;
                        if (!(rho > 0) || !std::isfinite(rho)) continue;
                        Real V = mesh_.cells[i].volume;
                        if (!(V > 0)) continue;
                        Real dt_i = dt_cell[i];
                        Real U_prov = q[i].rho_nu_tilde + (dt_i / V) * residual[i].turbulence;
                        PrimitiveState wp = w[i];
                        wp.nu_tilde = U_prov / rho;
                        Real T = w[i].p / std::max(w[i].rho, Real(1e-30));
                        Real mu = sutherland_viscosity(T, config.T_ref, config.sutherland_T);
                        if (!(mu > 0)) mu = config.mu_ref;
                        const PrimitiveGradient& ggrad =
                            (apply_limiting && limited.size() == q.size()) ? limited[i]
                            : (grads.size() == q.size() ? grads[i] : PrimitiveGradient{});
                        sources[i] = compute_rans_source(
                            wp, ggrad, mesh_.cells[i].wall_distance, mu, rho, config.Re);
                    }
                }
                for (std::size_t i = 0; i < q.size(); ++i) {
                    Real rho = q[i].rho;
                    if (!(rho > 0) || !std::isfinite(rho)) continue;
                    Real nu_tilde = (sa_pass == 0)
                        ? q[i].rho_nu_tilde / rho
                        : (q[i].rho_nu_tilde + (dt_cell[i] / mesh_.cells[i].volume) * residual[i].turbulence) / rho;
                    if (!std::isfinite(nu_tilde)) continue;

                    Real V = mesh_.cells[i].volume;
                    if (!(V > 0) || !std::isfinite(V)) continue;
                    Real dt_i = dt_cell[i];

                    Real S = sources[i].total_source / rho;
                    Real T = w[i].p / std::max(w[i].rho, Real(1e-30));
                    Real mu = sutherland_viscosity(T, config.T_ref, config.sutherland_T);
                    if (!(mu > 0)) mu = config.mu_ref;
                    Real nu_mol = mu / (rho * config.Re + 1e-30f);
                    Real S_lim = sa_limit_source(S, nu_tilde, nu_mol, dt_i, Real(1));
                    Real R_src = rho * S_lim * V;
                    sources[i].total_source = rho * S_lim;
                    residual[i].turbulence = R_flux[i] + R_src;

                    Real d = mesh_.cells[i].wall_distance;
                    Real g = sa_damping_rate(sources[i].dS_dnu, nu_tilde, nu_mol, d,
                                             mesh_.cells[i].h_min);
                    Real f = 1.0f / (1.0f + dt_i * g + 1e-30f);
                    Real dtv = dt_i / V;
                    Real U = q[i].rho_nu_tilde;
                    Real R = residual[i].turbulence;
                    residual[i].turbulence = R * f + U * (f - 1.0f) / (dtv + 1e-30f);
                }
            }
        }

        Real l2 = 0.0f;
        for (std::size_t i = 0; i < q.size(); ++i) {
            Real scale = dt_cell[i] / mesh_.cells[i].volume;
            q_next[i] = add_scaled(q[i], residual[i], scale);
            // SA-neg floor + soft ceiling (prevents chi^3 overflow runaway)
            if (config.turbulence_model != TurbulenceModel::LAMINAR &&
                config.turbulence_model != TurbulenceModel::SST &&
                q_next[i].rho > 0) {
                Real T = w[i].p / std::max(w[i].rho, Real(1e-30));
                Real mu = sutherland_viscosity(T, config.T_ref, config.sutherland_T);
                if (!(mu > 0)) mu = config.mu_ref;
                Real nu_mol = mu / (q_next[i].rho * config.Re + 1e-30f);
                Real nu_floor = sa_nu_tilde_floor(nu_mol);
                Real nu_ceil = sa_nu_tilde_ceil(nu_mol);
                Real nu_n = q_next[i].rho_nu_tilde / q_next[i].rho;
                if (nu_n < nu_floor)
                    q_next[i].rho_nu_tilde = nu_floor * q_next[i].rho;
                else if (nu_n > nu_ceil)
                    q_next[i].rho_nu_tilde = nu_ceil * q_next[i].rho;
            }
            l2 += residual[i].mass * residual[i].mass
                + residual[i].mom_x * residual[i].mom_x
                + residual[i].mom_y * residual[i].mom_y
                + residual[i].mom_z * residual[i].mom_z
                + residual[i].energy * residual[i].energy
                + residual[i].turbulence * residual[i].turbulence;
        }
        Real nvar_eff = static_cast<Real>(CFD_NVAR);
        if (config.turbulence_model == TurbulenceModel::SST) {
            nvar_eff += 2;
            for (std::size_t i = 0; i < q.size(); ++i) {
                l2_spatial += sst_residual_k_[i] * sst_residual_k_[i]
                    + sst_residual_omega_[i] * sst_residual_omega_[i];
            }
        }
        (void)l2; // post-PI residual kept for optional future diagnostics
        Real residual_l2 = std::sqrt(l2_spatial / (nvar_eff * static_cast<Real>(q.size())));
        // Residual-adaptive CFL: cut on spikes; recover on decrease or plateau.
        if (!summary.residual_history.empty() && std::isfinite(residual_l2)) {
            Real r_prev = summary.residual_history.back();
            if (r_prev > 0 && residual_l2 > r_prev * Real(2)) {
                cfl_adapt *= Real(0.5);
                if (cfl_adapt < Real(1e-3)) cfl_adapt = Real(1e-3);
            } else if (r_prev > 0 && residual_l2 < r_prev * Real(0.9)) {
                cfl_adapt = std::min(Real(1), cfl_adapt * Real(1.08));
            } else if (r_prev > 0) {
                Real rel = std::fabs(residual_l2 - r_prev) / r_prev;
                if (rel < Real(1e-3))
                    cfl_adapt = std::min(Real(1), cfl_adapt * Real(1.02));
            }
        }
        summary.residual_history.push_back(residual_l2);

        // Reject invalid updates: keep previous state and cut CFL instead of failing later.
        bool update_ok = true;
        for (std::size_t i = 0; i < q_next.size(); ++i) {
            PrimitiveState wn;
            if (!conservative_to_primitive(q_next[i], config.gamma, wn)) {
                update_ok = false;
                break;
            }
        }
        if (!update_ok) {
            cfl_adapt *= Real(0.25);
            if (cfl_adapt < Real(1e-4)) cfl_adapt = Real(1e-4);
            // Keep q; skip swap. Still recorded residual of the rejected step.
            continue;
        }
        q.swap(q_next);

        // SST explicit update for k and omega
        if (config.turbulence_model == TurbulenceModel::SST) {
            for (std::size_t i = 0; i < mesh_.cells.size(); ++i) {
                Real dt_over_V = dt_cell[i] / (mesh_.cells[i].volume + Real(1e-30));
                sst_k_[i] += dt_over_V * sst_residual_k_[i];
                sst_omega_[i] += dt_over_V * sst_residual_omega_[i];
                if (sst_k_[i] < Real(0)) sst_k_[i] = Real(0);
                if (sst_omega_[i] <= Real(0)) sst_omega_[i] = Real(1e-10);
            }
        }

        if (config.amr.enabled && iter > 0 && config.amr.interval > 0 && (iter % config.amr.interval) == 0) {
            CfdMesh mesh_old = mesh_;
            std::vector<ConservativeState> q_old = q;

            // Collect refinement requests from all active sensors
            std::vector<std::vector<RefinementRequest>> sensor_outputs;

            // 1. Euler gradient sensor (density/pressure/velocity jumps) — always active
            sensor_outputs.push_back(
                compute_gradient_sensor(mesh_, q, config.amr, config.gamma));

            // Turbulence-aware sensors only activate for non-LAMINAR models.
            // For LAMINAR this preserves exact Phase 12 regression behavior.
            if (config.turbulence_model != TurbulenceModel::LAMINAR) {
                // 2. y+ sensor (wall-adjacent cells)
                if (config.amr.yplus_target > Real(0)) {
                    sensor_outputs.push_back(
                        compute_yplus_sensor(mesh_, q, config.amr, config.gamma,
                            config.Re, config.mu_ref, config.T_ref,
                            config.sutherland_T, config.turbulence_model));
                }

                // 3. Q-criterion sensor (vortex/wake detection)
                sensor_outputs.push_back(
                    compute_qcriterion_sensor(mesh_, q, config.amr, config.gamma));

                // 4. Wake cone sensor (geometric refinement region)
                if (config.amr.wake_cone.length > Real(0)) {
                    sensor_outputs.push_back(
                        compute_wake_cone_sensor(mesh_, config.amr, config.amr.wake_cone));
                }

                // 5. TKE ratio sensor (k / 0.5*U² — requires SST k data; no-op for
                //    non-SST models via nullptr sst_k)
                if (config.amr.tke_ratio_threshold > Real(0)) {
                    const std::vector<Real>* sst_k_ptr =
                        (config.turbulence_model == TurbulenceModel::SST) ? &sst_k_ : nullptr;
                    sensor_outputs.push_back(
                        compute_tke_ratio_sensor(mesh_, q, config.amr, config.gamma,
                            config.turbulence_model, sst_k_ptr,
                            config.amr.tke_ratio_threshold));
                }

                // 6. Shear-layer sensor (resolved_k / (resolved_k + modeled_k) ratio)
                if (config.amr.shear_layer_threshold > Real(0)) {
                    const std::vector<Real>* sst_k_ptr =
                        (config.turbulence_model == TurbulenceModel::SST) ? &sst_k_ : nullptr;
                    sensor_outputs.push_back(
                        compute_shear_layer_sensor(mesh_, q, config.amr, config.gamma,
                            config.turbulence_model, sst_k_ptr,
                            config.amr.shear_layer_threshold));
                }
            }

            // Merge all sensor outputs: Refine dominates, all Coarsen → Coarsen
            auto requests = merge_refinement_requests(sensor_outputs);

            // Enforce 2:1 balance after coarsening: prevent coarsening a cell if
            // a face neighbor at a higher refinement level is not also being coarsened.
            {
                std::vector<bool> is_coarsening_request(mesh_.cells.size(), false);
                for (const auto& req : requests)
                    if (req.flag == RefinementFlag::Coarsen)
                        is_coarsening_request[req.cell_id] = true;
                for (auto& req : requests) {
                    if (req.flag != RefinementFlag::Coarsen) continue;
                    int cl = mesh_.cells[req.cell_id].refinement_level;
                    for (const auto& face : mesh_.faces) {
                        int nbr = -1;
                        if (face.left_cell == req.cell_id) nbr = face.right_cell;
                        if (face.right_cell == req.cell_id) nbr = face.left_cell;
                        if (nbr < 0) continue;
                        int nl = mesh_.cells[nbr].refinement_level;
                        if (nl > cl && !is_coarsening_request[nbr]) {
                            req.flag = RefinementFlag::Unchanged;
                            break;
                        }
                    }
                }
            }

            bool has_work = false;
            for (const auto& req : requests)
                if (req.flag != RefinementFlag::Unchanged) { has_work = true; break; }

            if (has_work) {
                // Anisotropic cascade: cells at or above anisotropic_layers use isotropic split
                if (config.amr.anisotropic_layers > 0) {
                    for (auto& req : requests) {
                        if (req.flag == RefinementFlag::Refine &&
                            req.dir != AnisotropicDir::NONE &&
                            mesh_.cells[req.cell_id].refinement_level >= config.amr.anisotropic_layers) {
                            req.dir = AnisotropicDir::NONE;
                        }
                    }
                }
                std::vector<RefinementRecord> new_records;
                std::vector<CoarsenInfo> coarsen_info;
                std::string err;
                const std::vector<RefinementRecord>* prev = amr_records.empty() ? nullptr : &amr_records;
                if (refine_cells(mesh_, requests, &new_records, &err, prev, &coarsen_info, config.amr.max_level)) {
                    // Save SST k/omega state before AMR changes arrays
                    std::vector<Real> sst_k_old, sst_omega_old;
                    if (config.turbulence_model == TurbulenceModel::SST) {
                        sst_k_old = sst_k_;
                        sst_omega_old = sst_omega_;
                    }

                    if (need_gradients && !grads.empty() && !w.empty()) {
                        const auto& gsrc = apply_limiting ? limited : grads;
                        prolongate_solution_order2(q_old, w, gsrc, mesh_old, mesh_, new_records, config.gamma, q);
                    } else {
                        prolongate_solution(q_old, mesh_old, mesh_, new_records, q);
                    }

                    // Prolong SST k/omega for refined children (injection from parent)
                    // and copy unchanged cells from old arrays using centroid matching.
                    if (config.turbulence_model == TurbulenceModel::SST && !sst_k_old.empty()) {
                        std::vector<Real> new_k(mesh_.cells.size(), Real(0));
                        std::vector<Real> new_w(mesh_.cells.size(), Real(0));
                        for (const auto& rec : new_records) {
                            int parent = rec.parent_cell_id;
                            if (parent < 0 || static_cast<std::size_t>(parent) >= sst_k_old.size()) continue;
                            for (int c = 0; c < rec.n_children; ++c) {
                                int child = rec.child_cell_ids[c];
                                if (child >= 0 && child < static_cast<int>(new_k.size())) {
                                    new_k[child] = sst_k_old[parent];
                                    new_w[child] = sst_omega_old[parent];
                                }
                            }
                        }
                        for (std::size_t i = 0; i < mesh_.cells.size(); ++i) {
                            if (new_k[i] != Real(0)) continue;
                            for (std::size_t j = 0; j < mesh_old.cells.size(); ++j) {
                                if (mesh_.cells[i].cx == mesh_old.cells[j].cx &&
                                    mesh_.cells[i].cy == mesh_old.cells[j].cy &&
                                    mesh_.cells[i].cz == mesh_old.cells[j].cz) {
                                    new_k[i] = sst_k_old[j];
                                    new_w[i] = sst_omega_old[j];
                                    break;
                                }
                            }
                        }
                        sst_k_.swap(new_k);
                        sst_omega_.swap(new_w);
                        for (auto& v : sst_k_) if (v < Real(0)) v = Real(0);
                        for (auto& v : sst_omega_) if (v <= Real(0)) v = Real(1e-10);
                    }

                    // Clamp SA turbulence variables on newly created cells
                    for (const auto& rec : new_records) {
                        for (int c = 0; c < rec.n_children; ++c) {
                            int child_id = rec.child_cell_ids[c];
                            if (child_id < 0 || child_id >= static_cast<int>(q.size())) continue;
                            if (config.turbulence_model == TurbulenceModel::SA ||
                                config.turbulence_model == TurbulenceModel::SA_DDES) {
                                PrimitiveState wc;
                                if (conservative_to_primitive(q[child_id], config.gamma, wc) && wc.nu_tilde < Real(1e-8))
                                    q[child_id].rho_nu_tilde = wc.rho * Real(1e-8);
                            }
                        }
                    }
                    for (const auto& ci : coarsen_info) {
                        Real vol_sum = 0.0f;
                        ConservativeState avg;
                        Real k_avg = 0, w_avg = 0;
                        for (int c = 0; c < ci.n_children; ++c) {
                            int child_id = ci.old_child_ids[c];
                            if (child_id < 0 || child_id >= static_cast<int>(q_old.size())) continue;
                            Real vol = mesh_old.cells[child_id].volume;
                            vol_sum += vol;
                            avg.rho += q_old[child_id].rho * vol;
                            avg.rho_u += q_old[child_id].rho_u * vol;
                            avg.rho_v += q_old[child_id].rho_v * vol;
                            avg.rho_w += q_old[child_id].rho_w * vol;
                            avg.rho_E += q_old[child_id].rho_E * vol;
                            avg.rho_nu_tilde += q_old[child_id].rho_nu_tilde * vol;
                            if (config.turbulence_model == TurbulenceModel::SST &&
                                !sst_k_old.empty() && static_cast<std::size_t>(child_id) < sst_k_old.size()) {
                                k_avg += sst_k_old[child_id] * vol;
                                w_avg += sst_omega_old[child_id] * vol;
                            }
                        }
                        if (vol_sum > 0.0f) {
                            Real inv = 1.0f / vol_sum;
                            avg.rho *= inv; avg.rho_u *= inv; avg.rho_v *= inv;
                            avg.rho_w *= inv; avg.rho_E *= inv; avg.rho_nu_tilde *= inv;
                            if (config.turbulence_model == TurbulenceModel::SST &&
                                !sst_k_old.empty() && ci.new_parent_id >= 0 &&
                                ci.new_parent_id < static_cast<int>(sst_k_.size())) {
                                sst_k_[ci.new_parent_id] = k_avg / vol_sum;
                                sst_omega_[ci.new_parent_id] = w_avg / vol_sum;
                            }
                        }
                        q[ci.new_parent_id] = avg;
                        // Clamp turbulence on coarsened parent
                        if (config.turbulence_model == TurbulenceModel::SA ||
                            config.turbulence_model == TurbulenceModel::SA_DDES) {
                            PrimitiveState wc;
                            if (conservative_to_primitive(q[ci.new_parent_id], config.gamma, wc) && wc.nu_tilde < Real(1e-8))
                                q[ci.new_parent_id].rho_nu_tilde = wc.rho * Real(1e-8);
                        }
                    }
                    amr_records.insert(amr_records.end(), new_records.begin(), new_records.end());
                    if (!coarsen_info.empty()) {
                        std::vector<int> stale_parents;
                        stale_parents.reserve(coarsen_info.size());
                        for (const auto& ci : coarsen_info)
                            stale_parents.push_back(ci.old_parent_id);
                        std::sort(stale_parents.begin(), stale_parents.end());
                        amr_records.erase(
                            std::remove_if(amr_records.begin(), amr_records.end(),
                                [&](const RefinementRecord& rec) {
                                    return std::binary_search(stale_parents.begin(), stale_parents.end(), rec.parent_cell_id);
                                }),
                            amr_records.end());
                    }
                    compute_mesh_metrics(mesh_);
                    compact_mesh_nodes(mesh_);
                    int n_new = static_cast<int>(mesh_.cells.size());
                    q_next.resize(n_new);
                    residual.resize(n_new);
                    grads.assign(n_new, PrimitiveGradient{});
                    w.assign(n_new, PrimitiveState{});
                    if (!limited.empty()) limited.assign(n_new, PrimitiveGradient{});
                    if (!sources.empty()) sources.assign(n_new, RansSource{});
                    if (config.turbulence_model == TurbulenceModel::SST) {
                        sst_residual_k_.assign(n_new, Real(0));
                        sst_residual_omega_.assign(n_new, Real(0));
                        sst_grad_k_.assign(n_new * 3, Real(0));
                        sst_grad_omega_.assign(n_new * 3, Real(0));
                        sst_f1_.assign(n_new, Real(0));
                        // sst_k_ and sst_omega_ were already resized in prolongation
                        if (static_cast<int>(sst_k_.size()) != n_new)
                            sst_k_.resize(n_new, Real(0));
                        if (static_cast<int>(sst_omega_.size()) != n_new)
                            sst_omega_.resize(n_new, Real(1e-10));
                    }
                    wall_face_indices_.clear();
                    for (int fi = 0; fi < static_cast<int>(mesh_.faces.size()); ++fi) {
                        auto bc = mesh_.faces[fi].boundary;
                        if (bc == BoundaryKind::SlipWall || bc == BoundaryKind::NoSlipWall)
                            wall_face_indices_.push_back(fi);
                    }
                    residual_l2 = config.convergence_tol;
                    // Recompute w and gradients after AMR so wall-force integration
                    // (which runs after the loop exit) uses valid, mesh-consistent data.
                    for (int i = 0; i < n_new; ++i)
                        conservative_to_primitive(q[i], config.gamma, w[i]);
                    if (need_gradients) {
                        grads = compute_green_gauss_gradients(mesh_, q, config.gamma, &w);
                        if (apply_limiting) {
                            auto amr_limiters = compute_barth_jespersen_limiters(mesh_, q, grads, config.gamma, &w);
                            limited.resize(grads.size());
                            for (std::size_t j = 0; j < grads.size(); ++j)
                                limited[j] = apply_limiter(grads[j], amr_limiters[j]);
                        }
                    }
                }
            }
        }

        if (residual_l2 < config.convergence_tol) {
            summary.converged = true;
            break;
        }
    }

    if (diagnostics_enabled) {
        StateBounds final_bounds = compute_state_bounds(q, config.gamma);
        summary.diagnostics.state_bounds_history.push_back(final_bounds);
    }

    {
        bool need_grads_at_end = config.viscous && !grads.empty();
        bool use_limited = config.reconstruction_order == 2 || (config.turbulence_model != TurbulenceModel::LAMINAR);
        const std::vector<PrimitiveGradient>* wall_grads = nullptr;
        if (need_grads_at_end)
            wall_grads = (use_limited && !limited.empty()) ? &limited : &grads;
        integrate_wall_forces(mesh_, wall_face_indices_, q, condition, config, summary.forces, wall_grads);
    }
    summary.forces.iterations = static_cast<int>(summary.residual_history.size());
    summary.forces.residual = summary.residual_history.empty() ? 0.0f : summary.residual_history.back();
    summary.final_state = q;
    if (config.turbulence_model == TurbulenceModel::SST) {
        summary.sst_final_k = sst_k_;
        summary.sst_final_omega = sst_omega_;
    }
    const char* tm_str = "laminar";
    if (config.turbulence_model == TurbulenceModel::SA) tm_str = "rans-sa";
    else if (config.turbulence_model == TurbulenceModel::SA_DDES) tm_str = "rans-sa-ddes";
    else if (config.turbulence_model == TurbulenceModel::SST) tm_str = "rans-sst";
    summary.forces.turbulence_model = tm_str;
    summary.forces.fidelity = "cfd-cpu";
    return summary;
}

bool assert_oracle_equivalent(
    const CfdSolveSummary& gpu,
    const CfdSolveSummary& cpu,
    Real tol_residual,
    Real tol_forces,
    std::string* error) {
    std::size_t n = std::min(gpu.residual_history.size(), cpu.residual_history.size());
    for (std::size_t i = 0; i < n; ++i) {
        Real diff = std::fabs(gpu.residual_history[i] - cpu.residual_history[i]);
        Real base = 1.0f + std::max(std::fabs(gpu.residual_history[i]), std::fabs(cpu.residual_history[i]));
        if (diff > tol_residual * base) {
            if (error) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "residual iter=%zu GPU=%g CPU=%g diff=%g", i,
                    gpu.residual_history[i], cpu.residual_history[i], diff);
                *error = buf;
            }
            return false;
        }
    }

    struct ForcePair { const char* name; Real g; Real c; };
    ForcePair pairs[] = {
        {"CX", gpu.forces.CX, cpu.forces.CX},
        {"CY", gpu.forces.CY, cpu.forces.CY},
        {"CZ", gpu.forces.CZ, cpu.forces.CZ},
        {"Cl", gpu.forces.Cl, cpu.forces.Cl},
        {"Cm", gpu.forces.Cm, cpu.forces.Cm},
        {"Cn", gpu.forces.Cn, cpu.forces.Cn},
        {"CD", gpu.forces.CD, cpu.forces.CD},
        {"CL", gpu.forces.CL, cpu.forces.CL},
        {"Q_wall", gpu.forces.Q_wall, cpu.forces.Q_wall},
    };
    for (const auto& p : pairs) {
        Real diff = std::fabs(p.g - p.c);
        Real base = 1.0f + std::max(std::fabs(p.g), std::fabs(p.c));
        if (diff > tol_forces * base) {
            if (error) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "force %s GPU=%g CPU=%g diff=%g", p.name, p.g, p.c, diff);
                *error = buf;
            }
            return false;
        }
    }
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp

