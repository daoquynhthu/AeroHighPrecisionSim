#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/rans.hpp"
#include "aero/cfd/reconstruction.hpp"
#include "aero/cfd/viscous.hpp"
#include "aero/cfd/amr_sensor.hpp"
#include "aero/cfd/amr_interpolate.hpp"
#include "aero/cfd/amr_hanging.hpp"

#include <algorithm>
#include <cmath>

#include <limits>

namespace aerosp {
namespace aero {
namespace cfd {

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

Real state_delta_l2(const ConservativeState& a, const ConservativeState& b) {
    Real d0 = a.rho - b.rho;
    Real d1 = a.rho_u - b.rho_u;
    Real d2 = a.rho_v - b.rho_v;
    Real d3 = a.rho_w - b.rho_w;
    Real d4 = a.rho_E - b.rho_E;
    Real d5 = a.rho_nu_tilde - b.rho_nu_tilde;
    return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5;
}

void integrate_wall_forces(const CfdMesh& mesh, const std::vector<int>& wall_face_indices,
    const std::vector<ConservativeState>& q, const FreestreamCondition& condition,
    const CfdConfig& config, CfdForceResult& result,
    const std::vector<PrimitiveGradient>* grads) {
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

} // namespace

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

    Real sx = 0.0f, sy = 0.0f, sz = 0.0f;
    Real total_area = 0.0f;
    for (std::size_t i = 0; i < mesh_.faces.size(); ++i) {
        const auto& face = mesh_.faces[i];
        if (face.boundary == BoundaryKind::SlipWall || face.boundary == BoundaryKind::NoSlipWall)
            wall_face_indices_.push_back(static_cast<int>(i));
        if (face.boundary != BoundaryKind::Interior) {
            sx += face.area * face.nx;
            sy += face.area * face.ny;
            sz += face.area * face.nz;
            total_area += face.area;
        }
    }

    Real closure_error = real_sqrt(sx*sx + sy*sy + sz*sz);
    if (closure_error > Real(1e-4) * (total_area + Real(1e-30))) return false;

    return true;
}

CfdSolveSummary CfdSolver::solve(const FreestreamCondition& condition, const CfdConfig& config) {
    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    w_inf.nu_tilde = condition.nu_tilde;
    if (condition.nu_tilde_ratio > 0.0f && config.viscous) {
        Real T_inf = w_inf.p / w_inf.rho;
        Real t_ratio = T_inf / config.T_ref;
        Real mu_inf = config.mu_ref * t_ratio * std::sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T_inf + config.sutherland_T);
        w_inf.nu_tilde = condition.nu_tilde_ratio * mu_inf / w_inf.rho;
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
    std::vector<ConservativeState> q_next;
    q_next.resize(mesh_.cells.size());
    std::vector<EulerFlux> residual;
    std::vector<PrimitiveGradient> grads, limited;
    std::vector<RansSource> sources;
    std::vector<PrimitiveState> w(mesh_.cells.size());
    std::vector<RefinementRecord> amr_records;
    bool diagnostics_enabled = config.diagnostic_level != DiagnosticLevel::Off;

    for (int iter = 0; iter < config.max_iter; ++iter) {
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
            Real dt = config.cfl * h_min_val / signal_speed;
            if (config.viscous) {
                Real T = w[i].p / w[i].rho;
                if (T > 0.0f) {
                    Real t_ratio = T / config.T_ref;
                    Real mu = config.mu_ref * t_ratio * std::sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T + config.sutherland_T);
                    if (mu > 0.0f) {
                        Real dt_visc = config.cfl * w[i].rho * h_min_val * h_min_val * config.Re / mu;
                        if (dt_visc < dt) dt = dt_visc;
                    }
                }
            }
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
        if (diagnostics_enabled) {
            summary.diagnostics.dt_limiter_history.push_back(dt_limiter);
            summary.diagnostics.state_bounds_history.push_back(iter_bounds);
        }

        bool need_gradients = (config.reconstruction_order == 2) || config.viscous || config.turbulence;
        std::vector<PrimitiveLimiter> limiters_vec;
        bool apply_limiting = config.reconstruction_order == 2 || config.turbulence;

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
            if (!compute_viscous_flux_cpu(mesh_, q, visc_grads, config.gamma,
                    config.prandtl, config.mu_ref, config.T_ref,
                    config.sutherland_T, config.Re, config.wall_temperature,
                    config.turbulence ? 1 : 0, residual, &w)) {
                summary.failed = true;
                if (diagnostics_enabled) {
                    summary.diagnostics.failure.reason = "viscous flux failed";
                    summary.diagnostics.failure.valid = true;
                    summary.diagnostics.failure.iteration = iter;
                }
                return summary;
            }
        }

        if (config.turbulence) {
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

        // Semi-implicit destruction treatment (match GPU apply_rans_implicit_gpu).
        // Skipped in MMS mode: the semi-implicit correction is a convergence acceleration
        // technique, not part of the spatial discretization being verified. Applying it
        // after MMS subtraction would break source consistency since the correction
        // transforms residual as R' = q*(f-1)/dtv + R*f, making it impossible for
        // S_mms to cancel R when q=q_exact.
        if (config.turbulence && config.mms_source.empty()) {
            for (std::size_t i = 0; i < q.size(); ++i) {
                Real wall_distance = mesh_.cells[i].h_min;
                if (wall_distance <= 0.0f) wall_distance = 1e30f;
                Real nu_tilde = q[i].rho_nu_tilde / q[i].rho;
                if (!std::isfinite(nu_tilde)) continue;
                constexpr Real cw1 = 0.1355f / (0.41f * 0.41f) + (1.0f + 0.622f) / (2.0f / 3.0f);
                constexpr Real karman = 0.41f;
                Real d_dest = 2.0f * cw1 * nu_tilde / (karman * karman * wall_distance * wall_distance + 1e-30f);
                Real dt_over_V = min_dt / (mesh_.cells[i].volume + 1e-30f);
                Real implicit_factor = 1.0f / (1.0f + dt_over_V * d_dest + 1e-30f);
                Real old_rhont = q[i].rho_nu_tilde;
                Real old_residual = residual[i].turbulence;
                residual[i].turbulence = old_rhont * (implicit_factor - 1.0f) / (dt_over_V + 1e-30f)
                                       + old_residual * implicit_factor;
            }
        }

        Real l2 = 0.0f;
        for (std::size_t i = 0; i < q.size(); ++i) {
            Real scale = min_dt / mesh_.cells[i].volume;
            q_next[i] = add_scaled(q[i], residual[i], scale);
            l2 += state_delta_l2(q_next[i], q[i]);
        }
        Real residual_l2 = std::sqrt(l2 / (static_cast<Real>(CFD_NVAR) * static_cast<Real>(q.size())));
        summary.residual_history.push_back(residual_l2);
        q.swap(q_next);

        if (config.amr.enabled && iter > 0 && config.amr.interval > 0 && (iter % config.amr.interval) == 0) {
            CfdMesh mesh_old = mesh_;
            std::vector<ConservativeState> q_old = q;

            auto requests = compute_gradient_sensor(mesh_, q, config.amr, config.gamma);

            // Enforce 2:1 balance after coarsening: prevent coarsening a cell if
            // a face neighbor at a higher refinement level is not also being coarsened.
            // (Coarsening would drop the cell's level by 1, potentially creating
            //  a level difference > 1 with neighbors that stay at their current level.)
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
                std::vector<RefinementRecord> new_records;
                std::vector<CoarsenInfo> coarsen_info;
                std::string err;
                const std::vector<RefinementRecord>* prev = amr_records.empty() ? nullptr : &amr_records;
                if (refine_cells(mesh_, requests, &new_records, &err, prev, &coarsen_info, config.amr.max_level)) {
                    if (need_gradients && !grads.empty() && !w.empty()) {
                        const auto& gsrc = apply_limiting ? limited : grads;
                        prolongate_solution_order2(q_old, w, gsrc, mesh_old, mesh_, new_records, config.gamma, q);
                    } else {
                        prolongate_solution(q_old, mesh_old, mesh_, new_records, q);
                    }
                    for (const auto& ci : coarsen_info) {
                        Real vol_sum = 0.0f;
                        ConservativeState avg;
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
                        }
                        if (vol_sum > 0.0f) {
                            Real inv = 1.0f / vol_sum;
                            avg.rho *= inv; avg.rho_u *= inv; avg.rho_v *= inv;
                            avg.rho_w *= inv; avg.rho_E *= inv; avg.rho_nu_tilde *= inv;
                        }
                        q[ci.new_parent_id] = avg;
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
        bool use_limited = config.reconstruction_order == 2 || config.turbulence;
        const std::vector<PrimitiveGradient>* wall_grads = nullptr;
        if (need_grads_at_end)
            wall_grads = (use_limited && !limited.empty()) ? &limited : &grads;
        integrate_wall_forces(mesh_, wall_face_indices_, q, condition, config, summary.forces, wall_grads);
    }
    summary.forces.iterations = static_cast<int>(summary.residual_history.size());
    summary.forces.residual = summary.residual_history.empty() ? 0.0f : summary.residual_history.back();
    summary.final_state = q;
    summary.forces.turbulence_model = config.turbulence ? "rans-sa" : "laminar";
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

