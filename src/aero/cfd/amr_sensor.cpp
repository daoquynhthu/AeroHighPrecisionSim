#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/amr_sensor.hpp"
#include "aero/cfd/turbulence_model.hpp"
#include "aero/cfd/reconstruction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// Sutherland viscosity: mu = mu_ref * (T/T_ref) * sqrt(T/T_ref) * (T_ref + S) / (T + S)
static inline Real sutherland_mu(Real T, Real T_ref, Real S, Real mu_ref) {
    if (!real_isfinite(T) || T <= Real(0)) return mu_ref;
    Real t_ratio = T / T_ref;
    return mu_ref * t_ratio * real_sqrt(t_ratio) * (T_ref + S) / (T + S);
}

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

std::vector<RefinementRequest> compute_yplus_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma,
    Real Re,
    Real mu_ref,
    Real T_ref,
    Real sutherland_T,
    TurbulenceModel turbulence_model) {

    int n_cells = static_cast<int>(mesh.cells.size());
    std::vector<RefinementRequest> requests(n_cells);
    for (int i = 0; i < n_cells; ++i)
        requests[i] = {i, RefinementFlag::Unchanged};

    if (n_cells == 0 || config.yplus_target <= Real(0) || Re <= Real(0))
        return requests;

    constexpr Real cv1 = 7.1f;
    constexpr Real cv13 = cv1 * cv1 * cv1;
    bool has_turbulence = (turbulence_model == TurbulenceModel::SA ||
                           turbulence_model == TurbulenceModel::SA_DDES);

    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::NoSlipWall &&
            face.boundary != BoundaryKind::SlipWall)
            continue;

        int cell_id = face.left_cell;
        if (cell_id < 0 || cell_id >= n_cells) continue;

        const CfdCell& cell = mesh.cells[cell_id];
        Real d = cell.wall_distance;
        if (d <= Real(0)) continue;

        // Compute primitive state
        PrimitiveState w;
        if (!conservative_to_primitive(q[cell_id], gamma, w)) continue;

        Real rho = std::fabs(w.rho);
        Real speed = std::sqrt(w.u * w.u + w.v * w.v + w.w * w.w);

        if (rho < Real(1e-10) || speed < Real(1e-12)) continue;

        // Molecular viscosity from Sutherland's law
        Real T = w.p / rho;
        Real mu = sutherland_mu(T, T_ref, sutherland_T, mu_ref);
        if (mu <= Real(0)) continue;

        // Effective viscosity with turbulence contribution
        Real mu_eff = mu;
        if (has_turbulence) {
            Real nu_tilde = std::fabs(w.nu_tilde);
            Real chi = rho * nu_tilde / mu;
            if (chi > Real(0)) {
                Real chi3 = chi * chi * chi;
                Real fv1 = chi3 / (chi3 + cv13 + Real(1e-30));
                mu_eff = mu + rho * nu_tilde * fv1;
            }
        }

        // y+ estimate using linear profile approximation
        // tau_wall ≈ (mu_eff/Re) * |u| / d  (non-dimensional)
        // u_tau = sqrt(tau_wall / rho)
        // y+ = d * u_tau * rho * Re / mu  = sqrt(Re*d*rho*|u|) * sqrt(mu_eff) / mu
        Real yplus = real_sqrt(Re * d * rho * speed) * real_sqrt(mu_eff) / mu;

        if (yplus > config.yplus_target) {
            requests[cell_id].flag = RefinementFlag::Refine;
            // Wall-adjacent cell: mark for anisotropic refinement in wall-normal direction
            requests[cell_id].dir = AnisotropicDir::WALL_NORMAL;
        } else if (yplus < Real(0.5) * config.yplus_target && cell.refinement_level > 0) {
            requests[cell_id].flag = RefinementFlag::Coarsen;
        }
    }

    return requests;
}

std::vector<RefinementRequest> compute_qcriterion_sensor(
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

    // Compute Green-Gauss velocity gradients
    auto grads = compute_green_gauss_gradients(mesh, q, gamma);

    // Per-cell Q-criterion
    std::vector<Real> cell_Q(n_cells, Real(0));
    Real Q_max = Real(0);

    for (int i = 0; i < n_cells; ++i) {
        const PrimitiveGradient& g = grads[i];
        // Q = -0.5*(du/dx² + dv/dy² + dw/dz²) - du/dy*dv/dx - du/dz*dw/dx - dv/dz*dw/dy
        Real du2 = g.du_dx * g.du_dx;
        Real dv2 = g.dv_dy * g.dv_dy;
        Real dw2 = g.dw_dz * g.dw_dz;
        Real cross_xy = g.du_dy * g.dv_dx;
        Real cross_xz = g.du_dz * g.dw_dx;
        Real cross_yz = g.dv_dz * g.dw_dy;

        Real Q = Real(-0.5) * (du2 + dv2 + dw2) - cross_xy - cross_xz - cross_yz;
        cell_Q[i] = Q;
        if (std::fabs(Q) > Q_max) Q_max = std::fabs(Q);
    }

    // Normalize thresholds by max |Q| if available
    Real refine_tol = config.refine_tol;
    Real coarsen_tol = config.coarsen_tol;
    if (Q_max > Real(1e-12)) {
        refine_tol *= Q_max;
        coarsen_tol *= Q_max;
    }

    for (int i = 0; i < n_cells; ++i) {
        if (cell_Q[i] > refine_tol) {
            requests[i].flag = RefinementFlag::Refine;
        } else if (cell_Q[i] < Real(0) && std::fabs(cell_Q[i]) < coarsen_tol &&
                   mesh.cells[i].refinement_level > 0) {
            requests[i].flag = RefinementFlag::Coarsen;
        }
    }

    return requests;
}

std::vector<RefinementRequest> compute_wake_cone_sensor(
    const CfdMesh& mesh,
    const AmrConfig& config,
    const WakeConeConfig& cone) {

    int n_cells = static_cast<int>(mesh.cells.size());
    std::vector<RefinementRequest> requests(n_cells);
    for (int i = 0; i < n_cells; ++i)
        requests[i] = {i, RefinementFlag::Unchanged};

    if (n_cells == 0 || cone.half_angle_deg <= Real(0) || cone.length <= Real(0))
        return requests;

    Real half_angle_rad = cone.half_angle_deg * Real(3.14159265358979323846) / Real(180.0);
    Real cos_angle = std::cos(half_angle_rad);
    Real inv_len = Real(1) / cone.length;

    for (int i = 0; i < n_cells; ++i) {
        const CfdCell& cell = mesh.cells[i];
        Real dx = cell.cx - cone.origin_x;
        Real dy = cell.cy - cone.origin_y;
        Real dz = cell.cz - cone.origin_z;

        // Distance along axis
        Real proj = dx * cone.axis_x + dy * cone.axis_y + dz * cone.axis_z;

        // Behind the body (proj > 0 away from origin along axis)
        // Within length and within cone half-angle
        if (proj > Real(0) && proj <= cone.length) {
            // Perpendicular distance from axis
            Real px = dx - proj * cone.axis_x;
            Real py = dy - proj * cone.axis_y;
            Real pz = dz - proj * cone.axis_z;
            Real dist = std::sqrt(px * px + py * py + pz * pz);

            // Cone radius at this axial position (linear taper)
            Real radius = proj * std::tan(half_angle_rad);

            if (dist <= radius) {
                if (config.refine_tol > Real(0)) {
                    requests[i].flag = RefinementFlag::Refine;
                    requests[i].dir = AnisotropicDir::STREAMWISE;
                }
            }
        }
    }

    return requests;
}

std::vector<RefinementRequest> merge_refinement_requests(
    const std::vector<std::vector<RefinementRequest>>& sensors) {

    if (sensors.empty()) return {};

    int n_cells = static_cast<int>(sensors[0].size());
    std::vector<RefinementRequest> merged(n_cells);
    for (int i = 0; i < n_cells; ++i)
        merged[i] = {i, RefinementFlag::Unchanged};

    for (int i = 0; i < n_cells; ++i) {
        bool any_refine = false;
        int coarsen_count = 0;
        int non_unchanged = 0;

        for (const auto& sensor : sensors) {
            if (i >= static_cast<int>(sensor.size())) continue;
            auto flag = sensor[i].flag;
            if (flag == RefinementFlag::Refine) {
                any_refine = true;
                break;
            } else if (flag == RefinementFlag::Coarsen) {
                ++coarsen_count;
                ++non_unchanged;
            } else if (flag != RefinementFlag::Unchanged) {
                ++non_unchanged;
            }
        }

        if (any_refine) {
            merged[i].flag = RefinementFlag::Refine;
        } else if (non_unchanged > 0 && coarsen_count == non_unchanged) {
            merged[i].flag = RefinementFlag::Coarsen;
        }
    }

    return merged;
}

std::vector<RefinementRequest> compute_tke_ratio_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma,
    TurbulenceModel turbulence_model,
    const std::vector<Real>* sst_k,
    Real tke_ratio_threshold) {

    int n_cells = static_cast<int>(mesh.cells.size());
    std::vector<RefinementRequest> requests(n_cells);
    for (int i = 0; i < n_cells; ++i)
        requests[i] = {i, RefinementFlag::Unchanged};

    // Non-SST models: no k available, return no-op
    if (turbulence_model != TurbulenceModel::SST || !sst_k) return requests;

    if (n_cells == 0 || config.coarsen_tol >= config.refine_tol || tke_ratio_threshold <= Real(0))
        return requests;

    for (int i = 0; i < n_cells; ++i) {
        PrimitiveState w;
        if (!conservative_to_primitive(q[i], gamma, w)) continue;

        Real U2 = w.u * w.u + w.v * w.v + w.w * w.w;
        if (U2 < Real(1e-20)) continue;

        Real k_val = (*sst_k)[i];
        if (!real_isfinite(k_val) || k_val < Real(0)) continue;

        Real ratio = k_val / (Real(0.5) * U2);

        if (ratio > tke_ratio_threshold) {
            requests[i].flag = RefinementFlag::Refine;
        } else if (ratio < Real(0.5) * tke_ratio_threshold && mesh.cells[i].refinement_level > 0) {
            requests[i].flag = RefinementFlag::Coarsen;
        }
    }

    return requests;
}

std::vector<RefinementRequest> compute_shear_layer_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma,
    TurbulenceModel turbulence_model,
    const std::vector<Real>* sst_k,
    Real shear_layer_threshold) {

    int n_cells = static_cast<int>(mesh.cells.size());
    std::vector<RefinementRequest> requests(n_cells);
    for (int i = 0; i < n_cells; ++i)
        requests[i] = {i, RefinementFlag::Unchanged};

    if (turbulence_model == TurbulenceModel::LAMINAR) return requests;
    if (n_cells == 0 || config.coarsen_tol >= config.refine_tol) return requests;

    bool is_sst = (turbulence_model == TurbulenceModel::SST);
    bool is_sa = (turbulence_model == TurbulenceModel::SA ||
                  turbulence_model == TurbulenceModel::SA_DDES);
    if (!is_sst && !is_sa) return requests;
    if (is_sst && !sst_k) return requests;

    constexpr Real a1 = 0.31f;
    constexpr Real cv1 = 7.1f;
    constexpr Real cv13 = cv1 * cv1 * cv1;

    auto grads = compute_green_gauss_gradients(mesh, q, gamma);

    std::vector<Real> cell_ratio(n_cells, Real(-1));  // -1 = skipped

    for (int i = 0; i < n_cells; ++i) {
        PrimitiveState w;
        if (!conservative_to_primitive(q[i], gamma, w)) continue;

        const PrimitiveGradient& g = grads[i];

        Real h = mesh.cells[i].h_min;
        if (h <= Real(0)) h = std::pow(mesh.cells[i].volume, Real(1.0 / 3.0));
        if (h <= Real(0)) continue;

        Real grad_u2 = g.du_dx * g.du_dx + g.du_dy * g.du_dy + g.du_dz * g.du_dz;
        Real grad_v2 = g.dv_dx * g.dv_dx + g.dv_dy * g.dv_dy + g.dv_dz * g.dv_dz;
        Real grad_w2 = g.dw_dx * g.dw_dx + g.dw_dy * g.dw_dy + g.dw_dz * g.dw_dz;
        Real resolved_k = Real(0.125) * h * h * (grad_u2 + grad_v2 + grad_w2);

        Real Sxx = g.du_dx;
        Real Syy = g.dv_dy;
        Real Szz = g.dw_dz;
        Real Sxy = Real(0.5) * (g.du_dy + g.dv_dx);
        Real Sxz = Real(0.5) * (g.du_dz + g.dw_dx);
        Real Syz = Real(0.5) * (g.dv_dz + g.dw_dy);
        Real S_mag = std::sqrt(Real(2.0) * (Sxx*Sxx + Syy*Syy + Szz*Szz +
                                             Real(2.0) * (Sxy*Sxy + Sxz*Sxz + Syz*Syz)));

        Real modeled_k = Real(0);
        if (is_sst) {
            modeled_k = std::fmax(Real(0), (*sst_k)[i]);
        } else if (is_sa) {
            Real nu_tilde = std::fmax(Real(0), w.nu_tilde);
            modeled_k = nu_tilde * S_mag / a1;
        }

        if (modeled_k < Real(1e-14)) continue;

        Real eps = Real(1e-30);
        cell_ratio[i] = resolved_k / (resolved_k + modeled_k + eps);
    }

    Real refine_th = Real(0.5) - shear_layer_threshold * Real(0.5);
    Real coarsen_th = Real(0.5) + shear_layer_threshold * Real(0.5);

    for (int i = 0; i < n_cells; ++i) {
        Real r = cell_ratio[i];
        if (r < Real(0)) continue;  // skipped
        if (r < refine_th) {
            requests[i].flag = RefinementFlag::Refine;
        } else if (r > coarsen_th && mesh.cells[i].refinement_level > 0) {
            requests[i].flag = RefinementFlag::Coarsen;
        }
    }

    return requests;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
