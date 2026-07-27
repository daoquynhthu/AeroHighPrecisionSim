#include "aero/cfd/rans.hpp"
#include "aero/cfd/viscous.hpp"
#include "aero/cfd/cfd_mesh.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {
constexpr Real kSaCv1 = 7.1f;
constexpr Real kSaCb1 = 0.1355f;
constexpr Real kSaCb2 = 0.622f;
constexpr Real kSaSigma = 2.0f / 3.0f;
constexpr Real kSaCw2 = 0.3f;
constexpr Real kSaCw3 = 2.0f;
constexpr Real kSaKarman = 0.41f;
constexpr Real kSaCt3 = 1.2f;
constexpr Real kSaCt4 = 0.5f;
constexpr Real kSaCn1 = 16.0f;
constexpr Real kSaCw1 = kSaCb1 / (kSaKarman * kSaKarman) + (1.0f + kSaCb2) / kSaSigma;
} // namespace

Real sa_vorticity(const PrimitiveGradient& grad) {
    Real vort_x = grad.dw_dy - grad.dv_dz;
    Real vort_y = grad.du_dz - grad.dw_dx;
    Real vort_z = grad.dv_dx - grad.du_dy;
    return std::sqrt(vort_x * vort_x + vort_y * vort_y + vort_z * vort_z);
}

Real sa_ft2(Real chi) {
    return kSaCt3 * std::exp(-kSaCt4 * chi * chi);
}

Real sa_fn(Real chi) {
    if (chi >= 0) return Real(1);
    Real chi3 = chi * chi * chi;
    return (kSaCn1 + chi3) / (kSaCn1 - chi3 + 1e-30f);
}

Real sa_diff_coeff(Real nu_tilde, Real rho, Real mu, Real Re) {
    if (!(rho > 0) || !(mu > 0) || !(Re > 0)) return Real(0);
    Real chi = rho * Re * nu_tilde / (mu + 1e-30f);
    Real fn = sa_fn(chi);
    // (1/sigma)*(mu/Re + rho*nu_tilde*fn)  — working-variable diffusivity, not mu_t
    return ((mu / Re) + rho * nu_tilde * fn) / kSaSigma;
}

Real sa_damping_rate(Real dS_dnu, Real nu_tilde, Real nu_mol, Real wall_distance,
                     Real h_min) {
    Real g = (dS_dnu < 0) ? -dS_dnu : Real(0);

    constexpr Real kWdFar = 1.0e10f;
    Real d = wall_distance;
    if (!(d > 0) || !std::isfinite(d) || d > kWdFar) d = kWdFar;
    Real d2 = d * d + 1e-30f;

    Real nu_ref = std::max(std::max(std::fabs(nu_tilde), std::max(nu_mol, Real(0))), Real(1e-30));
    Real g_dest = 2.0f * kSaCw1 * nu_ref / d2;
    if (g < g_dest) g = g_dest;

    Real h = d;
    if (h_min > 0 && std::isfinite(h_min)) h = std::min(h, h_min);
    if (!(h > 0) || !std::isfinite(h) || h > kWdFar) h = kWdFar;
    Real nu_eff = std::max(nu_mol, Real(0)) + std::max(nu_tilde, Real(0));
    Real g_diff = nu_eff / (kSaSigma * h * h + 1e-30f);
    if (g < g_diff) g = g_diff;

    return g;
}

Real sa_nu_tilde_floor(Real nu_mol) {
    return -kSaCv1 * std::max(nu_mol, Real(1e-30));
}

Real sa_nu_tilde_ceil(Real nu_mol, Real nu_cap_factor, Real nu_cap_abs) {
    Real cap = std::max(nu_cap_factor * std::max(nu_mol, Real(1e-30)), nu_cap_abs);
    return cap;
}

Real sa_freestream_nu_tilde(Real ratio, Real mu, Real rho, Real Re) {
    if (!(ratio > 0) || !(mu > 0) || !(rho > 0) || !(Re > 0)) return Real(0);
    return ratio * mu / (rho * Re);
}

Real sa_eddy_viscosity(Real nu_tilde, Real rho, Real mu, Real Re) {
    if (!(nu_tilde > 0) || !(rho > 0) || !(mu > 0) || !(Re > 0)) return Real(0);
    Real chi = rho * Re * nu_tilde / (mu + 1e-30f);
    if (chi > Real(1.0e6)) chi = Real(1.0e6);
    Real chi3 = chi * chi * chi;
    Real cv13 = kSaCv1 * kSaCv1 * kSaCv1;
    Real fv1 = chi3 / (chi3 + cv13 + 1e-30f);
    return nu_tilde * fv1;
}

Real sa_diffusive_rate(Real nu_tilde, Real nu_mol, Real h) {
    if (!(h > 0) || !std::isfinite(h)) return Real(0);
    Real nu_eff = std::max(nu_mol, Real(0)) + std::max(nu_tilde, Real(0));
    return nu_eff / (kSaSigma * h * h + 1e-30f);
}

Real sa_limit_source(Real S, Real nu_tilde, Real nu_mol, Real dt, Real C) {
    if (!(dt > 0) || !std::isfinite(S)) return 0;
    Real nu_scale = std::max({std::fabs(nu_tilde), std::max(nu_mol, Real(0)), Real(1e-12)});
    Real S_max = C * nu_scale / dt;
    if (!std::isfinite(S_max) || S_max <= 0) return 0;
    if (S > S_max) return S_max;
    if (S < -S_max) return -S_max;
    return S;
}

RansSource compute_rans_source(
    const PrimitiveState& w,
    const PrimitiveGradient& grad,
    Real wall_distance,
    Real mu,
    Real rho,
    Real Re,
    Real sa_r_max) {
    RansSource s;

    constexpr Real kWdFar = 1.0e10f;
    if (wall_distance <= 0.0f || !std::isfinite(wall_distance) || wall_distance > kWdFar) {
        wall_distance = kWdFar;
    }

    Real nu_tilde = w.nu_tilde;
    Real chi = rho * Re * nu_tilde / (mu + 1e-30f);
    if (chi > Real(1.0e6)) chi = Real(1.0e6);
    if (chi < Real(-1.0e6)) chi = Real(-1.0e6);

    Real vort = sa_vorticity(grad);

    Real grad_nu2 = grad.dnu_tilde_dx * grad.dnu_tilde_dx
                 + grad.dnu_tilde_dy * grad.dnu_tilde_dy
                 + grad.dnu_tilde_dz * grad.dnu_tilde_dz
                 + 1e-30f;
    Real diffusion = (kSaCb2 / kSaSigma) * grad_nu2;

    Real d = wall_distance;
    Real d2 = d * d + 1e-30f;

    Real r_nd = std::fabs(nu_tilde) / (d + 1e-30f);
    Real stiff_scale = Real(1);
    if (r_nd > sa_r_max && r_nd > 0) {
        Real r_cap = sa_r_max / r_nd;
        stiff_scale = r_cap * r_cap;
    }
    diffusion *= stiff_scale;

    Real source;
    Real dS_dnu;
    if (chi >= 0.0f) {
        Real chi3 = chi * chi * chi;
        Real cv13 = kSaCv1 * kSaCv1 * kSaCv1;
        Real fv1 = chi3 / (chi3 + cv13 + 1e-30f);

        Real fv2 = 1.0f - chi / (1.0f + chi * fv1 + 1e-30f);
        Real inv_kd2 = 1.0f / (kSaKarman * kSaKarman * d2);
        Real omega_tilde = vort + nu_tilde * fv2 * inv_kd2;
        Real s_floor = Real(0.3) * vort;
        if (omega_tilde < s_floor) omega_tilde = s_floor;

        Real ft2 = sa_ft2(chi);
        Real production = kSaCb1 * (1.0f - ft2) * omega_tilde * nu_tilde;

        Real r = nu_tilde / (omega_tilde * kSaKarman * kSaKarman * d2 + 1e-30f);
        if (r < 0.0f) r = 0.0f;
        if (r > 10.0f) r = 10.0f;
        Real r6 = r * r * r * r * r * r;
        Real fw_g = r + kSaCw2 * (r6 - r);
        Real fw_num = 1.0f + kSaCw3 * kSaCw3 * kSaCw3 * kSaCw3 * kSaCw3 * kSaCw3;
        Real fw_den = fw_g * fw_g * fw_g * fw_g * fw_g * fw_g
                    + kSaCw3 * kSaCw3 * kSaCw3 * kSaCw3 * kSaCw3 * kSaCw3 + 1e-30f;
        Real fw = fw_g * std::pow(fw_num / fw_den, 1.0f / 6.0f);
        // Full SA: (cw1*fw - cb1/κ² * ft2) * (ν̃/d)²
        Real dest_coeff = kSaCw1 * fw - (kSaCb1 / (kSaKarman * kSaKarman)) * ft2;
        Real destruction = dest_coeff * (nu_tilde / d) * (nu_tilde / d) * stiff_scale;

        source = production - destruction + diffusion;
        s.production = production;
        s.destruction = destruction;

        // Frozen S̃, fw, ft2 Jacobian (standard point-implicit diagonal)
        dS_dnu = kSaCb1 * (1.0f - ft2) * omega_tilde
               - 2.0f * dest_coeff * nu_tilde / d2 * stiff_scale;
    } else {
        // SA-neg (Allmaras et al. 2012): recovery + reduced production
        Real production = kSaCb1 * (1.0f - kSaCt3) * vort * nu_tilde;
        Real recovery = kSaCw1 * (nu_tilde / d) * (nu_tilde / d) * stiff_scale;
        source = production + recovery + diffusion;
        s.production = production;
        s.destruction = recovery;

        dS_dnu = kSaCb1 * (1.0f - kSaCt3) * vort
               + 2.0f * kSaCw1 * nu_tilde / d2 * stiff_scale;
    }

    s.diffusion = diffusion;
    s.dS_dnu = dS_dnu;
    s.total_source = rho * source;

    return s;
}

std::vector<RansSource> compute_rans_sources(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    Real gamma,
    Real Re,
    const std::vector<PrimitiveState>* primitive_override) {
    constexpr Real T_ref = 288.15f;
    constexpr Real S = 110.4f;
    std::vector<RansSource> sources(mesh.cells.size());
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        PrimitiveState w;
        if (primitive_override && primitive_override->size() == q.size()) {
            w = (*primitive_override)[i];
        } else if (!conservative_to_primitive(q[i], gamma, w)) {
            sources[i].total_source = std::numeric_limits<Real>::quiet_NaN();
            continue;
        }
        Real T = w.p / std::max(w.rho, Real(1e-30));
        Real mu = sutherland_viscosity(T, T_ref, S);
        if (mu <= 0.0f) mu = 1.0f;
        sources[i] = compute_rans_source(
            w, gradients[i], mesh.cells[i].wall_distance, mu, q[i].rho, Re);
    }
    return sources;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
