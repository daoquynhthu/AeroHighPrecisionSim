#include "aero/cfd/rans.hpp"
#include "aero/cfd/viscous.hpp"
#include "aero/cfd/cfd_mesh.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

namespace aerosp {
namespace aero {
namespace cfd {

Real sa_vorticity(const PrimitiveGradient& grad) {
    Real vort_x = grad.dw_dy - grad.dv_dz;
    Real vort_y = grad.du_dz - grad.dw_dx;
    Real vort_z = grad.dv_dx - grad.du_dy;
    return std::sqrt(vort_x*vort_x + vort_y*vort_y + vort_z*vort_z);
}

RansSource compute_rans_source(
    const PrimitiveState& w,
    const PrimitiveGradient& grad,
    Real wall_distance,
    Real mu,
    Real rho,
    Real Re) {
    RansSource s;

    if (wall_distance <= 0.0f || !std::isfinite(wall_distance)) {
        wall_distance = 1e30f;
    }

    constexpr Real cv1 = 7.1f;
    constexpr Real cb1 = 0.1355f;
    constexpr Real cb2 = 0.622f;
    constexpr Real sigma = 2.0f / 3.0f;
    constexpr Real cw2 = 0.3f;
    constexpr Real cw3 = 2.0f;
    constexpr Real karman = 0.41f;
    constexpr Real ct3 = 1.2f;
    constexpr Real ct4 = 0.5f;

    Real chi = rho * Re * w.nu_tilde / (mu + 1e-30f) + 1e-30f;
    Real vort = sa_vorticity(grad);

    Real grad_nu2 = grad.dnu_tilde_dx * grad.dnu_tilde_dx
                 + grad.dnu_tilde_dy * grad.dnu_tilde_dy
                 + grad.dnu_tilde_dz * grad.dnu_tilde_dz;
    Real diffusion = (cb2 / sigma) * grad_nu2;

    Real cw1 = cb1 / (karman*karman) + (1.0f + cb2) / sigma;

    Real source;
    if (chi >= 0.0f) {
        Real chi3 = chi*chi*chi;
        Real cv13 = cv1*cv1*cv1;
Real fv1 = chi3 / (chi3 + cv13 + 1e-30f);

        Real fv2 = 1.0f - chi / (1.0f + chi * fv1 + 1e-30f);
        Real inv_kd2 = 1.0f / (karman * karman * wall_distance * wall_distance + 1e-30f);
        Real omega_tilde = vort + w.nu_tilde * fv2 * inv_kd2;

        Real production = cb1 * omega_tilde * w.nu_tilde;

        Real r = w.nu_tilde / (omega_tilde * karman * karman * wall_distance * wall_distance + 1e-30f);
        if (r > 10.0f) r = 10.0f;
        Real r6 = r*r*r*r*r*r;
        Real fw_g = r + cw2 * (r6 - r);
        Real fw_num = 1.0f + cw3*cw3*cw3*cw3*cw3*cw3;
        Real fw_den = fw_g*fw_g*fw_g*fw_g*fw_g*fw_g + cw3*cw3*cw3*cw3*cw3*cw3 + 1e-30f;
        Real fw = fw_g * std::pow(fw_num / fw_den, 1.0f / 6.0f);
        Real destruction = cw1 * fw * (w.nu_tilde / wall_distance) * (w.nu_tilde / wall_distance);

        source = production - destruction + diffusion;
        s.production = production;
        s.destruction = destruction;
    } else {
        source = cb1 * (1.0f - ct3) * vort * w.nu_tilde
               + cw1 * (w.nu_tilde / wall_distance) * (w.nu_tilde / wall_distance)
               + diffusion;
        s.production = cb1 * (1.0f - ct3) * vort * w.nu_tilde;
        s.destruction = cw1 * (w.nu_tilde / wall_distance) * (w.nu_tilde / wall_distance);
    }

    s.diffusion = diffusion;
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