#pragma once

#include "aero/cfd/real.hpp"

#include <cmath>

namespace aerosp {
namespace aero {
namespace cfd {

struct ConservativeState {
    Real rho = 0.0f;
    Real rho_u = 0.0f;
    Real rho_v = 0.0f;
    Real rho_w = 0.0f;
    Real rho_E = 0.0f;
    Real rho_nu_tilde = 0.0f;
};

struct PrimitiveState {
    Real rho = 0.0f;
    Real u = 0.0f;
    Real v = 0.0f;
    Real w = 0.0f;
    Real p = 0.0f;
    Real nu_tilde = 0.0f;
};

struct EulerFlux {
    Real mass = 0.0f;
    Real mom_x = 0.0f;
    Real mom_y = 0.0f;
    Real mom_z = 0.0f;
    Real energy = 0.0f;
    Real turbulence = 0.0f;
};

AEROSP_REAL_HOST_DEVICE bool is_valid_primitive(const PrimitiveState& w) {
    return aerosp::real_isfinite(w.rho) && aerosp::real_isfinite(w.u) && aerosp::real_isfinite(w.v) &&
           aerosp::real_isfinite(w.w) && aerosp::real_isfinite(w.p) && aerosp::real_isfinite(w.nu_tilde) &&
           w.rho > 0.0f && w.p > 0.0f;
}

inline ConservativeState primitive_to_conservative(const PrimitiveState& w, Real gamma) {
    Real kinetic = 0.5f * (w.u*w.u + w.v*w.v + w.w*w.w);
    ConservativeState q;
    q.rho = w.rho;
    q.rho_u = w.rho * w.u;
    q.rho_v = w.rho * w.v;
    q.rho_w = w.rho * w.w;
    q.rho_E = w.p / (gamma - 1.0f) + w.rho * kinetic;
    q.rho_nu_tilde = w.rho * w.nu_tilde;
    return q;
}

inline bool conservative_to_primitive(const ConservativeState& q, Real gamma, PrimitiveState& w) {
    if (!std::isfinite(q.rho) || q.rho <= 0.0f) return false;
    Real inv_rho = 1.0f / q.rho;
    w.rho = q.rho;
    w.u = q.rho_u * inv_rho;
    w.v = q.rho_v * inv_rho;
    w.w = q.rho_w * inv_rho;
    Real kinetic = 0.5f * (w.u*w.u + w.v*w.v + w.w*w.w);
    w.p = (gamma - 1.0f) * (q.rho_E - q.rho * kinetic);
    w.nu_tilde = q.rho_nu_tilde * inv_rho;
    return is_valid_primitive(w);
}

inline Real speed_of_sound(const PrimitiveState& w, Real gamma) {
    return std::sqrt(gamma * w.p / w.rho);
}

inline EulerFlux physical_flux(const PrimitiveState& w, Real gamma, Real nx, Real ny, Real nz) {
    Real vn = w.u*nx + w.v*ny + w.w*nz;
    Real kinetic = 0.5f * (w.u*w.u + w.v*w.v + w.w*w.w);
    Real rho_E = w.p / (gamma - 1.0f) + w.rho * kinetic;

    EulerFlux f;
    f.mass = w.rho * vn;
    f.mom_x = w.rho * w.u * vn + w.p * nx;
    f.mom_y = w.rho * w.v * vn + w.p * ny;
    f.mom_z = w.rho * w.w * vn + w.p * nz;
    f.energy = (rho_E + w.p) * vn;
    f.turbulence = w.rho * w.nu_tilde * vn;
    return f;
}

inline EulerFlux slip_wall_flux(const PrimitiveState& w, Real nx, Real ny, Real nz) {
    EulerFlux f;
    f.mass = 0.0f;
    f.mom_x = w.p * nx;
    f.mom_y = w.p * ny;
    f.mom_z = w.p * nz;
    f.energy = 0.0f;
    f.turbulence = 0.0f;
    return f;
}

EulerFlux hllc_flux(const PrimitiveState& left, const PrimitiveState& right, Real gamma, Real nx, Real ny, Real nz);

PrimitiveState make_freestream(Real mach, Real alpha_deg, Real beta_deg, Real gamma);

PrimitiveState farfield_ghost_state(const PrimitiveState& left, const PrimitiveState& freestream, Real gamma,
    Real nx, Real ny, Real nz);

} // namespace cfd
} // namespace aero
} // namespace aerosp
