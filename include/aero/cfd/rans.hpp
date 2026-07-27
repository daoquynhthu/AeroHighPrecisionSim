#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/reconstruction.hpp"

#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

struct RansSource {
    Real production = 0.0f;
    Real destruction = 0.0f;
    Real diffusion = 0.0f;
    Real total_source = 0.0f;
    // d(nu_tilde)/dt form: analytic frozen-coefficient Jacobian ∂S/∂nu_tilde
    Real dS_dnu = 0.0f;
};

Real sa_vorticity(const PrimitiveGradient& grad);

// SA source S for d(nu_tilde)/dt, plus analytic frozen-coefficient Jacobian ∂S/∂nu.
// Quadratic stiffness capped via |nu|/d <= sa_r_max (default 200).
// Positive branch uses SA-R floor: S_tilde = max(S_tilde, 0.3 * vorticity).
RansSource compute_rans_source(
    const PrimitiveState& w,
    const PrimitiveGradient& grad,
    Real wall_distance,
    Real mu,
    Real rho,
    Real Re,
    Real sa_r_max = 200.0f);

std::vector<RansSource> compute_rans_sources(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    Real gamma,
    Real Re,
    const std::vector<PrimitiveState>* primitive_override = nullptr);

// Positive damping rate g (1/time) for nu_tilde point-implicit:
//   g = max(0, -∂S/∂nu) + near-wall destruction floor + viscous spectral radius.
Real sa_damping_rate(Real dS_dnu, Real nu_tilde, Real nu_mol, Real wall_distance,
                     Real h_min = 0.0f);

// Limit |S| so one step of size dt cannot change |nu| by more than
// C * max(|nu|, nu_mol, eps). Returns limited S.
Real sa_limit_source(Real S, Real nu_tilde, Real nu_mol, Real dt, Real C = 1.0f);

// SA-neg lower bound: nu_tilde >= -cv1 * nu_mol
Real sa_nu_tilde_floor(Real nu_mol);

// Soft upper bound to prevent chi^3 overflow / runaway production.
// nu_tilde <= max(nu_cap_factor * nu_mol, nu_cap_abs)
Real sa_nu_tilde_ceil(Real nu_mol, Real nu_cap_factor = 1.0e5f, Real nu_cap_abs = 1.0e3f);

} // namespace cfd
} // namespace aero
} // namespace aerosp
