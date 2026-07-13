#pragma once

#include "aero/cfd/real.hpp"

namespace aerosp {
namespace aero {
namespace cfd {

// Menter 2003 k-omega SST coefficients
namespace sst_coeff {

constexpr Real sigma_k1 = 0.85f;
constexpr Real sigma_w1 = 0.5f;
constexpr Real beta_1   = 0.075f;
constexpr Real gamma_1  = Real(5.0) / Real(9.0);

constexpr Real sigma_k2 = 1.0f;
constexpr Real sigma_w2 = 0.856f;
constexpr Real beta_2   = 0.0828f;
constexpr Real gamma_2  = 0.44f;

constexpr Real beta_star = 0.09f;
constexpr Real karman    = 0.41f;
constexpr Real a1        = 0.31f;

} // namespace sst_coeff

struct SstBlending {
    Real F1;
    Real F2;
    Real CD_kw;
};

AEROSP_REAL_HOST_DEVICE SstBlending compute_sst_blending(
    Real k, Real omega, Real d, Real rho, Real mu,
    Real gk_x, Real gk_y, Real gk_z,
    Real gw_x, Real gw_y, Real gw_z)
{
    SstBlending b;
    constexpr Real eps = Real(1e-30);

    Real d2 = d * d + eps;

    Real sqrt_k = real_sqrt(k * k + eps * eps);

    b.CD_kw = Real(2.0) * rho * sst_coeff::sigma_w2
            * (gk_x * gw_x + gk_y * gw_y + gk_z * gw_z) / (omega + eps);
    b.CD_kw = real_fmax(b.CD_kw, Real(1e-10));

    Real arg1a = sqrt_k / (sst_coeff::beta_star * omega * d + eps);
    Real arg1b = Real(500.0) * mu / (rho * d2 * omega + eps);
    Real arg1c = Real(4.0) * rho * sst_coeff::sigma_w2 * k / (b.CD_kw * d2 + eps);
    Real Phi_1 = real_fmin(real_fmax(arg1a, arg1b), arg1c);
    Real Phi_1_4 = Phi_1 * Phi_1 * Phi_1 * Phi_1;
    Phi_1_4 = real_fmin(Phi_1_4, Real(80.0));
    b.F1 = real_tanh(Phi_1_4);

    Real arg2a = Real(2.0) * sqrt_k / (sst_coeff::beta_star * omega * d + eps);
    Real arg2b = Real(500.0) * mu / (rho * d2 * omega + eps);
    Real Phi_2 = real_fmax(arg2a, arg2b);
    Real Phi_2_2 = Phi_2 * Phi_2;
    Phi_2_2 = real_fmin(Phi_2_2, Real(80.0));
    b.F2 = real_tanh(Phi_2_2);

    return b;
}

struct SstSource {
    Real source_k;
    Real source_w;
};

AEROSP_REAL_HOST_DEVICE SstSource compute_sst_source(
    Real k, Real omega, Real rho, Real mu_t,
    const SstBlending& b, Real S_mag)
{
    constexpr Real eps = Real(1e-30);

    Real beta  = b.F1 * sst_coeff::beta_1  + (Real(1.0) - b.F1) * sst_coeff::beta_2;
    Real gamma = b.F1 * sst_coeff::gamma_1 + (Real(1.0) - b.F1) * sst_coeff::gamma_2;

    Real nu_t = mu_t / (rho + eps);
    Real a1_omega = sst_coeff::a1 * omega + eps;
    Real F2_S = b.F2 * S_mag;
    Real nu_t_lim = sst_coeff::a1 * k / real_fmax(a1_omega, F2_S);

    Real P_k_raw = nu_t_lim * S_mag * S_mag;
    Real P_k_max = Real(10.0) * sst_coeff::beta_star * rho * k * omega;
    Real P_k = real_fmin(P_k_raw, P_k_max);

    Real P_w = gamma / (nu_t + eps) * P_k;

    Real D_k = sst_coeff::beta_star * rho * k * omega;
    Real D_w = beta * rho * omega * omega;

    Real cross = (Real(1.0) - b.F1) * b.CD_kw;

    SstSource s;
    s.source_k = P_k - D_k;
    s.source_w = P_w - D_w + cross;
    return s;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp