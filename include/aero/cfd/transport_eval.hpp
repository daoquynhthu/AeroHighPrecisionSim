#pragma once

#include "aero/cfd/thermo_db.hpp"
#include "aero/cfd/real.hpp"

#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// CEA log form: ln(mu) = A*ln(T) + B/T + C/T² + D
// coeffs[0..3] = {A, B, C, D}
AEROSP_REAL_HOST_DEVICE inline Real eval_mu(const Real* c, Real T) {
    Real lnT = real_log(T);
    Real invT = Real(1) / T;
    Real ln_mu = c[0] * lnT + c[1] * invT + c[2] * invT * invT + c[3];
    return real_exp(ln_mu) * Real(1e-7); // micropoise → Pa*s
}

AEROSP_REAL_HOST_DEVICE inline Real eval_kappa(const Real* c, Real T) {
    Real lnT = real_log(T);
    Real invT = Real(1) / T;
    Real ln_kappa = c[0] * lnT + c[1] * invT + c[2] * invT * invT + c[3];
    return real_exp(ln_kappa) * Real(1e-7); // micropoise → Pa*s
}

// Wilke interaction parameter:
// Φ_ij = [1 + sqrt(μ_i/μ_j) * (M_j/M_i)^(1/4)]^2 / sqrt(8*(1 + M_i/M_j))
AEROSP_REAL_HOST_DEVICE inline Real wilke_phi(Real mu_i, Real mu_j, Real M_i, Real M_j) {
    Real sqrt_ratio = real_sqrt(mu_i / mu_j);
    Real fourth_root = real_sqrt(real_sqrt(M_j / M_i));
    Real num = Real(1) + sqrt_ratio * fourth_root;
    Real den = real_sqrt(Real(8) * (Real(1) + M_i / M_j));
    return num * num / den;
}

// Species-level (CPU)
inline Real species_mu(const TransportRecord& rec, Real T) {
    if (rec.n_intervals < 1) return Real(0);
    int iv = 0;
    for (int i = 0; i < rec.n_intervals - 1; ++i) {
        if (T >= rec.T_max[i]) iv = i + 1;
    }
    return eval_mu(rec.mu_coeffs[iv], T);
}

inline Real species_kappa(const TransportRecord& rec, Real T) {
    if (rec.n_intervals < 1) return Real(0);
    int iv = 0;
    for (int i = 0; i < rec.n_intervals - 1; ++i) {
        if (T >= rec.T_max[i]) iv = i + 1;
    }
    return eval_kappa(rec.kappa_coeffs[iv], T);
}

// Simplified Wilke mixing rule (CPU)
Real mix_mu(Real T, const std::vector<Real>& Y,
            const std::vector<const TransportRecord*>& records);

Real mix_kappa(Real T, const std::vector<Real>& Y,
               const std::vector<const TransportRecord*>& records);

} // namespace cfd
} // namespace aero
} // namespace aerosp
