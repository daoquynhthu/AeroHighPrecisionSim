#include "aero/cfd/transport_eval.hpp"

#include <cmath>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

inline void mass_to_mole_fraction(const std::vector<Real>& Y,
                                   std::vector<Real>& X,
                                   const std::vector<const TransportRecord*>& records) {
    int nsp = static_cast<int>(Y.size());
    Real sum_moles = Real(0);
    for (int i = 0; i < nsp; ++i) {
        Real M = records[i]->M > Real(0) ? records[i]->M : Real(29);
        sum_moles += Y[i] / M;
    }
    if (sum_moles <= Real(0)) return;
    for (int i = 0; i < nsp; ++i) {
        Real M = records[i]->M > Real(0) ? records[i]->M : Real(29);
        X[i] = (Y[i] / M) / sum_moles;
    }
}

} // anonymous namespace

// Full Wilke mixing rule:
// μ_mix = Σ_i ( X_i * μ_i / Σ_j (X_j * Φ_ij) )
// where Φ_ij computed from pure-species μ_i, μ_j, M_i, M_j via wilke_phi().
// Φ_ii = 1 (self-interaction).

Real mix_mu(Real T, const std::vector<Real>& Y,
            const std::vector<const TransportRecord*>& records) {
    int nsp = static_cast<int>(Y.size());
    if (nsp == 0) return Real(0);

    // Mass fractions → mole fractions
    std::vector<Real> X(static_cast<size_t>(nsp));
    mass_to_mole_fraction(Y, X, records);

    // Evaluate pure species viscosities
    std::vector<Real> mu(static_cast<size_t>(nsp));
    for (int i = 0; i < nsp; ++i)
        mu[i] = species_mu(*records[i], T);

    // Full Wilke: μ_mix = Σ_i ( X_i * μ_i / Σ_j (X_j * Φ_ij) )
    Real mu_mix = Real(0);
    for (int i = 0; i < nsp; ++i) {
        Real denom = Real(0);
        for (int j = 0; j < nsp; ++j) {
            if (i == j) {
                denom += X[j];
            } else {
                denom += X[j] * wilke_phi(mu[i], mu[j],
                    records[i]->M > Real(0) ? records[i]->M : Real(29),
                    records[j]->M > Real(0) ? records[j]->M : Real(29));
            }
        }
        mu_mix += X[i] * mu[i] / denom;
    }
    return mu_mix;
}

Real mix_kappa(Real T, const std::vector<Real>& Y,
               const std::vector<const TransportRecord*>& records) {
    int nsp = static_cast<int>(Y.size());
    if (nsp == 0) return Real(0);

    // Mass fractions → mole fractions
    std::vector<Real> X(static_cast<size_t>(nsp));
    mass_to_mole_fraction(Y, X, records);

    // Evaluate pure species conductivities
    std::vector<Real> kappa(static_cast<size_t>(nsp));
    for (int i = 0; i < nsp; ++i)
        kappa[i] = species_kappa(*records[i], T);

    // Mason-Saxena analog of Wilke for conductivity:
    // κ_mix = Σ_i ( X_i * κ_i / Σ_j (X_j * Ψ_ij) )
    Real kappa_mix = Real(0);
    for (int i = 0; i < nsp; ++i) {
        Real denom = Real(0);
        for (int j = 0; j < nsp; ++j) {
            if (i == j) {
                denom += X[j];
            } else {
                denom += X[j] * wilke_phi(kappa[i], kappa[j],
                    records[i]->M > Real(0) ? records[i]->M : Real(29),
                    records[j]->M > Real(0) ? records[j]->M : Real(29));
            }
        }
        kappa_mix += X[i] * kappa[i] / denom;
    }
    return kappa_mix;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
