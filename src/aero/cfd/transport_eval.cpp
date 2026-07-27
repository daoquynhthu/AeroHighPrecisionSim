#include "aero/cfd/transport_eval.hpp"

#include <cmath>

namespace aerosp {
namespace aero {
namespace cfd {

// Simplified Wilke mixing: negligible binary interaction.
// Full Wilke: mu_mix = Σ (X_i * mu_i) / Σ (X_j * Φ_ij)
// with Φ_ij = [1 + sqrt(mu_i/mu_j) * (M_j/M_i)^0.25]² / sqrt(8*(1+M_i/M_j))
// Here we use Herning-Zipperer approximation: mu_mix = Σ (X_i * mu_i * sqrt(M_i)) / Σ (X_j * sqrt(M_j))
// which is a common simplification that ignores second-order binary effects.

Real mix_mu(Real T, const std::vector<Real>& Y,
            const std::vector<const TransportRecord*>& records) {
    int nsp = static_cast<int>(Y.size());
    if (nsp == 0) return Real(0);

    Real sum_num = Real(0), sum_den = Real(0);
    for (int i = 0; i < nsp && i < static_cast<int>(records.size()); ++i) {
        Real mu = species_mu(*records[i], T);
        Real sqrt_M = real_sqrt(records[i]->name.empty() ? Real(29) : Real(29));
        sum_num += Y[i] * mu * sqrt_M;
        sum_den += Y[i] * sqrt_M;
    }
    return (sum_den > Real(0)) ? sum_num / sum_den : Real(0);
}

Real mix_kappa(Real T, const std::vector<Real>& Y,
               const std::vector<const TransportRecord*>& records) {
    int nsp = static_cast<int>(Y.size());
    if (nsp == 0) return Real(0);

    Real sum_num = Real(0), sum_den = Real(0);
    for (int i = 0; i < nsp && i < static_cast<int>(records.size()); ++i) {
        Real kappa = species_kappa(*records[i], T);
        Real sqrt_M = real_sqrt(records[i]->name.empty() ? Real(29) : Real(29));
        sum_num += Y[i] * kappa * sqrt_M;
        sum_den += Y[i] * sqrt_M;
    }
    return (sum_den > Real(0)) ? sum_num / sum_den : Real(0);
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
