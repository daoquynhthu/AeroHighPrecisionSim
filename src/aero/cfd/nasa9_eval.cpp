#include "aero/cfd/nasa9_eval.hpp"

namespace aerosp {
namespace aero {
namespace cfd {

Real mix_R(const std::vector<Real>& Y,
           const std::vector<const SpeciesRecord*>& species) {
    Real R_mix = Real(0);
    int nsp = static_cast<int>(Y.size());
    for (int i = 0; i < nsp && i < static_cast<int>(species.size()); ++i) {
        R_mix += Y[i] * (R_UNIV / species[i]->M);
    }
    return R_mix * Real(1000); // [J/(kg*K)]
}

Real mix_cp(Real T, const std::vector<Real>& Y,
            const std::vector<const SpeciesRecord*>& species) {
    Real cp = Real(0);
    int nsp = static_cast<int>(Y.size());
    for (int i = 0; i < nsp && i < static_cast<int>(species.size()); ++i) {
        Real R_spec = R_UNIV / species[i]->M * Real(1000);
        cp += Y[i] * species_cp(R_spec, *species[i], T);
    }
    return cp;
}

Real mix_h(Real T, const std::vector<Real>& Y,
           const std::vector<const SpeciesRecord*>& species) {
    Real h = Real(0);
    int nsp = static_cast<int>(Y.size());
    for (int i = 0; i < nsp && i < static_cast<int>(species.size()); ++i) {
        Real R_spec = R_UNIV / species[i]->M * Real(1000);
        h += Y[i] * species_h(R_spec, *species[i], T);
    }
    return h;
}

Real mix_gamma(Real T, const std::vector<Real>& Y,
               const std::vector<const SpeciesRecord*>& species) {
    Real cp = mix_cp(T, Y, species);
    Real R_mix_val = mix_R(Y, species);
    return cp / (cp - R_mix_val);
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
