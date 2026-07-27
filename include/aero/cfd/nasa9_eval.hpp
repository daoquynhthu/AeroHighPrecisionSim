#pragma once

#include "aero/cfd/thermo_db.hpp"
#include "aero/cfd/real.hpp"

#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// ─── Single-interval evaluators ─────────────────────────────────────
// NASA-9 polynomial: cp/R = a1*T⁻² + a2*T⁻¹ + a3 + a4*T + a5*T² + a6*T³ + a7*T⁴
// coeffs[0..8] = {a1, a2, a3, a4, a5, a6, a7, b1, b2}

AEROSP_REAL_HOST_DEVICE inline Real eval_cp_R(const Real* a, Real T) {
    Real invT = Real(1) / T;
    Real T2 = T * T;
    Real T3 = T2 * T;
    Real T4 = T2 * T2;
    return (a[0] * invT + a[1]) * invT + a[2]
         + a[3] * T + a[4] * T2 + a[5] * T3 + a[6] * T4;
}

// h/RT = -a1*T⁻² + a2*lnT/T + a3 + a4*T/2 + a5*T²/3 + a6*T³/4 + a7*T⁴/5 + b1/T
AEROSP_REAL_HOST_DEVICE inline Real eval_h_RT(const Real* a, Real T) {
    Real invT = Real(1) / T;
    Real lnT = real_log(T);
    Real T2 = T * T;
    Real T3 = T2 * T;
    Real T4 = T2 * T2;
    return -a[0] * invT * invT
         + a[1] * lnT * invT
         + a[2]
         + a[3] * T * Real(0.5)
         + a[4] * T2 / Real(3)
         + a[5] * T3 * Real(0.25)
         + a[6] * T4 * Real(0.2)
         + a[7] * invT;
}

// s/R = -a1/(2*T²) - a2/T + a3*lnT + a4*T + a5*T²/2 + a6*T³/3 + a7*T⁴/4 + b2
AEROSP_REAL_HOST_DEVICE inline Real eval_s_R(const Real* a, Real T) {
    Real invT = Real(1) / T;
    Real lnT = real_log(T);
    Real T2 = T * T;
    Real T3 = T2 * T;
    Real T4 = T2 * T2;
    return -a[0] * Real(0.5) * invT * invT
         - a[1] * invT
         + a[2] * lnT
         + a[3] * T
         + a[4] * T2 * Real(0.5)
         + a[5] * T3 / Real(3)
         + a[6] * T4 * Real(0.25)
         + a[8];
}

// ─── Temperature interval selection (CPU) ────────────────────────

inline int select_interval(Real T, Real T_break0, Real T_break1, int n_intervals) {
    if (n_intervals <= 1) return 0;
    if (T <= T_break0) return 0;
    if (n_intervals == 2) return 1;
    if (T <= T_break1) return 1;
    return 2;
}

// ─── Species-level evaluators ────────────────────────────────────

inline Real species_cp_R(const SpeciesRecord& sp, Real T) {
    int seg = select_interval(T, sp.T_break[0], sp.T_break[1], sp.n_intervals);
    return eval_cp_R(sp.coeffs[seg], T);
}

inline Real species_h_RT(const SpeciesRecord& sp, Real T) {
    int seg = select_interval(T, sp.T_break[0], sp.T_break[1], sp.n_intervals);
    return eval_h_RT(sp.coeffs[seg], T);
}

inline Real species_s_R(const SpeciesRecord& sp, Real T) {
    int seg = select_interval(T, sp.T_break[0], sp.T_break[1], sp.n_intervals);
    return eval_s_R(sp.coeffs[seg], T);
}

inline Real species_cp(Real R_specific, const SpeciesRecord& sp, Real T) {
    return R_specific * species_cp_R(sp, T);
}

inline Real species_h(Real R_specific, const SpeciesRecord& sp, Real T) {
    return R_specific * T * species_h_RT(sp, T);
}

inline Real species_s(Real R_specific, const SpeciesRecord& sp, Real T) {
    return R_specific * species_s_R(sp, T);
}

// ─── Mixed-gas evaluators (CPU declarations) ─────────────────────

Real mix_cp(Real T, const std::vector<Real>& Y,
            const std::vector<const SpeciesRecord*>& species);

Real mix_h(Real T, const std::vector<Real>& Y,
           const std::vector<const SpeciesRecord*>& species);

Real mix_gamma(Real T, const std::vector<Real>& Y,
               const std::vector<const SpeciesRecord*>& species);

Real mix_R(const std::vector<Real>& Y,
           const std::vector<const SpeciesRecord*>& species);

} // namespace cfd
} // namespace aero
} // namespace aerosp
