#include "aero/cfd/thermo.hpp"

#include <cmath>
#include <stdexcept>

namespace aerosp {
namespace aero {
namespace cfd {

// ─── Species Database: NASA Glenn 7-Coeff Polynomials ────────────
//
// Source: McBride, B.J., M.J. Zehe, and S. Gordon, "NASA Glenn
//         Coefficients for Calculating Thermodynamic Properties of
//         Individual Species", NASA/TP-2002-211556, September 2002.
//
// Verified against:
//   - Cantera 3.0 gri30_highT.yaml (gas-phase NASA7 coefficients)
//   - Python NASA-7 library (Coolvictory/asgur, NASA-TP-2002-211556)
//
//   cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
//   h/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
//
//   R = 8314.462618 J/(kmol*K)

static const SpeciesThermo kSpeciesDB[] = {
    // ── N2 (nitrogen) ─────────────────────────────────────────
    {
        "N2",                          // name
        28.0134f,                      // M [kg/kmol]
        200.0f, 1000.0f, 6000.0f,     // T_low, T_break, T_high
        { 3.53100528E+00f, -1.23660988E-04f, -5.02999433E-07f,
          2.43530612E-09f, -1.40881235E-12f, -1.04697628E+03f,
          2.96747038E+00f },
        { 2.95257637E+00f,  1.39690040E-03f, -4.92631603E-07f,
          7.86010195E-11f, -4.60755204E-15f, -9.23948688E+02f,
          5.87188762E+00f }
    },
    // ── O2 (oxygen) ───────────────────────────────────────────
    {
        "O2",
        31.9988f,
        200.0f, 1000.0f, 6000.0f,
        { 3.78245636E+00f, -2.99673416E-03f,  9.84730201E-06f,
         -9.68129509E-09f,  3.24372837E-12f, -1.06394356E+03f,
          3.65767573E+00f },
        { 3.61221390E+00f,  7.48531660E-04f, -1.98206738E-07f,
          3.37490080E-11f, -2.39075099E-15f, -1.19781510E+03f,
          3.45264136E+00f }
    },
    // ── NO (nitric oxide) ─────────────────────────────────────
    {
        "NO",
        30.0061f,
        200.0f, 1000.0f, 6000.0f,
        { 4.21859896E+00f, -4.63988124E-03f,  1.10443049E-05f,
         -9.34055507E-09f,  2.80554874E-12f,  9.84509964E+03f,
          2.28061001E+00f },
        { 3.26071234E+00f,  1.19101135E-03f, -4.29122646E-07f,
          6.94481463E-11f, -4.03295681E-15f,  9.92143132E+03f,
          6.36900518E+00f }
    },
    // ── N (atomic nitrogen) ────────────────────────────────────
    // Cantera gri30_highT.yaml, RUS 78 (monatomic with electronic
    // excitation, cp/R ~ 2.5 at low T)
    {
        "N",
        14.0067f,
        200.0f, 1000.0f, 6000.0f,
        { 2.50000000E+00f,  0.0f,  0.0f,  0.0f,  0.0f,
          2.54736599E+04f, -4.46682853E-01f },
        { 2.50000286E+00f, -5.65334214E-09f,  3.63251723E-12f,
         -9.19949720E-16f,  7.95260746E-20f,  2.54736589E+04f,
         -4.46698494E-01f }
    },
    // ── O (atomic oxygen) ─────────────────────────────────────
    {
        "O",
        15.9994f,
        200.0f, 1000.0f, 6000.0f,
        { 3.16826710E+00f, -3.27931884E-03f,  6.64306396E-06f,
         -6.12806624E-09f,  2.11265971E-12f,  2.91222592E+04f,
          2.05193346E+00f },
        { 2.54363697E+00f, -2.73162486E-05f, -4.19029520E-09f,
          4.95481845E-12f, -4.79553694E-16f,  2.92260120E+04f,
          4.92229457E+00f }
    }
};

static_assert(sizeof(kSpeciesDB) / sizeof(kSpeciesDB[0]) == NUM_AIR_SPECIES,
    "Species database size mismatch");

const SpeciesThermo& species_db(SpeciesIdx s) {
    return kSpeciesDB[static_cast<int>(s)];
}

// ─── PerfectGasModel ─────────────────────────────────────────────

PerfectGasModel::PerfectGasModel(Real gamma, Real R_val)
    : gamma_(gamma), R_(R_val)
{
    cp_ = gamma_ * R_ / (gamma_ - Real(1));
    cv_ = R_ / (gamma_ - Real(1));
}

GasModelKind PerfectGasModel::kind() const { return GasModelKind::PERFECT_GAS; }

Real PerfectGasModel::cp(Real /*T*/) const { return cp_; }
Real PerfectGasModel::cv(Real /*T*/) const { return cv_; }
Real PerfectGasModel::gamma(Real /*T*/) const { return gamma_; }
Real PerfectGasModel::R() const { return R_; }

Real PerfectGasModel::h(Real T) const {
    return cp_ * T;
}

Real PerfectGasModel::e(Real T) const {
    return cv_ * T;
}

Real PerfectGasModel::T_from_e(Real e) const {
    return e / (cv_ + Real(1e-30));
}

// ─── EquilibriumAirModel ─────────────────────────────────────────

EquilibriumAirModel::EquilibriumAirModel() {
    Y_[SPECIES_N2] = Real(0.767);
    Y_[SPECIES_O2] = Real(0.233);
    Y_[SPECIES_NO] = Real(0);
    Y_[SPECIES_N]  = Real(0);
    Y_[SPECIES_O]  = Real(0);

    Real R_univ = Real(8314.462618);
    for (int s = 0; s < NUM_AIR_SPECIES; ++s)
        R_spec_[s] = R_univ / kSpeciesDB[s].M;

    R_mix_ = Real(0);
    for (int s = 0; s < NUM_AIR_SPECIES; ++s)
        R_mix_ += Y_[s] * R_spec_[s];
}

GasModelKind EquilibriumAirModel::kind() const {
    return GasModelKind::EQUILIBRIUM_AIR;
}

Real EquilibriumAirModel::evaluate_cp(Real T, const Real Y[5]) const {
    Real cp = Real(0);
    for (int s = 0; s < NUM_AIR_SPECIES; ++s) {
        const auto& sp = kSpeciesDB[s];
        cp += Y[s] * R_spec_[s] * eval_cp_R(pick_coeffs(sp, T), T);
    }
    return cp;
}

Real EquilibriumAirModel::evaluate_h(Real T, const Real Y[5]) const {
    Real h = Real(0);
    for (int s = 0; s < NUM_AIR_SPECIES; ++s) {
        const auto& sp = kSpeciesDB[s];
        h += Y[s] * R_spec_[s] * T * eval_h_RT(pick_coeffs(sp, T), T);
    }
    return h;
}

Real EquilibriumAirModel::cp(Real T) const {
    return evaluate_cp(T, Y_);
}

Real EquilibriumAirModel::cv(Real T) const {
    return cp(T) - R_mix_;
}

Real EquilibriumAirModel::gamma(Real T) const {
    Real cp_val = cp(T);
    return cp_val / (cp_val - R_mix_);
}

Real EquilibriumAirModel::R() const {
    return R_mix_;
}

Real EquilibriumAirModel::h(Real T) const {
    return evaluate_h(T, Y_);
}

Real EquilibriumAirModel::e(Real T) const {
    return h(T) - R_mix_ * T;
}

Real EquilibriumAirModel::T_from_e(Real target_e) const {
    Real cv_guess = Real(717.0);
    Real T = target_e / cv_guess;
    T = std::fmax(std::fmin(T, Real(6000.0)), Real(200.0));

    for (int iter = 0; iter < 50; ++iter) {
        Real e_cur = e(T);
        Real f = e_cur - target_e;
        if (std::fabs(f) < Real(1e-6) * std::fmax(Real(1), std::fabs(target_e)))
            break;
        Real cv_cur = cv(T);
        if (std::fabs(cv_cur) < Real(1e-30))
            break;
        Real dT = -f / cv_cur;
        dT = std::fmax(std::fmin(dT, Real(500.0)), Real(-500.0));
        T += dT;
        T = std::fmax(std::fmin(T, Real(6000.0)), Real(200.0));
    }
    return T;
}

// ─── Factory ─────────────────────────────────────────────────────

GasModel* create_gas_model(GasModelKind kind, Real gamma_fallback) {
    switch (kind) {
    case GasModelKind::PERFECT_GAS:
        return new PerfectGasModel(gamma_fallback);
    case GasModelKind::EQUILIBRIUM_AIR:
        return new EquilibriumAirModel();
    case GasModelKind::CHEM_NON_EQ:
        return new PerfectGasModel(gamma_fallback);
    }
    return new PerfectGasModel(gamma_fallback);
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
