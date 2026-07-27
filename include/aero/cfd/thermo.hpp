#pragma once

#include "aero/cfd/real.hpp"

#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

// ─── Gas Model Kind ──────────────────────────────────────────────

enum class GasModelKind : int {
    PERFECT_GAS = 0,
    EQUILIBRIUM_AIR = 1,
    CHEM_NON_EQ = 2
};

// ─── NASA 7-Coefficient Polynomial (McBride et al. 2002) ────────
//
// Format:
//   cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
//   h/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
//   s/R  = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7

struct McBride7 {
    Real a1, a2, a3, a4, a5, a6, a7;
};

// One species: two temperature ranges, molar mass, name
struct SpeciesThermo {
    const char* name;
    Real M;              // molar mass [kg/kmol]
    Real T_low, T_break, T_high;
    McBride7 low, high;
};

// Pre-defined air species indices
enum SpeciesIdx : int {
    SPECIES_N2 = 0,
    SPECIES_O2 = 1,
    SPECIES_NO = 2,
    SPECIES_N  = 3,
    SPECIES_O  = 4,
    NUM_AIR_SPECIES = 5
};

// Access the pre-defined 5-species air database
const SpeciesThermo& species_db(SpeciesIdx s);

// ─── Polynomial Evaluation (Host + Device) ───────────────────────

AEROSP_REAL_HOST_DEVICE inline Real eval_cp_R(const McBride7& c, Real T) {
    return c.a1 + T*(c.a2 + T*(c.a3 + T*(c.a4 + T*c.a5)));
}

AEROSP_REAL_HOST_DEVICE inline Real eval_h_RT(const McBride7& c, Real T) {
    return c.a1 + T*(c.a2*Real(0.5) + T*(c.a3/Real(3.0) + T*(c.a4*Real(0.25) + T*c.a5*Real(0.2)))) + c.a6/T;
}

// Select the correct coefficient set (low or high) based on T
AEROSP_REAL_HOST_DEVICE inline const McBride7& pick_coeffs(const SpeciesThermo& sp, Real T) {
    return T <= sp.T_break ? sp.low : sp.high;
}

// Per-species: cp = R_specific * cp_over_R(T)
AEROSP_REAL_HOST_DEVICE inline Real species_cp(const SpeciesThermo& sp, Real T) {
    Real R_s = Real(8314.462618) / sp.M;
    return R_s * eval_cp_R(pick_coeffs(sp, T), T);
}

// Per-species: h = R_specific * T * (h/RT)(T)
AEROSP_REAL_HOST_DEVICE inline Real species_h(const SpeciesThermo& sp, Real T) {
    Real R_s = Real(8314.462618) / sp.M;
    return R_s * T * eval_h_RT(pick_coeffs(sp, T), T);
}

// ─── GasModel Base (Host) ─────────────────────────────────────────

class GasModel {
public:
    virtual ~GasModel() = default;
    virtual GasModelKind kind() const = 0;

    virtual Real cp(Real T) const = 0;       // [J/(kg*K)]
    virtual Real cv(Real T) const = 0;
    virtual Real gamma(Real T) const = 0;
    virtual Real R() const = 0;              // specific gas constant [J/(kg*K)]

    virtual Real h(Real T) const = 0;        // specific enthalpy [J/kg]
    virtual Real e(Real T) const = 0;        // specific internal energy [J/kg]
    virtual Real T_from_e(Real e) const = 0;
};

// ─── PerfectGasModel ─────────────────────────────────────────────

class PerfectGasModel : public GasModel {
public:
    explicit PerfectGasModel(Real gamma, Real R_val = Real(287.058));
    GasModelKind kind() const override;
    Real cp(Real T) const override;
    Real cv(Real T) const override;
    Real gamma(Real T) const override;
    Real R() const override;
    Real h(Real T) const override;
    Real e(Real T) const override;
    Real T_from_e(Real e) const override;
private:
    Real gamma_;
    Real cp_;
    Real cv_;
    Real R_;
};

// ─── EquilibriumAirModel (fixed-composition 5-species air) ──────

class EquilibriumAirModel : public GasModel {
public:
    EquilibriumAirModel();
    GasModelKind kind() const override;
    Real cp(Real T) const override;
    Real cv(Real T) const override;
    Real gamma(Real T) const override;
    Real R() const override;
    Real h(Real T) const override;
    Real e(Real T) const override;
    Real T_from_e(Real e) const override;
private:
    Real Y_[NUM_AIR_SPECIES];
    Real R_spec_[NUM_AIR_SPECIES];
    Real R_mix_;
    Real evaluate_cp(Real T, const Real Y[5]) const;
    Real evaluate_h(Real T, const Real Y[5]) const;
};

// ─── Helper: create a GasModel from kind + gamma fallback ────────

GasModel* create_gas_model(GasModelKind kind, Real gamma_fallback = Real(1.4));

// ─── Device-Side Helpers (for kernel use in Phase 15.2+) ────────
//
// These accept raw coefficient arrays by pointer, suitable for passing
// as kernel arguments or storing in __constant__ memory.

// Per-species cp = R_s * (a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4)
AEROSP_REAL_HOST_DEVICE inline Real d_species_cp(const McBride7& c, Real R_s, Real T) {
    return R_s * eval_cp_R(c, T);
}

// Per-species h = R_s * T * (a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T)
AEROSP_REAL_HOST_DEVICE inline Real d_species_h(const McBride7& c, Real R_s, Real T) {
    return R_s * T * eval_h_RT(c, T);
}

// Mixture cp = Σ Y_s * R_s * cp_over_R_s(T)
AEROSP_REAL_HOST_DEVICE inline Real d_mix_cp(
    const Real Y[5], const Real R_s[5], const McBride7 coeffs[5], Real T)
{
    Real cp = Real(0);
    for (int s = 0; s < 5; ++s)
        cp += Y[s] * R_s[s] * eval_cp_R(coeffs[s], T);
    return cp;
}

// Mixture gas constant = Σ Y_s * R_s
AEROSP_REAL_HOST_DEVICE inline Real d_mix_R(const Real Y[5], const Real R_s[5]) {
    Real R = Real(0);
    for (int s = 0; s < 5; ++s)
        R += Y[s] * R_s[s];
    return R;
}

// Mixture gamma = cp / (cp - R)
AEROSP_REAL_HOST_DEVICE inline Real d_mix_gamma(
    const Real Y[5], const Real R_s[5], const McBride7 coeffs[5], Real T)
{
    Real cp = d_mix_cp(Y, R_s, coeffs, T);
    Real R = d_mix_R(Y, R_s);
    return cp / (cp - R);
}

AEROSP_REAL_HOST_DEVICE inline Real d_mix_h(
    const Real Y[5], const Real R_s[5], const McBride7 coeffs[5], Real T)
{
    Real h = Real(0);
    for (int s = 0; s < 5; ++s)
        h += Y[s] * R_s[s] * T * eval_h_RT(coeffs[s], T);
    return h;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
