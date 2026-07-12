#pragma once

#include <cmath>
#include <algorithm>
#include "aero/panel/aero_solver.hpp"

namespace aerosp {
namespace aero {
namespace eng {

struct EngineeringAeroCoeffs {
    double CX, CY, CZ;
    double Cl, Cm, Cn;
    double CD, CL;
};

inline EngineeringAeroCoeffs compute_engineering_coeffs(
    const aerosp::aero::panel::AeroGeometry& geo,
    double mach, double alpha_rad, double beta_rad)
{
    double ca = std::cos(alpha_rad);
    double sa = std::sin(alpha_rad);
    double cb = std::cos(beta_rad);
    double sb = std::sin(beta_rad);

    double Cf = 0.074 / std::pow(1.0e7, 0.2);
    double CD_skin = Cf * geo.wet_area / geo.ref_area;

    double CD_base = 0.0;
    if (geo.base_area > 0.0) {
        double Ab_ratio = geo.base_area / geo.ref_area;
        double m2 = mach * mach;
        CD_base = (mach < 1.0) ? 0.12 * Ab_ratio / (m2 + 0.1)
                               : 0.20 * Ab_ratio / m2;
    }

    double CD_wave = 0.0;
    if (mach > 1.0) {
        double beta = std::sqrt(mach * mach - 1.0);
        double ratio = 1.0 / (beta * geo.nose_fineness);
        CD_wave = (1.0 - std::cos(2.0 * std::atan(ratio))) * 0.5;
        if (mach < 1.2) {
            double peak = 0.15 * (1.0 + std::cos(3.14159265 * (mach - 1.0) / 0.4));
            if (peak > CD_wave) CD_wave = peak;
        }
    }

    double AR = geo.ref_span * geo.ref_span / geo.ref_area;
    double CL_alpha_0 = 2.0 * 3.14159265 * AR / (2.0 + std::sqrt(AR * AR + 4.0));

    double CL = 0.0;
    if (mach < 0.8) {
        double beta_pg = std::sqrt(std::max(1.0 - mach * mach, 0.01));
        double CL_alpha = CL_alpha_0 / beta_pg;
        CL = CL_alpha * std::sin(alpha_rad) * std::cos(alpha_rad);
    } else if (mach < 1.2) {
        double beta_sub = std::sqrt(std::max(1.0 - 0.64, 0.01));
        double CL_sub = (CL_alpha_0 / beta_sub) * std::sin(alpha_rad) * std::cos(alpha_rad);
        double CL_sup = (4.0 / std::sqrt(std::max(mach * mach - 1.0, 0.01)))
                       * std::sin(alpha_rad) * std::cos(alpha_rad);
        double t = (mach - 0.8) / 0.4;
        CL = (1.0 - t) * CL_sub + t * CL_sup;
    } else {
        double beta = std::sqrt(std::max(mach * mach - 1.0, 0.01));
        double CL_alpha = 4.0 / beta;
        CL = CL_alpha * std::sin(alpha_rad) * std::cos(alpha_rad);
    }

    if (mach > 5.0) {
        double CL_newt = 2.0 * std::sin(alpha_rad) * std::sin(alpha_rad) * std::cos(alpha_rad);
        double t = std::min(1.0, (mach - 5.0) / 3.0);
        CL = (1.0 - t) * CL + t * CL_newt;
    }

    double e = 0.8;
    double CD_ind = CL * CL / (3.14159265 * e * AR);
    double CD_wind = CD_skin + CD_base + CD_wave + CD_ind;

    double CY_beta = -CL_alpha_0 * 0.5;
    double CY = CY_beta * std::sin(beta_rad);

    double Fsx = -CD_wind;
    double Fsz = -CL;

    double CX = Fsx * ca * cb - CY * sb + Fsz * sa * cb;
    double CZ = -Fsx * sa + Fsz * ca;

    double static_margin = 0.05;
    double Cm = -static_margin * CL;
    double Cn = CY * static_margin;
    double Cl = 0.0;

    if (std::abs(beta_rad) < 1e-8) {
        CY = 0.0;
        Cn = 0.0;
    }

    return {CX, CY, CZ, Cl, Cm, Cn, CD_wind, CL};
}

} // namespace panel
} // namespace aero
} // namespace aerosp
