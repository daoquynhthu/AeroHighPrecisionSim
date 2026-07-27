#include "aero/cfd/t_from_e.hpp"
#include "aero/cfd/nasa9_eval.hpp"

#include <algorithm>
#include <cmath>

namespace aerosp {
namespace aero {
namespace cfd {

Real T_from_e(Real e_target,
              const std::vector<Real>& Y,
              const std::vector<const SpeciesRecord*>& species,
              Real R_mix,
              Real T_guess,
              Real T_min,
              Real T_max,
              Real tol,
              int max_iter) {
    (void)R_mix;

    Real T = T_guess;
    T = std::max(T_min, std::min(T_max, T));

    for (int iter = 0; iter < max_iter; ++iter) {
        Real h = mix_h(T, Y, species);
        Real R = mix_R(Y, species);
        Real f = h - e_target - R * T;

        if (std::abs(f) < tol * (std::abs(e_target) + Real(1))) {
            return T;
        }

        Real cp = mix_cp(T, Y, species);
        Real df = cp - R;

        if (std::abs(df) < Real(1e-15)) {
            break;
        }

        Real dT = -f / df;
        Real T_new = T + dT;

        if (T_new <= T_min || T_new >= T_max || !std::isfinite(T_new)) {
            break;
        }

        T = T_new;
    }

    // Fallback: bisection
    Real fa = mix_h(T_min, Y, species) - e_target - mix_R(Y, species) * T_min;
    Real fb = mix_h(T_max, Y, species) - e_target - mix_R(Y, species) * T_max;

    if (fa * fb >= Real(0)) {
        return -Real(1);
    }

    for (int iter = 0; iter < 100; ++iter) {
        Real T_mid = (T_min + T_max) * Real(0.5);
        Real fm = mix_h(T_mid, Y, species) - e_target - mix_R(Y, species) * T_mid;

        if (std::abs(fm) < tol * (std::abs(e_target) + Real(1)) ||
            (T_max - T_min) * Real(0.5) < tol) {
            return T_mid;
        }

        if (fa * fm < Real(0)) {
            T_max = T_mid;
            fb = fm;
        } else {
            T_min = T_mid;
            fa = fm;
        }
    }

    return -Real(1);
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
