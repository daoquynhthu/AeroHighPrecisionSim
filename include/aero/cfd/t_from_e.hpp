#pragma once

#include "aero/cfd/thermo_db.hpp"
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// Newton iteration to recover temperature from specific internal energy.
// Solves: mix_h(T, Y) - e_target - R_mix * T = 0
// Uses mix_cp - R_mix as Jacobian.
// Falls back to bisection [T_min, T_max] if Newton fails.
// Returns T on success, or -1 on failure.
Real T_from_e(Real e_target,
              const std::vector<Real>& Y,
              const std::vector<const SpeciesRecord*>& species,
              Real R_mix,
              Real T_guess = Real(1400),
              Real T_min = Real(200),
              Real T_max = Real(20000),
              Real tol = Real(1e-8),
              int max_iter = 50);

} // namespace cfd
} // namespace aero
} // namespace aerosp
