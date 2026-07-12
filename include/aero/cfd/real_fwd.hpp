#pragma once

namespace aerosp {

// Default precision: float (GPU-compatible, adequate for production).
// For CPU oracle double-precision calibration, build with -DAEROSIM_REAL_DOUBLE=ON.
// This toggles AEROSP_REAL_DOUBLE via CMakeLists.txt for both CPU and GPU.
#ifdef AEROSP_REAL_DOUBLE
    using Real = double;
#else
    using Real = float;
#endif

} // namespace aerosp
