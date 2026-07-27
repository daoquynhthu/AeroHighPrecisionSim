#pragma once

#include "aero/cfd/thermo_db.hpp"
#include "aero/cfd/real.hpp"

namespace aerosp {
namespace aero {
namespace cfd {

// Maximum species supported in GPU constant memory
static constexpr int GPU_MAX_SPECIES = 128;

// GPU constant memory layout (flat arrays)
struct GpuThermoTable {
    // Per-species data: packed as flat arrays
    Real M[GPU_MAX_SPECIES];                       // [kg/kmol]
    Real T_break_low[GPU_MAX_SPECIES];             // first interval T_min
    Real T_break_mid[GPU_MAX_SPECIES];             // second interval T_max (0 if only 1 interval)
    Real T_break_high[GPU_MAX_SPECIES];            // third interval T_max (0 if only 2 intervals)
    Real coeffs_a1[GPU_MAX_SPECIES];               // NASA-9 a1 (interval 0)
    Real coeffs_a2[GPU_MAX_SPECIES];
    Real coeffs_a3[GPU_MAX_SPECIES];
    Real coeffs_a4[GPU_MAX_SPECIES];
    Real coeffs_a5[GPU_MAX_SPECIES];
    Real coeffs_a6[GPU_MAX_SPECIES];
    Real coeffs_a7[GPU_MAX_SPECIES];
    Real coeffs_b1[GPU_MAX_SPECIES];
    Real coeffs_b2[GPU_MAX_SPECIES];
    // Intervals 1 and 2 (padded with zeros if not present)
    Real coeffs2[GPU_MAX_SPECIES][9];
    Real coeffs3[GPU_MAX_SPECIES][9];
    int n_intervals[GPU_MAX_SPECIES];
    int nspecies;
};

// Upload selected species from host ThermoDb to GPU __constant__ memory.
// On non-CUDA builds this is a no-op stub.
bool upload_thermo_table(const ThermoDb& db, const std::vector<int>& selected,
                         std::string* error = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp
