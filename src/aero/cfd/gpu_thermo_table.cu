#include "aero/cfd/gpu_thermo_table.hpp"

#include <cstring>
#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

// Device-side constant memory table
__constant__ GpuThermoTable d_thermo_table;

// Host-side upload: called from CfdSolver init (compiled by NVCC)
bool upload_thermo_table(const ThermoDb& db, const std::vector<int>& selected,
                         std::string* error) {
    int nsp = static_cast<int>(selected.size());
    if (nsp > GPU_MAX_SPECIES) {
        if (error) *error = "too many species: " + std::to_string(nsp) + " > " + std::to_string(GPU_MAX_SPECIES);
        return false;
    }

    GpuThermoTable h;
    std::memset(&h, 0, sizeof(h));
    h.nspecies = nsp;

    for (int i = 0; i < nsp; ++i) {
        const auto& rec = db.get_species(selected[i]);
        h.M[i] = rec.M;
        int nint = rec.n_intervals;
        h.n_intervals[i] = nint;

        for (int k = 0; k < 9; ++k)
            (&h.coeffs_a1[i])[k] = rec.coeffs[0][k];

        if (nint > 0) h.T_break_low[i] = rec.T_break[0];

        if (nint > 1) {
            h.T_break_mid[i] = rec.T_break[1];
            for (int k = 0; k < 9; ++k)
                h.coeffs2[i][k] = rec.coeffs[1][k];
        }

        if (nint > 2) {
            h.T_break_high[i] = Real(20000.0);
            for (int k = 0; k < 9; ++k)
                h.coeffs3[i][k] = rec.coeffs[2][k];
        }
    }

    cudaError_t ce = cudaMemcpyToSymbol(d_thermo_table, &h, sizeof(GpuThermoTable));
    if (ce != cudaSuccess) {
        if (error) *error = "cudaMemcpyToSymbol failed: " + std::string(cudaGetErrorString(ce));
        return false;
    }

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
