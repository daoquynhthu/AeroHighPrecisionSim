#include "aero/cfd/gpu_thermo_table.hpp"

namespace aerosp {
namespace aero {
namespace cfd {

#ifndef AEROSP_HAS_CUDA

// CPU-only stub — real implementation in gpu_thermo_table.cu (NVCC)
bool upload_thermo_table(const ThermoDb& db, const std::vector<int>& selected,
                         std::string* error) {
    (void)db;
    (void)selected;
    if (error) *error = "upload_thermo_table requires CUDA build";
    return false;
}

#endif

} // namespace cfd
} // namespace aero
} // namespace aerosp
