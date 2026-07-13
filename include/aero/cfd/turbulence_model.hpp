#pragma once

namespace aerosp {
namespace aero {
namespace cfd {

enum class TurbulenceModel : int {
    LAMINAR = 0,
    SA = 1,
    SA_DDES = 2,
    SST = 3
};

} // namespace cfd
} // namespace aero
} // namespace aerosp
