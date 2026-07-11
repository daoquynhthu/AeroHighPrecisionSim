#pragma once

namespace aerosp {

#ifdef AEROSP_REAL_DOUBLE
    using Real = double;
#else
    using Real = float;
#endif

} // namespace aerosp
