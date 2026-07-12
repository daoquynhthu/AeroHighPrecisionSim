#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct float3 { float x, y, z; };
struct double3 { double x, y, z; };
#endif

#ifdef __CUDACC__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST_DEVICE
#endif

namespace aerosp {
    enum class SimulationProfile {
        GLOBAL_BALLISTIC,  // Original missile logic
        LOCAL_TACTICAL     // Short-range dart logic
    };

    struct EnvironmentContext {
        SimulationProfile profile;
        double ref_alt;
        float3 gravity_local; // Constant gravity for local tactical
        double3 origin_ecef;
    };
}
