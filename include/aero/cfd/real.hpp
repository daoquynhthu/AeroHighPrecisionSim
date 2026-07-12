#pragma once

#include "aero/cfd/real_fwd.hpp"
#include <cmath>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace aerosp {

#ifdef __CUDACC__
#define AEROSP_REAL_DEVICE static __device__
#define AEROSP_REAL_HOST_DEVICE static __device__ __host__ inline
#else
#define AEROSP_REAL_DEVICE static inline
#define AEROSP_REAL_HOST_DEVICE static inline
#endif

#ifdef AEROSP_REAL_DOUBLE

    AEROSP_REAL_HOST_DEVICE Real real_sqrt(Real x) { return sqrt(x); }
    AEROSP_REAL_HOST_DEVICE Real real_fabs(Real x) { return fabs(x); }
    AEROSP_REAL_HOST_DEVICE Real real_fmin(Real x, Real y) { return fmin(x, y); }
    AEROSP_REAL_HOST_DEVICE Real real_fmax(Real x, Real y) { return fmax(x, y); }
    AEROSP_REAL_HOST_DEVICE bool real_isfinite(Real x) {
#ifdef __CUDA_ARCH__
        return static_cast<bool>(isfinite(x));
#else
        return static_cast<bool>(std::isfinite(x));
#endif
    }
    AEROSP_REAL_HOST_DEVICE Real real_copysign(Real x, Real y) { return copysign(x, y); }
    AEROSP_REAL_HOST_DEVICE Real real_cos(Real x) { return cos(x); }
    AEROSP_REAL_HOST_DEVICE Real real_sin(Real x) { return sin(x); }
    AEROSP_REAL_HOST_DEVICE Real real_pow(Real x, Real y) { return pow(x, y); }
    AEROSP_REAL_HOST_DEVICE Real real_exp(Real x) { return exp(x); }

#ifdef __CUDACC__
    AEROSP_REAL_DEVICE Real real_atomic_add(Real* addr, Real val) {
        return atomicAdd(addr, val);
    }

    AEROSP_REAL_DEVICE Real real_atomic_min(Real* addr, Real val) {
        unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
        unsigned long long old = *addr_as_ull;
        unsigned long long assumed;
        do {
            assumed = old;
            old = atomicCAS(addr_as_ull, assumed,
                __double_as_longlong(real_fmin(val, __longlong_as_double(assumed))));
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    AEROSP_REAL_DEVICE Real real_atomic_max(Real* addr, Real val) {
        unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
        unsigned long long old = *addr_as_ull;
        unsigned long long assumed;
        do {
            assumed = old;
            old = atomicCAS(addr_as_ull, assumed,
                __double_as_longlong(real_fmax(val, __longlong_as_double(assumed))));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
#endif
#else
    using Real = float;

    AEROSP_REAL_HOST_DEVICE Real real_sqrt(Real x) { return sqrtf(x); }
    AEROSP_REAL_HOST_DEVICE Real real_fabs(Real x) { return fabsf(x); }
    AEROSP_REAL_HOST_DEVICE Real real_fmin(Real x, Real y) { return fminf(x, y); }
    AEROSP_REAL_HOST_DEVICE Real real_fmax(Real x, Real y) { return fmaxf(x, y); }
    AEROSP_REAL_HOST_DEVICE bool real_isfinite(Real x) {
#ifdef __CUDA_ARCH__
        return static_cast<bool>(__finitef(x));
#else
        return static_cast<bool>(std::isfinite(x));
#endif
    }
    AEROSP_REAL_HOST_DEVICE Real real_copysign(Real x, Real y) { return copysignf(x, y); }
    AEROSP_REAL_HOST_DEVICE Real real_cos(Real x) { return cosf(x); }
    AEROSP_REAL_HOST_DEVICE Real real_sin(Real x) { return sinf(x); }
    AEROSP_REAL_HOST_DEVICE Real real_pow(Real x, Real y) { return powf(x, y); }
    AEROSP_REAL_HOST_DEVICE Real real_exp(Real x) { return expf(x); }

#ifdef __CUDACC__
    AEROSP_REAL_DEVICE Real real_atomic_add(Real* addr, Real val) {
        return atomicAdd(addr, val);
    }

    AEROSP_REAL_DEVICE Real real_atomic_min(Real* addr, Real val) {
        unsigned int* addr_as_int = reinterpret_cast<unsigned int*>(addr);
        unsigned int old = *addr_as_int;
        unsigned int assumed;
        do {
            assumed = old;
            old = atomicCAS(addr_as_int, assumed,
                __float_as_int(real_fmin(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }

    AEROSP_REAL_DEVICE Real real_atomic_max(Real* addr, Real val) {
        unsigned int* addr_as_int = reinterpret_cast<unsigned int*>(addr);
        unsigned int old = *addr_as_int;
        unsigned int assumed;
        do {
            assumed = old;
            old = atomicCAS(addr_as_int, assumed,
                __float_as_int(real_fmax(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }
#endif
#endif

} // namespace aerosp
