#pragma once

#include "aero/cfd/real_fwd.hpp"
#ifdef __CUDACC__
#  include <cuda_runtime.h>
#endif

namespace aerosp {
namespace aero {
namespace cfd {

bool ddot_gpu(const Real* x, const Real* y, int n, Real* result, cudaStream_t stream = nullptr);
bool daxpy_gpu(Real a, const Real* x, Real* y, int n, cudaStream_t stream = nullptr);
bool dnrm2_gpu(const Real* x, int n, Real* result, cudaStream_t stream = nullptr);
bool dscal_gpu(Real a, Real* x, int n, cudaStream_t stream = nullptr);
bool dcopy_gpu(const Real* src, Real* dst, int n, cudaStream_t stream = nullptr);
bool daxpby_gpu(Real a, const Real* x, Real b, Real* y, int n, cudaStream_t stream = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp
