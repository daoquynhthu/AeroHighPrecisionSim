#pragma once

#include <cuda_runtime.h>
#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

bool cuda_check(cudaError_t error, const char* action, std::string* message = nullptr);

template <typename T>
static inline void cuda_free_safe(T*& ptr) {
    if (ptr) { cudaFree(ptr); ptr = nullptr; }
}

} // namespace cfd
} // namespace aero
} // namespace aerosp

#ifdef NDEBUG
#define CUDA_KERNEL_CHECK(msg, error) ((void)(msg), (void)(error))
#else
#define CUDA_KERNEL_CHECK(msg, error) \
    do { if (!aerosp::aero::cfd::cuda_check(cudaGetLastError(), (msg), (error))) return false; } while(0)
#endif
