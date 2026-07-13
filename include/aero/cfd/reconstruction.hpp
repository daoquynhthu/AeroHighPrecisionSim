#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"

#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

namespace aerosp {
namespace aero {
namespace cfd {

struct PrimitiveGradient {
    Real drho_dx = 0.0f;
    Real drho_dy = 0.0f;
    Real drho_dz = 0.0f;
    Real du_dx = 0.0f;
    Real du_dy = 0.0f;
    Real du_dz = 0.0f;
    Real dv_dx = 0.0f;
    Real dv_dy = 0.0f;
    Real dv_dz = 0.0f;
    Real dw_dx = 0.0f;
    Real dw_dy = 0.0f;
    Real dw_dz = 0.0f;
    Real dp_dx = 0.0f;
    Real dp_dy = 0.0f;
    Real dp_dz = 0.0f;
    Real dnu_tilde_dx = 0.0f;
    Real dnu_tilde_dy = 0.0f;
    Real dnu_tilde_dz = 0.0f;
};

struct PrimitiveLimiter {
    Real rho = 1.0f;
    Real u = 1.0f;
    Real v = 1.0f;
    Real w = 1.0f;
    Real p = 1.0f;
    Real nu_tilde = 1.0f;
};

bool solve_3x3(Real a[3][3], Real b[3], Real x[3]);
bool lu_factor_3x3(const Real a[3][3], Real lu[3][3], int pivot[3]);
void lu_solve_3x3(const Real lu[3][3], const int pivot[3], const Real b[3], Real x[3]);

class DeviceMesh;

std::vector<PrimitiveGradient> compute_green_gauss_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    Real gamma,
    const std::vector<PrimitiveState>* primitive_override = nullptr);

std::vector<PrimitiveGradient> compute_least_squares_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    Real gamma,
    const std::vector<PrimitiveState>* primitive_override = nullptr);

std::vector<PrimitiveLimiter> compute_barth_jespersen_limiters(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    Real gamma,
    const std::vector<PrimitiveState>* primitive_override = nullptr);

PrimitiveGradient apply_limiter(const PrimitiveGradient& gradient, const PrimitiveLimiter& limiter);

bool apply_limiter_gpu(DeviceMesh& mesh, std::string* error = nullptr, cudaStream_t stream = nullptr);
bool apply_limiter_gpu(DeviceMesh& mesh, bool sync, std::string* error = nullptr, cudaStream_t stream = nullptr);

PrimitiveState reconstruct_primitive(
    const PrimitiveState& center,
    const PrimitiveGradient& gradient,
    Real dx,
    Real dy,
    Real dz);

PrimitiveState reconstruct_primitive_positive(
    const PrimitiveState& center,
    const PrimitiveGradient& gradient,
    Real dx,
    Real dy,
    Real dz,
    Real rho_floor,
    Real p_floor,
    Real* theta = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp
