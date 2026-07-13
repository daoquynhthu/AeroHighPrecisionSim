#include "aero/cfd/reconstruction.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/partition.hpp"
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

namespace {

__global__ void init_float_one_kernel(Real* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 1.0f;
}

__device__ bool d_conservative_to_primitive(const Real* q, int cell, int nvar, Real gamma,
    Real& rho, Real& u, Real& v, Real& w, Real& p) {
    rho = q[cell * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) return false;
    Real inv_rho = 1.0f / rho;
    u = q[cell * nvar + 1] * inv_rho;
    v = q[cell * nvar + 2] * inv_rho;
    w = q[cell * nvar + 3] * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    p = (gamma - 1.0f) * (q[cell * nvar + 4] - rho * kinetic);
    return real_isfinite(u) && real_isfinite(v) && real_isfinite(w) && real_isfinite(p) && p > 0.0f;
}

__device__ PrimitiveGradient d_apply_limiter(PrimitiveGradient gradient, PrimitiveLimiter limiter) {
    gradient.drho_dx *= limiter.rho;
    gradient.drho_dy *= limiter.rho;
    gradient.drho_dz *= limiter.rho;
    gradient.du_dx *= limiter.u;
    gradient.du_dy *= limiter.u;
    gradient.du_dz *= limiter.u;
    gradient.dv_dx *= limiter.v;
    gradient.dv_dy *= limiter.v;
    gradient.dv_dz *= limiter.v;
    gradient.dw_dx *= limiter.w;
    gradient.dw_dy *= limiter.w;
    gradient.dw_dz *= limiter.w;
    gradient.dp_dx *= limiter.p;
    gradient.dp_dy *= limiter.p;
    gradient.dp_dz *= limiter.p;
    return gradient;
}

__global__ void __launch_bounds__(256) apply_limiter_kernel(PrimitiveGradient* gradients, const PrimitiveLimiter* limiters, int cell_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cell_count) return;
    gradients[idx] = d_apply_limiter(gradients[idx], limiters[idx]);
}



__global__ void __launch_bounds__(128) gg_gradient_kernel_atomic(
    const Real* d_q,
    int nvar, int n_cells, int n_faces,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    const Real* d_nx, const Real* d_ny, const Real* d_nz, const Real* d_area,
    const Real* d_volume,
    Real gamma,
    Real* d_gradients,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_faces) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    int bnd = d_boundary[idx];

    Real rhoL, uL, vL, wL, pL;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    Real nx = d_nx[idx], ny = d_ny[idx], nz = d_nz[idx];
    Real area = d_area[idx];

    Real nu_tildeL = d_q[left * nvar + 5] * (1.0f / rhoL);
    Real rhoF, uF, vF, wF, pF, nu_tildeF;
    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) return;
        Real rhoR, uR, vR, wR, pR;
        if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }
        Real nu_tildeR = d_q[right * nvar + 5] * (1.0f / rhoR);
        rhoF = 0.5f * (rhoL + rhoR);
        uF = 0.5f * (uL + uR);
        vF = 0.5f * (vL + vR);
        wF = 0.5f * (wL + wR);
        pF = 0.5f * (pL + pR);
        nu_tildeF = 0.5f * (nu_tildeL + nu_tildeR);

        if (!real_isfinite(d_volume[right]) || d_volume[right] <= 0.0f) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }
        Real right_scale = -area / d_volume[right];
        Real* gR = d_gradients + right * DeviceMesh::NGRAD;
        Real drho_r = rhoF - rhoR;
        Real du_r = uF - uR;
        Real dv_r = vF - vR;
        Real dw_r = wF - wR;
        Real dp_r = pF - pR;
        Real dnu_r = nu_tildeF - nu_tildeR;
        real_atomic_add(&gR[0], drho_r * nx * right_scale);
        real_atomic_add(&gR[1], drho_r * ny * right_scale);
        real_atomic_add(&gR[2], drho_r * nz * right_scale);
        real_atomic_add(&gR[3], du_r * nx * right_scale);
        real_atomic_add(&gR[4], du_r * ny * right_scale);
        real_atomic_add(&gR[5], du_r * nz * right_scale);
        real_atomic_add(&gR[6], dv_r * nx * right_scale);
        real_atomic_add(&gR[7], dv_r * ny * right_scale);
        real_atomic_add(&gR[8], dv_r * nz * right_scale);
        real_atomic_add(&gR[9], dw_r * nx * right_scale);
        real_atomic_add(&gR[10], dw_r * ny * right_scale);
        real_atomic_add(&gR[11], dw_r * nz * right_scale);
        real_atomic_add(&gR[12], dp_r * nx * right_scale);
        real_atomic_add(&gR[13], dp_r * ny * right_scale);
        real_atomic_add(&gR[14], dp_r * nz * right_scale);
        real_atomic_add(&gR[15], dnu_r * nx * right_scale);
        real_atomic_add(&gR[16], dnu_r * ny * right_scale);
        real_atomic_add(&gR[17], dnu_r * nz * right_scale);
    } else {
        rhoF = rhoL; uF = uL; vF = vL; wF = wL; pF = pL;
        nu_tildeF = nu_tildeL;
    }

    if (!real_isfinite(d_volume[left]) || d_volume[left] <= 0.0f) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    Real left_scale = area / d_volume[left];
    Real drho = rhoF - rhoL;
    Real du = uF - uL;
    Real dv = vF - vL;
    Real dw = wF - wL;
    Real dp = pF - pL;
    Real dnu = nu_tildeF - nu_tildeL;

    Real* gL = d_gradients + left * DeviceMesh::NGRAD;
    real_atomic_add(&gL[0], drho * nx * left_scale);
    real_atomic_add(&gL[1], drho * ny * left_scale);
    real_atomic_add(&gL[2], drho * nz * left_scale);
    real_atomic_add(&gL[3], du * nx * left_scale);
    real_atomic_add(&gL[4], du * ny * left_scale);
    real_atomic_add(&gL[5], du * nz * left_scale);
    real_atomic_add(&gL[6], dv * nx * left_scale);
    real_atomic_add(&gL[7], dv * ny * left_scale);
    real_atomic_add(&gL[8], dv * nz * left_scale);
    real_atomic_add(&gL[9], dw * nx * left_scale);
    real_atomic_add(&gL[10], dw * ny * left_scale);
    real_atomic_add(&gL[11], dw * nz * left_scale);
    real_atomic_add(&gL[12], dp * nx * left_scale);
    real_atomic_add(&gL[13], dp * ny * left_scale);
    real_atomic_add(&gL[14], dp * nz * left_scale);
    real_atomic_add(&gL[15], dnu * nx * left_scale);
    real_atomic_add(&gL[16], dnu * ny * left_scale);
    real_atomic_add(&gL[17], dnu * nz * left_scale);
}

__global__ void __launch_bounds__(128) gg_gradient_kernel_colored(
    const Real* d_q,
    int nvar, int n_cells,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    const Real* d_nx, const Real* d_ny, const Real* d_nz, const Real* d_area,
    const Real* d_volume,
    Real gamma,
    int face_start, int face_end,
    Real* d_gradients,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + face_start;
    if (idx >= face_end) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    int bnd = d_boundary[idx];

    Real rhoL, uL, vL, wL, pL;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    Real nx = d_nx[idx], ny = d_ny[idx], nz = d_nz[idx];
    Real area = d_area[idx];

    Real nu_tildeL = d_q[left * nvar + 5] * (1.0f / rhoL);
    Real rhoF, uF, vF, wF, pF, nu_tildeF;
    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) return;
        Real rhoR, uR, vR, wR, pR;
        if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }
        Real nu_tildeR = d_q[right * nvar + 5] * (1.0f / rhoR);
        rhoF = 0.5f * (rhoL + rhoR);
        uF = 0.5f * (uL + uR);
        vF = 0.5f * (vL + vR);
        wF = 0.5f * (wL + wR);
        pF = 0.5f * (pL + pR);
        nu_tildeF = 0.5f * (nu_tildeL + nu_tildeR);

        if (!real_isfinite(d_volume[right]) || d_volume[right] <= 0.0f) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }
        Real right_scale = -area / d_volume[right];
        Real* gR = d_gradients + right * DeviceMesh::NGRAD;
        Real drho_r = rhoF - rhoR;
        Real du_r = uF - uR;
        Real dv_r = vF - vR;
        Real dw_r = wF - wR;
        Real dp_r = pF - pR;
        Real dnu_r = nu_tildeF - nu_tildeR;
        gR[0] += drho_r * nx * right_scale;
        gR[1] += drho_r * ny * right_scale;
        gR[2] += drho_r * nz * right_scale;
        gR[3] += du_r * nx * right_scale;
        gR[4] += du_r * ny * right_scale;
        gR[5] += du_r * nz * right_scale;
        gR[6] += dv_r * nx * right_scale;
        gR[7] += dv_r * ny * right_scale;
        gR[8] += dv_r * nz * right_scale;
        gR[9] += dw_r * nx * right_scale;
        gR[10] += dw_r * ny * right_scale;
        gR[11] += dw_r * nz * right_scale;
        gR[12] += dp_r * nx * right_scale;
        gR[13] += dp_r * ny * right_scale;
        gR[14] += dp_r * nz * right_scale;
        gR[15] += dnu_r * nx * right_scale;
        gR[16] += dnu_r * ny * right_scale;
        gR[17] += dnu_r * nz * right_scale;
    } else {
        rhoF = rhoL; uF = uL; vF = vL; wF = wL; pF = pL;
        nu_tildeF = nu_tildeL;
    }

    if (!real_isfinite(d_volume[left]) || d_volume[left] <= 0.0f) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    Real left_scale = area / d_volume[left];
    Real drho = rhoF - rhoL;
    Real du = uF - uL;
    Real dv = vF - vL;
    Real dw = wF - wL;
    Real dp = pF - pL;
    Real dnu = nu_tildeF - nu_tildeL;

    Real* gL = d_gradients + left * DeviceMesh::NGRAD;
    gL[0] += drho * nx * left_scale;
    gL[1] += drho * ny * left_scale;
    gL[2] += drho * nz * left_scale;
    gL[3] += du * nx * left_scale;
    gL[4] += du * ny * left_scale;
    gL[5] += du * nz * left_scale;
    gL[6] += dv * nx * left_scale;
    gL[7] += dv * ny * left_scale;
    gL[8] += dv * nz * left_scale;
    gL[9] += dw * nx * left_scale;
    gL[10] += dw * ny * left_scale;
    gL[11] += dw * nz * left_scale;
    gL[12] += dp * nx * left_scale;
    gL[13] += dp * ny * left_scale;
    gL[14] += dp * nz * left_scale;
    gL[15] += dnu * nx * left_scale;
    gL[16] += dnu * ny * left_scale;
    gL[17] += dnu * nz * left_scale;
}

constexpr int kMINMAX_STRIDE = 12;

__global__ void init_minmax_kernel(
    const Real* d_q, int nvar, int n_cells, Real gamma,
    Real* d_minmax,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    Real rho, u, v, w, p;
    if (!d_conservative_to_primitive(d_q, idx, nvar, gamma, rho, u, v, w, p)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        Real* m = d_minmax + idx * kMINMAX_STRIDE;
        m[0] = 1e10f;  m[1] = -1e10f;
        m[2] = 1e10f;  m[3] = -1e10f;
        m[4] = 1e10f;  m[5] = -1e10f;
        m[6] = 1e10f;  m[7] = -1e10f;
        m[8] = 1e10f;  m[9] = -1e10f;
        m[10] = 1e10f; m[11] = -1e10f;
        return;
    }
    Real* m = d_minmax + idx * kMINMAX_STRIDE;
    m[0] = rho; m[1] = rho;
    m[2] = u;   m[3] = u;
    m[4] = v;   m[5] = v;
    m[6] = w;   m[7] = w;
    m[8] = p;   m[9] = p;
    Real nu_tilde = d_q[idx * nvar + 5] * (1.0f / rho);
    m[10] = nu_tilde; m[11] = nu_tilde;
}

__global__ void __launch_bounds__(128) update_minmax_kernel(
    const Real* d_q, int nvar, int n_cells, int n_faces,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    Real gamma,
    Real* d_minmax,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_faces) return;
    int bnd = d_boundary[idx];
    if (bnd != static_cast<int>(BoundaryKind::Interior)) return;
    int left = d_left_cell[idx];
    int right = d_right_cell[idx];
    if (left < 0 || left >= n_cells || right < 0 || right >= n_cells) return;

    Real rhoL, uL, vL, wL, pL;
    Real rhoR, uR, vR, wR, pR;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    auto update = [](Real* min_addr, Real* max_addr, Real val) {
        real_atomic_min(min_addr, val);
        real_atomic_max(max_addr, val);
    };

    Real nu_tildeL = d_q[left * nvar + 5] * (1.0f / rhoL);
    Real nu_tildeR = d_q[right * nvar + 5] * (1.0f / rhoR);

    Real* mL = d_minmax + left * kMINMAX_STRIDE;
    update(&mL[0], &mL[1], rhoR);
    update(&mL[2], &mL[3], uR);
    update(&mL[4], &mL[5], vR);
    update(&mL[6], &mL[7], wR);
    update(&mL[8], &mL[9], pR);
    update(&mL[10], &mL[11], nu_tildeR);

    Real* mR = d_minmax + right * kMINMAX_STRIDE;
    update(&mR[0], &mR[1], rhoL);
    update(&mR[2], &mR[3], uL);
    update(&mR[4], &mR[5], vL);
    update(&mR[6], &mR[7], wL);
    update(&mR[8], &mR[9], pL);
    update(&mR[10], &mR[11], nu_tildeL);
}

__global__ void __launch_bounds__(128) update_minmax_kernel_colored(
    const Real* d_q, int nvar, int n_cells,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    Real gamma,
    int face_start, int face_end,
    Real* d_minmax,
    int* d_failed,
    const int* d_partition_owner, int my_rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + face_start;
    if (idx >= face_end) return;
    int bnd = d_boundary[idx];
    if (bnd != static_cast<int>(BoundaryKind::Interior)) return;
    int left = d_left_cell[idx];
    int right = d_right_cell[idx];
    if (left < 0 || left >= n_cells || right < 0 || right >= n_cells) return;
    if (d_partition_owner && d_partition_owner[left] != my_rank) return;

    Real rhoL, uL, vL, wL, pL;
    Real rhoR, uR, vR, wR, pR;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    Real nu_tildeL = d_q[left * nvar + 5] * (1.0f / rhoL);
    Real nu_tildeR = d_q[right * nvar + 5] * (1.0f / rhoR);

    Real* mL = d_minmax + left * kMINMAX_STRIDE;
    if (rhoR < mL[0]) mL[0] = rhoR; if (rhoR > mL[1]) mL[1] = rhoR;
    if (uR < mL[2]) mL[2] = uR; if (uR > mL[3]) mL[3] = uR;
    if (vR < mL[4]) mL[4] = vR; if (vR > mL[5]) mL[5] = vR;
    if (wR < mL[6]) mL[6] = wR; if (wR > mL[7]) mL[7] = wR;
    if (pR < mL[8]) mL[8] = pR; if (pR > mL[9]) mL[9] = pR;
    if (nu_tildeR < mL[10]) mL[10] = nu_tildeR; if (nu_tildeR > mL[11]) mL[11] = nu_tildeR;

    Real* mR = d_minmax + right * kMINMAX_STRIDE;
    if (rhoL < mR[0]) mR[0] = rhoL; if (rhoL > mR[1]) mR[1] = rhoL;
    if (uL < mR[2]) mR[2] = uL; if (uL > mR[3]) mR[3] = uL;
    if (vL < mR[4]) mR[4] = vL; if (vL > mR[5]) mR[5] = vL;
    if (wL < mR[6]) mR[6] = wL; if (wL > mR[7]) mR[7] = wL;
    if (pL < mR[8]) mR[8] = pL; if (pL > mR[9]) mR[9] = pL;
    if (nu_tildeL < mR[10]) mR[10] = nu_tildeL; if (nu_tildeL > mR[11]) mR[11] = nu_tildeL;
}

__device__ Real limiter_theta_device(Real center, Real reconstructed, Real min_val, Real max_val) {
    if (reconstructed > max_val) {
        Real denom = reconstructed - center;
        if (denom <= 0.0f) return 0.0f;
        return real_fmax(0.0f, real_fmin(1.0f, (max_val - center) / denom));
    }
    if (reconstructed < min_val) {
        Real denom = reconstructed - center;
        if (denom >= 0.0f) return 0.0f;
        return real_fmax(0.0f, real_fmin(1.0f, (min_val - center) / denom));
    }
    return 1.0f;
}

__global__ void __launch_bounds__(128) bj_limiter_kernel(
    const Real* d_q, int nvar, int n_cells, int n_faces,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    const Real* d_face_cx, const Real* d_face_cy, const Real* d_face_cz,
    const Real* d_cx, const Real* d_cy, const Real* d_cz,
    Real gamma,
    const Real* d_gradients,
    const Real* d_minmax,
    Real* d_limiters,
    int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_faces) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    int bnd = d_boundary[idx];

    Real rhoL, uL, vL, wL, pL;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    Real dxL = d_face_cx[idx] - d_cx[left];
    Real dyL = d_face_cy[idx] - d_cy[left];
    Real dzL = d_face_cz[idx] - d_cz[left];

    Real nu_tildeL = d_q[left * nvar + 5] * (1.0f / rhoL);

    const Real* gL = d_gradients + left * DeviceMesh::NGRAD;
    Real rec_rho = rhoL + gL[0]*dxL + gL[1]*dyL + gL[2]*dzL;
    Real rec_u = uL + gL[3]*dxL + gL[4]*dyL + gL[5]*dzL;
    Real rec_v = vL + gL[6]*dxL + gL[7]*dyL + gL[8]*dzL;
    Real rec_w = wL + gL[9]*dxL + gL[10]*dyL + gL[11]*dzL;
    Real rec_p = pL + gL[12]*dxL + gL[13]*dyL + gL[14]*dzL;
    Real rec_nu_tilde = nu_tildeL + gL[15]*dxL + gL[16]*dyL + gL[17]*dzL;

    const Real* mL = d_minmax + left * kMINMAX_STRIDE;
    Real t_rho = limiter_theta_device(rhoL, rec_rho, mL[0], mL[1]);
    Real t_u = limiter_theta_device(uL, rec_u, mL[2], mL[3]);
    Real t_v = limiter_theta_device(vL, rec_v, mL[4], mL[5]);
    Real t_w = limiter_theta_device(wL, rec_w, mL[6], mL[7]);
    Real t_p = limiter_theta_device(pL, rec_p, mL[8], mL[9]);
    Real t_nu = limiter_theta_device(nu_tildeL, rec_nu_tilde, mL[10], mL[11]);

    Real* limL = d_limiters + left * 6;
    real_atomic_min(&limL[0], t_rho);
    real_atomic_min(&limL[1], t_u);
    real_atomic_min(&limL[2], t_v);
    real_atomic_min(&limL[3], t_w);
    real_atomic_min(&limL[4], t_p);
    real_atomic_min(&limL[5], t_nu);

    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) return;
        Real rhoR, uR, vR, wR, pR;
        if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }

        Real dxR = d_face_cx[idx] - d_cx[right];
        Real dyR = d_face_cy[idx] - d_cy[right];
        Real dzR = d_face_cz[idx] - d_cz[right];

        Real nu_tildeR = d_q[right * nvar + 5] * (1.0f / rhoR);

        const Real* gR = d_gradients + right * DeviceMesh::NGRAD;
        rec_rho = rhoR + gR[0]*dxR + gR[1]*dyR + gR[2]*dzR;
        rec_u = uR + gR[3]*dxR + gR[4]*dyR + gR[5]*dzR;
        rec_v = vR + gR[6]*dxR + gR[7]*dyR + gR[8]*dzR;
        rec_w = wR + gR[9]*dxR + gR[10]*dyR + gR[11]*dzR;
        rec_p = pR + gR[12]*dxR + gR[13]*dyR + gR[14]*dzR;
        rec_nu_tilde = nu_tildeR + gR[15]*dxR + gR[16]*dyR + gR[17]*dzR;

        const Real* mR = d_minmax + right * kMINMAX_STRIDE;
        t_rho = limiter_theta_device(rhoR, rec_rho, mR[0], mR[1]);
        t_u = limiter_theta_device(uR, rec_u, mR[2], mR[3]);
        t_v = limiter_theta_device(vR, rec_v, mR[4], mR[5]);
        t_w = limiter_theta_device(wR, rec_w, mR[6], mR[7]);
        t_p = limiter_theta_device(pR, rec_p, mR[8], mR[9]);
        t_nu = limiter_theta_device(nu_tildeR, rec_nu_tilde, mR[10], mR[11]);

        Real* limR = d_limiters + right * 6;
        real_atomic_min(&limR[0], t_rho);
        real_atomic_min(&limR[1], t_u);
        real_atomic_min(&limR[2], t_v);
        real_atomic_min(&limR[3], t_w);
        real_atomic_min(&limR[4], t_p);
        real_atomic_min(&limR[5], t_nu);
    }
}

__global__ void __launch_bounds__(128) bj_limiter_kernel_colored(
    const Real* d_q, int nvar, int n_cells,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    const Real* d_face_cx, const Real* d_face_cy, const Real* d_face_cz,
    const Real* d_cx, const Real* d_cy, const Real* d_cz,
    Real gamma,
    const Real* d_gradients,
    const Real* d_minmax,
    int face_start, int face_end,
    Real* d_limiters,
    int* d_failed,
    const int* d_partition_owner, int my_rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + face_start;
    if (idx >= face_end) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    if (d_partition_owner && d_partition_owner[left] != my_rank) return;
    int bnd = d_boundary[idx];

    Real rhoL, uL, vL, wL, pL;
    if (!d_conservative_to_primitive(d_q, left, nvar, gamma, rhoL, uL, vL, wL, pL)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    Real dxL = d_face_cx[idx] - d_cx[left];
    Real dyL = d_face_cy[idx] - d_cy[left];
    Real dzL = d_face_cz[idx] - d_cz[left];

    Real nu_tildeL = d_q[left * nvar + 5] * (1.0f / rhoL);

    const Real* gL = d_gradients + left * DeviceMesh::NGRAD;
    Real rec_rho = rhoL + gL[0]*dxL + gL[1]*dyL + gL[2]*dzL;
    Real rec_u = uL + gL[3]*dxL + gL[4]*dyL + gL[5]*dzL;
    Real rec_v = vL + gL[6]*dxL + gL[7]*dyL + gL[8]*dzL;
    Real rec_w = wL + gL[9]*dxL + gL[10]*dyL + gL[11]*dzL;
    Real rec_p = pL + gL[12]*dxL + gL[13]*dyL + gL[14]*dzL;
    Real rec_nu_tilde = nu_tildeL + gL[15]*dxL + gL[16]*dyL + gL[17]*dzL;

    const Real* mL = d_minmax + left * kMINMAX_STRIDE;
    Real t_rho = limiter_theta_device(rhoL, rec_rho, mL[0], mL[1]);
    Real t_u = limiter_theta_device(uL, rec_u, mL[2], mL[3]);
    Real t_v = limiter_theta_device(vL, rec_v, mL[4], mL[5]);
    Real t_w = limiter_theta_device(wL, rec_w, mL[6], mL[7]);
    Real t_p = limiter_theta_device(pL, rec_p, mL[8], mL[9]);
    Real t_nu = limiter_theta_device(nu_tildeL, rec_nu_tilde, mL[10], mL[11]);

    Real* limL = d_limiters + left * 6;
    if (t_rho < limL[0]) limL[0] = t_rho;
    if (t_u   < limL[1]) limL[1] = t_u;
    if (t_v   < limL[2]) limL[2] = t_v;
    if (t_w   < limL[3]) limL[3] = t_w;
    if (t_p   < limL[4]) limL[4] = t_p;
    if (t_nu  < limL[5]) limL[5] = t_nu;

    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) return;
        Real rhoR, uR, vR, wR, pR;
        if (!d_conservative_to_primitive(d_q, right, nvar, gamma, rhoR, uR, vR, wR, pR)) {
            if (d_failed) atomicCAS(d_failed, 0, 1);
            return;
        }

        Real dxR = d_face_cx[idx] - d_cx[right];
        Real dyR = d_face_cy[idx] - d_cy[right];
        Real dzR = d_face_cz[idx] - d_cz[right];

        Real nu_tildeR = d_q[right * nvar + 5] * (1.0f / rhoR);

        const Real* gR = d_gradients + right * DeviceMesh::NGRAD;
        rec_rho = rhoR + gR[0]*dxR + gR[1]*dyR + gR[2]*dzR;
        rec_u = uR + gR[3]*dxR + gR[4]*dyR + gR[5]*dzR;
        rec_v = vR + gR[6]*dxR + gR[7]*dyR + gR[8]*dzR;
        rec_w = wR + gR[9]*dxR + gR[10]*dyR + gR[11]*dzR;
        rec_p = pR + gR[12]*dxR + gR[13]*dyR + gR[14]*dzR;
        rec_nu_tilde = nu_tildeR + gR[15]*dxR + gR[16]*dyR + gR[17]*dzR;

        const Real* mR = d_minmax + right * kMINMAX_STRIDE;
        t_rho = limiter_theta_device(rhoR, rec_rho, mR[0], mR[1]);
        t_u = limiter_theta_device(uR, rec_u, mR[2], mR[3]);
        t_v = limiter_theta_device(vR, rec_v, mR[4], mR[5]);
        t_w = limiter_theta_device(wR, rec_w, mR[6], mR[7]);
        t_p = limiter_theta_device(pR, rec_p, mR[8], mR[9]);
        t_nu = limiter_theta_device(nu_tildeR, rec_nu_tilde, mR[10], mR[11]);

        Real* limR = d_limiters + right * 6;
        if (t_rho < limR[0]) limR[0] = t_rho;
        if (t_u   < limR[1]) limR[1] = t_u;
        if (t_v   < limR[2]) limR[2] = t_v;
        if (t_w   < limR[3]) limR[3] = t_w;
        if (t_p   < limR[4]) limR[4] = t_p;
        if (t_nu  < limR[5]) limR[5] = t_nu;
    }
}

} // namespace

bool compute_gradients_gpu(DeviceMesh& mesh, Real gamma, std::string* error, int* d_failed,
    cudaStream_t stream) {
    if (mesh.cell_count() == 0 || mesh.face_count() == 0) return true;
    if (!mesh.gradients_device()) {
        if (error) *error = "gradients buffer not allocated";
        return false;
    }

    std::size_t grad_bytes = DeviceMesh::NGRAD * mesh.cell_count() * sizeof(Real);
    if (!cuda_check(cudaMemsetAsync(mesh.gradients_device(), 0, grad_bytes, stream), "cudaMemset gradients", error)) return false;
    if (d_failed && !cuda_check(cudaMemsetAsync(d_failed, 0, sizeof(int), stream), "cudaMemset d_failed", error)) return false;

    int block = 128;
    int nf = static_cast<int>(mesh.face_count());
    int nc = static_cast<int>(mesh.cell_count());
    int n_colors = mesh.color_count();
    DeviceFaceData fd = mesh.face_data();
    DeviceCellData cd = mesh.cell_data();

    if (n_colors > 0) {
        for (int c = 0; c < n_colors; ++c) {
            int start = mesh.host_color_offsets()[c];
            int end   = mesh.host_color_offsets()[c + 1];
            int nf_c  = end - start;
            int grid_c = (nf_c + block - 1) / block;
            gg_gradient_kernel_colored<<<grid_c, block, 0, stream>>>(
                mesh.state_device(),
                DeviceMesh::NVAR, nc,
                fd.left_cell, fd.right_cell, fd.boundary,
                fd.nx, fd.ny, fd.nz, fd.area,
                cd.volume, gamma,
                start, end,
                mesh.gradients_device(),
                d_failed);
            if (!cuda_check(cudaGetLastError(), "gg_gradient_kernel_colored", error)) return false;
        }
    } else {
        int grid = (nf + block - 1) / block;
        gg_gradient_kernel_atomic<<<grid, block, 0, stream>>>(
            mesh.state_device(),
            DeviceMesh::NVAR, nc, nf,
            fd.left_cell, fd.right_cell, fd.boundary,
            fd.nx, fd.ny, fd.nz, fd.area,
            cd.volume, gamma,
            mesh.gradients_device(),
            d_failed);
        if (!cuda_check(cudaGetLastError(), "gg_gradient_kernel_atomic", error)) return false;
    }
    return true;
}

bool compute_limiters_gpu(DeviceMesh& mesh, Real gamma, std::string* error, int* d_failed,
    cudaStream_t stream) {
    if (mesh.cell_count() == 0 || mesh.face_count() == 0) return true;
    if (!mesh.gradients_device() || !mesh.limiters_device()) {
        if (error) *error = "gradient/limiter buffers not allocated";
        return false;
    }
    if (d_failed && !cuda_check(cudaMemsetAsync(d_failed, 0, sizeof(int), stream), "cudaMemset d_failed", error)) return false;

    int nc = static_cast<int>(mesh.cell_count());
    int nf = static_cast<int>(mesh.face_count());

    int block = 128;
    int cell_grid = (nc + block - 1) / block;
    int face_grid = (nf + block - 1) / block;

    int n_colors = mesh.color_count();
    DeviceFaceData fd = mesh.face_data();
    DeviceCellData cd = mesh.cell_data();
    const GpuPartition* gpu_part = mesh.get_partition();
    const int* d_partition_owner = gpu_part ? gpu_part->d_partition_owner : nullptr;
    int my_rank = gpu_part ? gpu_part->my_rank : 0;

    Real* d_minmax = mesh.minmax_device();

    init_minmax_kernel<<<cell_grid, block, 0, stream>>>(
        mesh.state_device(), DeviceMesh::NVAR, nc, gamma, d_minmax, d_failed);
    if (!cuda_check(cudaGetLastError(), "init_minmax_kernel", error)) return false;

    if (n_colors > 0) {
        for (int c = 0; c < n_colors; ++c) {
            int start = mesh.host_color_offsets()[c];
            int end   = mesh.host_color_offsets()[c + 1];
            int nf_c  = end - start;
            int grid_c = (nf_c + block - 1) / block;
            update_minmax_kernel_colored<<<grid_c, block, 0, stream>>>(
                mesh.state_device(), DeviceMesh::NVAR, nc,
                fd.left_cell, fd.right_cell, fd.boundary,
                gamma,
                start, end,
                d_minmax, d_failed,
                d_partition_owner, my_rank);
            if (!cuda_check(cudaGetLastError(), "update_minmax_kernel_colored", error)) return false;
        }
    } else {
        update_minmax_kernel<<<face_grid, block, 0, stream>>>(
            mesh.state_device(), DeviceMesh::NVAR, nc, nf,
            fd.left_cell, fd.right_cell, fd.boundary,
            gamma, d_minmax, d_failed);
        if (!cuda_check(cudaGetLastError(), "update_minmax_kernel", error)) return false;
    }

    int limiter_grid = (nc * 6 + block - 1) / block;
    init_float_one_kernel<<<limiter_grid, block, 0, stream>>>(mesh.limiters_device(), nc * 6);
    if (!cuda_check(cudaGetLastError(), "init_float_one limiters", error)) return false;

    if (n_colors > 0) {
        for (int c = 0; c < n_colors; ++c) {
            int start = mesh.host_color_offsets()[c];
            int end   = mesh.host_color_offsets()[c + 1];
            int nf_c  = end - start;
            int grid_c = (nf_c + block - 1) / block;
            bj_limiter_kernel_colored<<<grid_c, block, 0, stream>>>(
                mesh.state_device(), DeviceMesh::NVAR, nc,
                fd.left_cell, fd.right_cell, fd.boundary,
                fd.cx, fd.cy, fd.cz,
                cd.cx, cd.cy, cd.cz,
                gamma,
                mesh.gradients_device(), d_minmax,
                start, end,
                mesh.limiters_device(), d_failed,
                d_partition_owner, my_rank);
            if (!cuda_check(cudaGetLastError(), "bj_limiter_kernel_colored", error)) return false;
        }
    } else {
        bj_limiter_kernel<<<face_grid, block, 0, stream>>>(
            mesh.state_device(), DeviceMesh::NVAR, nc, nf,
            fd.left_cell, fd.right_cell, fd.boundary,
            fd.cx, fd.cy, fd.cz,
            cd.cx, cd.cy, cd.cz,
            gamma,
            mesh.gradients_device(), d_minmax,
            mesh.limiters_device(), d_failed);
        if (!cuda_check(cudaGetLastError(), "bj_limiter_kernel", error)) return false;
    }
    return true;
}

bool apply_limiter_gpu(DeviceMesh& mesh, std::string* error, cudaStream_t stream) {
    return apply_limiter_gpu(mesh, true, error, stream);
}

bool apply_limiter_gpu(DeviceMesh& mesh, bool sync, std::string* error, cudaStream_t stream) {
    if (mesh.cell_count() == 0 || !mesh.gradients_device() || !mesh.limiters_device()) {
        if (error) *error = "GPU gradient/limiter buffers are not ready";
        return false;
    }

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    apply_limiter_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<PrimitiveGradient*>(mesh.gradients_device()),
        reinterpret_cast<const PrimitiveLimiter*>(mesh.limiters_device()),
        nc);
    if (!cuda_check(cudaGetLastError(), "apply_limiter_kernel launch", error)) return false;
    if (sync && !stream && !cuda_check(cudaDeviceSynchronize(), "apply_limiter_kernel synchronize", error)) return false;
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp




