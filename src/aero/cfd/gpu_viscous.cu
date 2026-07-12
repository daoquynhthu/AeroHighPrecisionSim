#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/partition.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

namespace {

__device__ void primitive_from_q(const Real* d_q, int idx, int nvar, Real gamma,
    Real& rho, Real& u, Real& v, Real& w, Real& p) {
    rho = d_q[idx * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) { rho = -1.0f; return; }
    Real inv_rho = 1.0f / rho;
    u = d_q[idx * nvar + 1] * inv_rho;
    v = d_q[idx * nvar + 2] * inv_rho;
    w = d_q[idx * nvar + 3] * inv_rho;
    Real E = d_q[idx * nvar + 4];
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    p = (gamma - 1.0f) * (E - rho * kinetic);
}

__device__ Real sutherland_mu(Real T, Real T_ref, Real S, Real mu_ref) {
    if (!real_isfinite(T) || T <= 0.0f) return mu_ref;
    Real t_ratio = T / T_ref;
    return mu_ref * t_ratio * real_sqrt(t_ratio) * (T_ref + S) / (T + S);
}

__global__ void viscous_flux_kernel_atomic(
    const Real* d_q, int nvar, Real gamma,
    const Real* d_nx, const Real* d_ny, const Real* d_nz,
    const Real* d_area,
    const int* d_left_cell, const int* d_right_cell,
    const int* d_boundary,
    const Real* d_cell_cx, const Real* d_cell_cy, const Real* d_cell_cz,
    const Real* d_face_cx, const Real* d_face_cy, const Real* d_face_cz,
    const Real* d_gradients,
    Real* d_residual,
    int face_count, int n_cells,
    Real prandtl, Real mu_ref, Real T_ref, Real sutherland_T,
    Real inv_Re, Real Re, Real wall_T,
    int turbulence,
    int* d_failed,
    const int* d_partition_owner, int my_rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= face_count) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    if (d_partition_owner && d_partition_owner[left] != my_rank) return;
    int right = d_right_cell[idx];

    int bnd = d_boundary[idx];
    if (bnd != static_cast<int>(BoundaryKind::Interior) &&
        bnd != static_cast<int>(BoundaryKind::NoSlipWall)) return;

    Real rho_L, u_L, v_L, w_L, p_L;
    primitive_from_q(d_q, left, nvar, gamma, rho_L, u_L, v_L, w_L, p_L);
    if (!real_isfinite(rho_L) || rho_L <= 0.0f || !real_isfinite(p_L) || p_L <= 0.0f) return;

    Real inv_rho_L = 1.0f / rho_L;
    Real inv_rho_L2 = inv_rho_L * inv_rho_L;
    Real T_L = p_L * inv_rho_L;

    int gL = left * DeviceMesh::NGRAD;
    Real du_dx_L = d_gradients[gL + 3];
    Real du_dy_L = d_gradients[gL + 4];
    Real du_dz_L = d_gradients[gL + 5];
    Real dv_dx_L = d_gradients[gL + 6];
    Real dv_dy_L = d_gradients[gL + 7];
    Real dv_dz_L = d_gradients[gL + 8];
    Real dw_dx_L = d_gradients[gL + 9];
    Real dw_dy_L = d_gradients[gL + 10];
    Real dw_dz_L = d_gradients[gL + 11];
    Real drho_dx_L = d_gradients[gL + 0];
    Real drho_dy_L = d_gradients[gL + 1];
    Real drho_dz_L = d_gradients[gL + 2];
    Real dp_dx_L = d_gradients[gL + 12];
    Real dp_dy_L = d_gradients[gL + 13];
    Real dp_dz_L = d_gradients[gL + 14];

    Real dT_dx_L = (rho_L * dp_dx_L - p_L * drho_dx_L) * inv_rho_L2;
    Real dT_dy_L = (rho_L * dp_dy_L - p_L * drho_dy_L) * inv_rho_L2;
    Real dT_dz_L = (rho_L * dp_dz_L - p_L * drho_dz_L) * inv_rho_L2;

    Real face_u, face_v, face_w, face_T;
    Real grad_du_dx, grad_du_dy, grad_du_dz;
    Real grad_dv_dx, grad_dv_dy, grad_dv_dz;
    Real grad_dw_dx, grad_dw_dy, grad_dw_dz;
    Real grad_dT_dx, grad_dT_dy, grad_dT_dz;

    if (right >= 0 && bnd == static_cast<int>(BoundaryKind::Interior)) {
        Real rho_R, u_R, v_R, w_R, p_R;
        primitive_from_q(d_q, right, nvar, gamma, rho_R, u_R, v_R, w_R, p_R);
        if (!real_isfinite(rho_R) || rho_R <= 0.0f || !real_isfinite(p_R) || p_R <= 0.0f) return;

        Real inv_rho_R = 1.0f / rho_R;
        Real inv_rho_R2 = inv_rho_R * inv_rho_R;
        Real T_R = p_R * inv_rho_R;

        int gR = right * DeviceMesh::NGRAD;
        Real drho_dx_R = d_gradients[gR + 0];
        Real drho_dy_R = d_gradients[gR + 1];
        Real drho_dz_R = d_gradients[gR + 2];
        Real du_dx_R = d_gradients[gR + 3];
        Real du_dy_R = d_gradients[gR + 4];
        Real du_dz_R = d_gradients[gR + 5];
        Real dv_dx_R = d_gradients[gR + 6];
        Real dv_dy_R = d_gradients[gR + 7];
        Real dv_dz_R = d_gradients[gR + 8];
        Real dw_dx_R = d_gradients[gR + 9];
        Real dw_dy_R = d_gradients[gR + 10];
        Real dw_dz_R = d_gradients[gR + 11];
        Real dp_dx_R = d_gradients[gR + 12];
        Real dp_dy_R = d_gradients[gR + 13];
        Real dp_dz_R = d_gradients[gR + 14];

        Real dT_dx_R = (rho_R * dp_dx_R - p_R * drho_dx_R) * inv_rho_R2;
        Real dT_dy_R = (rho_R * dp_dy_R - p_R * drho_dy_R) * inv_rho_R2;
        Real dT_dz_R = (rho_R * dp_dz_R - p_R * drho_dz_R) * inv_rho_R2;

        grad_du_dx = 0.5f * (du_dx_L + du_dx_R);
        grad_du_dy = 0.5f * (du_dy_L + du_dy_R);
        grad_du_dz = 0.5f * (du_dz_L + du_dz_R);
        grad_dv_dx = 0.5f * (dv_dx_L + dv_dx_R);
        grad_dv_dy = 0.5f * (dv_dy_L + dv_dy_R);
        grad_dv_dz = 0.5f * (dv_dz_L + dv_dz_R);
        grad_dw_dx = 0.5f * (dw_dx_L + dw_dx_R);
        grad_dw_dy = 0.5f * (dw_dy_L + dw_dy_R);
        grad_dw_dz = 0.5f * (dw_dz_L + dw_dz_R);
        grad_dT_dx = 0.5f * (dT_dx_L + dT_dx_R);
        grad_dT_dy = 0.5f * (dT_dy_L + dT_dy_R);
        grad_dT_dz = 0.5f * (dT_dz_L + dT_dz_R);

        Real dr_x = d_cell_cx[right] - d_cell_cx[left];
        Real dr_y = d_cell_cy[right] - d_cell_cy[left];
        Real dr_z = d_cell_cz[right] - d_cell_cz[left];
        Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
        if (d2 > 1e-30f) {
            Real inv_d2 = 1.0f / d2;
            Real proj_du = grad_du_dx*dr_x + grad_du_dy*dr_y + grad_du_dz*dr_z;
            Real proj_dv = grad_dv_dx*dr_x + grad_dv_dy*dr_y + grad_dv_dz*dr_z;
            Real proj_dw = grad_dw_dx*dr_x + grad_dw_dy*dr_y + grad_dw_dz*dr_z;
            Real proj_dT = grad_dT_dx*dr_x + grad_dT_dy*dr_y + grad_dT_dz*dr_z;
            Real du_corr = ((u_R - u_L) - proj_du) * inv_d2;
            Real dv_corr = ((v_R - v_L) - proj_dv) * inv_d2;
            Real dw_corr = ((w_R - w_L) - proj_dw) * inv_d2;
            Real dT_corr = ((T_R - T_L) - proj_dT) * inv_d2;
            grad_du_dx += du_corr * dr_x; grad_du_dy += du_corr * dr_y; grad_du_dz += du_corr * dr_z;
            grad_dv_dx += dv_corr * dr_x; grad_dv_dy += dv_corr * dr_y; grad_dv_dz += dv_corr * dr_z;
            grad_dw_dx += dw_corr * dr_x; grad_dw_dy += dw_corr * dr_y; grad_dw_dz += dw_corr * dr_z;
            grad_dT_dx += dT_corr * dr_x; grad_dT_dy += dT_corr * dr_y; grad_dT_dz += dT_corr * dr_z;
        }

        face_u = 0.5f * (u_L + u_R);
        face_v = 0.5f * (v_L + v_R);
        face_w = 0.5f * (w_L + w_R);
        face_T = 0.5f * (T_L + T_R);
    } else {
        grad_du_dx = du_dx_L; grad_du_dy = du_dy_L; grad_du_dz = du_dz_L;
        grad_dv_dx = dv_dx_L; grad_dv_dy = dv_dy_L; grad_dv_dz = dv_dz_L;
        grad_dw_dx = dw_dx_L; grad_dw_dy = dw_dy_L; grad_dw_dz = dw_dz_L;
        grad_dT_dx = dT_dx_L; grad_dT_dy = dT_dy_L; grad_dT_dz = dT_dz_L;

        Real dr_x = d_face_cx[idx] - d_cell_cx[left];
        Real dr_y = d_face_cy[idx] - d_cell_cy[left];
        Real dr_z = d_face_cz[idx] - d_cell_cz[left];
        Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
        if (d2 > 1e-30f) {
            Real inv_d2 = 1.0f / d2;
            Real proj_du = grad_du_dx*dr_x + grad_du_dy*dr_y + grad_du_dz*dr_z;
            Real proj_dv = grad_dv_dx*dr_x + grad_dv_dy*dr_y + grad_dv_dz*dr_z;
            Real proj_dw = grad_dw_dx*dr_x + grad_dw_dy*dr_y + grad_dw_dz*dr_z;
            Real proj_dT = grad_dT_dx*dr_x + grad_dT_dy*dr_y + grad_dT_dz*dr_z;
            Real du_corr = ((0.0f - u_L) - proj_du) * inv_d2;
            Real dv_corr = ((0.0f - v_L) - proj_dv) * inv_d2;
            Real dw_corr = ((0.0f - w_L) - proj_dw) * inv_d2;
            Real dT_corr = ((wall_T - T_L) - proj_dT) * inv_d2;
            grad_du_dx += du_corr * dr_x; grad_du_dy += du_corr * dr_y; grad_du_dz += du_corr * dr_z;
            grad_dv_dx += dv_corr * dr_x; grad_dv_dy += dv_corr * dr_y; grad_dv_dz += dv_corr * dr_z;
            grad_dw_dx += dw_corr * dr_x; grad_dw_dy += dw_corr * dr_y; grad_dw_dz += dw_corr * dr_z;
            grad_dT_dx += dT_corr * dr_x; grad_dT_dy += dT_corr * dr_y; grad_dT_dz += dT_corr * dr_z;
        }

        face_u = 0.0f;
        face_v = 0.0f;
        face_w = 0.0f;
        face_T = wall_T;
    }

    if (face_T <= 0.0f) return;
    Real mu_face = sutherland_mu(face_T, T_ref, sutherland_T, mu_ref);
    if (mu_face <= 0.0f) return;

    Real nx = d_nx[idx];
    Real ny = d_ny[idx];
    Real nz = d_nz[idx];
    Real area = d_area[idx];

    Real div_u = grad_du_dx + grad_dv_dy + grad_dw_dz;
    Real tau_xx = 2.0f * (grad_du_dx - div_u / 3.0f);
    Real tau_yy = 2.0f * (grad_dv_dy - div_u / 3.0f);
    Real tau_zz = 2.0f * (grad_dw_dz - div_u / 3.0f);
    Real tau_xy = (grad_du_dy + grad_dv_dx);
    Real tau_xz = (grad_du_dz + grad_dw_dx);
    Real tau_yz = (grad_dv_dz + grad_dw_dy);

    Real mu_invRe = mu_face * inv_Re;
    Real visc_mom_x = (tau_xx*nx + tau_xy*ny + tau_xz*nz) * mu_invRe * area;
    Real visc_mom_y = (tau_xy*nx + tau_yy*ny + tau_yz*nz) * mu_invRe * area;
    Real visc_mom_z = (tau_xz*nx + tau_yz*ny + tau_zz*nz) * mu_invRe * area;

    Real dT_dn = grad_dT_dx*nx + grad_dT_dy*ny + grad_dT_dz*nz;
    Real kappa_over_Re = mu_face * gamma / ((gamma - 1.0f) * prandtl) * inv_Re;
    Real visc_energy = (face_u * visc_mom_x + face_v * visc_mom_y + face_w * visc_mom_z
        + kappa_over_Re * dT_dn * area);

    real_atomic_add(&d_residual[left * nvar + 1], visc_mom_x);
    real_atomic_add(&d_residual[left * nvar + 2], visc_mom_y);
    real_atomic_add(&d_residual[left * nvar + 3], visc_mom_z);
    real_atomic_add(&d_residual[left * nvar + 4], visc_energy);

    if (right >= 0 && bnd == static_cast<int>(BoundaryKind::Interior)) {
        real_atomic_add(&d_residual[right * nvar + 1], -visc_mom_x);
        real_atomic_add(&d_residual[right * nvar + 2], -visc_mom_y);
        real_atomic_add(&d_residual[right * nvar + 3], -visc_mom_z);
        real_atomic_add(&d_residual[right * nvar + 4], -visc_energy);
    }

    if (!turbulence) return;

    // SA conservative diffusion: (1/sigma) * div((mu/Re + rho*nu_tilde*fv1) * grad(nu_tilde))
    constexpr Real sigma_sa = 2.0f / 3.0f;
    constexpr Real cv1 = 7.1f;
    constexpr Real cv13 = cv1 * cv1 * cv1;

    Real nu_tilde_L = d_q[left * nvar + 5] / rho_L;
    if (!real_isfinite(nu_tilde_L)) return;
    Real dnu_dx_L = d_gradients[gL + 15];
    Real dnu_dy_L = d_gradients[gL + 16];
    Real dnu_dz_L = d_gradients[gL + 17];

    Real grad_dnu_dx, grad_dnu_dy, grad_dnu_dz;
    Real face_nu_tilde, face_rho;

    if (right >= 0 && bnd == static_cast<int>(BoundaryKind::Interior)) {
        Real rho_R, u_R, v_R, w_R, p_R;
        primitive_from_q(d_q, right, nvar, gamma, rho_R, u_R, v_R, w_R, p_R);
        if (!real_isfinite(rho_R) || rho_R <= 0.0f || !real_isfinite(p_R) || p_R <= 0.0f) return;
        Real nu_tilde_R = d_q[right * nvar + 5] / rho_R;
        if (!real_isfinite(nu_tilde_R)) return;

        int gR = right * DeviceMesh::NGRAD;
        Real dnu_dx_R = d_gradients[gR + 15];
        Real dnu_dy_R = d_gradients[gR + 16];
        Real dnu_dz_R = d_gradients[gR + 17];

        grad_dnu_dx = 0.5f * (dnu_dx_L + dnu_dx_R);
        grad_dnu_dy = 0.5f * (dnu_dy_L + dnu_dy_R);
        grad_dnu_dz = 0.5f * (dnu_dz_L + dnu_dz_R);

        Real dr_x = d_cell_cx[right] - d_cell_cx[left];
        Real dr_y = d_cell_cy[right] - d_cell_cy[left];
        Real dr_z = d_cell_cz[right] - d_cell_cz[left];
        Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
        if (d2 > 1e-30f) {
            Real inv_d2 = 1.0f / d2;
            Real proj_dnu = grad_dnu_dx*dr_x + grad_dnu_dy*dr_y + grad_dnu_dz*dr_z;
            Real dnu_corr = ((nu_tilde_R - nu_tilde_L) - proj_dnu) * inv_d2;
            grad_dnu_dx += dnu_corr * dr_x;
            grad_dnu_dy += dnu_corr * dr_y;
            grad_dnu_dz += dnu_corr * dr_z;
        }

        face_nu_tilde = 0.5f * (nu_tilde_L + nu_tilde_R);
        face_rho = 0.5f * (rho_L + rho_R);
    } else {
        grad_dnu_dx = dnu_dx_L;
        grad_dnu_dy = dnu_dy_L;
        grad_dnu_dz = dnu_dz_L;

        Real dr_x = d_face_cx[idx] - d_cell_cx[left];
        Real dr_y = d_face_cy[idx] - d_cell_cy[left];
        Real dr_z = d_face_cz[idx] - d_cell_cz[left];
        Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
        if (d2 > 1e-30f) {
            Real inv_d2 = 1.0f / d2;
            Real proj_dnu = grad_dnu_dx*dr_x + grad_dnu_dy*dr_y + grad_dnu_dz*dr_z;
            Real dnu_corr = ((0.0f - nu_tilde_L) - proj_dnu) * inv_d2;
            grad_dnu_dx += dnu_corr * dr_x;
            grad_dnu_dy += dnu_corr * dr_y;
            grad_dnu_dz += dnu_corr * dr_z;
        }

        face_nu_tilde = 0.5f * nu_tilde_L;
        face_rho = rho_L;
    }

    Real dnu_dn = grad_dnu_dx*nx + grad_dnu_dy*ny + grad_dnu_dz*nz;

    Real chi_face = Re * face_rho * face_nu_tilde / (mu_face + 1e-30f);
    Real chi3 = chi_face * chi_face * chi_face;
    Real fv1_face = chi3 / (chi3 + cv13 + 1e-30f);
    Real mu_tilde = face_rho * face_nu_tilde * fv1_face / sigma_sa;
    Real mu_total = (mu_face * inv_Re) / sigma_sa + mu_tilde;
    Real visc_nu = mu_total * dnu_dn * area;

    real_atomic_add(&d_residual[left * nvar + 5], visc_nu);
    if (right >= 0 && bnd == static_cast<int>(BoundaryKind::Interior)) {
        real_atomic_add(&d_residual[right * nvar + 5], -visc_nu);
    }
}

__global__ void viscous_flux_kernel_colored(
    const Real* d_q, int nvar, Real gamma,
    const Real* d_nx, const Real* d_ny, const Real* d_nz,
    const Real* d_area,
    const int* d_left_cell, const int* d_right_cell,
    const int* d_boundary,
    const Real* d_cell_cx, const Real* d_cell_cy, const Real* d_cell_cz,
    const Real* d_face_cx, const Real* d_face_cy, const Real* d_face_cz,
    const Real* d_gradients,
    Real* d_residual,
    int face_start, int face_end, int n_cells,
    Real prandtl, Real mu_ref, Real T_ref, Real sutherland_T,
    Real inv_Re, Real Re, Real wall_T,
    int turbulence,
    int* d_failed,
    const int* d_partition_owner, int my_rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + face_start;
    if (idx >= face_end) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    if (d_partition_owner && d_partition_owner[left] != my_rank) return;
    int right = d_right_cell[idx];

    int bnd = d_boundary[idx];
    if (bnd != static_cast<int>(BoundaryKind::Interior) &&
        bnd != static_cast<int>(BoundaryKind::NoSlipWall)) return;

    Real rho_L, u_L, v_L, w_L, p_L;
    primitive_from_q(d_q, left, nvar, gamma, rho_L, u_L, v_L, w_L, p_L);
    if (!real_isfinite(rho_L) || rho_L <= 0.0f || !real_isfinite(p_L) || p_L <= 0.0f) return;

    Real inv_rho_L = 1.0f / rho_L;
    Real inv_rho_L2 = inv_rho_L * inv_rho_L;
    Real T_L = p_L * inv_rho_L;

    int gL = left * DeviceMesh::NGRAD;
    Real du_dx_L = d_gradients[gL + 3];
    Real du_dy_L = d_gradients[gL + 4];
    Real du_dz_L = d_gradients[gL + 5];
    Real dv_dx_L = d_gradients[gL + 6];
    Real dv_dy_L = d_gradients[gL + 7];
    Real dv_dz_L = d_gradients[gL + 8];
    Real dw_dx_L = d_gradients[gL + 9];
    Real dw_dy_L = d_gradients[gL + 10];
    Real dw_dz_L = d_gradients[gL + 11];
    Real drho_dx_L = d_gradients[gL + 0];
    Real drho_dy_L = d_gradients[gL + 1];
    Real drho_dz_L = d_gradients[gL + 2];
    Real dp_dx_L = d_gradients[gL + 12];
    Real dp_dy_L = d_gradients[gL + 13];
    Real dp_dz_L = d_gradients[gL + 14];

    Real dT_dx_L = (rho_L * dp_dx_L - p_L * drho_dx_L) * inv_rho_L2;
    Real dT_dy_L = (rho_L * dp_dy_L - p_L * drho_dy_L) * inv_rho_L2;
    Real dT_dz_L = (rho_L * dp_dz_L - p_L * drho_dz_L) * inv_rho_L2;

    Real face_u, face_v, face_w, face_T;
    Real grad_du_dx, grad_du_dy, grad_du_dz;
    Real grad_dv_dx, grad_dv_dy, grad_dv_dz;
    Real grad_dw_dx, grad_dw_dy, grad_dw_dz;
    Real grad_dT_dx, grad_dT_dy, grad_dT_dz;

    if (right >= 0 && bnd == static_cast<int>(BoundaryKind::Interior)) {
        Real rho_R, u_R, v_R, w_R, p_R;
        primitive_from_q(d_q, right, nvar, gamma, rho_R, u_R, v_R, w_R, p_R);
        if (!real_isfinite(rho_R) || rho_R <= 0.0f || !real_isfinite(p_R) || p_R <= 0.0f) return;

        Real inv_rho_R = 1.0f / rho_R;
        Real inv_rho_R2 = inv_rho_R * inv_rho_R;
        Real T_R = p_R * inv_rho_R;

        int gR = right * DeviceMesh::NGRAD;
        Real drho_dx_R = d_gradients[gR + 0];
        Real drho_dy_R = d_gradients[gR + 1];
        Real drho_dz_R = d_gradients[gR + 2];
        Real du_dx_R = d_gradients[gR + 3];
        Real du_dy_R = d_gradients[gR + 4];
        Real du_dz_R = d_gradients[gR + 5];
        Real dv_dx_R = d_gradients[gR + 6];
        Real dv_dy_R = d_gradients[gR + 7];
        Real dv_dz_R = d_gradients[gR + 8];
        Real dw_dx_R = d_gradients[gR + 9];
        Real dw_dy_R = d_gradients[gR + 10];
        Real dw_dz_R = d_gradients[gR + 11];
        Real dp_dx_R = d_gradients[gR + 12];
        Real dp_dy_R = d_gradients[gR + 13];
        Real dp_dz_R = d_gradients[gR + 14];

        Real dT_dx_R = (rho_R * dp_dx_R - p_R * drho_dx_R) * inv_rho_R2;
        Real dT_dy_R = (rho_R * dp_dy_R - p_R * drho_dy_R) * inv_rho_R2;
        Real dT_dz_R = (rho_R * dp_dz_R - p_R * drho_dz_R) * inv_rho_R2;

        grad_du_dx = 0.5f * (du_dx_L + du_dx_R);
        grad_du_dy = 0.5f * (du_dy_L + du_dy_R);
        grad_du_dz = 0.5f * (du_dz_L + du_dz_R);
        grad_dv_dx = 0.5f * (dv_dx_L + dv_dx_R);
        grad_dv_dy = 0.5f * (dv_dy_L + dv_dy_R);
        grad_dv_dz = 0.5f * (dv_dz_L + dv_dz_R);
        grad_dw_dx = 0.5f * (dw_dx_L + dw_dx_R);
        grad_dw_dy = 0.5f * (dw_dy_L + dw_dy_R);
        grad_dw_dz = 0.5f * (dw_dz_L + dw_dz_R);
        grad_dT_dx = 0.5f * (dT_dx_L + dT_dx_R);
        grad_dT_dy = 0.5f * (dT_dy_L + dT_dy_R);
        grad_dT_dz = 0.5f * (dT_dz_L + dT_dz_R);

        Real dr_x = d_cell_cx[right] - d_cell_cx[left];
        Real dr_y = d_cell_cy[right] - d_cell_cy[left];
        Real dr_z = d_cell_cz[right] - d_cell_cz[left];
        Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
        if (d2 > 1e-30f) {
            Real inv_d2 = 1.0f / d2;
            Real proj_du = grad_du_dx*dr_x + grad_du_dy*dr_y + grad_du_dz*dr_z;
            Real proj_dv = grad_dv_dx*dr_x + grad_dv_dy*dr_y + grad_dv_dz*dr_z;
            Real proj_dw = grad_dw_dx*dr_x + grad_dw_dy*dr_y + grad_dw_dz*dr_z;
            Real proj_dT = grad_dT_dx*dr_x + grad_dT_dy*dr_y + grad_dT_dz*dr_z;
            Real du_corr = ((u_R - u_L) - proj_du) * inv_d2;
            Real dv_corr = ((v_R - v_L) - proj_dv) * inv_d2;
            Real dw_corr = ((w_R - w_L) - proj_dw) * inv_d2;
            Real dT_corr = ((T_R - T_L) - proj_dT) * inv_d2;
            grad_du_dx += du_corr * dr_x; grad_du_dy += du_corr * dr_y; grad_du_dz += du_corr * dr_z;
            grad_dv_dx += dv_corr * dr_x; grad_dv_dy += dv_corr * dr_y; grad_dv_dz += dv_corr * dr_z;
            grad_dw_dx += dw_corr * dr_x; grad_dw_dy += dw_corr * dr_y; grad_dw_dz += dw_corr * dr_z;
            grad_dT_dx += dT_corr * dr_x; grad_dT_dy += dT_corr * dr_y; grad_dT_dz += dT_corr * dr_z;
        }

        face_u = 0.5f * (u_L + u_R);
        face_v = 0.5f * (v_L + v_R);
        face_w = 0.5f * (w_L + w_R);
        face_T = 0.5f * (T_L + T_R);
    } else {
        grad_du_dx = du_dx_L; grad_du_dy = du_dy_L; grad_du_dz = du_dz_L;
        grad_dv_dx = dv_dx_L; grad_dv_dy = dv_dy_L; grad_dv_dz = dv_dz_L;
        grad_dw_dx = dw_dx_L; grad_dw_dy = dw_dy_L; grad_dw_dz = dw_dz_L;
        grad_dT_dx = dT_dx_L; grad_dT_dy = dT_dy_L; grad_dT_dz = dT_dz_L;

        Real dr_x = d_face_cx[idx] - d_cell_cx[left];
        Real dr_y = d_face_cy[idx] - d_cell_cy[left];
        Real dr_z = d_face_cz[idx] - d_cell_cz[left];
        Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
        if (d2 > 1e-30f) {
            Real inv_d2 = 1.0f / d2;
            Real proj_du = grad_du_dx*dr_x + grad_du_dy*dr_y + grad_du_dz*dr_z;
            Real proj_dv = grad_dv_dx*dr_x + grad_dv_dy*dr_y + grad_dv_dz*dr_z;
            Real proj_dw = grad_dw_dx*dr_x + grad_dw_dy*dr_y + grad_dw_dz*dr_z;
            Real proj_dT = grad_dT_dx*dr_x + grad_dT_dy*dr_y + grad_dT_dz*dr_z;
            Real du_corr = ((0.0f - u_L) - proj_du) * inv_d2;
            Real dv_corr = ((0.0f - v_L) - proj_dv) * inv_d2;
            Real dw_corr = ((0.0f - w_L) - proj_dw) * inv_d2;
            Real dT_corr = ((wall_T - T_L) - proj_dT) * inv_d2;
            grad_du_dx += du_corr * dr_x; grad_du_dy += du_corr * dr_y; grad_du_dz += du_corr * dr_z;
            grad_dv_dx += dv_corr * dr_x; grad_dv_dy += dv_corr * dr_y; grad_dv_dz += dv_corr * dr_z;
            grad_dw_dx += dw_corr * dr_x; grad_dw_dy += dw_corr * dr_y; grad_dw_dz += dw_corr * dr_z;
            grad_dT_dx += dT_corr * dr_x; grad_dT_dy += dT_corr * dr_y; grad_dT_dz += dT_corr * dr_z;
        }

        face_u = 0.0f;
        face_v = 0.0f;
        face_w = 0.0f;
        face_T = wall_T;
    }

    if (face_T <= 0.0f) return;
    Real mu_face = sutherland_mu(face_T, T_ref, sutherland_T, mu_ref);
    if (mu_face <= 0.0f) return;

    Real nx = d_nx[idx];
    Real ny = d_ny[idx];
    Real nz = d_nz[idx];
    Real area = d_area[idx];

    Real div_u = grad_du_dx + grad_dv_dy + grad_dw_dz;
    Real tau_xx = 2.0f * (grad_du_dx - div_u / 3.0f);
    Real tau_yy = 2.0f * (grad_dv_dy - div_u / 3.0f);
    Real tau_zz = 2.0f * (grad_dw_dz - div_u / 3.0f);
    Real tau_xy = (grad_du_dy + grad_dv_dx);
    Real tau_xz = (grad_du_dz + grad_dw_dx);
    Real tau_yz = (grad_dv_dz + grad_dw_dy);

    Real mu_invRe = mu_face * inv_Re;
    Real visc_mom_x = (tau_xx*nx + tau_xy*ny + tau_xz*nz) * mu_invRe * area;
    Real visc_mom_y = (tau_xy*nx + tau_yy*ny + tau_yz*nz) * mu_invRe * area;
    Real visc_mom_z = (tau_xz*nx + tau_yz*ny + tau_zz*nz) * mu_invRe * area;

    Real dT_dn = grad_dT_dx*nx + grad_dT_dy*ny + grad_dT_dz*nz;
    Real kappa_over_Re = mu_face * gamma / ((gamma - 1.0f) * prandtl) * inv_Re;
    Real visc_energy = (face_u * visc_mom_x + face_v * visc_mom_y + face_w * visc_mom_z
        + kappa_over_Re * dT_dn * area);

    d_residual[left * nvar + 1] += visc_mom_x;
    d_residual[left * nvar + 2] += visc_mom_y;
    d_residual[left * nvar + 3] += visc_mom_z;
    d_residual[left * nvar + 4] += visc_energy;

    if (right >= 0 && bnd == static_cast<int>(BoundaryKind::Interior)) {
        d_residual[right * nvar + 1] += -visc_mom_x;
        d_residual[right * nvar + 2] += -visc_mom_y;
        d_residual[right * nvar + 3] += -visc_mom_z;
        d_residual[right * nvar + 4] += -visc_energy;
    }

    if (!turbulence) return;

    // SA conservative diffusion: (1/sigma) * div((mu/Re + rho*nu_tilde*fv1) * grad(nu_tilde))
    constexpr Real sigma_sa = 2.0f / 3.0f;
    constexpr Real cv1 = 7.1f;
    constexpr Real cv13 = cv1 * cv1 * cv1;

    Real nu_tilde_L = d_q[left * nvar + 5] / rho_L;
    if (!real_isfinite(nu_tilde_L)) return;
    Real dnu_dx_L = d_gradients[gL + 15];
    Real dnu_dy_L = d_gradients[gL + 16];
    Real dnu_dz_L = d_gradients[gL + 17];

    Real grad_dnu_dx, grad_dnu_dy, grad_dnu_dz;
    Real face_nu_tilde, face_rho;

    if (right >= 0 && bnd == static_cast<int>(BoundaryKind::Interior)) {
        Real rho_R, u_R, v_R, w_R, p_R;
        primitive_from_q(d_q, right, nvar, gamma, rho_R, u_R, v_R, w_R, p_R);
        if (!real_isfinite(rho_R) || rho_R <= 0.0f || !real_isfinite(p_R) || p_R <= 0.0f) return;
        Real nu_tilde_R = d_q[right * nvar + 5] / rho_R;
        if (!real_isfinite(nu_tilde_R)) return;

        int gR = right * DeviceMesh::NGRAD;
        Real dnu_dx_R = d_gradients[gR + 15];
        Real dnu_dy_R = d_gradients[gR + 16];
        Real dnu_dz_R = d_gradients[gR + 17];

        grad_dnu_dx = 0.5f * (dnu_dx_L + dnu_dx_R);
        grad_dnu_dy = 0.5f * (dnu_dy_L + dnu_dy_R);
        grad_dnu_dz = 0.5f * (dnu_dz_L + dnu_dz_R);

        Real dr_x = d_cell_cx[right] - d_cell_cx[left];
        Real dr_y = d_cell_cy[right] - d_cell_cy[left];
        Real dr_z = d_cell_cz[right] - d_cell_cz[left];
        Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
        if (d2 > 1e-30f) {
            Real inv_d2 = 1.0f / d2;
            Real proj_dnu = grad_dnu_dx*dr_x + grad_dnu_dy*dr_y + grad_dnu_dz*dr_z;
            Real dnu_corr = ((nu_tilde_R - nu_tilde_L) - proj_dnu) * inv_d2;
            grad_dnu_dx += dnu_corr * dr_x;
            grad_dnu_dy += dnu_corr * dr_y;
            grad_dnu_dz += dnu_corr * dr_z;
        }

        face_nu_tilde = 0.5f * (nu_tilde_L + nu_tilde_R);
        face_rho = 0.5f * (rho_L + rho_R);
    } else {
        grad_dnu_dx = dnu_dx_L;
        grad_dnu_dy = dnu_dy_L;
        grad_dnu_dz = dnu_dz_L;

        Real dr_x = d_face_cx[idx] - d_cell_cx[left];
        Real dr_y = d_face_cy[idx] - d_cell_cy[left];
        Real dr_z = d_face_cz[idx] - d_cell_cz[left];
        Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
        if (d2 > 1e-30f) {
            Real inv_d2 = 1.0f / d2;
            Real proj_dnu = grad_dnu_dx*dr_x + grad_dnu_dy*dr_y + grad_dnu_dz*dr_z;
            Real dnu_corr = ((0.0f - nu_tilde_L) - proj_dnu) * inv_d2;
            grad_dnu_dx += dnu_corr * dr_x;
            grad_dnu_dy += dnu_corr * dr_y;
            grad_dnu_dz += dnu_corr * dr_z;
        }

        face_nu_tilde = 0.5f * nu_tilde_L;
        face_rho = rho_L;
    }

    Real dnu_dn = grad_dnu_dx*nx + grad_dnu_dy*ny + grad_dnu_dz*nz;

    Real chi_face = Re * face_rho * face_nu_tilde / (mu_face + 1e-30f);
    Real chi3 = chi_face * chi_face * chi_face;
    Real fv1_face = chi3 / (chi3 + cv13 + 1e-30f);
    Real mu_tilde = face_rho * face_nu_tilde * fv1_face / sigma_sa;
    Real mu_total = (mu_face * inv_Re) / sigma_sa + mu_tilde;
    Real visc_nu = mu_total * dnu_dn * area;

    d_residual[left * nvar + 5] += visc_nu;
    if (right >= 0 && bnd == static_cast<int>(BoundaryKind::Interior)) {
        d_residual[right * nvar + 5] += -visc_nu;
    }
}

} // namespace

bool compute_viscous_flux_gpu(DeviceMesh& mesh, Real gamma, Real prandtl,
    Real mu_ref, Real T_ref, Real sutherland_T, Real Re, Real wall_T,
    int turbulence, int* d_failed, cudaStream_t stream) {
    Real inv_Re = 1.0f / (Re > 0.0f ? Re : 1e6f);

    int block = 128;
    int nf = static_cast<int>(mesh.face_count());
    int grid = (nf + block - 1) / block;

    DeviceFaceData fd = mesh.face_data();
    DeviceCellData cd = mesh.cell_data();

    int nc = static_cast<int>(mesh.cell_count());

    const GpuPartition* gpu_part = mesh.get_partition();
    const int* d_partition_owner = gpu_part ? gpu_part->d_partition_owner : nullptr;
    int my_rank = gpu_part ? gpu_part->my_rank : 0;

    int n_colors = mesh.color_count();

    if (n_colors > 0) {
        for (int c = 0; c < n_colors; ++c) {
            int start = mesh.host_color_offsets()[c];
            int end   = mesh.host_color_offsets()[c + 1];
            int nf_c  = end - start;
            int grid_c = (nf_c + block - 1) / block;
            viscous_flux_kernel_colored<<<grid_c, block, 0, stream>>>(
                mesh.state_device(), DeviceMesh::NVAR, gamma,
                fd.nx, fd.ny, fd.nz, fd.area,
                fd.left_cell, fd.right_cell, fd.boundary,
                cd.cx, cd.cy, cd.cz,
                fd.cx, fd.cy, fd.cz,
                mesh.gradients_device(),
                mesh.residual_device(),
                start, end, nc,
                prandtl, mu_ref, T_ref, sutherland_T,
                inv_Re, Re, wall_T,
                turbulence,
                d_failed,
                d_partition_owner, my_rank);
            if (!cuda_check(cudaGetLastError(), "viscous_flux_kernel_colored")) return false;
        }
    } else {
        viscous_flux_kernel_atomic<<<grid, block, 0, stream>>>(
            mesh.state_device(), DeviceMesh::NVAR, gamma,
            fd.nx, fd.ny, fd.nz, fd.area,
            fd.left_cell, fd.right_cell, fd.boundary,
            cd.cx, cd.cy, cd.cz,
            fd.cx, fd.cy, fd.cz,
            mesh.gradients_device(),
            mesh.residual_device(),
            nf, nc,
            prandtl, mu_ref, T_ref, sutherland_T,
            inv_Re, Re, wall_T,
            turbulence,
            d_failed,
            d_partition_owner, my_rank);
            if (!cuda_check(cudaGetLastError(), "viscous_flux_kernel_atomic")) return false;
    }
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
