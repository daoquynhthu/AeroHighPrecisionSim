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

__global__ void init_floatN_zero_kernel(Real* ptr, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n; ++i) ptr[i] = 0.0f;
    }
}

__global__ void wall_force_kernel(
    const Real* d_q, int nvar, Real gamma,
    const Real* d_nx, const Real* d_ny, const Real* d_nz,
    const Real* d_area,
    const int* d_left_cell,
    const int* d_boundary,
    const Real* d_cx, const Real* d_cy, const Real* d_cz,
    int face_count, int n_cells,
    Real* d_forces,
    const Real* d_gradients,
    const Real* d_cell_cx, const Real* d_cell_cy, const Real* d_cell_cz,
    bool viscous, Real prandtl, Real mu_ref, Real T_ref, Real sutherland_T,
    Real inv_Re, Real wall_T,
    const int* d_partition_owner, int my_rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= face_count) return;

    int bnd = d_boundary[idx];
    if (bnd != static_cast<int>(BoundaryKind::SlipWall) &&
        bnd != static_cast<int>(BoundaryKind::NoSlipWall)) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    if (d_partition_owner && d_partition_owner[left] != my_rank) return;
    Real rho = d_q[left * nvar + 0];
    if (!real_isfinite(rho) || rho <= 0.0f) return;
    Real inv_rho = 1.0f / rho;
    Real u = d_q[left * nvar + 1] * inv_rho;
    Real v = d_q[left * nvar + 2] * inv_rho;
    Real w = d_q[left * nvar + 3] * inv_rho;
    Real E = d_q[left * nvar + 4];
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (E - rho * kinetic);
    if (!real_isfinite(p) || p <= 0.0f) return;

    Real dr_x = d_cx[idx] - d_cell_cx[left];
    Real dr_y = d_cy[idx] - d_cell_cy[left];
    Real dr_z = d_cz[idx] - d_cell_cz[left];

    if (d_gradients) {
        int g_base = left * DeviceMesh::NGRAD;
        p += d_gradients[g_base + 12] * dr_x;
        p += d_gradients[g_base + 13] * dr_y;
        p += d_gradients[g_base + 14] * dr_z;
        if (!real_isfinite(p) || p <= 0.0f) p = (gamma - 1.0f) * (E - rho * kinetic);
    }

    Real nx = d_nx[idx];
    Real ny = d_ny[idx];
    Real nz = d_nz[idx];
    Real area = d_area[idx];

    Real px = -p * nx * area;
    Real py = -p * ny * area;
    Real pz = -p * nz * area;

    Real fcx = d_cx[idx];
    Real fcy = d_cy[idx];
    Real fcz = d_cz[idx];

    real_atomic_add(&d_forces[0], px);
    real_atomic_add(&d_forces[1], py);
    real_atomic_add(&d_forces[2], pz);
    real_atomic_add(&d_forces[3], fcy * pz - fcz * py);
    real_atomic_add(&d_forces[4], fcz * px - fcx * pz);
    real_atomic_add(&d_forces[5], fcx * py - fcy * px);

    if (viscous && d_gradients && bnd == static_cast<int>(BoundaryKind::NoSlipWall)) {
        int g_base = left * DeviceMesh::NGRAD;

        Real du_dx = d_gradients[g_base + 3];
        Real du_dy = d_gradients[g_base + 4];
        Real du_dz = d_gradients[g_base + 5];
        Real dv_dx = d_gradients[g_base + 6];
        Real dv_dy = d_gradients[g_base + 7];
        Real dv_dz = d_gradients[g_base + 8];
        Real dw_dx = d_gradients[g_base + 9];
        Real dw_dy = d_gradients[g_base + 10];
        Real dw_dz = d_gradients[g_base + 11];
        Real drho_dx = d_gradients[g_base + 0];
        Real drho_dy = d_gradients[g_base + 1];
        Real drho_dz = d_gradients[g_base + 2];
        Real dp_dx = d_gradients[g_base + 12];
        Real dp_dy = d_gradients[g_base + 13];
        Real dp_dz = d_gradients[g_base + 14];

        Real inv_rho2 = 1.0f / (rho * rho + 1e-30f);
        Real dT_dx = (rho * dp_dx - p * drho_dx) * inv_rho2;
        Real dT_dy = (rho * dp_dy - p * drho_dy) * inv_rho2;
        Real dT_dz = (rho * dp_dz - p * drho_dz) * inv_rho2;

        Real d2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z;
        if (d2 > 1e-30f) {
            Real inv_d2 = 1.0f / d2;
            Real proj_du = du_dx*dr_x + du_dy*dr_y + du_dz*dr_z;
            Real proj_dv = dv_dx*dr_x + dv_dy*dr_y + dv_dz*dr_z;
            Real proj_dw = dw_dx*dr_x + dw_dy*dr_y + dw_dz*dr_z;
            Real proj_dT = dT_dx*dr_x + dT_dy*dr_y + dT_dz*dr_z;
            Real du_corr = ((0.0f - u) - proj_du) * inv_d2;
            Real dv_corr = ((0.0f - v) - proj_dv) * inv_d2;
            Real dw_corr = ((0.0f - w) - proj_dw) * inv_d2;
            Real dT_corr = ((wall_T - p * inv_rho) - proj_dT) * inv_d2;
            du_dx += du_corr * dr_x; du_dy += du_corr * dr_y; du_dz += du_corr * dr_z;
            dv_dx += dv_corr * dr_x; dv_dy += dv_corr * dr_y; dv_dz += dv_corr * dr_z;
            dw_dx += dw_corr * dr_x; dw_dy += dw_corr * dr_y; dw_dz += dw_corr * dr_z;
            dT_dx += dT_corr * dr_x; dT_dy += dT_corr * dr_y; dT_dz += dT_corr * dr_z;
        }

        Real div_u = du_dx + dv_dy + dw_dz;
        Real tau_xx = 2.0f * (du_dx - div_u / 3.0f);
        Real tau_yy = 2.0f * (dv_dy - div_u / 3.0f);
        Real tau_zz = 2.0f * (dw_dz - div_u / 3.0f);
        Real tau_xy = (du_dy + dv_dx);
        Real tau_xz = (du_dz + dw_dx);
        Real tau_yz = (dv_dz + dw_dy);

        Real T_face = wall_T;
        Real mu_face = mu_ref * (T_face / T_ref) * real_sqrt(T_face / T_ref) * (T_ref + sutherland_T) / (T_face + sutherland_T);
        if (mu_face <= 0.0f) mu_face = mu_ref;

        Real mu_invRe = mu_face * inv_Re;

        Real tx = (tau_xx*nx + tau_xy*ny + tau_xz*nz) * mu_invRe * area;
        Real ty = (tau_xy*nx + tau_yy*ny + tau_yz*nz) * mu_invRe * area;
        Real tz = (tau_xz*nx + tau_yz*ny + tau_zz*nz) * mu_invRe * area;

        real_atomic_add(&d_forces[0], tx);
        real_atomic_add(&d_forces[1], ty);
        real_atomic_add(&d_forces[2], tz);
        real_atomic_add(&d_forces[3], fcy * tz - fcz * ty);
        real_atomic_add(&d_forces[4], fcz * tx - fcx * tz);
        real_atomic_add(&d_forces[5], fcx * ty - fcy * tx);

        Real conductivity = mu_face / ((gamma - 1.0f) * prandtl + 1e-30f);
        Real dT_dn = dT_dx * nx + dT_dy * ny + dT_dz * nz;
        Real q_face = -conductivity * dT_dn * inv_Re * area;
        real_atomic_add(&d_forces[6], q_face);
    }
}

} // namespace

bool compute_wall_forces_gpu(DeviceMesh& mesh, Real gamma, Real* d_forces) {
    return compute_wall_forces_gpu(mesh, gamma, d_forces, false, 0.72f, 1.0f, 288.15f, 110.4f, 1e6f, 300.0f);
}

bool compute_wall_forces_gpu(DeviceMesh& mesh, Real gamma, Real* d_forces,
    bool viscous, Real prandtl, Real mu_ref, Real T_ref, Real sutherland_T,
    Real Re, Real wall_T) {
    init_floatN_zero_kernel<<<1, 1>>>(d_forces, 7);
    if (!cuda_check(cudaGetLastError(), "init_forces kernel launch")) return false;

    int block = 128;
    int nf = static_cast<int>(mesh.face_count());
    DeviceFaceData fd = mesh.face_data();
    DeviceCellData cd = mesh.cell_data();

    int nc = static_cast<int>(mesh.cell_count());
    Real inv_Re = 1.0f / (Re > 0.0f ? Re : 1e6f);

    const GpuPartition* gpu_part = mesh.get_partition();
    const int* d_partition_owner = gpu_part ? gpu_part->d_partition_owner : nullptr;
    int my_rank = gpu_part ? gpu_part->my_rank : 0;

    int grid = (nf + block - 1) / block;
    wall_force_kernel<<<grid, block>>>(
        mesh.state_device(), DeviceMesh::NVAR, gamma,
        fd.nx, fd.ny, fd.nz, fd.area,
        fd.left_cell, fd.boundary,
        fd.cx, fd.cy, fd.cz,
        nf, nc, d_forces,
        mesh.gradients_device(),
        cd.cx, cd.cy, cd.cz,
        viscous, prandtl, mu_ref, T_ref, sutherland_T,
        inv_Re, wall_T,
        d_partition_owner, my_rank);
    if (!cuda_check(cudaGetLastError(), "wall_force kernel launch")) return false;
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
