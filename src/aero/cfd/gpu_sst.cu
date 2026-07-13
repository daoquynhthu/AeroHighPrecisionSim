#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include "aero/cfd/rans_sst.hpp"
#include <cuda_runtime.h>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

__device__ Real d_face_vn(const Real* d_q, int cell, int nvar,
    Real nx, Real ny, Real nz)
{
    Real rho = d_q[cell * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) return 0.0f;
    Real inv_rho = 1.0f / rho;
    Real u = d_q[cell * nvar + 1] * inv_rho;
    Real v = d_q[cell * nvar + 2] * inv_rho;
    Real w = d_q[cell * nvar + 3] * inv_rho;
    return u * nx + v * ny + w * nz;
}

__global__ void __launch_bounds__(128) sst_gradient_kernel(
    const Real* d_q_k, const Real* d_q_omega,
    int n_cells, int n_faces, int nvar,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    const Real* d_nx, const Real* d_ny, const Real* d_nz, const Real* d_area,
    const Real* d_volume,
    Real* d_grad_k, Real* d_grad_omega,
    int* d_failed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_faces) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) return;
    int bnd = d_boundary[idx];

    Real kL = d_q_k[left];
    Real wL = d_q_omega[left];

    Real kF, wF;
    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) return;
        Real kR = d_q_k[right];
        Real wR = d_q_omega[right];
        kF = 0.5f * (kL + kR);
        wF = 0.5f * (wL + wR);
    } else {
        kF = kL;
        wF = wL;
    }

    Real nx = d_nx[idx], ny = d_ny[idx], nz = d_nz[idx];
    Real area = d_area[idx];
    Real weight = area;

    real_atomic_add(&d_grad_k[left * 3 + 0], kF * nx * weight);
    real_atomic_add(&d_grad_k[left * 3 + 1], kF * ny * weight);
    real_atomic_add(&d_grad_k[left * 3 + 2], kF * nz * weight);
    real_atomic_add(&d_grad_omega[left * 3 + 0], wF * nx * weight);
    real_atomic_add(&d_grad_omega[left * 3 + 1], wF * ny * weight);
    real_atomic_add(&d_grad_omega[left * 3 + 2], wF * nz * weight);
}

__global__ void __launch_bounds__(256) sst_divide_volume_kernel(
    Real* d_grad_k, Real* d_grad_omega,
    const Real* d_volume, int n_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    Real inv_vol = 1.0f / (d_volume[idx] + 1e-30f);
    d_grad_k[idx * 3 + 0] *= inv_vol;
    d_grad_k[idx * 3 + 1] *= inv_vol;
    d_grad_k[idx * 3 + 2] *= inv_vol;
    d_grad_omega[idx * 3 + 0] *= inv_vol;
    d_grad_omega[idx * 3 + 1] *= inv_vol;
    d_grad_omega[idx * 3 + 2] *= inv_vol;
}

__global__ void __launch_bounds__(128) sst_advection_kernel(
    const Real* d_q,
    const Real* d_q_k, const Real* d_q_omega,
    Real* d_residual_k, Real* d_residual_omega,
    int n_cells, int n_faces, int nvar,
    const int* d_left_cell, const int* d_right_cell, const int* d_boundary,
    const Real* d_nx, const Real* d_ny, const Real* d_nz, const Real* d_area,
    Real inf_k, Real inf_omega,
    int* d_failed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_faces) return;

    int left = d_left_cell[idx];
    if (left < 0 || left >= n_cells) { atomicExch(d_failed, 1); return; }
    int bnd = d_boundary[idx];
    Real nx = d_nx[idx], ny = d_ny[idx], nz = d_nz[idx];
    Real area = d_area[idx];

    Real rhoL = d_q[left * nvar + 0];
    Real kL = d_q_k[left];
    Real wL = d_q_omega[left];
    Real vnL = d_face_vn(d_q, left, nvar, nx, ny, nz);

    Real rhoF, kF, wF, vnF;

    if (bnd == static_cast<int>(BoundaryKind::Interior)) {
        int right = d_right_cell[idx];
        if (right < 0 || right >= n_cells) { atomicExch(d_failed, 1); return; }
        Real rhoR = d_q[right * nvar + 0];
        Real kR = d_q_k[right];
        Real wR = d_q_omega[right];
        Real vnR = d_face_vn(d_q, right, nvar, nx, ny, nz);

        vnF = 0.5f * (vnL + vnR);
        if (vnF >= 0.0f) {
            rhoF = rhoL; kF = kL; wF = wL;
        } else {
            rhoF = rhoR; kF = kR; wF = wR;
        }

        Real flux_k = vnF * rhoF * kF * area;
        Real flux_w = vnF * rhoF * wF * area;

        real_atomic_add(&d_residual_k[left], -flux_k);
        real_atomic_add(&d_residual_omega[left], -flux_w);
        real_atomic_add(&d_residual_k[right], flux_k);
        real_atomic_add(&d_residual_omega[right], flux_w);
    } else if (bnd == static_cast<int>(BoundaryKind::SlipWall) ||
               bnd == static_cast<int>(BoundaryKind::NoSlipWall) ||
               bnd == static_cast<int>(BoundaryKind::Symmetry)) {
        return;
    } else if (bnd == static_cast<int>(BoundaryKind::Farfield)) {
        Real rho_inf = d_q[left * nvar + 0];
        if (vnL >= 0.0f) {
            rhoF = rhoL; kF = kL; wF = wL;
        } else {
            rhoF = rho_inf; kF = inf_k; wF = inf_omega;
        }
        vnF = vnL;
        Real flux_k = vnF * rhoF * kF * area;
        Real flux_w = vnF * rhoF * wF * area;
        real_atomic_add(&d_residual_k[left], -flux_k);
        real_atomic_add(&d_residual_omega[left], -flux_w);
    } else {
        atomicExch(d_failed, 1);
    }
}

__global__ void __launch_bounds__(256) sst_source_kernel(
    const Real* d_q,
    const Real* d_q_k, const Real* d_q_omega,
    Real* d_residual_k, Real* d_residual_omega,
    const Real* d_grad_k, const Real* d_grad_omega,
    const Real* d_vel_grad,
    const Real* d_wall_distance,
    const Real* d_volume,
    int n_cells, int nvar, int ngrad,
    Real gamma, Real Re, Real mu_ref, Real T_ref, Real sutherland_T,
    int* d_failed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real rho = d_q[idx * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    Real inv_rho = 1.0f / rho;
    Real u = d_q[idx * nvar + 1] * inv_rho;
    Real v = d_q[idx * nvar + 2] * inv_rho;
    Real w = d_q[idx * nvar + 3] * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (d_q[idx * nvar + 4] - rho * kinetic);
    if (!real_isfinite(p) || p <= 0.0f) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }
    Real T = p * inv_rho;
    Real mu = mu_ref;
    if (T > 0.0f) {
        Real t_ratio = T / T_ref;
        mu = mu_ref * t_ratio * real_sqrt(t_ratio) * (T_ref + sutherland_T) / (T + sutherland_T);
    }
    if (mu <= 0.0f) mu = mu_ref;

    Real k = d_q_k[idx];
    Real omega = d_q_omega[idx];
    if (!real_isfinite(k) || k < 0.0f) k = 0.0f;
    if (!real_isfinite(omega) || omega < 0.0f) omega = 1e-10f;

    Real wall_distance = d_wall_distance[idx];
    if (wall_distance <= 0.0f || !real_isfinite(wall_distance)) {
        wall_distance = 1e30f;
    }

    Real gk_x = d_grad_k[idx * 3 + 0];
    Real gk_y = d_grad_k[idx * 3 + 1];
    Real gk_z = d_grad_k[idx * 3 + 2];
    Real gw_x = d_grad_omega[idx * 3 + 0];
    Real gw_y = d_grad_omega[idx * 3 + 1];
    Real gw_z = d_grad_omega[idx * 3 + 2];

    SstBlending b = compute_sst_blending(k, omega, wall_distance, rho, mu,
        gk_x, gk_y, gk_z, gw_x, gw_y, gw_z);

    Real mu_t = rho * k / (omega + 1e-30f);

    Real S_mag = 0.0f;
    {
        constexpr Real eps = 1e-30f;
        const Real* g = d_vel_grad + idx * ngrad;
        Real du_dx = g[3]; Real du_dy = g[4]; Real du_dz = g[5];
        Real dv_dx = g[6]; Real dv_dy = g[7]; Real dv_dz = g[8];
        Real dw_dx = g[9]; Real dw_dy = g[10]; Real dw_dz = g[11];
        Real S11 = du_dx;
        Real S22 = dv_dy;
        Real S33 = dw_dz;
        Real S12 = 0.5f * (du_dy + dv_dx);
        Real S13 = 0.5f * (du_dz + dw_dx);
        Real S23 = 0.5f * (dv_dz + dw_dy);
        S_mag = real_sqrt(2.0f * (S11*S11 + S22*S22 + S33*S33
                    + 2.0f * (S12*S12 + S13*S13 + S23*S23)) + eps);
    }

    SstSource s = compute_sst_source(k, omega, rho, mu_t, b, S_mag);

    Real vol_k = s.source_k;
    Real vol_w = s.source_w;

    if (!real_isfinite(vol_k) || !real_isfinite(vol_w)) {
        if (d_failed) atomicCAS(d_failed, 0, 1);
        return;
    }

    d_residual_k[idx] += vol_k;
    d_residual_omega[idx] += vol_w;
}

__global__ void __launch_bounds__(256) sst_init_kernel(
    Real* d_q_k, Real* d_q_omega, Real k_val, Real omega_val, int n_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    d_q_k[idx] = k_val;
    d_q_omega[idx] = omega_val;
}

__global__ void __launch_bounds__(256) clear_sst_residual_kernel(
    Real* d_residual_k, Real* d_residual_omega, int n_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    d_residual_k[idx] = 0.0f;
    d_residual_omega[idx] = 0.0f;
}

__global__ void __launch_bounds__(256) sst_update_kernel(
    const Real* d_q_k, const Real* d_q_omega,
    const Real* d_residual_k, const Real* d_residual_omega,
    const Real* d_min_dt, const Real* d_volume,
    Real* d_q_k_next, Real* d_q_omega_next,
    int n_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    Real k = d_q_k[idx];
    Real w = d_q_omega[idx];
    if (k < 0.0f || !real_isfinite(k)) k = 0.0f;
    if (w < 0.0f || !real_isfinite(w)) w = 1e-10f;

    Real dt = __ldg(d_min_dt);
    Real vol = d_volume[idx];
    if (vol <= 0.0f) { vol = 1e-30f; }

    Real dtv = dt / vol;
    Real k_new = k + dtv * d_residual_k[idx];
    Real w_new = w + dtv * d_residual_omega[idx];

    if (k_new < 0.0f || !real_isfinite(k_new)) k_new = k * 0.5f + 1e-16f;
    if (w_new < 0.0f || !real_isfinite(w_new)) w_new = w * 0.5f + 1e-10f;

    d_q_k_next[idx] = k_new;
    d_q_omega_next[idx] = w_new;
}

} // namespace

bool compute_sst_gradients_gpu(DeviceMesh& mesh, int* d_failed,
    std::string* error, cudaStream_t stream)
{
    if (mesh.cell_count() == 0 || mesh.face_count() == 0) return true;
    if (!mesh.has_sst()) {
        if (error) *error = "SST buffers not allocated for gradients";
        return false;
    }
    int block = 128;
    int nf = static_cast<int>(mesh.face_count());
    int nc = static_cast<int>(mesh.cell_count());
    int grid_grad = (nf + block - 1) / block;
    int grid_div = (nc + block - 1) / block;

    DeviceCellData cd = mesh.cell_data();
    DeviceFaceData fd = mesh.face_data();

    sst_gradient_kernel<<<grid_grad, block, 0, stream>>>(
        mesh.q_k_device(), mesh.q_omega_device(),
        nc, nf, DeviceMesh::NVAR,
        fd.left_cell, fd.right_cell, fd.boundary,
        fd.nx, fd.ny, fd.nz, fd.area,
        cd.volume,
        mesh.grad_k_device(), mesh.grad_omega_device(),
        d_failed);
    if (!cuda_check(cudaGetLastError(), "sst_gradient_kernel launch", error)) return false;

    sst_divide_volume_kernel<<<grid_div, block, 0, stream>>>(
        mesh.grad_k_device(), mesh.grad_omega_device(),
        cd.volume, nc);
    if (!cuda_check(cudaGetLastError(), "sst_divide_volume_kernel launch", error)) return false;

    return true;
}

bool compute_sst_advection_gpu(DeviceMesh& mesh,
    Real inf_k, Real inf_omega,
    int* d_failed, std::string* error, cudaStream_t stream)
{
    if (mesh.cell_count() == 0 || mesh.face_count() == 0) return true;
    if (!mesh.has_sst()) {
        if (error) *error = "SST buffers not allocated for advection";
        return false;
    }
    int block = 128;
    int nf = static_cast<int>(mesh.face_count());
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nf + block - 1) / block;

    DeviceFaceData fd = mesh.face_data();

    sst_advection_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(),
        mesh.q_k_device(), mesh.q_omega_device(),
        mesh.residual_k_device(), mesh.residual_omega_device(),
        nc, nf, DeviceMesh::NVAR,
        fd.left_cell, fd.right_cell, fd.boundary,
        fd.nx, fd.ny, fd.nz, fd.area,
        inf_k, inf_omega,
        d_failed);
    if (!cuda_check(cudaGetLastError(), "sst_advection_kernel launch", error)) return false;

    return true;
}

bool compute_sst_source_gpu(DeviceMesh& mesh, Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    int* d_failed, std::string* error, cudaStream_t stream)
{
    if (mesh.cell_count() == 0) return true;
    if (!mesh.has_sst()) {
        if (error) *error = "SST buffers not allocated for source";
        return false;
    }
    if (!mesh.gradients_device()) {
        if (error) *error = "gradients not allocated for SST source";
        return false;
    }

    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    sst_source_kernel<<<grid, block, 0, stream>>>(
        mesh.state_device(),
        mesh.q_k_device(), mesh.q_omega_device(),
        mesh.residual_k_device(), mesh.residual_omega_device(),
        mesh.grad_k_device(), mesh.grad_omega_device(),
        mesh.gradients_device(),
        cd.wall_distance, cd.volume,
        nc, DeviceMesh::NVAR, DeviceMesh::NGRAD,
        gamma, Re, mu_ref, T_ref, sutherland_T,
        d_failed);
    if (!cuda_check(cudaGetLastError(), "sst_source_kernel launch", error)) return false;

    return true;
}

bool compute_sst_update_gpu(DeviceMesh& mesh, const Real* d_min_dt,
    std::string* error, cudaStream_t stream)
{
    if (mesh.cell_count() == 0) return true;
    if (!mesh.has_sst()) {
        if (error) *error = "SST buffers not allocated for update";
        return false;
    }
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;
    DeviceCellData cd = mesh.cell_data();

    sst_update_kernel<<<grid, block, 0, stream>>>(
        mesh.q_k_device(), mesh.q_omega_device(),
        mesh.residual_k_device(), mesh.residual_omega_device(),
        d_min_dt, cd.volume,
        mesh.q_k_device(), mesh.q_omega_device(),
        nc);
    if (!cuda_check(cudaGetLastError(), "sst_update_kernel launch", error)) return false;

    return true;
}

bool compute_sst_init_gpu(DeviceMesh& mesh,
    Real k_val, Real omega_val,
    std::string* error, cudaStream_t stream)
{
    if (mesh.cell_count() == 0) return true;
    if (!mesh.has_sst()) {
        if (error) *error = "SST buffers not allocated for init";
        return false;
    }
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;

    sst_init_kernel<<<grid, block, 0, stream>>>(
        mesh.q_k_device(), mesh.q_omega_device(),
        k_val, omega_val, nc);
    if (!cuda_check(cudaGetLastError(), "sst_init_kernel launch", error)) return false;

    return true;
}

bool clear_sst_residual_gpu(DeviceMesh& mesh,
    std::string* error, cudaStream_t stream)
{
    if (mesh.cell_count() == 0) return true;
    if (!mesh.has_sst()) return true;
    int block = 128;
    int nc = static_cast<int>(mesh.cell_count());
    int grid = (nc + block - 1) / block;

    clear_sst_residual_kernel<<<grid, block, 0, stream>>>(
        mesh.residual_k_device(), mesh.residual_omega_device(), nc);
    if (!cuda_check(cudaGetLastError(), "clear_sst_residual_kernel launch", error)) return false;

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp