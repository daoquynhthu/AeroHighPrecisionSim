#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/cfd_result.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/fgmres.hpp"
#include "aero/cfd/gpu_communicator.hpp"
#include "aero/cfd/gpu_solver.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include "aero/cfd/krylov_ops.hpp"
#include "aero/cfd/lusgs.hpp"
#include "aero/cfd/partition.hpp"
#include "aero/cfd/diagnostics.hpp"
#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/amr_sensor.hpp"
#include "aero/cfd/amr_interpolate.hpp"
#include <cfloat>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cuda_runtime.h>
namespace aerosp {
namespace aero {
namespace cfd {

namespace {

// Temporary diagnostic kernel: find and print the first cell that fails
// d_conservative_to_primitive, so we can understand why d_failed is set.
__global__ void d_find_invalid_cell(const Real* d_q, int n_cells, int nvar, Real gamma,
    Real* d_out_rho, Real* d_out_u, Real* d_out_v, Real* d_out_w, Real* d_out_p, int* d_out_cell) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;
    Real rho = d_q[cell * nvar + 0];
    if (rho <= 0.0f || !real_isfinite(rho)) {
        if (atomicExch(d_out_cell, cell) == -1) {
            *d_out_rho = rho; *d_out_u = 0; *d_out_v = 0; *d_out_w = 0; *d_out_p = 0;
        }
        return;
    }
    Real inv_rho = 1.0f / rho;
    Real u = d_q[cell * nvar + 1] * inv_rho;
    Real v = d_q[cell * nvar + 2] * inv_rho;
    Real w = d_q[cell * nvar + 3] * inv_rho;
    Real kinetic = 0.5f * (u*u + v*v + w*w);
    Real p = (gamma - 1.0f) * (d_q[cell * nvar + 4] - rho * kinetic);
    Real nu_tilde = d_q[cell * nvar + 5] * inv_rho;
    if (!real_isfinite(u) || !real_isfinite(v) || !real_isfinite(w) ||
        !real_isfinite(p) || !real_isfinite(nu_tilde) || p <= 0.0f) {
        if (atomicExch(d_out_cell, cell) == -1) {
            *d_out_rho = rho; *d_out_u = u; *d_out_v = v; *d_out_w = w; *d_out_p = p;
        }
    }
}

__global__ void check_array_valid_kernel(const Real* d_arr, int n, int* d_failed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (!real_isfinite(d_arr[idx])) {
        atomicExch(d_failed, 1);
    }
}

__global__ void check_status_kernel(
    const int* d_failed,
    const Real* d_l2_sum,
    int nvar_ncells,
    Real convergence_tol,
    Real* d_residual_history_slot) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (*d_failed != 0 || !real_isfinite(*d_l2_sum)) {
        *d_residual_history_slot = -1.0f;
        return;
    }
    Real l2 = real_sqrt(*d_l2_sum / static_cast<Real>(nvar_ncells));
    *d_residual_history_slot = l2;
}

__global__ void newton_l2_check_kernel(
    Real* d_l2_sum,
    int nvar_ncells,
    Real sufficient_decrease,
    int* d_newton_accepted) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    Real l2_new = real_sqrt(d_l2_sum[0] / static_cast<Real>(nvar_ncells));
    int accepted = (l2_new < sufficient_decrease * d_l2_sum[1]) ? 1 : 0;
    if (accepted) {
        d_l2_sum[1] = l2_new;
    }
    *d_newton_accepted = accepted;
}

__global__ void init_l2_old_kernel(
    Real* d_l2_sum,
    int nvar_ncells) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    d_l2_sum[1] = real_sqrt(d_l2_sum[0] / static_cast<Real>(nvar_ncells));
}

__global__ void sst_l2_accumulate_kernel(
    Real* d_l2_sum,
    const Real* d_sst_sums) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    d_l2_sum[0] += d_sst_sums[0] + d_sst_sums[1];
}

} // namespace

bool reallocate_implicit_buffers(
    int nvar,
    int old_n_cells,
    int new_n_cells,
    Real*& d_dq,
    Real*& d_dt_cell,
    Real*& d_neg_r,
    Real*& d_r_saved,
    Real*& d_q_backup,
    Real*& d_scratch,
    int*& d_newton_accepted,
    std::string* error) {
    auto safe_free = [](auto*& p) { cuda_free_safe(p); };
    safe_free(d_dq);
    safe_free(d_dt_cell);
    safe_free(d_neg_r);
    safe_free(d_r_saved);
    safe_free(d_q_backup);
    safe_free(d_scratch);
    safe_free(d_newton_accepted);
    if (new_n_cells <= 0) return true;
    int new_nvar_cells = new_n_cells * nvar;
    if (!cuda_check(cudaMalloc(&d_dq, new_nvar_cells * sizeof(Real)), "realloc d_dq", error)) return false;
    if (!cuda_check(cudaMemset(d_dq, 0, new_nvar_cells * sizeof(Real)), "zero d_dq", error)) return false;
    if (!cuda_check(cudaMalloc(&d_dt_cell, new_n_cells * sizeof(Real)), "realloc d_dt_cell", error)) return false;
    if (!cuda_check(cudaMalloc(&d_neg_r, new_nvar_cells * sizeof(Real)), "realloc d_neg_r", error)) return false;
    if (!cuda_check(cudaMalloc(&d_r_saved, new_nvar_cells * sizeof(Real)), "realloc d_r_saved", error)) return false;
    if (!cuda_check(cudaMalloc(&d_q_backup, new_nvar_cells * sizeof(Real)), "realloc d_q_backup", error)) return false;
    if (!cuda_check(cudaMalloc(&d_scratch, 4 * new_nvar_cells * sizeof(Real)), "realloc d_scratch", error)) return false;
    if (!cuda_check(cudaMalloc(&d_newton_accepted, sizeof(int)), "realloc d_newton_accepted", error)) return false;
    return true;
}

static void solve_gpu_free(int* d_failed, Real* d_min_dt, Real* d_l2_sum, Real* d_forces,
    Real* d_residual_history, Real* d_state_bounds_history, int* d_failure_cell, Real* d_failure_state) {
    cuda_free_safe(d_failed); cuda_free_safe(d_min_dt); cuda_free_safe(d_l2_sum); cuda_free_safe(d_forces);
    cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history);
    cuda_free_safe(d_failure_cell); cuda_free_safe(d_failure_state);
}

static CfdSolveSummary solve_gpu_impl(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    int* d_failed,
    Real* d_min_dt,
    Real* d_l2_sum,
    Real* d_forces,
    Real* d_residual_history,
    Real* d_state_bounds_history,
    int* d_failure_cell,
    Real* d_failure_state,
    bool owned_buffers,
    std::string* error,
    GpuCommunicator* comm = nullptr,
    const GpuPartition* gpu_part = nullptr,
    CfdMesh* host_mesh = nullptr,
    std::vector<ConservativeState>* host_state = nullptr,
    AmrCycleCallback amr_callback = nullptr) {
    CfdSolveSummary summary;
    std::vector<Real> host_residual_history;
    int host_failed = 0;
    bool diagnostics_enabled = config.diagnostic_level != DiagnosticLevel::Off;
    Real cfl_multiplier = 1.0f;

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    w_inf.nu_tilde = condition.nu_tilde;
    if (condition.nu_tilde_ratio > 0.0f && config.viscous) {
        Real T_inf = w_inf.p / w_inf.rho;
        Real t_ratio = T_inf / config.T_ref;
        Real mu_inf = config.mu_ref * t_ratio * real_sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T_inf + config.sutherland_T);
        // chi_inf = ratio: nu_tilde = ratio * mu/(rho*Re)
        w_inf.nu_tilde = condition.nu_tilde_ratio * mu_inf
                       / (w_inf.rho * config.Re + Real(1e-30));
    }
    int nvar_ncells_flow = d_mesh.nvar() * static_cast<int>(d_mesh.cell_count());
    int nvar_ncells = nvar_ncells_flow;
    int n_cells = static_cast<int>(d_mesh.cell_count());
    int nvar = d_mesh.nvar();
    int nvar_cells = n_cells * nvar;

    FgmresSolver* fgmres = nullptr;
    LusgsPreconditioner* lusgs = nullptr;
    Real* d_dq = nullptr;
    Real* d_dt_cell = nullptr;
    Real* d_neg_r = nullptr;
    Real* d_r_saved = nullptr;
    Real* d_q_backup = nullptr;
    Real* d_scratch = nullptr;
    int* d_newton_accepted = nullptr;

    cudaStream_t stream_main = nullptr, stream_pre = nullptr;
    cudaStreamCreate(&stream_main);
    cudaStreamCreate(&stream_pre);
    cudaEvent_t event_timestep_done = nullptr;
    cudaEventCreate(&event_timestep_done);
    cudaEvent_t event_update_done = nullptr;
    cudaEventCreate(&event_update_done);
    cudaEventRecord(event_update_done, stream_main); // signaled for first iter

    if (config.implicit || config.local_time_stepping || config.mean_flow_point_implicit) {
        if (!cuda_check(cudaMalloc(&d_dt_cell, n_cells * sizeof(Real)), "cudaMalloc d_dt_cell", error)) goto fail;
    }
    if (config.implicit) {
        d_dq = nullptr; d_r_saved = nullptr; d_q_backup = nullptr;
        if (!cuda_check(cudaMalloc(&d_dq, nvar_cells * sizeof(Real)), "cudaMalloc d_dq", error)) goto fail;
        if (!cuda_check(cudaMemset(d_dq, 0, nvar_cells * sizeof(Real)), "zero d_dq", error)) goto fail;
        if (!cuda_check(cudaMalloc(&d_neg_r, nvar_cells * sizeof(Real)), "cudaMalloc d_neg_r", error)) goto fail;
        if (!cuda_check(cudaMalloc(&d_r_saved, nvar_cells * sizeof(Real)), "cudaMalloc d_r_saved", error)) goto fail;
        if (!cuda_check(cudaMalloc(&d_q_backup, nvar_cells * sizeof(Real)), "cudaMalloc d_q_backup", error)) goto fail;
        if (!cuda_check(cudaMalloc(&d_scratch, 4 * nvar_cells * sizeof(Real)), "cudaMalloc d_scratch", error)) goto fail;
        if (!cuda_check(cudaMalloc(&d_newton_accepted, sizeof(int)), "cudaMalloc d_newton_accepted", error)) goto fail;

        fgmres = new FgmresSolver(nvar_cells, config.fgmres_restart, config.fgmres_max_iter, config.fgmres_tol);
        if (!fgmres->allocate(error)) goto fail;

        lusgs = new LusgsPreconditioner();
        if (!lusgs->allocate(d_mesh, error)) goto fail;
    }

#ifdef MPI_ENABLED
    cudaStream_t stream_comm;
    cudaStreamCreate(&stream_comm);
    if (comm && gpu_part && comm->size() > 1) {
        d_mesh.set_partition(gpu_part);
    }
#else
    (void)comm;
    (void)gpu_part;
#endif

    Real inf_k = 0.0f, inf_omega = 0.0f;
    if (config.turbulence_model == TurbulenceModel::SST) {
        Real U_inf = real_sqrt(w_inf.u * w_inf.u + w_inf.v * w_inf.v + w_inf.w * w_inf.w);
        Real tu = 0.001f;
        inf_k = 1.5f * (tu * tu) * U_inf * U_inf;
        Real T_inf = w_inf.p / w_inf.rho;
        Real t_ratio = T_inf / config.T_ref;
        Real mu_inf = config.mu_ref * t_ratio * real_sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T_inf + config.sutherland_T);
        Real nu_inf = mu_inf / w_inf.rho;
        Real mu_t_mu_ratio = 0.1f;
        Real nu_t_inf = mu_t_mu_ratio * nu_inf;
        inf_omega = inf_k / (0.09f * nu_t_inf + 1e-30f);
        if (!d_mesh.has_sst() && !d_mesh.allocate_sst()) {
            if (error) *error = "allocate_sst failed";
            goto fail;
        }
        if (!compute_sst_init_gpu(d_mesh, inf_k, inf_omega, error, stream_main)) {
            goto fail;
        }
    }

    if (config.turbulence_model == TurbulenceModel::SST && d_mesh.has_sst()) {
        nvar_ncells = nvar_ncells_flow + 2 * n_cells;
    }

    if (config.viscous && !d_mesh.allocate_viscous()) {
        if (error) *error = "allocate_viscous failed";
        goto fail;
    }
    if (config.turbulence_model == TurbulenceModel::SA_DDES && !d_mesh.has_delta_ddes() && !d_mesh.allocate_ddes()) {
        if (error) *error = "allocate_ddes failed";
        goto fail;
    }

    // DIAG: verify initial state right before entering solve_gpu_impl
    { Real hq[6]; cudaMemcpy(hq, d_mesh.state_device(), 6*sizeof(Real), cudaMemcpyDeviceToHost);
      std::fprintf(stderr, "DIAG_in_solve_gpu: cell0 rho=%.6g rhou=%.6g rhoE=%.6g\n", hq[0], hq[1], hq[4]);
      int nc = static_cast<int>(d_mesh.cell_count()); int bad = -1;
      for (int i = 0; i < std::min(nc,1000); ++i) { Real r; cudaMemcpy(&r, d_mesh.state_device()+i*6, sizeof(Real), cudaMemcpyDeviceToHost); if (r <= 0) { bad = i; break; } }
      std::fprintf(stderr, "DIAG_in_solve_gpu: nc=%d first_bad=%d\n", nc, bad); }

    for (int iter = 0; iter < config.max_iter; ++iter) {
#ifdef MPI_ENABLED
        if (comm && comm->size() > 1 && d_mesh.has_halo()) {
            exchange_halo_gpu(d_mesh, *gpu_part, *comm, stream_comm);
            cudaStreamSynchronize(stream_comm);
        }
#endif

        // AMR trigger: at the top of each iteration (after iter 0),
        // before gradient/residual computation. Caller provides the
        // CPU-side AMR cycle via amr_callback; we handle GPU-side rebuild.
        if (config.amr.enabled && iter > 0 && config.amr.interval > 0 &&
                (iter % config.amr.interval) == 0 && host_mesh && host_state && amr_callback) {
            std::size_t n_cells_host = host_mesh->cells.size();
            host_state->resize(n_cells_host);
            if (!d_mesh.download_state(*host_state, error))
                goto fail;
            if (!amr_callback(*host_mesh, *host_state, iter, config, error))
                goto fail;
            bool mesh_changed = (host_mesh->cells.size() != n_cells_host);
            if (!mesh_changed) {
                if (!d_mesh.upload_state(*host_state, error))
                    goto fail;
                n_cells = static_cast<int>(d_mesh.cell_count());
                nvar_cells = n_cells * nvar;
            } else {
                if (!d_mesh.upload_mesh(*host_mesh, error))
                    goto fail;
                if (!d_mesh.upload_state(*host_state, error))
                    goto fail;
                int old_n_cells = n_cells;
                n_cells = static_cast<int>(d_mesh.cell_count());
                nvar_cells = n_cells * nvar;
                if (config.viscous && !d_mesh.allocate_viscous()) {
                    if (error) *error = "AMR re-allocate_viscous failed";
                    goto fail;
                }
                if (config.turbulence_model == TurbulenceModel::SA_DDES &&
                        !d_mesh.allocate_ddes()) {
                    if (error) *error = "AMR re-allocate_ddes failed";
                    goto fail;
                }
                if (config.turbulence_model == TurbulenceModel::SST) {
                    if (!d_mesh.allocate_sst()) {
                        if (error) *error = "AMR re-allocate_sst failed";
                        goto fail;
                    }
                    if (!compute_sst_init_gpu(d_mesh, inf_k, inf_omega, error, stream_main))
                        goto fail;
                }
                nvar_ncells = d_mesh.nvar() * n_cells;
                if (config.turbulence_model == TurbulenceModel::SST && d_mesh.has_sst()) {
                    nvar_ncells += 2 * n_cells;
                }
                if (config.implicit) {
                    if (!lusgs->rebuild_coloring(d_mesh, error)) goto fail;
                    if (!reallocate_implicit_buffers(nvar, old_n_cells, n_cells,
                            d_dq, d_dt_cell, d_neg_r, d_r_saved, d_q_backup,
                            d_scratch, d_newton_accepted, error))
                        goto fail;
                    delete fgmres;
                    fgmres = new FgmresSolver(nvar_cells, config.fgmres_restart,
                        config.fgmres_max_iter, config.fgmres_tol);
                    if (!fgmres->allocate(error)) goto fail;
                }
            }
            d_mesh.clear_residual(error);
        }

        if (config.reconstruction_order == 2 || config.viscous || (config.turbulence_model != TurbulenceModel::LAMINAR)) {
            if (!compute_gradients_gpu(d_mesh, config.gamma, error, d_failed, stream_main)) goto fail;
            if (config.reconstruction_order == 2 || (config.turbulence_model != TurbulenceModel::LAMINAR)) {
                if (!compute_limiters_gpu(d_mesh, config.gamma, error, d_failed, stream_main)) goto fail;
                if (!apply_limiter_gpu(d_mesh, false, error, stream_main)) goto fail;
            }
        }

        cudaStreamWaitEvent(stream_pre, event_update_done, 0);
        {
            const bool sa_dt = (config.turbulence_model == TurbulenceModel::SA ||
                               config.turbulence_model == TurbulenceModel::SA_DDES);
            Real cfl_now = config.cfl;
            if (config.implicit || config.cfl_ramp) {
                Real c0 = config.cfl_start > 0 ? config.cfl_start : config.cfl;
                Real c1 = config.cfl_end > c0 ? config.cfl_end : c0;
                int n_ramp = config.cfl_ramp_steps > 0 ? config.cfl_ramp_steps : 1;
                Real t = static_cast<Real>(iter) / static_cast<Real>(n_ramp);
                if (t > Real(1)) t = Real(1);
                cfl_now = c0 * real_pow(c1 / c0, t);
            }
            if (!compute_timestep_gpu(d_mesh, config.gamma, cfl_now, d_min_dt,
                    config.viscous, config.mu_ref, config.T_ref, config.sutherland_T, config.Re,
                    stream_pre, sa_dt)) {
                if (error) *error = "timestep kernel failed";
                goto fail;
            }
            if (d_dt_cell) {
                if (config.local_time_stepping) {
                    if (!compute_local_timestep_gpu(d_mesh, config.gamma, cfl_now, d_dt_cell,
                            config.viscous, config.mu_ref, config.T_ref, config.sutherland_T, config.Re,
                            error, stream_pre, sa_dt)) {
                        goto fail;
                    }
                    if (!cap_local_timestep_gpu(d_dt_cell, d_min_dt, config.lts_dt_ratio_max,
                            n_cells, error, stream_pre)) {
                        goto fail;
                    }
                } else {
                    // Global dt: every cell uses min_dt
                    if (!cap_local_timestep_gpu(d_dt_cell, d_min_dt, Real(1),
                            n_cells, error, stream_pre)) {
                        goto fail;
                    }
                }
            }
        }
        cudaEventRecord(event_timestep_done, stream_pre);
        cudaStreamWaitEvent(stream_main, event_timestep_done, 0);

        if (!launch_euler_residual_kernel(d_mesh, w_inf, config.gamma, d_failed, nullptr, error,
                config.reconstruction_order, stream_main)) {
            goto fail;
        }
        // DIAG: check d_failed right after Euler residual
        { int df = 0; cudaDeviceSynchronize(); cudaMemcpy(&df, d_failed, sizeof(int), cudaMemcpyDeviceToHost);
          if (df != 0) {
            Real dr=0, du=0, dv=0, dw=0, dp=0; int dc = -1;
            Real* d_diag_rho=nullptr, *d_diag_u=nullptr, *d_diag_v=nullptr, *d_diag_w=nullptr, *d_diag_p=nullptr; int* d_diag_cell=nullptr;
            cudaMalloc(&d_diag_rho, sizeof(Real)); cudaMalloc(&d_diag_u, sizeof(Real)); cudaMalloc(&d_diag_v, sizeof(Real));
            cudaMalloc(&d_diag_w, sizeof(Real)); cudaMalloc(&d_diag_p, sizeof(Real)); cudaMalloc(&d_diag_cell, sizeof(int));
            cudaMemset(d_diag_cell, 0xFF, sizeof(int));
            { int g = (n_cells+127)/128; d_find_invalid_cell<<<g,128>>>(d_mesh.state_device(), n_cells, nvar, config.gamma, d_diag_rho, d_diag_u, d_diag_v, d_diag_w, d_diag_p, d_diag_cell); cudaDeviceSynchronize(); }
            cudaMemcpy(&dc, d_diag_cell, sizeof(int), cudaMemcpyDeviceToHost);
            if (dc >= 0) {
                cudaMemcpy(&dr, d_diag_rho, sizeof(Real), cudaMemcpyDeviceToHost);
                cudaMemcpy(&du, d_diag_u, sizeof(Real), cudaMemcpyDeviceToHost);
                cudaMemcpy(&dv, d_diag_v, sizeof(Real), cudaMemcpyDeviceToHost);
                cudaMemcpy(&dw, d_diag_w, sizeof(Real), cudaMemcpyDeviceToHost);
                cudaMemcpy(&dp, d_diag_p, sizeof(Real), cudaMemcpyDeviceToHost);
                std::fprintf(stderr, "DIAG d_failed=1 after Euler residual (iter=%d) first bad cell=%d rho=%.6g u=%.6g v=%.6g w=%.6g p=%.6g\n", iter, dc, dr, du, dv, dw, dp);
            }
            cudaFree(d_diag_rho); cudaFree(d_diag_u); cudaFree(d_diag_v); cudaFree(d_diag_w); cudaFree(d_diag_p); cudaFree(d_diag_cell);
          }
        }

if (config.viscous) {
            if (!compute_viscous_flux_gpu(d_mesh, config.gamma, config.prandtl,
                    config.mu_ref, config.T_ref, config.sutherland_T,
                    config.Re, config.wall_temperature, static_cast<int>(config.turbulence_model), d_failed, stream_main)) {
                if (error) *error = "viscous flux kernel failed";
                goto fail;
            }
        }

        if (!compute_turbulence_source_gpu(d_mesh, config, d_failed, error, stream_main, inf_k, inf_omega, w_inf.rho, d_min_dt)) {
            if (error && error->empty()) *error = "turbulence source kernel failed";
            goto fail;
        }

        // Spatial residual L2 before point-implicit (steady convergence metric).
        // For implicit Newton path, residual is re-normed later from d_r_saved.
        if (!config.implicit) {
            if (!dnrm2_gpu(d_mesh.residual_device(), nvar_cells, d_l2_sum, stream_main)) {
                if (error) *error = "spatial residual L2 failed"; goto fail;
            }
            if (config.turbulence_model == TurbulenceModel::SST && d_mesh.has_sst()) {
                if (!dnrm2_gpu(d_mesh.residual_k_device(), n_cells, d_l2_sum + 2, stream_main)) {
                    if (error) *error = "SST k spatial L2 failed"; goto fail;
                }
                if (!dnrm2_gpu(d_mesh.residual_omega_device(), n_cells, d_l2_sum + 3, stream_main)) {
                    if (error) *error = "SST omega spatial L2 failed"; goto fail;
                }
                sst_l2_accumulate_kernel<<<1, 1, 0, stream_main>>>(d_l2_sum, d_l2_sum + 2);
                if (!cuda_check(cudaGetLastError(), "sst_l2_accumulate spatial", error)) goto fail;
            }
            check_status_kernel<<<1, 1, 0, stream_main>>>(
                d_failed, d_l2_sum, nvar_ncells,
                config.convergence_tol, d_residual_history + iter);
            if (!cuda_check(cudaGetLastError(), "check_status spatial residual", error)) goto fail;
        }

        if (!config.implicit && config.mean_flow_point_implicit && d_dt_cell) {
            const bool sa_pi = (config.turbulence_model == TurbulenceModel::SA ||
                                config.turbulence_model == TurbulenceModel::SA_DDES);
            if (!apply_mean_flow_point_implicit_gpu(d_mesh, d_dt_cell, config.gamma,
                    config.viscous, config.mu_ref, config.T_ref, config.sutherland_T,
                    config.Re, sa_pi, error, stream_main)) {
                goto fail;
            }
        }

        if (config.turbulence_model != TurbulenceModel::LAMINAR &&
                config.turbulence_model != TurbulenceModel::SST && !config.implicit) {
            if (d_dt_cell) {
                if (!apply_rans_implicit_per_cell_gpu(d_mesh, config.Re, d_dt_cell, error, stream_main)) {
                    if (error && error->empty()) *error = "RANS implicit per-cell kernel failed";
                    goto fail;
                }
            } else if (!apply_rans_implicit_gpu(d_mesh, config.Re, d_min_dt, error, stream_main)) {
                if (error && error->empty()) *error = "RANS implicit kernel failed";
                goto fail;
            }
        }

        if (config.implicit) {
            Real cfl_ramp = config.cfl_start * real_pow(config.cfl_end / config.cfl_start,
                static_cast<Real>(iter) / static_cast<Real>(config.cfl_ramp_steps > 0 ? config.cfl_ramp_steps : 1));
            cfl_ramp = real_fmin(cfl_ramp, config.cfl_end);
            cfl_ramp *= cfl_multiplier;

            {
                const bool sa_dt = (config.turbulence_model == TurbulenceModel::SA ||
                                   config.turbulence_model == TurbulenceModel::SA_DDES);
                if (!compute_local_timestep_gpu(d_mesh, config.gamma, cfl_ramp, d_dt_cell,
                        config.viscous, config.mu_ref, config.T_ref, config.sutherland_T, config.Re,
                        error, stream_main, sa_dt)) {
                    goto fail;
                }
            }
            { int df = 0; cudaMemcpy(&df, d_failed, sizeof(int), cudaMemcpyDeviceToHost);
              if (df != 0) std::fprintf(stderr, "DIAG d_failed=1 after local timestep (iter=%d)\n", iter); }

            if (!lusgs->compute_diagonal(d_mesh, d_dt_cell, config.gamma,
                    config.viscous, d_mesh.mu_device(), config.Re, error)) {
                goto fail;
            }
            { int df = 0; cudaMemcpy(&df, d_failed, sizeof(int), cudaMemcpyDeviceToHost);
              if (df != 0) std::fprintf(stderr, "DIAG d_failed=1 after compute_diagonal (iter=%d)\n", iter); }

            if (!cuda_check(cudaMemcpy(d_r_saved, d_mesh.residual_device(),
                    nvar_cells * sizeof(Real), cudaMemcpyDeviceToDevice), "save raw R(Q^n)", error)) goto fail;

            if (config.turbulence_model != TurbulenceModel::LAMINAR &&
                    config.turbulence_model != TurbulenceModel::SST) {
                if (!apply_rans_implicit_per_cell_gpu(d_mesh, config.Re, d_dt_cell, error, stream_main)) {
                    if (error && error->empty()) *error = "RANS implicit per-cell kernel failed";
                    goto fail;
                }
            }

            if (!dcopy_gpu(d_mesh.residual_device(), d_neg_r, nvar_cells, stream_main)) { if (error) *error = "copy R failed"; goto fail; }
            if (!dscal_gpu(-1, d_neg_r, nvar_cells, stream_main)) { if (error) *error = "negate R failed"; goto fail; }

            if (!dnrm2_gpu(d_r_saved, nvar_cells, d_l2_sum, stream_main)) { if (error) *error = "L2 norm failed"; goto fail; }
            if (config.turbulence_model == TurbulenceModel::SST && d_mesh.has_sst()) {
                if (!dnrm2_gpu(d_mesh.residual_k_device(), n_cells, d_l2_sum + 2, stream_main)) { if (error) *error = "SST k L2 norm failed"; goto fail; }
                if (!dnrm2_gpu(d_mesh.residual_omega_device(), n_cells, d_l2_sum + 3, stream_main)) { if (error) *error = "SST omega L2 norm failed"; goto fail; }
                sst_l2_accumulate_kernel<<<1, 1, 0, stream_main>>>(d_l2_sum, d_l2_sum + 2);
                if (!cuda_check(cudaGetLastError(), "sst_l2_accumulate kernel", error)) goto fail;
            }
            init_l2_old_kernel<<<1, 1, 0, stream_main>>>(d_l2_sum, nvar_ncells);
            if (!cuda_check(cudaGetLastError(), "init_l2_old kernel launch", error)) goto fail;

            // Adaptive JFV epsilon: eps = sqrt(machine_eps) * max(1, ||q||_RMS)
            // Krylov vectors are L2-unit-normalized by FGMRES, so ||v||_2 ≈ 1,
            // making the standard FD-optimal epsilon = sqrt(eps_mach) * scale.
            // The state-RMS scaling ensures proper perturbation magnitudes across
            // different flow regimes (e.g., low-density vs high-enthalpy).
            Real eps_jfv = real_sqrt(std::numeric_limits<Real>::epsilon());
            {
                if (!dnrm2_gpu(d_mesh.state_device(), nvar_cells, d_l2_sum, stream_main)) {
                    if (error) *error = "JFV norm_q failed"; goto fail;
                }
                Real state_norm_sq = 0;
                if (!cuda_check(cudaMemcpy(&state_norm_sq, d_l2_sum, sizeof(Real),
                        cudaMemcpyDeviceToHost), "read state_norm_sq", error)) goto fail;
                Real state_rms = real_sqrt(state_norm_sq / static_cast<Real>(nvar_cells > 0 ? nvar_cells : 1));
                eps_jfv *= real_fmax(Real(1.0), state_rms);
                eps_jfv = real_fmin(eps_jfv, Real(0.01));
            }

            // Save pre-Newton state for CFL retry on Newton failure.
            Real* d_q_before_newton = d_scratch + 3 * nvar_cells;
            if (!dcopy_gpu(d_mesh.state_device(), d_q_before_newton, nvar_cells, stream_main)) {
                if (error) *error = "save Q before Newton failed"; goto fail;
            }

            bool newt_converged = false;

            for (int newt = 0; newt < config.newton_max_iter; ++newt) {
                if (!cuda_check(cudaMemcpy(d_q_backup, d_mesh.state_device(),
                        nvar_cells * sizeof(Real), cudaMemcpyDeviceToDevice), "backup Q", error)) goto fail;

                auto matvec = [&](const Real* v, Real* w, std::string* err) -> bool {
                    return compute_jfv_product(d_mesh, v, w, d_r_saved, eps_jfv, config, w_inf, d_scratch, d_failed, err, stream_main);
                };
                auto prec = [&](const Real* v, Real* z, std::string* err) -> bool {
                    return lusgs->apply(d_mesh, v, z, config.gamma, err);
                };

                fgmres->set_preconditioner(prec);
                if (!fgmres->solve(matvec, d_neg_r, d_dq, error)) {
                    if (error) *error = "FGMRES solve failed in Newton iteration";
                    goto fail;
                }
                { int df = 0; cudaMemcpy(&df, d_failed, sizeof(int), cudaMemcpyDeviceToHost);
                  if (df != 0) std::fprintf(stderr, "DIAG d_failed=1 after FGMRES solve (iter=%d newt=%d)\n", iter, newt); }

                {
                    int btry = 0;
                    Real* d_dq_full = d_scratch + 2 * nvar_cells;
                    while (btry < 8) {
                        if (btry == 0) {
                            if (!dcopy_gpu(d_dq, d_dq_full, nvar_cells, stream_main)) {
                                if (error) *error = "Newton save dq_full failed";
                                goto fail;
                            }
                        } else {
                            if (!dscal_gpu(0.5f, d_dq_full, nvar_cells, stream_main)) {
                                if (error) *error = "Newton backtrack scale failed";
                                goto fail;
                            }
                            if (!dcopy_gpu(d_dq_full, d_dq, nvar_cells, stream_main)) {
                                if (error) *error = "Newton restore dq_full failed";
                                goto fail;
                            }
                        }
                        {   cudaMemsetAsync(d_failed, 0, sizeof(int), stream_main);
                            int dq_g = (nvar_cells + 127) / 128;
                            check_array_valid_kernel<<<dq_g, 128, 0, stream_main>>>(d_dq, nvar_cells, d_failed);
                            int dq_invalid = 0; cudaMemcpy(&dq_invalid, d_failed, sizeof(int), cudaMemcpyDeviceToHost);
                            if (dq_invalid) {
                                cudaMemsetAsync(d_dq, 0, nvar_cells * sizeof(Real), stream_main);
                                cudaMemsetAsync(d_failed, 0, sizeof(int), stream_main);
                            }
                        }
                        if (!daxpy_gpu(1, d_dq, d_mesh.state_device(), nvar_cells, stream_main)) {
                            if (error) *error = "Q += dq failed"; goto fail;
                        }

                        d_mesh.clear_residual(error);
                        if (!launch_euler_residual_kernel(d_mesh, w_inf, config.gamma, d_failed,
                                nullptr, error, config.reconstruction_order, stream_main)) {
                            goto fail;
                        }
                        { int df = 0; cudaMemcpy(&df, d_failed, sizeof(int), cudaMemcpyDeviceToHost);
                          if (df != 0) std::fprintf(stderr, "DIAG d_failed=1 after Newton Euler residual (iter=%d newt=%d btry=%d)\n", iter, newt, btry); }
                        if (config.viscous) {
                            if (!compute_viscous_flux_gpu(d_mesh, config.gamma, config.prandtl,
                                    config.mu_ref, config.T_ref, config.sutherland_T,
                    config.Re, config.wall_temperature, static_cast<int>(config.turbulence_model), d_failed, stream_main)) {
                                if (error) *error = "Newton viscous flux failed";
                                goto fail;
                            }
                        }
                        if (!compute_turbulence_source_gpu(d_mesh, config, d_failed, error, stream_main, inf_k, inf_omega, w_inf.rho)) {
                            if (error && error->empty()) *error = "Newton turbulence source failed";
                            goto fail;
                        }

                        if (!dnrm2_gpu(d_mesh.residual_device(), nvar_cells, d_l2_sum, stream_main)) {
                            if (error) *error = "new L2 norm failed"; goto fail;
                        }
                        if (config.turbulence_model == TurbulenceModel::SST && d_mesh.has_sst()) {
                            if (!dnrm2_gpu(d_mesh.residual_k_device(), n_cells, d_l2_sum + 2, stream_main)) { if (error) *error = "SST k L2 norm failed"; goto fail; }
                            if (!dnrm2_gpu(d_mesh.residual_omega_device(), n_cells, d_l2_sum + 3, stream_main)) { if (error) *error = "SST omega L2 norm failed"; goto fail; }
                            sst_l2_accumulate_kernel<<<1, 1, 0, stream_main>>>(d_l2_sum, d_l2_sum + 2);
                            if (!cuda_check(cudaGetLastError(), "sst_l2_accumulate kernel", error)) goto fail;
                        }
                        newton_l2_check_kernel<<<1, 1, 0, stream_main>>>(
                            d_l2_sum, nvar_ncells, config.newton_sufficient_decrease, d_newton_accepted);
                        if (!cuda_check(cudaGetLastError(), "newton L2 check kernel launch", error)) goto fail;
                        int accepted = 0;
                        if (!cuda_check(cudaMemcpy(&accepted, d_newton_accepted, sizeof(int), cudaMemcpyDeviceToHost), "read accepted", error)) goto fail;

                        if (accepted) {
                            if (!dcopy_gpu(d_mesh.residual_device(), d_r_saved, nvar_cells, stream_main)) {
                                if (error) *error = "update saved R failed"; goto fail;
                            }
                            newt_converged = true;
                            goto newton_accepted;
                        }

                        if (!daxpy_gpu(-1, d_dq, d_mesh.state_device(), nvar_cells, stream_main)) {
                            if (error) *error = "Newton backtrack restore failed"; goto fail;
                        }
                        ++btry;
                    }
                }
            }
newton_accepted:
            ;

            if (!newt_converged) {
                // Newton failed (backtracking exhausted). Restore pre-Newton state and
                // reduce CFL for the next outer iteration. The pre-Newton residual is
                // already in d_r_saved, so the L2 check below will show no progress
                // (but also no divergence). Future iterations try again at lower CFL.
                { int df = 0; cudaMemcpy(&df, d_failed, sizeof(int), cudaMemcpyDeviceToHost);
                  if (df != 0) std::fprintf(stderr, "DIAG d_failed=1 after Newton failed (iter=%d)\n", iter); }
                if (!dcopy_gpu(d_q_before_newton, d_mesh.state_device(), nvar_cells, stream_main)) {
                    if (error) *error = "Newton restore Q failed"; goto fail;
                }
                cfl_multiplier *= 0.5f;
            }

            if (!dnrm2_gpu(d_r_saved, nvar_cells, d_l2_sum, stream_main)) {
                if (error) *error = "final L2 norm failed"; goto fail;
            }
        } else {
            if (!compute_update_gpu(d_mesh, d_min_dt, config.gamma, d_l2_sum, d_failed,
                    d_failure_cell, d_failure_state, stream_main, d_dt_cell)) {
                if (error) *error = "update kernel failed";
                goto fail;
            }
            // Spatial residual already written to history before PI; do not overwrite.
        }

        if (config.implicit) {
            if (config.turbulence_model == TurbulenceModel::SST && d_mesh.has_sst()) {
                if (!dnrm2_gpu(d_mesh.residual_k_device(), n_cells, d_l2_sum + 2, stream_main)) { if (error) *error = "SST k L2 norm failed"; goto fail; }
                if (!dnrm2_gpu(d_mesh.residual_omega_device(), n_cells, d_l2_sum + 3, stream_main)) { if (error) *error = "SST omega L2 norm failed"; goto fail; }
                sst_l2_accumulate_kernel<<<1, 1, 0, stream_main>>>(d_l2_sum, d_l2_sum + 2);
                if (!cuda_check(cudaGetLastError(), "sst_l2_accumulate kernel", error)) goto fail;
            }
            { int df = 0; Real l2 = 0;
              cudaMemcpy(&df, d_failed, sizeof(int), cudaMemcpyDeviceToHost);
              cudaMemcpy(&l2, d_l2_sum, sizeof(Real), cudaMemcpyDeviceToHost);
              if (df != 0 || !real_isfinite(l2))
                  std::fprintf(stderr, "DIAG before check_status: d_failed=%d l2=%.6e (iter=%d)\n", df, l2, iter); }
            check_status_kernel<<<1, 1, 0, stream_main>>>(
                d_failed, d_l2_sum, nvar_ncells,
                config.convergence_tol, d_residual_history + iter);
            if (!cuda_check(cudaGetLastError(), "check_status kernel launch", error)) goto fail;
        }

        if (diagnostics_enabled) {
            if (!compute_state_bounds_gpu(d_mesh, config.gamma, d_state_bounds_history + iter * 6, stream_main)) {
                if (error) *error = "state bounds kernel failed";
                goto fail;
            }
        }

        cudaEventRecord(event_update_done, stream_main);

#ifdef MPI_ENABLED
        if (d_mesh.has_halo()) {
            cudaStreamSynchronize(stream_comm);
        }
        if (comm && comm->size() > 1) {
            comm->barrier(nullptr);
        }
#endif
    }

    if (!cuda_check(cudaDeviceSynchronize(), "post-loop sync", error)) goto fail;

    if (!cuda_check(cudaMemcpy(&host_failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost), "read d_failed", error)) goto fail;

    host_residual_history.assign(config.max_iter, 0.0f);
    if (!cuda_check(cudaMemcpy(host_residual_history.data(), d_residual_history,
            config.max_iter * sizeof(Real), cudaMemcpyDeviceToHost), "read residual history", error)) goto fail;

    if (host_failed != 0) {
        if (error) *error = "GPU solver failed during iteration";
        summary.failed = true;
    }

    {
        int valid_iters = 0;
        for (int i = 0; i < config.max_iter; ++i) {
            if (host_residual_history[i] < 0.0f) break;
            summary.residual_history.push_back(host_residual_history[i]);
            valid_iters++;
        }
        if (valid_iters > 0 && host_residual_history[valid_iters - 1] < config.convergence_tol) {
            summary.converged = true;
        }
    }

    if (diagnostics_enabled) {
        std::vector<Real> bounds_host(config.max_iter * 6);
        if (!cuda_check(cudaMemcpy(bounds_host.data(), d_state_bounds_history,
                config.max_iter * 6 * sizeof(Real), cudaMemcpyDeviceToHost), "read state bounds history", error)) goto fail;

        for (int i = 0; i < config.max_iter; ++i) {
            StateBounds sb;
            sb.min_rho = bounds_host[i * 6 + 0];
            sb.max_rho = bounds_host[i * 6 + 1];
            sb.min_p = bounds_host[i * 6 + 2];
            sb.max_p = bounds_host[i * 6 + 3];
            sb.min_mach = bounds_host[i * 6 + 4];
            sb.max_mach = bounds_host[i * 6 + 5];
            sb.valid = true;
            summary.diagnostics.state_bounds_history.push_back(sb);
        }

        if (host_failed != 0 && d_failure_cell) {
            int host_failure_cell = -1;
            Real host_failure_state[5] = {0.0f};
            if (!cuda_check(cudaMemcpy(&host_failure_cell, d_failure_cell, sizeof(int), cudaMemcpyDeviceToHost), "read d_failure_cell", error)) goto fail;
            if (!cuda_check(cudaMemcpy(host_failure_state, d_failure_state, 5 * sizeof(Real), cudaMemcpyDeviceToHost), "read d_failure_state", error)) goto fail;

            if (host_failure_cell >= 0) {
                int fail_iter = 0;
                for (int i = 0; i < config.max_iter; ++i) {
                    if (host_residual_history[i] < 0.0f) { fail_iter = i; break; }
                }
                ConservativeState fail_q;
                fail_q.rho = host_failure_state[0];
                fail_q.rho_u = host_failure_state[1];
                fail_q.rho_v = host_failure_state[2];
                fail_q.rho_w = host_failure_state[3];
                fail_q.rho_E = host_failure_state[4];
                summary.diagnostics.failure = make_failure_snapshot(
                    fail_iter + 1, host_failure_cell, "NaN/non-positive state", fail_q, config.gamma);
            }
        }
    }

    if (!summary.failed) {
        if (!compute_wall_forces_gpu(d_mesh, config.gamma, d_forces,
                config.viscous, config.prandtl, config.mu_ref, config.T_ref,
                config.sutherland_T, config.Re, config.wall_temperature)) {
            if (error) *error = "wall force kernel failed";
            goto fail;
        }
        if (!cuda_check(cudaDeviceSynchronize(), "wall force sync", error)) goto fail;

        Real forces[7];
        if (!cuda_check(cudaMemcpy(forces, d_forces, 7 * sizeof(Real), cudaMemcpyDeviceToHost), "read d_forces", error)) goto fail;

        Real q_inf = 0.5f * condition.mach * condition.mach;
        Real inv_force_ref = 1.0f / real_fmax(q_inf * config.ref_area, 1e-30f);
        summary.forces.CX = forces[0] * inv_force_ref;
        summary.forces.CY = forces[1] * inv_force_ref;
        summary.forces.CZ = forces[2] * inv_force_ref;
        summary.forces.Cl = forces[3] / real_fmax(q_inf * config.ref_area * config.ref_span, 1e-30f);
        summary.forces.Cm = forces[4] / real_fmax(q_inf * config.ref_area * config.ref_length, 1e-30f);
        summary.forces.Cn = forces[5] / real_fmax(q_inf * config.ref_area * config.ref_area, 1e-30f);
        summary.forces.Q_wall = forces[6] * inv_force_ref;

        constexpr Real kPi = 3.14159265358979323846;
        Real alpha = condition.alpha_deg * kPi / 180.0f;
        Real beta = condition.beta_deg * kPi / 180.0f;
        Real ca = real_cos(alpha);
        Real sa = real_sin(alpha);
        Real cb = real_cos(beta);
        Real sb = real_sin(beta);
        Real fsx = summary.forces.CX * ca * cb + summary.forces.CY * sb + summary.forces.CZ * sa * cb;
        Real fsz = -summary.forces.CX * sa + summary.forces.CZ * ca;
        summary.forces.CD = -fsx;
        summary.forces.CL = -fsz;

        summary.forces.iterations = static_cast<int>(summary.residual_history.size());
        summary.forces.residual = summary.residual_history.empty() ? 0.0f : summary.residual_history.back();
        const char* tm_str = "laminar";
        if (config.turbulence_model == TurbulenceModel::SA) tm_str = "rans-sa";
        else if (config.turbulence_model == TurbulenceModel::SA_DDES) tm_str = "rans-sa-ddes";
        else if (config.turbulence_model == TurbulenceModel::SST) tm_str = "rans-sst";
        summary.forces.turbulence_model = tm_str;
        summary.forces.fidelity = "cfd-gpu";
    }

    goto cleanup;

fail:
    summary.failed = true;

cleanup:
    d_mesh.set_partition(nullptr);
    if (fgmres) { delete fgmres; fgmres = nullptr; }
    if (lusgs) { delete lusgs; lusgs = nullptr; }
    cuda_free_safe(d_dq);
    cuda_free_safe(d_dt_cell);
    cuda_free_safe(d_neg_r);
    cuda_free_safe(d_r_saved);
    cuda_free_safe(d_q_backup);
    cuda_free_safe(d_scratch);
    cuda_free_safe(d_newton_accepted);
    cudaEventDestroy(event_timestep_done);
    cudaEventDestroy(event_update_done);
    cudaStreamDestroy(stream_pre);
    cudaStreamDestroy(stream_main);
#ifdef MPI_ENABLED
    cudaStreamDestroy(stream_comm);
#endif
    if (owned_buffers) solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history,
        d_state_bounds_history, d_failure_cell, d_failure_state);
    return summary;
}

CfdSolveSummary solve_gpu(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    std::string* error,
    CfdMesh* host_mesh,
    std::vector<ConservativeState>* host_state,
    AmrCycleCallback amr_callback) {
    if (d_mesh.cell_count() == 0 || d_mesh.face_count() == 0) {
        CfdSolveSummary s;
        if (error) *error = "DeviceMesh is empty";
        s.failed = true;
        return s;
    }

    int* d_failed = nullptr;
    Real* d_min_dt = nullptr;
    Real* d_l2_sum = nullptr;
    Real* d_forces = nullptr;
    Real* d_residual_history = nullptr;
    Real* d_state_bounds_history = nullptr;
    int* d_failure_cell = nullptr;
    Real* d_failure_state = nullptr;

    if (!cuda_check(cudaMalloc(&d_failed, sizeof(int)), "cudaMalloc d_failed", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_min_dt, sizeof(Real)), "cudaMalloc d_min_dt", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_l2_sum, 4 * sizeof(Real)), "cudaMalloc d_l2_sum", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_forces, 7 * sizeof(Real)), "cudaMalloc d_forces", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    if (!cuda_check(cudaMalloc(&d_residual_history, config.max_iter * sizeof(Real)), "cudaMalloc d_residual_history", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }

    bool diag = config.diagnostic_level != DiagnosticLevel::Off;
    if (diag) {
        if (!cuda_check(cudaMalloc(&d_state_bounds_history, config.max_iter * 6 * sizeof(Real)), "cudaMalloc d_state_bounds_history", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_cell, sizeof(int)), "cudaMalloc d_failure_cell", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_state, 5 * sizeof(Real)), "cudaMalloc d_failure_state", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMemset(d_failure_cell, 0xFF, sizeof(int)), "init d_failure_cell", error)) { solve_gpu_free(d_failed, d_min_dt, d_l2_sum, d_forces, d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    }

    return solve_gpu_impl(d_mesh, condition, config, d_failed, d_min_dt, d_l2_sum, d_forces,
        d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state, true, error,
        nullptr, nullptr, host_mesh, host_state, amr_callback);
}

CfdSolveSummary solve_gpu(
    DeviceMesh& d_mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    int* d_failed,
    Real* d_min_dt,
    Real* d_l2_sum,
    Real* d_forces,
    std::string* error,
    CfdMesh* host_mesh,
    std::vector<ConservativeState>* host_state,
    AmrCycleCallback amr_callback) {
    if (d_mesh.cell_count() == 0 || d_mesh.face_count() == 0) {
        CfdSolveSummary s;
        if (error) *error = "DeviceMesh is empty";
        s.failed = true;
        return s;
    }

    Real* d_residual_history = nullptr;
    Real* d_state_bounds_history = nullptr;
    int* d_failure_cell = nullptr;
    Real* d_failure_state = nullptr;

    if (!cuda_check(cudaMalloc(&d_residual_history, config.max_iter * sizeof(Real)), "cudaMalloc d_residual_history", error)) { cuda_free_safe(d_residual_history); CfdSolveSummary s; s.failed = true; return s; }

    bool diag = config.diagnostic_level != DiagnosticLevel::Off;
    if (diag) {
        if (!cuda_check(cudaMalloc(&d_state_bounds_history, config.max_iter * 6 * sizeof(Real)), "cudaMalloc d_state_bounds_history", error)) { cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_cell, sizeof(int)), "cudaMalloc d_failure_cell", error)) { cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history); cuda_free_safe(d_failure_cell); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMalloc(&d_failure_state, 5 * sizeof(Real)), "cudaMalloc d_failure_state", error)) { cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history); cuda_free_safe(d_failure_cell); cuda_free_safe(d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
        if (!cuda_check(cudaMemset(d_failure_cell, 0xFF, sizeof(int)), "init d_failure_cell", error)) { cuda_free_safe(d_residual_history); cuda_free_safe(d_state_bounds_history); cuda_free_safe(d_failure_cell); cuda_free_safe(d_failure_state); CfdSolveSummary s; s.failed = true; return s; }
    }

    CfdSolveSummary result = solve_gpu_impl(d_mesh, condition, config, d_failed, d_min_dt, d_l2_sum, d_forces,
        d_residual_history, d_state_bounds_history, d_failure_cell, d_failure_state, false, error,
        nullptr, nullptr, host_mesh, host_state, amr_callback);
    cuda_free_safe(d_residual_history);
    cuda_free_safe(d_state_bounds_history);
    cuda_free_safe(d_failure_cell);
    cuda_free_safe(d_failure_state);
    return result;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp




