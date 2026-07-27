#include "aero/cfd/cfd_solver_gpu.hpp"
#include "aero/cfd/gpu_solver.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include "aero/cfd/lusgs.hpp"
#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/amr_sensor.hpp"
#include "aero/cfd/amr_interpolate.hpp"
#include "aero/cfd/reconstruction.hpp"
#include "aero/cfd/diagnostics.hpp"
#include "aero/cfd/rans.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

// Persist prev_records across gpu_amr_cycle calls within a single solve.
std::vector<RefinementRecord> g_gpu_amr_prev_records;

void reset_gpu_amr_prev_records() {
    g_gpu_amr_prev_records.clear();
}

// Host-side AMR cycle callback for solve_gpu.
// Runs sensors → merges → refines → prolongates → rebuilds metrics.
bool gpu_amr_cycle(CfdMesh& mesh, std::vector<ConservativeState>& q,
    int iter, const CfdConfig& config, std::string* error) {
    (void)iter;
    CfdMesh mesh_old = mesh;
    std::vector<ConservativeState> q_old = q;

    std::vector<std::vector<RefinementRequest>> sensor_outputs;
    sensor_outputs.push_back(
        compute_gradient_sensor(mesh, q, config.amr, config.gamma));

    if (config.turbulence_model != TurbulenceModel::LAMINAR) {
        if (config.amr.yplus_target > Real(0)) {
            sensor_outputs.push_back(
                compute_yplus_sensor(mesh, q, config.amr, config.gamma,
                    config.Re, config.mu_ref, config.T_ref,
                    config.sutherland_T, config.turbulence_model));
        }
        sensor_outputs.push_back(
            compute_qcriterion_sensor(mesh, q, config.amr, config.gamma));
        if (config.amr.wake_cone.length > Real(0)) {
            sensor_outputs.push_back(
                compute_wake_cone_sensor(mesh, config.amr, config.amr.wake_cone));
        }
        if (config.amr.tke_ratio_threshold > Real(0)) {
            sensor_outputs.push_back(
                compute_tke_ratio_sensor(mesh, q, config.amr, config.gamma,
                    config.turbulence_model, nullptr,
                    config.amr.tke_ratio_threshold));
        }
        if (config.amr.shear_layer_threshold > Real(0)) {
            sensor_outputs.push_back(
                compute_shear_layer_sensor(mesh, q, config.amr, config.gamma,
                    config.turbulence_model, nullptr,
                    config.amr.shear_layer_threshold));
        }
    }

    auto requests = merge_refinement_requests(sensor_outputs);

    std::vector<RefinementRecord> new_records;
    if (!refine_cells(mesh, requests, &new_records, error,
            g_gpu_amr_prev_records.empty() ? nullptr : &g_gpu_amr_prev_records,
            nullptr, config.amr.max_level))
        return false;

    g_gpu_amr_prev_records = std::move(new_records);

    // 1st-order prolongation (2nd-order needs CPU gradients not available here)
    prolongate_solution(q_old, mesh_old, mesh, g_gpu_amr_prev_records, q);

    compute_mesh_metrics(mesh, true);
    return true;
}

} // anonymous namespace

CfdSolveSummary solve_gpu_dispatch(
    const CfdMesh& mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    std::string* error) {

    // Shared: freestream state for initialization
    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    w_inf.nu_tilde = condition.nu_tilde;
    if (condition.nu_tilde_ratio > 0.0f && config.viscous) {
        Real T_inf = w_inf.p / w_inf.rho;
        Real t_ratio = T_inf / config.T_ref;
        Real mu_inf = config.mu_ref * t_ratio * std::sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T_inf + config.sutherland_T);
        w_inf.nu_tilde = sa_freestream_nu_tilde(condition.nu_tilde_ratio, mu_inf, w_inf.rho, config.Re);
    }
    ConservativeState q_inf = primitive_to_conservative(w_inf, config.gamma);
    std::vector<ConservativeState> q(mesh.cells.size(), q_inf);

    // GPU path with optional AMR integration.
    // When amr.enabled, pass mutable host mesh/state and the AMR callback
    // so solve_gpu can trigger CPU-side refinement cycles internally.
    DeviceMesh d_mesh;
    if (!d_mesh.upload_mesh(mesh, error)) {
        CfdSolveSummary s;
        s.failed = true;
        return s;
    }
    if (!d_mesh.upload_state(q, error)) {
        CfdSolveSummary s;
        s.failed = true;
        return s;
    }
    CfdSolveSummary gpu_result;
    if (config.amr.enabled) {
        reset_gpu_amr_prev_records();
        CfdMesh mutable_mesh = mesh;
        gpu_result = solve_gpu(d_mesh, condition, config, error,
            &mutable_mesh, &q, gpu_amr_cycle);
    } else {
        gpu_result = solve_gpu(d_mesh, condition, config, error);
    }

    if (config.cpu_oracle && !gpu_result.failed) {
        CfdConfig cpu_cfg = config;
        cpu_cfg.use_gpu = false;
        CfdSolver cpu_solver;
        if (!cpu_solver.load_mesh(mesh)) {
            if (error) *error = "cpu_oracle: load_mesh failed";
            CfdSolveSummary s;
            s.failed = true;
            return s;
        }
        CfdSolveSummary cpu_result = cpu_solver.solve_from_state(condition, cpu_cfg, q);
        std::string oracle_error;
        if (!assert_oracle_equivalent(gpu_result, cpu_result, 1e-6f, 1e-6f, &oracle_error)) {
            std::fprintf(stderr, "[CPU Oracle] FAIL: %s\n", oracle_error.c_str());
            if (error) *error = oracle_error;
            CfdSolveSummary s;
            s.failed = true;
            return s;
        }
    }

    return gpu_result;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp