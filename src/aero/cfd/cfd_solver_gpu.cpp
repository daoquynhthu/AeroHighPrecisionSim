#include "aero/cfd/cfd_solver_gpu.hpp"
#include "aero/cfd/gpu_solver.hpp"
#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/gpu_solver_internal.hpp"
#include "aero/cfd/lusgs.hpp"
#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/amr_sensor.hpp"
#include "aero/cfd/amr_interpolate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace aerosp {
namespace aero {
namespace cfd {

CfdSolveSummary solve_gpu_dispatch(
    const CfdMesh& mesh,
    const FreestreamCondition& condition,
    const CfdConfig& config,
    std::string* error) {

    // AMR requires host-side mesh refinement. For explicit+AMR, use the
    // CPU solver path which handles AMR correctly (the solver iterate
    // between AMR cycles runs on CPU). For implicit+AMR, GPU solver
    // infrastructure (rebuild_coloring, JFV realloc) is implemented but
    // the full integration loop is deferred.
    if (config.amr.enabled) {
        if (config.implicit) {
            if (error) *error =
                "GPU implicit solver with AMR is not yet supported. "
                "Use explicit mode or set use_gpu=false to use the CPU "
                "solver which handles AMR + implicit.";
            CfdSolveSummary s;
            s.failed = true;
            return s;
        }

        // For explicit+AMR: use CPU solver with AMR (functionally correct,
        // GPU acceleration of the stencil between AMR cycles deferred).
        // The solver below handles the full AMR loop on CPU:
        //   sensor -> refine -> prolongate -> rebuild -> iterate -> repeat
        CfdSolver cpu_solver;
        if (!cpu_solver.load_mesh(mesh)) {
            if (error) *error = "solve_gpu_dispatch(AMR): load_mesh failed";
            CfdSolveSummary s;
            s.failed = true;
            return s;
        }
        CfdConfig cpu_cfg = config;
        cpu_cfg.use_gpu = false;
        PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
        w_inf.nu_tilde = condition.nu_tilde;
        if (condition.nu_tilde_ratio > 0.0f && config.viscous) {
            Real T_inf = w_inf.p / w_inf.rho;
            Real t_ratio = T_inf / config.T_ref;
            Real mu_inf = config.mu_ref * t_ratio * std::sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T_inf + config.sutherland_T);
            w_inf.nu_tilde = condition.nu_tilde_ratio * mu_inf / w_inf.rho;
        }
        ConservativeState q_inf = primitive_to_conservative(w_inf, config.gamma);
        std::vector<ConservativeState> q(mesh.cells.size(), q_inf);
        return cpu_solver.solve_from_state(condition, cpu_cfg, q);
    }

    // Standard GPU path (no AMR)
    DeviceMesh d_mesh;
    if (!d_mesh.upload_mesh(mesh, error)) {
        CfdSolveSummary s;
        s.failed = true;
        return s;
    }

    PrimitiveState w_inf = make_freestream(condition.mach, condition.alpha_deg, condition.beta_deg, config.gamma);
    w_inf.nu_tilde = condition.nu_tilde;
    if (condition.nu_tilde_ratio > 0.0f && config.viscous) {
        Real T_inf = w_inf.p / w_inf.rho;
        Real t_ratio = T_inf / config.T_ref;
        Real mu_inf = config.mu_ref * t_ratio * std::sqrt(t_ratio) * (config.T_ref + config.sutherland_T) / (T_inf + config.sutherland_T);
        w_inf.nu_tilde = condition.nu_tilde_ratio * mu_inf / w_inf.rho;
    }
    ConservativeState q_inf = primitive_to_conservative(w_inf, config.gamma);
    std::vector<ConservativeState> q(mesh.cells.size(), q_inf);

    if (!d_mesh.upload_state(q, error)) {
        CfdSolveSummary s;
        s.failed = true;
        return s;
    }

    CfdSolveSummary gpu_result = solve_gpu(d_mesh, condition, config, error);

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