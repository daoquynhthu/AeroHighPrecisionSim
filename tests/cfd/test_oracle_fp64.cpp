#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/turbulence_model.hpp"

#include <cmath>
#include <cstdio>

namespace aerosp {
namespace aero {
namespace cfd {

static int pass_count = 0;
static int test_count = 0;

#define TEST(name) do { ++test_count; std::printf("[Test] %s ... ", name); } while (0)
#define PASS do { ++pass_count; std::printf("PASS\n"); return 0; } while (0)
#define FAIL(...) do { std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); return 1; } while (0)

static int test_fp64_flat_plate_euler() {
    TEST("FP64-ORACLE-1 flat plate Euler (FP64)");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5, 0.05, 0.1, 1e-5, 1.12, 8, 3, 6);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 0.5;
        cond.alpha_deg = 0.0;
        cond.nu_tilde_ratio = 0.1;

        CfdConfig cfg;
        cfg.use_gpu = false;
        cfg.cfl = 0.5;
        cfg.max_iter = 1;
        cfg.convergence_tol = 1e-14;
        cfg.viscous = false;
        cfg.local_time_stepping = false;
        cfg.mean_flow_point_implicit = false;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");
        CfdSolveSummary result = solver.solve(cond, cfg);
        if (result.failed) FAIL("solver failed");

        if (!std::isfinite(result.forces.CD)) FAIL("CD not finite: %g", result.forces.CD);
        if (!std::isfinite(result.forces.CL)) FAIL("CL not finite: %g", result.forces.CL);

        std::printf("  forces: CD=%.15e CL=%.15e CX=%.15e CY=%.15e CZ=%.15e\n",
            result.forces.CD, result.forces.CL,
            result.forces.CX, result.forces.CY, result.forces.CZ);
        std::printf("  iterations=%zu converged=%d\n",
            result.residual_history.size(), result.converged);

        PASS;
    }
    return 0;
}

static int test_fp64_flat_plate_viscous() {
    TEST("FP64-ORACLE-2 flat plate viscous (FP64)");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5, 0.05, 0.1, 1e-5, 1.12, 8, 3, 6);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 0.5;
        cond.alpha_deg = 0.0;
        cond.nu_tilde_ratio = 0.1;

        CfdConfig cfg;
        cfg.use_gpu = false;
        cfg.cfl = 0.2;
        cfg.max_iter = 20;
        cfg.convergence_tol = 1e-14;
        cfg.viscous = true;
        cfg.Re = 1e5;
        cfg.prandtl = 0.72;
        cfg.wall_temperature = 288.15;
        cfg.T_ref = 288.15;
        cfg.mu_ref = 1.0;
        cfg.sutherland_T = 110.4;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");
        CfdSolveSummary result = solver.solve(cond, cfg);
        if (result.failed) FAIL("solver failed");

        if (!std::isfinite(result.forces.CD)) FAIL("CD not finite: %g", result.forces.CD);
        if (!std::isfinite(result.forces.CL)) FAIL("CL not finite: %g", result.forces.CL);

        std::printf("  forces: CD=%.15e CL=%.15e CX=%.15e CY=%.15e CZ=%.15e\n",
            result.forces.CD, result.forces.CL,
            result.forces.CX, result.forces.CY, result.forces.CZ);
        std::printf("  iterations=%zu converged=%d\n",
            result.residual_history.size(), result.converged);

        PASS;
    }
    return 0;
}

static int test_fp64_rans_flat_plate() {
    TEST("FP64-ORACLE-3 flat plate RANS (FP64)");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5, 0.05, 0.1, 1e-5, 1.12, 8, 3, 6);
        compute_mesh_metrics(mesh);

        FreestreamCondition cond;
        cond.mach = 0.5;
        cond.alpha_deg = 0.0;
        cond.nu_tilde_ratio = 3.0;

        CfdConfig cfg;
        cfg.use_gpu = false;
        cfg.cfl = 0.5;
        cfg.cfl_ramp = true;
        cfg.cfl_start = 0.2;
        cfg.cfl_end = 2.5;
        cfg.cfl_ramp_steps = 40;
        cfg.max_iter = 300;
        cfg.convergence_tol = 1e-10;
        cfg.viscous = true;
        cfg.Re = 1e5;
        cfg.prandtl = 0.72;
        cfg.wall_temperature = 288.15;
        cfg.T_ref = 288.15;
        cfg.mu_ref = 1.0;
        cfg.sutherland_T = 110.4;
        cfg.turbulence_model = TurbulenceModel::SA;
        cfg.local_time_stepping = true;
        // Spectral CFL only (no mean-flow PI double-damping); SA keeps full residual PI.
        cfg.mean_flow_point_implicit = false;
        cfg.sa_sub_iters = 3;
        cfg.diagnostic_level = DiagnosticLevel::Basic;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");
        CfdSolveSummary result = solver.solve(cond, cfg);

        std::printf("  residual history (%zu):", result.residual_history.size());
        for (std::size_t i = 0; i < result.residual_history.size(); ++i) {
            if (i < 5 || i + 5 >= result.residual_history.size() || i % 10 == 0)
                std::printf(" [%zu]=%.6e", i, result.residual_history[i]);
        }
        std::printf("\n");
        if (result.diagnostics.failure.valid) {
            std::printf("  failure: iter=%d cell=%d reason=%s\n",
                result.diagnostics.failure.iteration,
                result.diagnostics.failure.cell,
                result.diagnostics.failure.reason.c_str());
        }

        if (result.failed) FAIL("solver failed: %s",
            result.diagnostics.failure.valid
                ? result.diagnostics.failure.reason.c_str()
                : "unknown");

        if (!std::isfinite(result.forces.CD)) FAIL("CD not finite: %g", result.forces.CD);
        if (!std::isfinite(result.forces.CL)) FAIL("CL not finite: %g", result.forces.CL);

        if (result.residual_history.size() < 10)
            FAIL("too few residual samples: %zu", result.residual_history.size());
        Real r0 = result.residual_history.front();
        Real rN = result.residual_history.back();
        Real r_peak = r0;
        Real r_mid = result.residual_history[result.residual_history.size() / 2];
        for (Real r : result.residual_history)
            if (r > r_peak) r_peak = r;
        if (!std::isfinite(rN) || !std::isfinite(r_peak))
            FAIL("non-finite residual peak=%g final=%g", r_peak, rN);
        // Spatial residual gates (pre-PI metric): form BL, then drop & control.
        if (r_peak > Real(1e1))
            FAIL("residual peak too large: %g", r_peak);
        if (r_peak < Real(1e-7))
            FAIL("residual never developed (peak=%g) — check SA source path", r_peak);
        // ≥2× drop from peak (coarse mesh truncation limits absolute floor)
        if (rN > r_peak * Real(0.5))
            FAIL("residual not reduced from peak: peak=%g final=%g", r_peak, rN);
        if (rN > Real(5e-3))
            FAIL("final residual too large: %g (target <= 5e-3)", rN);
        if (rN > r_mid * Real(3) && rN > Real(1e-3))
            FAIL("residual grew late: mid=%g final=%g", r_mid, rN);

        std::printf("  forces: CD=%.15e CL=%.15e CX=%.15e CY=%.15e CZ=%.15e\n",
            result.forces.CD, result.forces.CL,
            result.forces.CX, result.forces.CY, result.forces.CZ);
        std::printf("  iterations=%zu converged=%d r0=%.3e rN=%.3e peak=%.3e mid=%.3e\n",
            result.residual_history.size(), result.converged, r0, rN, r_peak, r_mid);

        PASS;
    }
    return 0;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp

int main() {
    using namespace aerosp::aero::cfd;
    int result = 0;
    result |= test_fp64_flat_plate_euler();
    result |= test_fp64_flat_plate_viscous();
    result |= test_fp64_rans_flat_plate();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
