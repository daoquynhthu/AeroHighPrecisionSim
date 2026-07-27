#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/mesh_validator.hpp"

#include <cstdio>
#include <cmath>
#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define FAIL(...) do { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); return 1; } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)

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
        cfg.cfl = 0.4;
        cfg.max_iter = 20;
        cfg.convergence_tol = 1e-14;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");
        CfdSolveSummary result = solver.solve(cond, cfg);
        if (result.failed) FAIL("solver failed");

        if (!std::isfinite(result.forces.CD)) FAIL("CD not finite: %g", result.forces.CD);
        if (!std::isfinite(result.forces.CL)) FAIL("CL not finite: %g", result.forces.CL);
        if (!std::isfinite(result.forces.CX)) FAIL("CX not finite: %g", result.forces.CX);
        if (!std::isfinite(result.forces.CY)) FAIL("CY not finite: %g", result.forces.CY);
        if (!std::isfinite(result.forces.CZ)) FAIL("CZ not finite: %g", result.forces.CZ);

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
        cfg.cfl = 0.3;
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
        cfg.turbulence_model = TurbulenceModel::SA;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load mesh failed");
        CfdSolveSummary result = solver.solve(cond, cfg);

        std::printf("  residual history (%zu):", result.residual_history.size());
        for (std::size_t i = 0; i < result.residual_history.size(); ++i) {
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

        std::printf("  forces: CD=%.15e CL=%.15e CX=%.15e CY=%.15e CZ=%.15e\n",
            result.forces.CD, result.forces.CL,
            result.forces.CX, result.forces.CY, result.forces.CZ);
        std::printf("  iterations=%zu converged=%d\n",
            result.residual_history.size(), result.converged);

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
