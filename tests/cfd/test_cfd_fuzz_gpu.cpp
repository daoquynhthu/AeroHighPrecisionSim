#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/gpu_solver.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace aerosp;
using namespace aerosp::aero::cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static bool all_forces_finite(const CfdForceResult& f) {
    return std::isfinite(f.CX) && std::isfinite(f.CY) && std::isfinite(f.CZ) &&
           std::isfinite(f.Cl) && std::isfinite(f.Cm) && std::isfinite(f.Cn);
}

static int test_fuzz_mach_cfl() {
    TEST("FUZZ-MACH-1 CFL sweep 0.01 to 10.0 (log scale, 20 values)");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        FreestreamCondition cond;
        cond.mach = 2.0f; cond.alpha_deg = 0.0f; cond.beta_deg = 0.0f;

        CfdConfig cfg;
        cfg.max_iter = 10;
        cfg.reconstruction_order = 1;
        cfg.use_gpu = true;

        for (int i = 0; i < 20; ++i) {
            cfg.cfl = 0.01f * std::pow(10.0f, i / 19.0f);
            CfdSolveSummary s = solver.solve(cond, cfg);
            if (s.failed) continue;
            if (!all_forces_finite(s.forces)) FAIL("CFL=%g: forces not finite", cfg.cfl);
        }
        PASS;
    }
    return 0;
}

static int test_fuzz_reynolds() {
    TEST("FUZZ-MACH-2 Re sweep 1e3 to 1e8 (log scale, 20 values)");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        FreestreamCondition cond;
        cond.mach = 0.5f; cond.alpha_deg = 0.0f; cond.beta_deg = 0.0f;

        CfdConfig cfg;
        cfg.max_iter = 10;
        cfg.reconstruction_order = 1;
        cfg.use_gpu = true;
        cfg.viscous = true;

        for (int i = 0; i < 20; ++i) {
            cfg.Re = 1e3f * std::pow(1e5f, i / 19.0f);
            CfdSolveSummary s = solver.solve(cond, cfg);
            if (s.failed) continue;
            if (!all_forces_finite(s.forces)) FAIL("Re=%g: forces not finite", cfg.Re);
        }
        PASS;
    }
    return 0;
}

static int test_fuzz_gamma() {
    TEST("FUZZ-MACH-3 gamma sweep 1.05 to 1.67 (20 values)");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        FreestreamCondition cond;
        cond.mach = 2.0f; cond.alpha_deg = 0.0f; cond.beta_deg = 0.0f;

        CfdConfig cfg;
        cfg.max_iter = 10;
        cfg.reconstruction_order = 1;
        cfg.use_gpu = true;

        for (int i = 0; i < 20; ++i) {
            cfg.gamma = 1.05f + (1.67f - 1.05f) * (i / 19.0f);
            CfdSolveSummary s = solver.solve(cond, cfg);
            if (s.failed) continue;
            if (!all_forces_finite(s.forces)) FAIL("gamma=%g: forces not finite", cfg.gamma);
        }
        PASS;
    }
    return 0;
}

static int test_fuzz_alpha() {
    TEST("FUZZ-MACH-4 alpha sweep -30 to +30 deg (20 values)");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        FreestreamCondition cond;
        cond.mach = 2.0f; cond.beta_deg = 0.0f;

        CfdConfig cfg;
        cfg.max_iter = 20;
        cfg.reconstruction_order = 1;
        cfg.use_gpu = true;

        for (int i = 0; i < 20; ++i) {
            cond.alpha_deg = -30.0f + 60.0f * (i / 19.0f);
            CfdSolveSummary s = solver.solve(cond, cfg);
            if (s.failed) continue;
            if (!all_forces_finite(s.forces)) FAIL("alpha=%g: forces not finite", cond.alpha_deg);
            if (std::fabs(s.forces.CY) > 1e-4f) FAIL("alpha=%g: CY=%g not zero", cond.alpha_deg, s.forces.CY);
        }
        PASS;
    }
    return 0;
}

static int test_fuzz_nu_tilde_ratio() {
    TEST("FUZZ-RANS-1 nu_tilde_ratio sweep 0.001 to 10.0 (10 values)");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        FreestreamCondition cond;
        cond.mach = 2.0f; cond.alpha_deg = 0.0f; cond.beta_deg = 0.0f;

        CfdConfig cfg;
        cfg.max_iter = 10;
        cfg.reconstruction_order = 1;
        cfg.use_gpu = true;
        cfg.turbulence = true;
        cfg.viscous = true;

        for (int i = 0; i < 10; ++i) {
            cond.nu_tilde_ratio = 0.001f * std::pow(10000.0f, i / 9.0f);
            CfdSolveSummary s = solver.solve(cond, cfg);
            if (s.failed) continue;
            if (!all_forces_finite(s.forces)) FAIL("nu_tilde_ratio=%g: forces not finite", cond.nu_tilde_ratio);
        }
        PASS;
    }
    return 0;
}

static int test_fuzz_wall_temperature() {
    TEST("FUZZ-RANS-2 wall_temperature sweep 0.5 to 5.0 (10 values)");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        FreestreamCondition cond;
        cond.mach = 2.0f; cond.alpha_deg = 0.0f; cond.beta_deg = 0.0f;

        CfdConfig cfg;
        cfg.max_iter = 10;
        cfg.reconstruction_order = 1;
        cfg.use_gpu = true;
        cfg.viscous = true;

        for (int i = 0; i < 10; ++i) {
            Real tw_ratio = 0.5f + 4.5f * (i / 9.0f);
            PrimitiveState w_inf = make_freestream(cond.mach, cond.alpha_deg, cond.beta_deg, cfg.gamma);
            cfg.wall_temperature = w_inf.p / w_inf.rho * tw_ratio;
            CfdSolveSummary s = solver.solve(cond, cfg);
            if (s.failed) continue;
            if (!all_forces_finite(s.forces)) FAIL("Twall_ratio=%g: forces not finite", tw_ratio);
            if (!std::isfinite(s.forces.Q_wall)) FAIL("Twall_ratio=%g: Q_wall not finite", tw_ratio);
        }
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_fuzz_mach_cfl();
    result |= test_fuzz_reynolds();
    result |= test_fuzz_gamma();
    result |= test_fuzz_alpha();
    result |= test_fuzz_nu_tilde_ratio();
    result |= test_fuzz_wall_temperature();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
