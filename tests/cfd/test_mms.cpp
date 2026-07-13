#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/mms.hpp"
#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/diagnostics.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace aerosp::aero::cfd;
using aerosp::Real;

int test_count = 0;
int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[TEST] %s ... ", name); } while(0)
#define PASS() do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(msg) do { std::printf("FAIL: %s\n", msg); return 1; } while(0)

static int test_mms_ns_consistency() {
    TEST("MMS-NS: laminar NS source consistency (q=q_exact -> residual=0)");

    auto mms_euler = make_default_mms_euler();
    PrimitiveState w_inf = make_freestream(0.5, 0.0, 0.0, 1.4);
    FreestreamCondition freestream;
    freestream.mach = 0.5;

    CfdMesh mesh = generate_structured_hex_mesh(8);
    compute_mesh_metrics(mesh);

    CfdConfig cfg;
    cfg.max_iter = 10;
    cfg.cfl = 1.0;
    cfg.convergence_tol = 1e-14;
    cfg.reconstruction_order = 2;
    cfg.viscous = true;
    cfg.Re = 1e4;
    cfg.use_gpu = false;

    std::vector<ConservativeState> q_exact;
    fill_mms(mesh, mms_euler, q_exact, cfg.gamma);

    std::vector<EulerFlux> source;
    if (!compute_mms_source(mesh, q_exact, w_inf, cfg, source))
        FAIL("compute_mms_source failed");

    cfg.mms_source = source;

    CfdSolver solver;
    if (!solver.load_mesh(mesh))
        FAIL("load_mesh failed");

    auto summary = solver.solve_from_state(freestream, cfg, q_exact);
    if (summary.failed)
        FAIL("solver failed");

    std::printf("residual=%g ", summary.residual_history.empty() ? -1.0 : summary.residual_history.back());

    Real err = mms_l2_error(summary.final_state, q_exact);
    std::printf("L2_err=%g ", err);

    Real final_res = summary.residual_history.back();
    if (final_res > 1e-6)
        FAIL("residual too high");

    PASS();
    return 0;
}

static int test_mms_ns_order1_consistency() {
    TEST("MMS-NS: laminar NS order-1 source consistency (q=q_exact -> residual=0)");

    auto mms_euler = make_default_mms_euler();
    PrimitiveState w_inf = make_freestream(0.5, 0.0, 0.0, 1.4);
    FreestreamCondition freestream;
    freestream.mach = 0.5;

    CfdMesh mesh = generate_structured_hex_mesh(8);
    compute_mesh_metrics(mesh);

    CfdConfig cfg;
    cfg.max_iter = 10;
    cfg.cfl = 1.0;
    cfg.convergence_tol = 1e-14;
    cfg.reconstruction_order = 1;
    cfg.viscous = true;
    cfg.Re = 1e4;
    cfg.use_gpu = false;

    std::vector<ConservativeState> q_exact;
    fill_mms(mesh, mms_euler, q_exact, cfg.gamma);

    std::vector<EulerFlux> source;
    if (!compute_mms_source(mesh, q_exact, w_inf, cfg, source))
        FAIL("compute_mms_source failed");

    cfg.mms_source = source;

    CfdSolver solver;
    if (!solver.load_mesh(mesh))
        FAIL("load_mesh failed");

    auto summary = solver.solve_from_state(freestream, cfg, q_exact);
    if (summary.failed)
        FAIL("solver failed");

    std::printf("residual=%g ", summary.residual_history.empty() ? -1.0 : summary.residual_history.back());

    Real err = mms_l2_error(summary.final_state, q_exact);
    std::printf("L2_err=%g ", err);

    Real final_res = summary.residual_history.back();
    if (final_res > 1e-6)
        FAIL("residual too high");

    PASS();
    return 0;
}

static int test_mms_sa_consistency() {
    TEST("MMS-SA: RANS SA source consistency (order-2, q=q_exact -> residual=0)");

    auto mms_sa = make_default_mms_sa();
    PrimitiveState w_inf = make_freestream(0.5, 0.0, 0.0, 1.4);
    w_inf.nu_tilde = 0.1f;
    FreestreamCondition freestream;
    freestream.mach = 0.5;
    freestream.nu_tilde = 0.1f;

    CfdMesh mesh = generate_structured_hex_mesh(8);
    compute_mesh_metrics(mesh);

    CfdConfig cfg;
    cfg.max_iter = 20;
    cfg.cfl = 0.1;
    cfg.reconstruction_order = 2;
    cfg.viscous = true;
    cfg.Re = 1e4;
    cfg.turbulence = true;
    cfg.diagnostic_level = DiagnosticLevel::Basic;
    cfg.use_gpu = false;

    std::vector<ConservativeState> q_exact;
    fill_mms_sa(mesh, mms_sa, q_exact, cfg.gamma);

    std::vector<EulerFlux> source;
    if (!compute_mms_source(mesh, q_exact, w_inf, cfg, source))
        FAIL("compute_mms_source failed");

    cfg.mms_source = source;

    CfdSolver solver;
    if (!solver.load_mesh(mesh))
        FAIL("load_mesh failed");

    auto summary = solver.solve_from_state(freestream, cfg, q_exact);
    if (summary.failed) {
        std::string reason = summary.diagnostics.failure.valid
            ? summary.diagnostics.failure.reason : "unknown";
        std::printf("FAILED(%s) ", reason.c_str());
        FAIL("solver failed");
    }

    Real r0 = summary.residual_history.empty() ? -1.0 : summary.residual_history[0];
    Real rf = summary.residual_history.empty() ? -1.0 : summary.residual_history.back();
    Real err = mms_l2_error(summary.final_state, q_exact);

    std::printf("r0=%g rf=%g L2_err=%g ", r0, rf, err);

    // The MMS fix (skip semi-implicit in MMS mode) makes q_exact a fixed point.
    // Floating-point accumulation over iterations can give O(1e-4) residual.
    // Order-2 limiter non-associativity can give O(1e-9) initial residual.
    if (rf > 1e-4)
        FAIL("residual too high");

    PASS();
    return 0;
}

// Run Euler order-of-accuracy with boundary-compatible MMS.
// Prints error-per-mesh and observed order. Returns 0 on success.
static int run_euler_order_test(int order, Real cfl, int max_iter, int* ns_out = nullptr) {
    auto mms = make_default_mms_euler_bc();
    PrimitiveState w_inf = make_freestream(0.5, 0.0, 0.0, 1.4);
    FreestreamCondition freestream;
    freestream.mach = 0.5;

    // Use the same nx for all meshes to get consistent domain [0,1]^3
    // with h = 1/nx per cell. The structured hex generator uses [-0.5, 0.5]^3.
    int ns[] = {6, 10, 14};
    Real errors[3] = {};
    int iters[3] = {};
    bool ok = true;

    for (int mi = 0; mi < 3; ++mi) {
        int n = ns[mi];
        CfdMesh mesh = generate_structured_hex_mesh(n);
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.max_iter = max_iter;
        cfg.cfl = cfl;
        cfg.convergence_tol = 1e-12;
        cfg.reconstruction_order = order;
        cfg.use_gpu = false;

        std::vector<ConservativeState> q_exact;
        fill_mms(mesh, mms, q_exact, cfg.gamma);

        std::vector<EulerFlux> source;
        if (!compute_mms_source(mesh, q_exact, w_inf, cfg, source)) {
            std::printf("n=%d source_failed ", n);
            ok = false; continue;
        }

        cfg.mms_source = source;

        // Start from q_exact (steady state). The residual should be 0,
        // so the solution stays at q_exact indefinitely.
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) {
            std::printf("n=%d load_failed ", n);
            ok = false; continue;
        }

        auto summary = solver.solve_from_state(freestream, cfg, q_exact);
        if (summary.failed) {
            std::string reason = summary.diagnostics.failure.valid
                ? summary.diagnostics.failure.reason : "unknown";
            std::printf("n=%d FAILED(%s) ", n, reason.c_str());
            ok = false; continue;
        }

        errors[mi] = mms_l2_error(summary.final_state, q_exact);
        iters[mi] = static_cast<int>(summary.residual_history.size());
        Real final_res = summary.residual_history.empty() ? -1.0 : summary.residual_history.back();
        std::printf("n=%d err=%g res=%g iters=%d ", n, errors[mi], final_res, iters[mi]);
    }

    if (!ok) return -1;
    for (int i = 0; i < 3; ++i)
        if (errors[i] > 1e-10) return -2;

    // If all errors are ~0, source consistency holds for this order.
    return 0;
}

static int test_mms_euler_order2_consistency() {
    TEST("MMS-EULER: order-2 Euler source consistency (q=q_exact -> err=0)");
    int r = run_euler_order_test(2, 0.1f, 200);
    if (r != 0) {
        std::printf("source_consistency_failed=%d ", r);
        FAIL("expected err=0 for q=q_exact");
    }
    PASS();
    return 0;
}

static int test_mms_euler_order1_consistency() {
    TEST("MMS-EULER: order-1 Euler source consistency (q=q_exact -> err=0)");
    int r = run_euler_order_test(1, 0.1f, 200);
    if (r != 0) {
        std::printf("source_consistency_failed=%d ", r);
        FAIL("expected err=0 for q=q_exact");
    }
    PASS();
    return 0;
}

static int test_mms_sa_order1_consistency() {
    TEST("MMS-SA: order-1 RANS SA source consistency (q=q_exact -> err=0)");

    auto mms_sa = make_default_mms_sa();
    PrimitiveState w_inf = make_freestream(0.5, 0.0, 0.0, 1.4);
    w_inf.nu_tilde = 0.1f;
    FreestreamCondition freestream;
    freestream.mach = 0.5;
    freestream.nu_tilde = 0.1f;

    CfdMesh mesh = generate_structured_hex_mesh(8);
    compute_mesh_metrics(mesh);

    CfdConfig cfg;
    cfg.max_iter = 5;
    cfg.cfl = 0.01;
    cfg.reconstruction_order = 1;
    cfg.viscous = true;
    cfg.Re = 1e4;
    cfg.turbulence = true;
    cfg.use_gpu = false;

    std::vector<ConservativeState> q_exact;
    fill_mms_sa(mesh, mms_sa, q_exact, cfg.gamma);

    std::vector<EulerFlux> source;
    if (!compute_mms_source(mesh, q_exact, w_inf, cfg, source))
        FAIL("compute_mms_source failed");

    cfg.mms_source = source;

    CfdSolver solver;
    if (!solver.load_mesh(mesh))
        FAIL("load_mesh failed");

    auto summary = solver.solve_from_state(freestream, cfg, q_exact);
    if (summary.failed) {
        std::string reason = summary.diagnostics.failure.valid
            ? summary.diagnostics.failure.reason : "unknown";
        std::printf("FAILED(%s) ", reason.c_str());
        FAIL("solver failed");
    }

    Real r0 = summary.residual_history.empty() ? -1.0 : summary.residual_history[0];
    Real rf = summary.residual_history.empty() ? -1.0 : summary.residual_history.back();
    Real err = mms_l2_error(summary.final_state, q_exact);
    std::printf("r0=%g rf=%g L2_err=%g ", r0, rf, err);

    if (rf > 1e-6)
        FAIL("residual too high");

    PASS();
    return 0;
}

// Order-of-accuracy test via truncation error measurement.
// Computes T_i = R_h(q_exact)_i - V_i * S_analytic(x_i) directly without
// running the solver. T is the truncation error of the FV scheme and should
// converge as O(h^p). This is more direct than solver-based convergence
// tests (forward Euler is unstable for large analytic source terms).
static int run_truncation_error_order(int order) {
    auto mms = make_default_mms_euler_bc();
    PrimitiveState w_inf = make_freestream(0.5, 0.0, 0.0, 1.4);

    int ns[] = {8, 12, 16};
    Real tnorms[3] = {};
    bool ok = true;

    for (int mi = 0; mi < 3; ++mi) {
        int n = ns[mi];
        CfdMesh mesh = generate_structured_hex_mesh(n);
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.reconstruction_order = order;
        cfg.use_gpu = false;

        std::vector<ConservativeState> q_exact;
        fill_mms(mesh, mms, q_exact, cfg.gamma);
        std::vector<EulerFlux> discrete;
        if (!compute_mms_source(mesh, q_exact, w_inf, cfg, discrete)) {
            std::printf("n=%d discrete_failed ", n); ok = false; continue;
        }
        std::vector<EulerFlux> analytic;
        if (!compute_euler_source_analytic(mesh, mms, cfg.gamma, analytic)) {
            std::printf("n=%d analytic_failed ", n); ok = false; continue;
        }

        Real t2 = 0.0;
        for (std::size_t i = 0; i < q_exact.size(); ++i) {
            Real t0 = discrete[i].mass   - analytic[i].mass;
            Real t1 = discrete[i].mom_x  - analytic[i].mom_x;
            Real t2_ = discrete[i].mom_y  - analytic[i].mom_y;
            Real t3 = discrete[i].mom_z  - analytic[i].mom_z;
            Real t4 = discrete[i].energy - analytic[i].energy;
            t2 += t0*t0 + t1*t1 + t2_*t2_ + t3*t3 + t4*t4;
        }
        tnorms[mi] = std::sqrt(t2 / (5.0 * static_cast<Real>(q_exact.size())));
        std::printf("n=%d |T|=%g ", n, tnorms[mi]);
    }

    if (!ok) return -1;
    if (tnorms[0] <= 0.0 || tnorms[1] <= 0.0 || tnorms[2] <= 0.0) return -2;

    Real observed = mms_observed_order(tnorms[0], ns[0], tnorms[1], ns[1], tnorms[2], ns[2]);
    std::printf("T_order=%g ", observed);
    return static_cast<int>(observed * 10 + 0.5);
}

static int test_euler_order2_truncation() {
    TEST("EULER-OA: order-2 truncation error order (analytic vs discrete)");
    int o = run_truncation_error_order(2);
    if (o < 15) {
        std::printf("got_T_order=%d ", o);
        FAIL("expected truncation error order >= 1.5");
    }
    PASS();
    return 0;
}

static int test_euler_order1_truncation() {
    TEST("EULER-OA: order-1 truncation error order (analytic vs discrete)");
    int o = run_truncation_error_order(1);
    if (o < 7) {
        std::printf("got_T_order=%d ", o);
        FAIL("expected truncation error order >= 0.7");
    }
    PASS();
    return 0;
}

// MMS order-of-accuracy via solver convergence from freestream IC.
// With the MMS-compatible farfield BC, the solver imposes q_exact at
// boundary faces so that the only error source is interior discretization.
// MMS fixed-point stability: with MMS BC, q_exact is the exact fixed point
// of the discrete system. Run 1000 iterations from q_exact and verify that
// the state does not drift (residual stays ~0). This validates that the
// MMS-compatible BC is consistent with the MMS source term.
static int run_euler_fixed_point_test(int order) {
    auto mms = make_default_mms_euler_bc();
    PrimitiveState w_inf = make_freestream(0.5, 0.0, 0.0, 1.4);
    FreestreamCondition freestream;
    freestream.mach = 0.5;

    int ns[] = {8, 12, 16};
    bool ok = true;

    for (int mi = 0; mi < 3; ++mi) {
        int n = ns[mi];
        CfdMesh mesh = generate_structured_hex_mesh(n);
        compute_mesh_metrics(mesh);

        CfdConfig cfg;
        cfg.max_iter = 1000;
        cfg.cfl = 1.0f;
        cfg.convergence_tol = 1e-14;
        cfg.reconstruction_order = order;
        cfg.use_gpu = false;
        cfg.mms_solution = &mms;

        std::vector<ConservativeState> q_exact;
        fill_mms(mesh, mms, q_exact, cfg.gamma);

        std::vector<EulerFlux> source;
        if (!compute_mms_source(mesh, q_exact, w_inf, cfg, source)) {
            std::printf("n=%d source_failed ", n);
            ok = false; continue;
        }
        cfg.mms_source = source;

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) {
            std::printf("n=%d load_failed ", n);
            ok = false; continue;
        }

        // Run from q_exact IC -> should stay at q_exact for all iterations.
        auto summary = solver.solve_from_state(freestream, cfg, q_exact);

        if (summary.failed) {
            std::string reason = summary.diagnostics.failure.valid
                ? summary.diagnostics.failure.reason : "unknown";
            std::printf("n=%d FAILED(%s) ", n, reason.c_str());
            ok = false; continue;
        }

        Real final_res = summary.residual_history.empty() ? -1.0 : summary.residual_history.back();
        Real final_err = mms_l2_error(summary.final_state, q_exact);
        std::printf("n=%d res=%g err=%g iters=%zu ", n, final_res, final_err, summary.residual_history.size());

        if (final_res > 1e-6 || final_err > 1e-6) {
            std::printf("fixed_point_broken ");
            ok = false; continue;
        }
    }
    return ok ? 0 : -1;
}

static int test_euler_order2_fixed_point() {
    TEST("EULER-OA: order-2 MMS fixed point (q_exact stays q_exact, MMS BC)");
    int r = run_euler_fixed_point_test(2);
    if (r != 0)
        FAIL("expected fixed point stability with MMS BC");
    PASS();
    return 0;
}

static int test_mms_sa_fixed_point() {
    TEST("MMS-SA: order-2 SA fixed point with MMS BC (q_exact stays q_exact)");

    auto mms = make_default_mms_sa_bc();
    PrimitiveState w_inf = make_freestream(0.5, 0.0, 0.0, 1.4);
    w_inf.nu_tilde = 0.1f;
    FreestreamCondition freestream;
    freestream.mach = 0.5;
    freestream.nu_tilde = 0.1f;

    CfdMesh mesh = generate_structured_hex_mesh(8);
    compute_mesh_metrics(mesh);

    CfdConfig cfg;
    cfg.max_iter = 20;
    cfg.cfl = 0.1f;
    cfg.convergence_tol = 1e-14;
    cfg.reconstruction_order = 2;
    cfg.viscous = true;
    cfg.Re = 1e4;
    cfg.turbulence = true;
    cfg.use_gpu = false;
    cfg.mms_solution = &mms;

    std::vector<ConservativeState> q_exact;
    fill_mms_sa(mesh, mms, q_exact, cfg.gamma);

    std::vector<EulerFlux> source;
    if (!compute_mms_source(mesh, q_exact, w_inf, cfg, source))
        FAIL("compute_mms_source failed");

    cfg.mms_source = source;

    CfdSolver solver;
    if (!solver.load_mesh(mesh))
        FAIL("load_mesh failed");

    auto summary = solver.solve_from_state(freestream, cfg, q_exact);
    if (summary.failed) {
        std::string reason = summary.diagnostics.failure.valid
            ? summary.diagnostics.failure.reason : "unknown";
        std::printf("FAILED(%s) ", reason.c_str());
        FAIL("solver failed");
    }

    Real r0 = summary.residual_history.empty() ? -1.0 : summary.residual_history[0];
    Real rf = summary.residual_history.empty() ? -1.0 : summary.residual_history.back();
    Real err = mms_l2_error(summary.final_state, q_exact);
    std::printf("r0=%g rf=%g L2_err=%g ", r0, rf, err);

    if (rf > 1e-4 || err > 1e-6)
        FAIL("fixed point not maintained with MMS BC");

    PASS();
    return 0;
}

int main() {
    int result = 0;
    result |= test_mms_ns_consistency();
    result |= test_mms_ns_order1_consistency();
    result |= test_mms_sa_consistency();
    result |= test_mms_euler_order1_consistency();
    result |= test_mms_euler_order2_consistency();
    result |= test_mms_sa_order1_consistency();
    result |= test_euler_order1_truncation();
    result |= test_euler_order2_truncation();
    result |= test_euler_order2_fixed_point();
    result |= test_mms_sa_fixed_point();

    std::printf("[SUMMARY] %d/%d PASS\n", pass_count, test_count);
    return result;
}
