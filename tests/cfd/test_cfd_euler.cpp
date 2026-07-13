#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/diagnostics.hpp"
#include "aero/cfd/real.hpp"

#include <cmath>
#include <cstdio>

using namespace aerosp;
using namespace aerosp::aero::cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static int test_state_roundtrip() {
    TEST("CFD-EULER-1 primitive/conservative roundtrip");
    {
        PrimitiveState w;
        w.rho = 1.2f;
        w.u = -2.0f;
        w.v = 0.1f;
        w.w = 0.3f;
        w.p = 0.9f;
        auto q = primitive_to_conservative(w, 1.4f);
        PrimitiveState out;
        if (!conservative_to_primitive(q, 1.4f, out)) FAIL("conversion failed");
        if (std::fabs(out.rho - w.rho) > 1e-6f) FAIL("rho=%g", out.rho);
        if (std::fabs(out.u - w.u) > 1e-6f) FAIL("u=%g", out.u);
        if (std::fabs(out.p - w.p) > 1e-6f) FAIL("p=%g", out.p);
        PASS;
    }
    return 0;
}

static int test_fluxes() {
    TEST("CFD-EULER-2 HLLC equals physical flux for identical states");
    {
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        auto f = hllc_flux(w, w, 1.4f, 1.0f, 0.0f, 0.0f);
        auto exact = physical_flux(w, 1.4f, 1.0f, 0.0f, 0.0f);
        if (std::fabs(f.mass - exact.mass) > 1e-6f) FAIL("mass=%g exact=%g", f.mass, exact.mass);
        if (std::fabs(f.energy - exact.energy) > 1e-5f) FAIL("energy=%g exact=%g", f.energy, exact.energy);
        PASS;
    }

    TEST("CFD-EULER-3 slip wall mass and energy flux are zero");
    {
        auto w = make_freestream(5.0f, 10.0f, 0.0f, 1.4f);
        auto f = slip_wall_flux(w, 0.0f, 0.0f, 1.0f);
        if (std::fabs(f.mass) > 1e-8f) FAIL("mass=%g", f.mass);
        if (std::fabs(f.energy) > 1e-8f) FAIL("energy=%g", f.energy);
        if (std::fabs(f.mom_z - w.p) > 1e-6f) FAIL("mom_z=%g p=%g", f.mom_z, w.p);
        PASS;
    }
    return 0;
}

static int test_solver_uniform() {
    TEST("CFD-EULER-4 farfield-only uniform freestream remains steady");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        for (auto& face : mesh.faces) {
            if (face.boundary == BoundaryKind::NoSlipWall) face.boundary = BoundaryKind::Farfield;
        }
        compute_mesh_metrics(mesh);

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.max_iter = 3;
        cfg.cfl = 0.1f;
        cfg.convergence_tol = 1e-12f;
        auto summary = solver.solve({2.0f, 0.0f, 0.0f}, cfg);
        if (summary.failed) FAIL("solver failed");
        if (summary.residual_history.empty()) FAIL("missing residual");
        if (summary.residual_history.back() > 1e-7f) FAIL("residual=%g", summary.residual_history.back());
        PASS;
    }
    return 0;
}

static int test_farfield_boundary() {
    TEST("CFD-EULER-5 supersonic farfield inflow uses freestream state");
    {
        PrimitiveState left;
        left.rho = 0.7f;
        left.u = -0.4f;
        left.v = 0.2f;
        left.w = 0.0f;
        left.p = 0.5f;
        auto inf = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        auto ghost = farfield_ghost_state(left, inf, 1.4f, 1.0f, 0.0f, 0.0f);
        if (std::fabs(ghost.rho - inf.rho) > 1e-6f) FAIL("rho=%g", ghost.rho);
        if (std::fabs(ghost.u - inf.u) > 1e-6f) FAIL("u=%g", ghost.u);
        if (std::fabs(ghost.p - inf.p) > 1e-6f) FAIL("p=%g", ghost.p);
        PASS;
    }

    TEST("CFD-EULER-6 supersonic farfield outflow extrapolates interior state");
    {
        PrimitiveState left;
        left.rho = 0.7f;
        left.u = -0.4f;
        left.v = 0.2f;
        left.w = 0.0f;
        left.p = 0.5f;
        auto inf = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        auto ghost = farfield_ghost_state(left, inf, 1.4f, -1.0f, 0.0f, 0.0f);
        if (std::fabs(ghost.rho - left.rho) > 1e-6f) FAIL("rho=%g", ghost.rho);
        if (std::fabs(ghost.u - left.u) > 1e-6f) FAIL("u=%g", ghost.u);
        if (std::fabs(ghost.p - left.p) > 1e-6f) FAIL("p=%g", ghost.p);
        PASS;
    }

    return 0;
}

static int test_wall_forces() {
    TEST("CFD-EULER-7 symmetric cube has zero lateral force in uniform pressure field");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 13);
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.max_iter = 0;
        cfg.ref_area = 4.0f;
        cfg.ref_length = 2.0f;
        cfg.ref_span = 2.0f;
        auto summary = solver.solve({2.0f, 0.0f, 0.0f}, cfg);
        if (summary.failed) FAIL("solver failed");
        if (std::fabs(summary.forces.CY) > 1e-6f) {
            Real sx = 0.0f;
            Real sy = 0.0f;
            Real sz = 0.0f;
            for (const auto& face : mesh.faces) {
                if (face.boundary != BoundaryKind::SlipWall) continue;
                sx += face.nx * face.area;
                sy += face.ny * face.area;
                sz += face.nz * face.area;
            }
            FAIL("CY=%g normal_area=[%g,%g,%g]", summary.forces.CY, sx, sy, sz);
        }
        if (std::fabs(summary.forces.CZ) > 1e-6f) FAIL("CZ=%g", summary.forces.CZ);
        if (std::fabs(summary.forces.CX) > 1e-5f) FAIL("CX=%g", summary.forces.CX);
        PASS;
    }

    TEST("CFD-EULER-8 farfield-only mesh contributes no body force");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        for (auto& face : mesh.faces) {
            if (face.boundary == BoundaryKind::NoSlipWall) face.boundary = BoundaryKind::Farfield;
        }
        compute_mesh_metrics(mesh);

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.max_iter = 0;
        cfg.ref_area = 0.025f;
        auto summary = solver.solve({2.0f, 0.0f, 0.0f}, cfg);
        if (summary.failed) FAIL("solver failed");
        if (std::fabs(summary.forces.CX) > 1e-8f) FAIL("CX=%g", summary.forces.CX);
        if (std::fabs(summary.forces.CY) > 1e-8f) FAIL("CY=%g", summary.forces.CY);
        if (std::fabs(summary.forces.CZ) > 1e-8f) FAIL("CZ=%g", summary.forces.CZ);
        PASS;
    }
    return 0;
}

static int test_amr_solver_loop() {
    TEST("CFD-EULER-9 AMR solver loop refines cells near shock");
    {
        // Cube with embedded body — supersonic flow creates a shock
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        int cells_before = static_cast<int>(mesh.cells.size());

        CfdSolver solver;
        std::printf("  (cells=%zu, faces=%zu)\n", mesh.cells.size(), mesh.faces.size());
        if (!solver.load_mesh(mesh)) {
            auto rpt = compute_mesh_metrics(mesh);
            FAIL("load_mesh failed: %s", rpt.message.c_str());
        }

        CfdConfig cfg;
        cfg.max_iter = 11;
        cfg.cfl = 0.1f;
        cfg.convergence_tol = 1e-12f;
        cfg.amr.enabled = true;
        cfg.amr.interval = 10;
        cfg.amr.refine_tol = 0.03f;

        auto summary = solver.solve({2.0f, 0.0f, 0.0f}, cfg);
        if (summary.failed) {
            std::printf("  (solver failed: iter=%d, cell=%d, reason=%s)\n",
                summary.diagnostics.failure.iteration,
                summary.diagnostics.failure.cell,
                summary.diagnostics.failure.reason.c_str());
            FAIL("solver failed");
        }

        int cells_after = static_cast<int>(summary.final_state.size());
        std::printf("  (before=%d after=%d res=%.2e)\n", cells_before, cells_after,
            summary.residual_history.empty() ? -1.0 : summary.residual_history.back());
        if (cells_after <= cells_before) FAIL("AMR did not refine");

        CfdMesh mesh_after = solver.mesh();
        auto qr = compute_mesh_metrics(mesh_after);
        if (qr.min_volume <= 0.0f) FAIL("min volume after AMR=%g", qr.min_volume);
        if (qr.negative_jacobian_count != 0) FAIL("negative_jacobian_count=%d after AMR", qr.negative_jacobian_count);
        PASS;
    }
    return 0;
}

static int test_euler_residual_cpu_uniform() {
    TEST("CFD-EULER-COV1-1 uniform freestream gives zero residual");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        for (auto& face : mesh.faces) {
            if (face.boundary == BoundaryKind::SlipWall) face.boundary = BoundaryKind::Farfield;
        }
        auto freestream = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        auto q = std::vector<ConservativeState>(mesh.cells.size());
        for (auto& c : q) c = primitive_to_conservative(freestream, 1.4f);

        std::vector<EulerFlux> residual;
        bool ok = compute_euler_residual_cpu(mesh, q, freestream, 1.4f, residual);
        if (!ok) FAIL("returned false");

        Real max_res = 0.0f;
        for (const auto& r : residual) {
            Real n = std::fabs(r.mass) + std::fabs(r.mom_x) + std::fabs(r.mom_y) + std::fabs(r.mom_z) + std::fabs(r.energy);
            if (n > max_res) max_res = n;
        }
        if (max_res > 1e-6f) FAIL("max residual=%g expected ~0", max_res);
        PASS;
    }

    TEST("CFD-EULER-COV1-2 size mismatch returns false");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 5);
        compute_mesh_metrics(mesh);
        auto freestream = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q; // empty
        std::vector<EulerFlux> residual;
        bool ok = compute_euler_residual_cpu(mesh, q, freestream, 1.4f, residual);
        if (ok) FAIL("expected false for size mismatch");
        PASS;
    }
    return 0;
}

static int test_edge_case_meshes() {
    TEST("CFD-EULER-10 empty mesh (0 cells) returns false from solver");
    {
        CfdMesh mesh;
        compute_mesh_metrics(mesh);
        FreestreamCondition cond;
        cond.mach = 2.0f; cond.alpha_deg = 0.0f; cond.beta_deg = 0.0f;
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed on empty mesh");
        CfdConfig cfg;
        cfg.use_gpu = false;
        auto summary = solver.solve(cond, cfg);
        if (!summary.failed) FAIL("expected failure for empty mesh");
        PASS;
    }

    TEST("CFD-EULER-11 single-cell mesh solver runs without crash");
    {
        CfdMesh mesh = generate_structured_hex_mesh(2);
        compute_mesh_metrics(mesh);
        if (mesh.cells.size() != 1) FAIL("expected 1 cell, got %zu", mesh.cells.size());
        FreestreamCondition cond;
        cond.mach = 2.0f; cond.alpha_deg = 0.0f; cond.beta_deg = 0.0f;
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");
        CfdConfig cfg;
        cfg.max_iter = 2;
        cfg.cfl = 1.0f;
        cfg.use_gpu = false;
        auto summary = solver.solve(cond, cfg);
        if (summary.failed) {
            std::string reason = summary.diagnostics.failure.valid
                ? summary.diagnostics.failure.reason : "unknown";
            FAIL("solver failed: %s", reason.c_str());
        }
        if (summary.forces.iterations <= 0) FAIL("no iterations");
        PASS;
    }
    return 0;
}

static int test_tet_mesh_solver() {
    TEST("CFD-EULER-12 tet-only mesh (structured cube) solver produces finite forces");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 7);
        compute_mesh_metrics(mesh);
        FreestreamCondition cond;
        cond.mach = 2.0f; cond.alpha_deg = 0.0f; cond.beta_deg = 0.0f;
        CfdConfig cfg;
        cfg.max_iter = 3;
        cfg.cfl = 0.1f;
        cfg.use_gpu = false;
        cfg.diagnostic_level = DiagnosticLevel::Verbose;
        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");
        auto summary = solver.solve(cond, cfg);
        if (summary.failed) {
            std::string reason = summary.diagnostics.failure.valid
                ? summary.diagnostics.failure.reason : "unknown";
            FAIL("solver failed: %s", reason.c_str());
        }
        for (auto f : {summary.forces.CX, summary.forces.CY, summary.forces.CZ})
            if (!std::isfinite(f) || std::fabs(f) > 1e6f) FAIL("non-finite force component");
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_state_roundtrip();
    result |= test_fluxes();
    result |= test_solver_uniform();
    result |= test_farfield_boundary();
    result |= test_wall_forces();
    result |= test_amr_solver_loop();
    result |= test_euler_residual_cpu_uniform();
    result |= test_edge_case_meshes();
    result |= test_tet_mesh_solver();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}


