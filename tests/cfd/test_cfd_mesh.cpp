#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/element_types.hpp"
#include "aero/cfd/mesh_io.hpp"
#include "aero/cfd/mesh_io_cgns.hpp"
#include "aero/cfd/mesh_validator.hpp"
#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/amr_sensor.hpp"
#include "aero/cfd/amr_interpolate.hpp"
#include "aero/cfd/amr_hanging.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_residual.hpp"
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

static int test_cube_mesh() {
    TEST("CFD-MESH-1 structured cube has valid metrics");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 13);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("%s", report.message.c_str());
        if (report.nodes != 13*13*13) FAIL("nodes=%d", report.nodes);
        if (report.cells <= 0) FAIL("cells=%d", report.cells);
        if (report.slip_wall_faces <= 0) FAIL("slip_wall_faces=%d", report.slip_wall_faces);
        if (report.farfield_faces <= 0) FAIL("farfield_faces=%d", report.farfield_faces);
        if (report.min_volume <= 0.0f) FAIL("min_volume=%g", report.min_volume);
        PASS;
    }

    TEST("CFD-MESH-2 cube wall classification survives off-surface vertices");
    {
        auto mesh13 = generate_structured_cube_mesh(5.0f, 13);
        auto report13 = compute_mesh_metrics(mesh13);
        auto mesh17 = generate_structured_cube_mesh(5.0f, 17);
        auto report17 = compute_mesh_metrics(mesh17);
        if (report13.slip_wall_faces <= 0) FAIL("n=13 wall faces=%d", report13.slip_wall_faces);
        if (report17.slip_wall_faces <= 0) FAIL("n=17 wall faces=%d", report17.slip_wall_faces);
        if (report13.no_slip_wall_faces != 0) FAIL("unexpected no-slip wall faces=%d", report13.no_slip_wall_faces);
        PASS;
    }

    TEST("CFD-MESH-3 cube wall normal area is closed");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 13);
        Real sx = 0.0f;
        Real sy = 0.0f;
        Real sz = 0.0f;
        for (const auto& face : mesh.faces) {
            if (face.boundary != BoundaryKind::SlipWall) continue;
            sx += face.nx * face.area;
            sy += face.ny * face.area;
            sz += face.nz * face.area;
        }
        if (std::fabs(sx) > 1e-5f) FAIL("sx=%g", sx);
        if (std::fabs(sy) > 1e-5f) FAIL("sy=%g", sy);
        if (std::fabs(sz) > 1e-5f) FAIL("sz=%g", sz);
        PASS;
    }

    return 0;
}

static int test_flat_plate_mesh() {
    TEST("CFD-MESH-4 flat plate wall area matches geometry");
    {
        Real length = 0.5f;
        Real width = 0.05f;
        auto mesh = generate_flat_plate_mesh(length, width, 0.1f, 1e-5f, 1.12f, 30, 3, 50);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("%s", report.message.c_str());
        if (report.no_slip_wall_faces <= 0) FAIL("wall faces=%d", report.no_slip_wall_faces);
        Real area = boundary_area(mesh, BoundaryKind::NoSlipWall);
        Real expected = length * width;
        Real rel = std::fabs(area - expected) / expected;
        if (rel > 1e-5f) FAIL("wall area=%g expected=%g rel=%g", area, expected, rel);
        if (report.min_wall_distance <= 0.0f) FAIL("min wall distance=%g", report.min_wall_distance);
        if (std::fabs(report.min_wall_distance - 2.5e-6f) > 5e-7f) {
            FAIL("min wall distance=%g", report.min_wall_distance);
        }
        PASS;
    }

    TEST("CFD-MESH-5 flat plate farfield exists and mesh is positive");
    {
        auto mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 10, 3, 12);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("%s", report.message.c_str());
        if (report.farfield_faces <= 0) FAIL("farfield_faces=%d", report.farfield_faces);
        if (report.min_h <= 0.0f || report.max_h <= report.min_h) {
            FAIL("h range=[%g,%g]", report.min_h, report.max_h);
        }
        PASS;
    }

    return 0;
}

static int test_su2_round_trip() {
    TEST("CFD-MESH-IO-1 SU2 round-trip: hex mesh read/write/read matches");
    {
        CfdMesh original = generate_structured_hex_mesh(6);
        compute_mesh_metrics(original);

        const char* tmp_path = "test_mesh_su2_temp.su2";
        std::string err;
        if (!write_mesh_su2(original, tmp_path, &err)) FAIL("write: %s", err.c_str());

        CfdMesh reloaded;
        if (!read_mesh_su2(tmp_path, reloaded, &err)) FAIL("read: %s", err.c_str());
        compute_mesh_metrics(reloaded);

        if (reloaded.nodes.size() != original.nodes.size()) FAIL("node count: %zu vs %zu", reloaded.nodes.size(), original.nodes.size());
        if (reloaded.cells.size() != original.cells.size()) FAIL("cell count: %zu vs %zu", reloaded.cells.size(), original.cells.size());

        for (std::size_t i = 0; i < reloaded.cells.size(); ++i) {
            if (reloaded.cells[i].type != original.cells[i].type) FAIL("cell %zu type mismatch", i);
        }

        std::remove(tmp_path);
        PASS;
    }
    return 0;
}

static int test_su2_invalid_file() {
    TEST("CFD-MESH-IO-3 SU2 import with missing NELEM: read fails gracefully");
    {
        const char* tmp_path = "test_mesh_su2_bad.su2";
        std::FILE* f = std::fopen(tmp_path, "w");
        if (!f) FAIL("cannot create temp file");
        std::fprintf(f, "NDIME= 3\n");
        std::fprintf(f, "NPOIN= 0\n");
        std::fprintf(f, "NMARK= 0\n");
        std::fclose(f);

        CfdMesh mesh;
        std::string err;
        bool ok = read_mesh_su2(tmp_path, mesh, &err);
        if (ok) FAIL("expected read to fail, but it succeeded");

        std::remove(tmp_path);
        PASS;
    }
    return 0;
}

static int test_mesh_quality_detail() {
    TEST("CFD-MESH-IO-4 mesh quality detail on flat plate");
    {
        CfdMesh mesh = generate_flat_plate_mesh();
        compute_mesh_metrics(mesh);
        MeshQualityReport r = compute_mesh_quality_detail(mesh);
        std::printf("\n  flat plate quality report:\n");
        std::printf("  cells=%d faces=%d\n", r.cells, r.faces);
        std::printf("  neg_jac=%d | aspect=[%.2f .. %.2f] avg=%.2f\n", r.negative_jacobian_count, r.min_aspect_ratio, r.max_aspect_ratio, r.avg_aspect_ratio);
        std::printf("  skew=[%.4f .. %.4f] avg=%.4f\n", r.min_skewness, r.max_skewness, r.avg_skewness);
        std::printf("  ortho=[%.1f .. %.1f] avg=%.1f deg\n", r.min_orthogonality, r.max_orthogonality, r.avg_orthogonality);
        std::printf("  closed_surf_err=%g\n", r.closed_surface_error);
        if (!r.valid) FAIL("detail: %s", r.message.c_str());
        if (r.negative_jacobian_count != 0) FAIL("neg jac=%d", r.negative_jacobian_count);
        if (r.min_volume <= 0.0f) FAIL("min vol=%g", r.min_volume);
        PASS;
    }

    TEST("CFD-MESH-IO-5 mesh quality on cube mesh (25^3)");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 25);
        compute_mesh_metrics(mesh);
        MeshQualityReport r = compute_mesh_quality_detail(mesh);
        std::printf("\n  cube quality report:\n");
        std::printf("  cells=%d faces=%d\n", r.cells, r.faces);
        std::printf("  neg_jac=%d | aspect=[%.2f .. %.2f] avg=%.2f\n", r.negative_jacobian_count, r.min_aspect_ratio, r.max_aspect_ratio, r.avg_aspect_ratio);
        std::printf("  skew=[%.4f .. %.4f] avg=%.4f\n", r.min_skewness, r.max_skewness, r.avg_skewness);
        std::printf("  ortho=[%.1f .. %.1f] avg=%.1f deg\n", r.min_orthogonality, r.max_orthogonality, r.avg_orthogonality);
        std::printf("  vol=[%g .. %g]\n", r.min_volume, r.max_volume);
        std::printf("  closed_surf_err=%g\n", r.closed_surface_error);
        if (!r.valid) FAIL("detail: %s", r.message.c_str());
        if (r.negative_jacobian_count != 0) FAIL("neg jac=%d", r.negative_jacobian_count);
        if (r.min_volume <= 0.0f) FAIL("min vol=%g", r.min_volume);
        PASS;
    }

    TEST("CFD-MESH-IO-6 mesh quality on hex mesh");
    {
        CfdMesh mesh = generate_structured_hex_mesh(6);
        compute_mesh_metrics(mesh);
        MeshQualityReport r = compute_mesh_quality_detail(mesh);
        std::printf("\n  hex quality report:\n");
        std::printf("  cells=%d faces=%d\n", r.cells, r.faces);
        std::printf("  neg_jac=%d | aspect=[%.2f .. %.2f] avg=%.2f\n", r.negative_jacobian_count, r.min_aspect_ratio, r.max_aspect_ratio, r.avg_aspect_ratio);
        std::printf("  skew=[%.4f .. %.4f] avg=%.4f\n", r.min_skewness, r.max_skewness, r.avg_skewness);
        std::printf("  ortho=[%.1f .. %.1f] avg=%.1f deg\n", r.min_orthogonality, r.max_orthogonality, r.avg_orthogonality);
        std::printf("  closed_surf_err=%g\n", r.closed_surface_error);
        if (!r.valid) FAIL("detail: %s", r.message.c_str());
        if (r.negative_jacobian_count != 0) FAIL("neg jac=%d", r.negative_jacobian_count);
        if (r.min_volume <= 0.0f) FAIL("min vol=%g", r.min_volume);
        PASS;
    }

    return 0;
}

static int test_hex8_refinement() {
    TEST("CFD-AMR-3 single hex refinement: 1->8, volume sum conserved");
    {
        CfdMesh mesh = generate_structured_hex_mesh(3); // 3x3x3 = 8 hexes
        Real vol_before = 0.0f;
        for (const auto& c : mesh.cells) vol_before += c.volume;

        // Refine the first hex only
        std::vector<RefinementRequest> requests = {{0, RefinementFlag::Refine}};
        std::vector<RefinementRecord> records;
        std::string err;
        bool ok = refine_cells(mesh, requests, &records, &err);
        if (!ok) FAIL("refine_cells failed: %s", err.c_str());

        // 1 hex replaced by 8 = original 8 - 1 + 8 = 15 cells
        if (mesh.cells.size() != 15u) FAIL("expected 15 cells, got %zu (before=%d)",
            mesh.cells.size(), mesh.cells.size() - 7);

        Real vol_sum = 0.0f;
        int hex_count = 0;
        for (const auto& c : mesh.cells) {
            if (c.volume <= 0.0f) FAIL("cell has zero/negative volume=%g", c.volume);
            if (c.type == ElementType::HEX8) ++hex_count;
            vol_sum += c.volume;
        }
        if (hex_count != 15) FAIL("expected 15 HEX8 cells, got %d", hex_count);

        Real rel_err = std::fabs(vol_sum - vol_before) / vol_before;
        if (rel_err > 2e-6f) FAIL("volume mismatch: before=%g, after=%g, rel_err=%g",
                                    vol_before, vol_sum, rel_err);

        auto rep = compute_mesh_metrics(mesh);
        if (rep.min_volume <= 0.0f) FAIL("min volume after refine=%g", rep.min_volume);
        PASS;
    }

    TEST("CFD-AMR-4 full hex mesh refinement: volume sum conserved");
    {
        CfdMesh mesh = generate_structured_hex_mesh(4); // 4x4x4 = 27 hexes
        Real vol_before = 0.0f;
        for (const auto& c : mesh.cells) vol_before += c.volume;

        // Refine all cells
        std::vector<RefinementRequest> requests;
        for (int i = 0; i < static_cast<int>(mesh.cells.size()); ++i)
            requests.push_back({i, RefinementFlag::Refine});

        std::string err;
        bool ok = refine_cells(mesh, requests, nullptr, &err);
        if (!ok) FAIL("refine_cells failed: %s", err.c_str());

        if (mesh.cells.size() != 27u * 8) FAIL("expected 216 cells, got %zu", mesh.cells.size());

        Real vol_sum = 0.0f;
        for (const auto& c : mesh.cells) {
            if (c.volume <= 0.0f) FAIL("cell volume=%g", c.volume);
            vol_sum += c.volume;
        }
        Real rel_err = std::fabs(vol_sum - vol_before) / vol_before;
        // Allow FP32 accumulation error over 216 child cells
        if (rel_err > 2e-6f) FAIL("volume mismatch: before=%g, after=%g, rel_err=%g",
                                    vol_before, vol_sum, rel_err);
        PASS;
    }
    return 0;
}

static int test_gradient_sensor() {
    TEST("CFD-AMR-5 gradient sensor flags shock-like discontinuity");
    {
        // Build a small cube mesh with two regions: left half rho=1, right half rho=10
        // The sensor should flag cells at the interface.
        CfdMesh mesh = generate_structured_cube_mesh(2.0f, 7); // 6x6x6 = 216 cells
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            Real cx = mesh.cells[i].cx;
            q[i].rho = (cx < 0.0f) ? 1.0f : 10.0f;
            q[i].rho_u = q[i].rho_v = q[i].rho_w = 0.0f;
            q[i].rho_E = 1.0f;
        }

        AmrConfig cfg;
        cfg.refine_tol = 0.3f;
        cfg.coarsen_tol = 0.05f;

        auto requests = compute_gradient_sensor(mesh, q, cfg);
        int n_refine = 0, n_coarsen = 0;
        for (const auto& r : requests) {
            if (r.flag == RefinementFlag::Refine) ++n_refine;
            if (r.flag == RefinementFlag::Coarsen) ++n_coarsen;
        }
        // At the x=0 interface plane, ~6*6 = 36 cells on each side
        // should be flagged for refine. Total depends on mesh topology.
        if (n_refine == 0) FAIL("expected some cells flagged for refine, got 0");
        std::printf("  (refine=%d, coarsen=%d, total=%d)", n_refine, n_coarsen, n_cells);
        PASS;
    }

    TEST("CFD-AMR-6 gradient sensor with uniform flow: no refinement");
    {
        CfdMesh mesh = generate_structured_cube_mesh(2.0f, 7);
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            q[i].rho = 1.225f;
            q[i].rho_u = q[i].rho_v = q[i].rho_w = 0.0f;
            q[i].rho_E = 1.0f;
        }

        AmrConfig cfg;
        cfg.refine_tol = 0.5f;
        cfg.coarsen_tol = 0.1f;

        auto requests = compute_gradient_sensor(mesh, q, cfg);
        for (const auto& r : requests) {
            if (r.flag == RefinementFlag::Refine) FAIL("uniform flow should not trigger refine");
        }
        PASS;
    }
    return 0;
}

static int test_yplus_sensor() {
    TEST("CFD-AMR-YPLUS-1 coarse wall mesh: y+ > target triggers refine");
    {
        auto mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-3f, 1.12f, 10, 3, 12);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("mesh metrics: %s", report.message.c_str());

        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        Real gamma = 1.4f;
        Real mach = 0.5f;
        Real speed = mach / std::sqrt(gamma);
        Real inv_gm1 = 1.0f / (gamma - 1.0f);
        for (int i = 0; i < n_cells; ++i) {
            q[i].rho = 1.0f;
            q[i].rho_u = speed;
            q[i].rho_v = 0.0f;
            q[i].rho_w = 0.0f;
            q[i].rho_E = inv_gm1 + 0.5f * speed * speed;
        }

        AmrConfig cfg;
        cfg.yplus_target = 1.0f;

        auto requests = compute_yplus_sensor(mesh, q, cfg, gamma, 1e6f, 1.0f, 288.15f, 110.4f,
                                             TurbulenceModel::LAMINAR);
        int n_refine = 0;
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine) ++n_refine;
        if (n_refine == 0) FAIL("expected wall cells flagged for refine, got 0");
        std::printf("  (refine=%d, total=%d)", n_refine, n_cells);
        PASS;
    }

    TEST("CFD-AMR-YPLUS-2 fine wall mesh: low Re, no refine");
    {
        // Low Re (1e3) with fine first cell (1e-5) → y+ < yplus_target → no refine
        auto mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 10, 3, 12);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("mesh metrics: %s", report.message.c_str());

        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        Real gamma = 1.4f;
        Real mach = 0.5f;
        Real speed = mach / std::sqrt(gamma);
        Real inv_gm1 = 1.0f / (gamma - 1.0f);
        for (int i = 0; i < n_cells; ++i) {
            q[i].rho = 1.0f;
            q[i].rho_u = speed;
            q[i].rho_v = 0.0f;
            q[i].rho_w = 0.0f;
            q[i].rho_E = inv_gm1 + 0.5f * speed * speed;
        }

        AmrConfig cfg;
        cfg.yplus_target = 5.0f;

        auto requests = compute_yplus_sensor(mesh, q, cfg, gamma, 1e3f, 1.0f, 288.15f, 110.4f,
                                             TurbulenceModel::LAMINAR);
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine) FAIL("fine mesh should not trigger refine");
        PASS;
    }

    TEST("CFD-AMR-QCRIT-1 Q-criterion on uniform flow: no refinement");
    {
        auto mesh = generate_structured_cube_mesh(2.0f, 7);
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        Real gamma = 1.4f;
        Real speed = 0.5f / std::sqrt(gamma);
        Real inv_gm1 = 1.0f / (gamma - 1.0f);
        for (int i = 0; i < n_cells; ++i) {
            q[i].rho = 1.0f;
            q[i].rho_u = speed;
            q[i].rho_v = 0.0f;
            q[i].rho_w = 0.0f;
            q[i].rho_E = inv_gm1 + 0.5f * speed * speed;
        }

        AmrConfig cfg;
        cfg.refine_tol = 0.1f;
        cfg.coarsen_tol = 0.01f;

        auto requests = compute_qcriterion_sensor(mesh, q, cfg, gamma);
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine) FAIL("uniform flow should not trigger Q refine");
        PASS;
    }

    TEST("CFD-AMR-QCRIT-2 Q-criterion detects rotational flow");
    {
        // Create rotational flow: u = 0.5 - 0.3*y, v = 0.3*x, w = 0
        // du/dy = -0.3, dv/dx = 0.3 → Q = -(-0.3*0.3) = 0.09 > 0 (vortex cores)
        auto mesh = generate_structured_cube_mesh(2.0f, 9);
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        Real gamma = 1.4f;
        Real inv_gm1 = 1.0f / (gamma - 1.0f);
        for (int i = 0; i < n_cells; ++i) {
            Real cx = mesh.cells[i].cx;
            Real cy = mesh.cells[i].cy;
            Real u = 0.5f - 0.3f * cy;
            Real v = 0.3f * cx;
            q[i].rho = 1.0f;
            q[i].rho_u = u;
            q[i].rho_v = v;
            q[i].rho_w = 0.0f;
            q[i].rho_E = inv_gm1 + 0.5f * (u*u + v*v);
        }

        AmrConfig cfg;
        cfg.refine_tol = 0.05f;
        cfg.coarsen_tol = 0.005f;

        auto requests = compute_qcriterion_sensor(mesh, q, cfg, gamma);
        int n_refine = 0;
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine) ++n_refine;
        if (n_refine == 0) FAIL("expected Q-criterion refine in rotational flow, got 0");
        std::printf("  (refine=%d, total=%d)", n_refine, n_cells);
        PASS;
    }

    TEST("CFD-AMR-WAKE-1 wake cone sensor flags cells inside cone");
    {
        // Cube mesh spans [-2, 2] with 7 nodes/dim → ~1/3 spacing
        auto mesh = generate_structured_cube_mesh(2.0f, 7);
        compute_mesh_metrics(mesh);

        WakeConeConfig cone;
        cone.origin_x = -2.0f;       // apex at left face
        cone.origin_y = 0.0f;
        cone.origin_z = 0.0f;
        cone.axis_x = 1.0f;
        cone.axis_y = 0.0f;
        cone.axis_z = 0.0f;
        cone.half_angle_deg = 45.0f; // wide cone → many cells inside
        cone.length = 4.0f;

        AmrConfig cfg;
        cfg.refine_tol = 1.0f;

        auto requests = compute_wake_cone_sensor(mesh, cfg, cone);
        int n_refine = 0;
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine) ++n_refine;
        if (n_refine == 0) FAIL("expected some cells inside 45-deg wake cone, got 0");
        std::printf("  (refine=%d, total=%d)", n_refine, static_cast<int>(mesh.cells.size()));
        PASS;
    }

    TEST("CFD-AMR-WAKE-2 wake cone sensor: cells outside narrow cone not flagged");
    {
        auto mesh = generate_structured_cube_mesh(2.0f, 7);
        compute_mesh_metrics(mesh);

        WakeConeConfig cone;
        cone.origin_x = -2.0f;
        cone.origin_y = 0.0f;
        cone.origin_z = 0.0f;
        cone.axis_x = 1.0f;
        cone.axis_y = 0.0f;
        cone.axis_z = 0.0f;
        cone.half_angle_deg = 1.0f;  // very narrow cone
        cone.length = 4.0f;

        AmrConfig cfg;
        cfg.refine_tol = 1.0f;

        auto requests = compute_wake_cone_sensor(mesh, cfg, cone);
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine)
                FAIL("cells should not be inside 1-deg wake cone");
        PASS;
    }

    TEST("CFD-AMR-TKE-1 TKE ratio flags high-turbulence cells");
    {
        auto mesh = generate_structured_cube_mesh(2.0f, 7);
        compute_mesh_metrics(mesh);
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        std::vector<Real> sst_k(n_cells);
        Real gamma = 1.4f;
        Real speed = 0.5f;
        Real inv_gm1 = 1.0f / (gamma - 1.0f);
        for (int i = 0; i < n_cells; ++i) {
            q[i].rho = 1.0f;
            q[i].rho_u = speed;
            q[i].rho_v = 0.0f;
            q[i].rho_w = 0.0f;
            q[i].rho_E = inv_gm1 + 0.5f * speed * speed;
            // TKE ratio = k / 0.5*U² = 0.02 / (0.5*0.25) = 0.02/0.125 = 0.16 > 0.05 → refine
            sst_k[i] = 0.02f;
        }

        AmrConfig cfg;
        cfg.refine_tol = 0.01f;
        cfg.coarsen_tol = 0.001f;

        auto requests = compute_tke_ratio_sensor(mesh, q, cfg, gamma, TurbulenceModel::SST, &sst_k, 0.05f);
        int n_refine = 0;
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine) ++n_refine;
        if (n_refine == 0) FAIL("expected TKE ratio refine with k=0.02, U=0.5");
        std::printf("  (refine=%d, total=%d)", n_refine, n_cells);
        PASS;
    }

    TEST("CFD-AMR-TKE-2 TKE ratio no-op for non-SST");
    {
        auto mesh = generate_structured_cube_mesh(2.0f, 7);
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        std::vector<Real> sst_k(n_cells, 0.02f);

        AmrConfig cfg;
        auto requests = compute_tke_ratio_sensor(mesh, q, cfg, 1.4f, TurbulenceModel::LAMINAR, &sst_k, 0.05f);
        for (const auto& r : requests)
            if (r.flag != RefinementFlag::Unchanged)
                FAIL("non-SST should produce no refinement");
        PASS;
    }

    TEST("CFD-AMR-TKE-3 TKE ratio with zero k: no refinement");
    {
        auto mesh = generate_structured_cube_mesh(2.0f, 7);
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        std::vector<Real> sst_k(n_cells, 0.0f);
        Real gamma = 1.4f;
        Real speed = 0.5f;
        Real inv_gm1 = 1.0f / (gamma - 1.0f);
        for (int i = 0; i < n_cells; ++i) {
            q[i].rho = 1.0f;
            q[i].rho_u = speed;
            q[i].rho_v = 0.0f;
            q[i].rho_w = 0.0f;
            q[i].rho_E = inv_gm1 + 0.5f * speed * speed;
        }

        AmrConfig cfg;
        auto requests = compute_tke_ratio_sensor(mesh, q, cfg, gamma, TurbulenceModel::SST, &sst_k, 0.05f);
        for (const auto& r : requests)
            if (r.flag != RefinementFlag::Unchanged)
                FAIL("zero k should not trigger refine");
        PASS;
    }

    TEST("CFD-AMR-SL-1 shear-layer sensor: LAMINAR no-op");
    {
        auto mesh = generate_structured_cube_mesh(2.0f, 9);
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        AmrConfig cfg;
        auto requests = compute_shear_layer_sensor(mesh, q, cfg, 1.4f, TurbulenceModel::LAMINAR, nullptr, 0.3f);
        for (const auto& r : requests)
            if (r.flag != RefinementFlag::Unchanged)
                FAIL("LAMINAR should produce no refinement");
        PASS;
    }

    TEST("CFD-AMR-SL-2 shear-layer sensor: SA under-resolved shear flags refine");
    {
        // SA with high nu_tilde (> 0.0054) so modeled_k dominates resolved_k
        // → ratio < 0.35 → under-resolved → refine
        auto mesh = generate_structured_cube_mesh(2.0f, 9);
        compute_mesh_metrics(mesh);
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        Real gamma = 1.4f;
        Real inv_gm1 = 1.0f / (gamma - 1.0f);
        for (int i = 0; i < n_cells; ++i) {
            Real cy = mesh.cells[i].cy;
            Real u = 0.5f + 0.3f * cy;
            q[i].rho = 1.0f;
            q[i].rho_u = u;
            q[i].rho_v = 0.0f;
            q[i].rho_w = 0.0f;
            q[i].rho_E = inv_gm1 + 0.5f * u * u;
            q[i].rho_nu_tilde = 0.01f;  // high modeled TKE → under-resolved
        }

        AmrConfig cfg;

        auto requests = compute_shear_layer_sensor(mesh, q, cfg, gamma, TurbulenceModel::SA, nullptr, 0.3f);
        int n_refine = 0;
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine) ++n_refine;
        if (n_refine == 0) FAIL("expected shear-layer refine with nu_tilde=0.01 in shear");
        std::printf("  (refine=%d, total=%d)", n_refine, n_cells);
        PASS;
    }

    TEST("CFD-AMR-SL-3 shear-layer sensor: SA uniform flow nearly zero");
    {
        // Uniform flow → gradients nearly zero → skip cells (modeled_k too small)
        auto mesh = generate_structured_cube_mesh(2.0f, 9);
        compute_mesh_metrics(mesh);
        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        Real gamma = 1.4f;
        Real inv_gm1 = 1.0f / (gamma - 1.0f);
        Real speed = 0.5f;
        for (int i = 0; i < n_cells; ++i) {
            q[i].rho = 1.0f;
            q[i].rho_u = speed;
            q[i].rho_v = 0.0f;
            q[i].rho_w = 0.0f;
            q[i].rho_E = inv_gm1 + 0.5f * speed * speed;
            q[i].rho_nu_tilde = 0.01f;
        }

        AmrConfig cfg;

        auto requests = compute_shear_layer_sensor(mesh, q, cfg, gamma, TurbulenceModel::SA, nullptr, 0.3f);
        int n_refine = 0;
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine) ++n_refine;
        // Boundary gradient noise on tet mesh may trigger a small fraction of cells
        if (n_refine > n_cells / 4)
            FAIL("uniform flow: too many cells flagged (got %d/%d)", n_refine, n_cells);
        std::printf("  (refine=%d, total=%d)", n_refine, n_cells);
        PASS;
    }

    TEST("CFD-AMR-YPLUS-3 y+ estimate matches expected range");
    {
        // Known wall-adjacent cell: d=first_height/2, rho=1, u=speed
        // y+ = sqrt(Re*d*rho*|u|)*sqrt(mu_eff)/mu
        // With first_height=1e-3, d≈5e-4, Re=1e6, u=0.422 → y+ ≈ 550 (laminar)
        // Verify at least one cell has y+ > 1.0 (triggers refine) and all y+ > 0
        auto mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-3f, 1.12f, 10, 3, 12);
        compute_mesh_metrics(mesh);

        int n_cells = static_cast<int>(mesh.cells.size());
        std::vector<ConservativeState> q(n_cells);
        Real gamma = 1.4f;
        Real speed = 0.5f / std::sqrt(gamma);
        Real inv_gm1 = 1.0f / (gamma - 1.0f);
        for (int i = 0; i < n_cells; ++i) {
            q[i].rho = 1.0f;
            q[i].rho_u = speed;
            q[i].rho_v = 0.0f;
            q[i].rho_w = 0.0f;
            q[i].rho_E = inv_gm1 + 0.5f * speed * speed;
        }

        AmrConfig cfg;
        cfg.yplus_target = 1.0f;

        auto requests = compute_yplus_sensor(mesh, q, cfg, gamma, 1e6f, 1.0f, 288.15f, 110.4f,
                                             TurbulenceModel::LAMINAR);
        int n_refine = 0;
        for (const auto& r : requests)
            if (r.flag == RefinementFlag::Refine) ++n_refine;
        if (n_refine == 0) FAIL("expected some refine from coarse mesh");
        std::printf("  (refine=%d, total=%d)", n_refine, n_cells);
        PASS;
    }
    return 0;
}

static int test_tet4_refinement() {
    TEST("CFD-AMR-1 single tet refinement: 1->8, volume sum conserved");
    {
        // Build a single TET4: unit right tetrahedron
        CfdMesh mesh;
        mesh.nodes.resize(4);
        mesh.nodes[0] = {0.0f, 0.0f, 0.0f};
        mesh.nodes[1] = {1.0f, 0.0f, 0.0f};
        mesh.nodes[2] = {0.0f, 1.0f, 0.0f};
        mesh.nodes[3] = {0.0f, 0.0f, 1.0f};

        CfdCell cell;
        cell.type = ElementType::TET4;
        cell.node[0] = 0; cell.node[1] = 1; cell.node[2] = 2; cell.node[3] = 3;
        mesh.cells.push_back(cell);
        rebuild_mesh_faces(mesh);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("initial mesh invalid: %s", report.message.c_str());
        Real vol_before = mesh.cells[0].volume;
        if (vol_before <= 0.0f) FAIL("initial cell volume=%g", vol_before);

        // Refine
        std::vector<RefinementRequest> requests = {{0, RefinementFlag::Refine}};
        std::vector<RefinementRecord> records;
        std::string err;
        bool ok = refine_cells(mesh, requests, &records, &err);
        if (!ok) FAIL("refine_cells failed: %s", err.c_str());

        // Check cell count
        if (mesh.cells.size() != 8u) FAIL("expected 8 cells, got %zu", mesh.cells.size());

        // Volume sum conservation
        Real vol_sum = 0.0f;
        for (const auto& c : mesh.cells) {
            if (c.volume <= 0.0f) FAIL("child cell has zero/negative volume=%g", c.volume);
            vol_sum += c.volume;
        }
        Real rel_err = std::fabs(vol_sum - vol_before) / vol_before;
        // Allow FP32 accumulation error (~5 ULP for 8 tet volumes sum)
        if (rel_err > 1e-6f) FAIL("volume mismatch: before=%g, after=%g, rel_err=%g",
                                    vol_before, vol_sum, rel_err);

        // Face count: 4 faces * 8 tets / 2 (shared) = 16 interior + 4 boundary = 20
        // (each original tet face becomes 4 smaller faces)
        auto rep2 = compute_mesh_metrics(mesh);
        if (rep2.min_volume <= 0.0f) FAIL("min volume after refine=%g", rep2.min_volume);
        if (rep2.negative_jacobian_count != 0) FAIL("negative_jacobian_count=%d after refine", rep2.negative_jacobian_count);
        PASS;
    }

    TEST("CFD-AMR-2 refine-coarsen record consistency");
    {
        // Same single tet as above
        CfdMesh mesh2;
        mesh2.nodes.resize(4);
        mesh2.nodes[0] = {0.0f, 0.0f, 0.0f};
        mesh2.nodes[1] = {1.0f, 0.0f, 0.0f};
        mesh2.nodes[2] = {0.0f, 1.0f, 0.0f};
        mesh2.nodes[3] = {0.0f, 0.0f, 1.0f};
        CfdCell cell2;
        cell2.type = ElementType::TET4;
        cell2.node[0] = 0; cell2.node[1] = 1; cell2.node[2] = 2; cell2.node[3] = 3;
        mesh2.cells.push_back(cell2);
        rebuild_mesh_faces(mesh2);
        compute_mesh_metrics(mesh2);

        std::vector<RefinementRequest> reqs = {{0, RefinementFlag::Refine}};
        std::vector<RefinementRecord> recs;
        std::string err;
        refine_cells(mesh2, reqs, &recs, &err);

        if (recs.size() != 1u) FAIL("expected 1 refinement record, got %zu", recs.size());
        if (recs[0].parent_cell_id != 0) FAIL("parent_cell_id expected 0, got %d", recs[0].parent_cell_id);
        if (recs[0].n_children != 8) FAIL("expected 8 children, got %d", recs[0].n_children);
        for (int i = 0; i < 8; ++i) {
            if (recs[0].child_cell_ids[i] < 0) FAIL("child %d has invalid id %d", i, recs[0].child_cell_ids[i]);
        }
        // Each child should have parent_id = 0
        for (const auto& c : mesh2.cells) {
            if (c.parent_id != 0) FAIL("child cell parent_id should be 0, got %d", c.parent_id);
        }
        PASS;
    }

    return 0;
}

static int test_cgns_fallback() {
    TEST("CFD-MESH-IO-2 CGNS fallback when library unavailable");
    {
        CfdMesh mesh;
        std::string err;
        bool ok = read_mesh_cgns("nonexistent.cgns", mesh, &err);
        if (ok) FAIL("expected CGNS read to fail but it succeeded");
        if (err.empty()) FAIL("expected non-empty error message");
        std::printf("  CGNS fallback message: \"%s\"\n", err.c_str());
        PASS;
    }
    return 0;
}

static int test_hanging_faces() {
    TEST("CFD-AMR-9 hanging face detection with 2-hex mesh");
    {
        CfdMesh mesh;
        mesh.nodes.resize(12);
        mesh.nodes[0] = {0,0,0}; mesh.nodes[1] = {1,0,0}; mesh.nodes[2] = {1,1,0}; mesh.nodes[3] = {0,1,0};
        mesh.nodes[4] = {0,0,1}; mesh.nodes[5] = {1,0,1}; mesh.nodes[6] = {1,1,1}; mesh.nodes[7] = {0,1,1};
        mesh.nodes[8] = {2,0,0}; mesh.nodes[9] = {2,1,0}; mesh.nodes[10] = {2,0,1}; mesh.nodes[11] = {2,1,1};

        mesh.cells.resize(2);
        mesh.cells[0].type = ElementType::HEX8;
        mesh.cells[0].node[0]=0; mesh.cells[0].node[1]=1; mesh.cells[0].node[2]=2; mesh.cells[0].node[3]=3;
        mesh.cells[0].node[4]=4; mesh.cells[0].node[5]=5; mesh.cells[0].node[6]=6; mesh.cells[0].node[7]=7;
        mesh.cells[1].type = ElementType::HEX8;
        mesh.cells[1].node[0]=1; mesh.cells[1].node[1]=8; mesh.cells[1].node[2]=9; mesh.cells[1].node[3]=2;
        mesh.cells[1].node[4]=5; mesh.cells[1].node[5]=10; mesh.cells[1].node[6]=11; mesh.cells[1].node[7]=6;

        mesh.cells[0].refinement_level = 1;

        rebuild_mesh_faces(mesh);
        compute_mesh_metrics(mesh);

        auto hanging = detect_hanging_faces(mesh);
        if (hanging.empty()) FAIL("expected at least one hanging face");
        bool found = false;
        for (const auto& h : hanging) {
            if (h.coarse_cell_id == 1 && h.fine_cell_id == 0) { found = true; break; }
        }
        if (!found) FAIL("hanging face between cell 0(level=1) and cell 1(level=0) not found");
        PASS;
    }

    TEST("CFD-AMR-10 hanging face interpolation with non-zero gradient");
    {
        CfdMesh mesh;
        mesh.nodes.resize(12);
        mesh.nodes[0] = {0,0,0}; mesh.nodes[1] = {1,0,0}; mesh.nodes[2] = {1,1,0}; mesh.nodes[3] = {0,1,0};
        mesh.nodes[4] = {0,0,1}; mesh.nodes[5] = {1,0,1}; mesh.nodes[6] = {1,1,1}; mesh.nodes[7] = {0,1,1};
        mesh.nodes[8] = {2,0,0}; mesh.nodes[9] = {2,1,0}; mesh.nodes[10] = {2,0,1}; mesh.nodes[11] = {2,1,1};

        mesh.cells.resize(2);
        mesh.cells[0].type = ElementType::HEX8;
        mesh.cells[0].node[0]=0; mesh.cells[0].node[1]=1; mesh.cells[0].node[2]=2; mesh.cells[0].node[3]=3;
        mesh.cells[0].node[4]=4; mesh.cells[0].node[5]=5; mesh.cells[0].node[6]=6; mesh.cells[0].node[7]=7;
        mesh.cells[1].type = ElementType::HEX8;
        mesh.cells[1].node[0]=1; mesh.cells[1].node[1]=8; mesh.cells[1].node[2]=9; mesh.cells[1].node[3]=2;
        mesh.cells[1].node[4]=5; mesh.cells[1].node[5]=10; mesh.cells[1].node[6]=11; mesh.cells[1].node[7]=6;

        mesh.cells[0].refinement_level = 1;

        rebuild_mesh_faces(mesh);
        compute_mesh_metrics(mesh);

        auto hanging = detect_hanging_faces(mesh);
        if (hanging.empty()) FAIL("expected hanging faces");

        int nc = static_cast<int>(mesh.cells.size());
        int nf = static_cast<int>(mesh.faces.size());
        std::vector<ConservativeState> q_cell(nc);
        q_cell[0] = {2.0f, 20.0f, 0.0f, 0.0f, 40000.0f, 0.0f};
        q_cell[1] = {1.0f, 10.0f, 0.0f, 0.0f, 20000.0f, 0.0f};

        std::vector<CellGradient3> grad_rho(nc), grad_rho_u(nc), grad_rho_v(nc),
            grad_rho_w(nc), grad_rho_E(nc), grad_rho_nu(nc);
        grad_rho[1].gx = 0.5f;

        std::vector<ConservativeState> qL(nf), qR(nf);
        for (int i = 0; i < nf; ++i) {
            int l = mesh.faces[i].left_cell;
            int r = mesh.faces[i].right_cell;
            if (l >= 0) qL[i] = q_cell[l];
            if (r >= 0) qR[i] = q_cell[r];
        }

        apply_hanging_interpolation(mesh, hanging, q_cell,
            grad_rho, grad_rho_u, grad_rho_v, grad_rho_w, grad_rho_E, grad_rho_nu,
            qL, qR);

        for (const auto& h : hanging) {
            if (h.coarse_cell_id == 1 && h.fine_cell_id == 0) {
                Real expected_rho = 1.0f + 0.5f * (-0.5f);
                Real rho_at_face;
                if (mesh.faces[h.face_id].left_cell == 1)
                    rho_at_face = qL[h.face_id].rho;
                else
                    rho_at_face = qR[h.face_id].rho;
                if (std::fabs(rho_at_face - expected_rho) > 1e-6f)
                    FAIL("interpolated rho=%g, expected %g", rho_at_face, expected_rho);
                break;
            }
        }
        PASS;
    }

    TEST("CFD-AMR-11 tet refine-coarsen roundtrip: 1->8->1, cells and volume conserved");
    {
        CfdMesh mesh;
        mesh.nodes.resize(4);
        mesh.nodes[0] = {0.0f, 0.0f, 0.0f};
        mesh.nodes[1] = {1.0f, 0.0f, 0.0f};
        mesh.nodes[2] = {0.0f, 1.0f, 0.0f};
        mesh.nodes[3] = {0.0f, 0.0f, 1.0f};

        CfdCell cell;
        cell.type = ElementType::TET4;
        cell.node[0] = 0; cell.node[1] = 1; cell.node[2] = 2; cell.node[3] = 3;
        mesh.cells.push_back(cell);
        rebuild_mesh_faces(mesh);
        compute_mesh_metrics(mesh);
        Real vol_orig = mesh.cells[0].volume;

        // Refine
        std::vector<RefinementRequest> refine_req = {{0, RefinementFlag::Refine}};
        std::vector<RefinementRecord> recs;
        std::string err;
        if (!refine_cells(mesh, refine_req, &recs, &err)) FAIL("refine failed: %s", err.c_str());
        if (mesh.cells.size() != 8u) FAIL("expected 8 after refine, got %zu", mesh.cells.size());

        // Coarsen using records from refine step
        std::vector<RefinementRequest> coarsen_req;
        for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci)
            coarsen_req.push_back({ci, RefinementFlag::Coarsen});
        std::vector<RefinementRecord> recs2;
        if (!refine_cells(mesh, coarsen_req, &recs2, &err, &recs)) FAIL("coarsen failed: %s", err.c_str());
        if (mesh.cells.size() != 1u) FAIL("expected 1 after coarsen, got %zu", mesh.cells.size());

        compute_mesh_metrics(mesh);
        Real vol_final = mesh.cells[0].volume;
        Real rel_err = std::fabs(vol_final - vol_orig) / vol_orig;
        if (rel_err > 1e-6f) FAIL("volume mismatch: orig=%g final=%g rel_err=%g", vol_orig, vol_final, rel_err);
        PASS;
    }

    return 0;
}

static int test_solution_interpolation() {
    TEST("CFD-AMR-7 tet refinement prolongation: children inherit parent state");
    {
        // Single TET4 with known conservative state
        CfdMesh mesh;
        mesh.nodes.resize(4);
        mesh.nodes[0] = {0.0f, 0.0f, 0.0f};
        mesh.nodes[1] = {1.0f, 0.0f, 0.0f};
        mesh.nodes[2] = {0.0f, 1.0f, 0.0f};
        mesh.nodes[3] = {0.0f, 0.0f, 1.0f};

        CfdCell cell;
        cell.type = ElementType::TET4;
        cell.node[0] = 0; cell.node[1] = 1; cell.node[2] = 2; cell.node[3] = 3;
        mesh.cells.push_back(cell);
        rebuild_mesh_faces(mesh);
        compute_mesh_metrics(mesh);
        Real vol_before = mesh.cells[0].volume;
        if (vol_before <= 0.0f) FAIL("initial cell volume=%g", vol_before);

        // Set parent solution
        constexpr int N = 1;
        std::vector<ConservativeState> q_old(N);
        q_old[0] = {1.225f, 100.0f, 0.0f, 0.0f, 200000.0f, 0.0f};

        // Refine
        std::vector<RefinementRequest> requests = {{0, RefinementFlag::Refine}};
        std::vector<RefinementRecord> records;
        std::string err;
        CfdMesh mesh_ref = mesh;
        bool ok = refine_cells(mesh_ref, requests, &records, &err);
        if (!ok) FAIL("refine_cells failed: %s", err.c_str());
        if (mesh_ref.cells.size() != 8u) FAIL("expected 8 cells, got %zu", mesh_ref.cells.size());

        // Prolongate
        std::vector<ConservativeState> q_new;
        prolongate_solution(q_old, mesh, mesh_ref, records, q_new);
        if (q_new.size() != 8u) FAIL("expected 8 states, got %zu", q_new.size());

        // Verify all children inherit parent's conservative state exactly
        for (int i = 0; i < 8; ++i) {
            if (std::fabs(q_new[i].rho - 1.225f) > 1e-7f) FAIL("child %d rho mismatch: %g", i, q_new[i].rho);
            if (std::fabs(q_new[i].rho_u - 100.0f) > 1e-7f) FAIL("child %d rho_u mismatch: %g", i, q_new[i].rho_u);
            if (std::fabs(q_new[i].rho_E - 200000.0f) > 1e-4f) FAIL("child %d rho_E mismatch: %g", i, q_new[i].rho_E);
        }
        PASS;
    }

    TEST("CFD-AMR-8 tet refinement restriction: volume-weighted average recovers parent");
    {
        CfdMesh mesh;
        mesh.nodes.resize(4);
        mesh.nodes[0] = {0.0f, 0.0f, 0.0f};
        mesh.nodes[1] = {1.0f, 0.0f, 0.0f};
        mesh.nodes[2] = {0.0f, 1.0f, 0.0f};
        mesh.nodes[3] = {0.0f, 0.0f, 1.0f};

        CfdCell cell;
        cell.type = ElementType::TET4;
        cell.node[0] = 0; cell.node[1] = 1; cell.node[2] = 2; cell.node[3] = 3;
        mesh.cells.push_back(cell);
        rebuild_mesh_faces(mesh);
        compute_mesh_metrics(mesh);
        Real vol_before = mesh.cells[0].volume;
        if (vol_before <= 0.0f) FAIL("initial cell volume=%g", vol_before);

        constexpr int N = 1;
        std::vector<ConservativeState> q_old(N);
        q_old[0] = {1.225f, 100.0f, 0.0f, 0.0f, 200000.0f, 0.0f};

        std::vector<RefinementRequest> requests = {{0, RefinementFlag::Refine}};
        std::vector<RefinementRecord> records;
        std::string err;
        CfdMesh mesh_ref = mesh;
        refine_cells(mesh_ref, requests, &records, &err);

        std::vector<ConservativeState> q_new;
        prolongate_solution(q_old, mesh, mesh_ref, records, q_new);

        // Coarsen back: restrict children → parent
        std::vector<ConservativeState> q_restored;
        restrict_solution(q_new, mesh_ref, records, q_restored);
        if (q_restored.size() != 1u) FAIL("expected 1 restored state, got %zu", q_restored.size());

        // Since all children have same state (injection), volume-weighted avg = same state
        Real rel_err_rho = std::fabs(q_restored[0].rho - 1.225f) / 1.225f;
        Real rel_err_rho_u = std::fabs(q_restored[0].rho_u - 100.0f) / 100.0f;
        if (rel_err_rho > 1e-6f) FAIL("restored rho mismatch: %g (rel_err=%g)", q_restored[0].rho, rel_err_rho);
        if (rel_err_rho_u > 1e-6f) FAIL("restored rho_u mismatch: %g (rel_err=%g)", q_restored[0].rho_u, rel_err_rho_u);
        PASS;
    }

    return 0;
}

static int test_amr_disabled_regression() {
    TEST("CFD-AMR-12 solver converges with AMR disabled");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 7);
        compute_mesh_metrics(mesh);

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.max_iter = 50;
        cfg.cfl = 0.1f;
        cfg.convergence_tol = 1e-10f;
        cfg.amr.enabled = false;

        auto summary = solver.solve({2.0f, 0.0f, 0.0f}, cfg);
        if (summary.failed) FAIL("solver failed: %s",
            summary.diagnostics.failure.reason.c_str());
        if (summary.residual_history.empty()) FAIL("no residual history");
        PASS;
    }
    return 0;
}

static int test_amr_max_level() {
    TEST("CFD-AMR-13 max_level=1 enforcement limits refinement depth");
    {
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 7);
        compute_mesh_metrics(mesh);

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.max_iter = 11;
        cfg.cfl = 0.1f;
        cfg.convergence_tol = 1e-12f;
        cfg.amr.enabled = true;
        cfg.amr.interval = 10;
        cfg.amr.max_level = 1;
        cfg.amr.refine_tol = 0.0f;

        auto summary = solver.solve({2.0f, 0.0f, 0.0f}, cfg);
        if (summary.failed) FAIL("solver failed");

        const CfdMesh& m = solver.mesh();
        for (int i = 0; i < static_cast<int>(m.cells.size()); ++i) {
            if (m.cells[i].refinement_level > 1) FAIL("cell %d has refinement_level=%d > 1", i, m.cells[i].refinement_level);
        }
        PASS;
    }
    return 0;
}

// Helper: residual L2 norm = sqrt(sum(R^2) / (n_cells * CFD_NVAR))
static Real residual_l2(const std::vector<EulerFlux>& res) {
    Real sum_sq = 0.0f;
    for (const auto& r : res) {
        sum_sq += r.mass * r.mass + r.mom_x * r.mom_x + r.mom_y * r.mom_y +
                  r.mom_z * r.mom_z + r.energy * r.energy + r.turbulence * r.turbulence;
    }
    int n = static_cast<int>(res.size());
    return std::sqrt(sum_sq / (static_cast<Real>(n) * static_cast<Real>(CFD_NVAR)));
}

static int test_amr_order2_hanging() {
    TEST("CFD-AMR-14 2nd-order prolongation: child mass = parent mass after reconstruction");
    {
        CfdMesh mesh;
        mesh.nodes.resize(4);
        mesh.nodes[0] = {0.0f, 0.0f, 0.0f};
        mesh.nodes[1] = {1.0f, 0.0f, 0.0f};
        mesh.nodes[2] = {0.0f, 1.0f, 0.0f};
        mesh.nodes[3] = {0.0f, 0.0f, 1.0f};

        CfdCell cell;
        cell.type = ElementType::TET4;
        cell.node[0] = 0; cell.node[1] = 1; cell.node[2] = 2; cell.node[3] = 3;
        mesh.cells.push_back(cell);
        rebuild_mesh_faces(mesh);
        compute_mesh_metrics(mesh);
        Real vol_parent = mesh.cells[0].volume;
        if (vol_parent <= 0.0f) FAIL("parent volume=%g", vol_parent);

        Real gamma = 1.4f;
        PrimitiveState wp;
        wp.rho = 1.225f; wp.u = 50.0f; wp.v = 10.0f; wp.w = 5.0f;
        wp.p = 101325.0f; wp.nu_tilde = 0.0f;
        ConservativeState q_parent = primitive_to_conservative(wp, gamma);
        Real mass_parent = q_parent.rho * vol_parent;

        std::vector<PrimitiveState> w_old = {wp};
        std::vector<PrimitiveGradient> grad_old(1);
        grad_old[0].drho_dx = 0.1f; grad_old[0].drho_dy = 0.05f; grad_old[0].drho_dz = 0.02f;
        grad_old[0].du_dx = 2.0f; grad_old[0].dp_dx = 1000.0f;

        std::vector<RefinementRequest> requests = {{0, RefinementFlag::Refine}};
        std::vector<RefinementRecord> records;
        std::string err;
        CfdMesh mesh_ref = mesh;
        bool ok = refine_cells(mesh_ref, requests, &records, &err);
        if (!ok) FAIL("refine failed: %s", err.c_str());
        if (mesh_ref.cells.size() != 8u) FAIL("expected 8 cells, got %zu", mesh_ref.cells.size());

        std::vector<ConservativeState> q_new;
        std::vector<ConservativeState> q_old = {q_parent};
        prolongate_solution_order2(q_old, w_old, grad_old, mesh, mesh_ref, records, gamma, q_new);
        if (q_new.size() != 8u) FAIL("expected 8 states, got %zu", q_new.size());

        Real mass_children = 0.0f;
        for (int i = 0; i < 8; ++i)
            mass_children += q_new[i].rho * mesh_ref.cells[i].volume;
        Real rel_diff = std::fabs(mass_children - mass_parent) / mass_parent;
        if (rel_diff > 1e-6f) FAIL("mass rel diff=%g (parent=%g, children=%g)", rel_diff, mass_parent, mass_children);
        PASS;
    }

    // ----- Helper: build 2-hex mesh with hanging face (cell 0 = level 1, cell 1 = level 0) -----
    struct HangingMesh {
        CfdMesh mesh;
        std::vector<HangingFaceInfo> hanging;
    };
    auto make_hanging_mesh = []() -> HangingMesh {
        HangingMesh hm;
        hm.mesh.nodes.resize(12);
        hm.mesh.nodes[0] = {0,0,0}; hm.mesh.nodes[1] = {1,0,0};
        hm.mesh.nodes[2] = {1,1,0}; hm.mesh.nodes[3] = {0,1,0};
        hm.mesh.nodes[4] = {0,0,1}; hm.mesh.nodes[5] = {1,0,1};
        hm.mesh.nodes[6] = {1,1,1}; hm.mesh.nodes[7] = {0,1,1};
        hm.mesh.nodes[8] = {2,0,0}; hm.mesh.nodes[9] = {2,1,0};
        hm.mesh.nodes[10]= {2,0,1}; hm.mesh.nodes[11]= {2,1,1};

        hm.mesh.cells.resize(2);
        hm.mesh.cells[0].type = ElementType::HEX8;
        hm.mesh.cells[0].node[0]=0; hm.mesh.cells[0].node[1]=1;
        hm.mesh.cells[0].node[2]=2; hm.mesh.cells[0].node[3]=3;
        hm.mesh.cells[0].node[4]=4; hm.mesh.cells[0].node[5]=5;
        hm.mesh.cells[0].node[6]=6; hm.mesh.cells[0].node[7]=7;
        hm.mesh.cells[1].type = ElementType::HEX8;
        hm.mesh.cells[1].node[0]=1; hm.mesh.cells[1].node[1]=8;
        hm.mesh.cells[1].node[2]=9; hm.mesh.cells[1].node[3]=2;
        hm.mesh.cells[1].node[4]=5; hm.mesh.cells[1].node[5]=10;
        hm.mesh.cells[1].node[6]=11; hm.mesh.cells[1].node[7]=6;

        hm.mesh.cells[0].refinement_level = 1;
        rebuild_mesh_faces(hm.mesh);
        compute_mesh_metrics(hm.mesh);
        hm.hanging = detect_hanging_faces(hm.mesh);
        return hm;
    };

    TEST("CFD-AMR-15 uniform flow: 2nd-order hanging preserves uniform state exactly");
    {
        auto hm = make_hanging_mesh();
        if (hm.hanging.empty()) FAIL("expected hanging faces");

        Real gamma = 1.4f;
        int nc = static_cast<int>(hm.mesh.cells.size());
        std::vector<PrimitiveState> w(nc, {1.225f, 100.0f, 0.0f, 0.0f, 101325.0f, 0.0f});
        std::vector<ConservativeState> q(nc);
        PrimitiveState freestream = w[0];
        for (int i = 0; i < nc; ++i) q[i] = primitive_to_conservative(w[i], gamma);

        auto grads = compute_green_gauss_gradients(hm.mesh, q, gamma, &w);
        if (grads.size() != static_cast<std::size_t>(nc)) FAIL("grads size mismatch");

        auto limiters = compute_barth_jespersen_limiters(hm.mesh, q, grads, gamma, &w);
        std::vector<PrimitiveGradient> limited(nc);
        for (int i = 0; i < nc; ++i) limited[i] = apply_limiter(grads[i], limiters[i]);

        std::vector<EulerFlux> residual(nc);
        if (!compute_euler_residual_cpu_order2(hm.mesh, q, freestream, gamma, limited, residual, &w))
            FAIL("order2 residual failed");

        apply_hanging_flux_correction_primitive(hm.mesh, hm.hanging, q, w, limited, gamma, residual);

        Real l2 = residual_l2(residual);
        if (l2 > 1e-12f) FAIL("uniform flow L2=%g > 1e-12", l2);
        PASS;
    }

    TEST("CFD-AMR-16 smooth flow: hanging correction changes flux at hanging face without NaN");
    {
        // Use the 2-hex mesh with manually-assigned refinement_level (as in AMR-9/10).
        auto hm = make_hanging_mesh();
        if (hm.hanging.empty()) FAIL("expected hanging faces");

        Real gamma = 1.4f;
        int nc = static_cast<int>(hm.mesh.cells.size());

        // Non-uniform smooth flow: left cell denser/hotter than right cell (linear-ish gradient)
        std::vector<PrimitiveState> w(nc);
        w[0] = {1.225f, 100.0f, 0.0f, 0.0f, 101325.0f, 0.0f};
        w[1] = {1.0f, 50.0f, 0.0f, 0.0f, 50000.0f, 0.0f};
        PrimitiveState freestream = w[0];

        std::vector<ConservativeState> q(nc);
        for (int i = 0; i < nc; ++i) q[i] = primitive_to_conservative(w[i], gamma);

        auto grads = compute_green_gauss_gradients(hm.mesh, q, gamma, &w);
        auto limiters = compute_barth_jespersen_limiters(hm.mesh, q, grads, gamma, &w);
        std::vector<PrimitiveGradient> limited(nc);
        for (int i = 0; i < nc; ++i) limited[i] = apply_limiter(grads[i], limiters[i]);

        // Order-2 residual before and after hanging correction
        std::vector<EulerFlux> res_before(nc);
        if (!compute_euler_residual_cpu_order2(hm.mesh, q, freestream, gamma, limited, res_before, &w))
            FAIL("order-2 residual failed");
        auto res_after = res_before;
        apply_hanging_flux_correction_primitive(hm.mesh, hm.hanging, q, w, limited, gamma, res_after);

        // Check all residual values are finite
        for (const auto& r : res_after) {
            if (!std::isfinite(r.mass) || !std::isfinite(r.mom_x) ||
                !std::isfinite(r.mom_y) || !std::isfinite(r.mom_z) ||
                !std::isfinite(r.energy) || !std::isfinite(r.turbulence))
                FAIL("NaN/Inf after hanging correction");
        }

        // Verify L2 norm after correction is finite
        Real l2 = residual_l2(res_after);
        if (!std::isfinite(l2)) FAIL("L2=%g after hanging correction", l2);
        PASS;
    }

    TEST("CFD-AMR-17 shocked flow: primitive-space hanging reconstruction avoids negative p");
    {
        auto hm = make_hanging_mesh();
        if (hm.hanging.empty()) FAIL("expected hanging faces");

        Real gamma = 1.4f;
        int nc = static_cast<int>(hm.mesh.cells.size());
        // Strong shock: left cell (level 1) high pressure, right cell (level 0) low pressure
        std::vector<PrimitiveState> w(nc);
        w[0] = {8.0f, 0.0f, 0.0f, 0.0f, 800000.0f, 0.0f};   // high p left
        w[1] = {1.0f, 0.0f, 0.0f, 0.0f, 10000.0f, 0.0f};     // low p right
        PrimitiveState freestream = w[0];

        std::vector<ConservativeState> q(nc);
        for (int i = 0; i < nc; ++i) q[i] = primitive_to_conservative(w[i], gamma);

        auto grads = compute_green_gauss_gradients(hm.mesh, q, gamma, &w);
        auto limiters = compute_barth_jespersen_limiters(hm.mesh, q, grads, gamma, &w);
        std::vector<PrimitiveGradient> limited(nc);
        for (int i = 0; i < nc; ++i) limited[i] = apply_limiter(grads[i], limiters[i]);

        std::vector<EulerFlux> residual(nc);
        if (!compute_euler_residual_cpu_order2(hm.mesh, q, freestream, gamma, limited, residual, &w))
            FAIL("order-2 residual failed");

        apply_hanging_flux_correction_primitive(hm.mesh, hm.hanging, q, w, limited, gamma, residual);

        // Check for NaN/Inf in residual
        bool has_nan = false;
        for (const auto& r : residual) {
            if (!std::isfinite(r.mass) || !std::isfinite(r.mom_x) || !std::isfinite(r.mom_y) ||
                !std::isfinite(r.mom_z) || !std::isfinite(r.energy) || !std::isfinite(r.turbulence)) {
                has_nan = true; break;
            }
        }
        if (has_nan) FAIL("NaN/Inf in shocked hanging residual");

        Real l2 = residual_l2(residual);
        if (!std::isfinite(l2) || l2 <= 0.0f) FAIL("invalid L2=%g", l2);
        PASS;
    }
    return 0;
}

static int test_compact_mesh_nodes() {
    TEST("CFD-MESH-COV3-1 no-op when all nodes used");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 5);
        compute_mesh_metrics(mesh);
        std::size_t n_before = mesh.nodes.size();
        compact_mesh_nodes(mesh);
        if (mesh.nodes.size() != n_before) FAIL("nodes changed: %zu -> %zu", n_before, mesh.nodes.size());
        PASS;
    }

    TEST("CFD-MESH-COV3-2 removes unused trailing nodes");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 5);
        compute_mesh_metrics(mesh);
        std::size_t n_orig = mesh.nodes.size();
        mesh.nodes.push_back({100.0f, 200.0f, 300.0f});
        compact_mesh_nodes(mesh);
        if (mesh.nodes.size() != n_orig) FAIL("expected %zu nodes after compact, got %zu", n_orig, mesh.nodes.size());
        PASS;
    }

    TEST("CFD-MESH-COV3-3 remaps cell node indices after compaction");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 5);
        compute_mesh_metrics(mesh);
        std::size_t n_orig = mesh.nodes.size();
        mesh.nodes.push_back({100.0f, 200.0f, 300.0f});
        int max_node_before = 0;
        for (const auto& cell : mesh.cells) {
            for (int i = 0; i < 8; ++i) {
                if (cell.node[i] > max_node_before) max_node_before = cell.node[i];
            }
        }
        compact_mesh_nodes(mesh);
        int max_node_after = 0;
        for (const auto& cell : mesh.cells) {
            for (int i = 0; i < 8; ++i) {
                if (cell.node[i] > max_node_after) max_node_after = cell.node[i];
            }
        }
        if (max_node_after >= static_cast<int>(mesh.nodes.size()))
            FAIL("node index %d out of range (n_nodes=%zu)", max_node_after, mesh.nodes.size());
        if (mesh.nodes.size() != n_orig) FAIL("expected %zu nodes, got %zu", n_orig, mesh.nodes.size());
        PASS;
    }

    TEST("CFD-MESH-COV3-4 empty mesh does not crash");
    {
        CfdMesh empty;
        compact_mesh_nodes(empty);
        PASS;
    }
    return 0;
}

static int test_turbulence_amr() {
    TEST("CFD-AMR-TURB-1 Flat plate SA: AMR with y+ sensor refines wall cells");
    {
        auto mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-3f, 1.12f, 10, 3, 12);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("mesh metrics: %s", report.message.c_str());
        int cells_before = static_cast<int>(mesh.cells.size());

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.max_iter = 5;
        cfg.cfl = 0.1f;
        cfg.convergence_tol = 1e-12f;
        cfg.viscous = true;
        cfg.Re = 1e5f;
        cfg.turbulence_model = TurbulenceModel::SA;
        cfg.amr.enabled = false;

        FreestreamCondition freestream;
        freestream.mach = 0.5f;

        auto summary = solver.solve(freestream, cfg);
        if (summary.failed) {
            std::printf("  (fail: iter=%d cell=%d reason=%s)",
                summary.diagnostics.failure.iteration,
                summary.diagnostics.failure.cell,
                summary.diagnostics.failure.reason.c_str());
            FAIL("solver failed with SA");
        }

        std::printf("  (cells=%d res=%.2e)",
            cells_before,
            summary.residual_history.empty() ? -1.0 : summary.residual_history.back());
        PASS;
    }

    TEST("CFD-AMR-TURB-2 Cube SA: supersonic solver & multi-sensor AMR");
    {
        // 1) Verify SA solver stays stable for 11+ iterations (no AMR).
        CfdMesh mesh = generate_structured_cube_mesh(5.0f, 9);
        CfdMesh mesh_amr = mesh; // copy for second solver
        {
            CfdSolver solver;
            if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");
            CfdConfig cfg;
            cfg.max_iter = 12;
            cfg.cfl = 0.03f;
            cfg.convergence_tol = 1e-12f;
            cfg.viscous = true;
            cfg.Re = 1e5f;
            cfg.turbulence_model = TurbulenceModel::SA;
            cfg.diagnostic_level = DiagnosticLevel::Verbose;
            FreestreamCondition freestream;
            freestream.mach = 2.0f;
            auto s = solver.solve(freestream, cfg);
            if (s.failed) {
                std::printf("  (fail: iter=%d cell=%d reason=%s)",
                    s.diagnostics.failure.iteration,
                    s.diagnostics.failure.cell,
                    s.diagnostics.failure.reason.c_str());
                FAIL("SA solver unstable without AMR");
            }
        }

        // 2) Separate solver: multi-sensor AMR at iter=10, verify mesh and state.
        int cells_before = static_cast<int>(mesh_amr.cells.size());
        CfdSolver solver2;
        if (!solver2.load_mesh(mesh_amr)) FAIL("load_mesh failed");
        CfdConfig cfg2;
        cfg2.max_iter = 12;
        cfg2.cfl = 0.03f;
        cfg2.convergence_tol = 1e-12f;
        cfg2.viscous = true;
        cfg2.Re = 1e5f;
        cfg2.turbulence_model = TurbulenceModel::SA;
        cfg2.amr.enabled = true;
        cfg2.amr.interval = 10;
        cfg2.amr.max_level = 2;
        cfg2.amr.yplus_target = 1.0f;
        cfg2.amr.tke_ratio_threshold = 0.05f;
        cfg2.amr.shear_layer_threshold = 0.3f;
        cfg2.amr.wake_cone.length = 2.0f;
        cfg2.amr.wake_cone.half_angle_deg = 30.0f;
        cfg2.amr.wake_cone.origin_x = 0.0f;
        cfg2.amr.wake_cone.axis_x = 1.0f;
        cfg2.diagnostic_level = DiagnosticLevel::Verbose;
        FreestreamCondition freestream2;
        freestream2.mach = 2.0f;

        auto summary = solver2.solve(freestream2, cfg2);
        if (summary.failed) {
            std::printf("  (fail: iter=%d cell=%d reason=%s)",
                summary.diagnostics.failure.iteration,
                summary.diagnostics.failure.cell,
                summary.diagnostics.failure.reason.c_str());
            FAIL("solver failed SA multi-sensor AMR");
        }

        int cells_after = static_cast<int>(summary.final_state.size());
        std::printf("  (before=%d after=%d res=%.2e ratio=%.2f)",
            cells_before, cells_after,
            summary.residual_history.empty() ? -1.0 : summary.residual_history.back(),
            static_cast<Real>(cells_after) / static_cast<Real>(cells_before));

        CfdMesh mesh_after = solver2.mesh();
        auto qr = compute_mesh_metrics(mesh_after);
        if (qr.min_volume <= 0.0f) FAIL("min volume=%g", qr.min_volume);
        if (qr.negative_jacobian_count != 0) FAIL("negative Jacobians=%d", qr.negative_jacobian_count);
        PASS;
    }

    TEST("CFD-AMR-TURB-3 Flat plate SST: CPU solver with AMR produces finite forces");
    {
        auto mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-3f, 1.12f, 10, 3, 12);
        auto report = compute_mesh_metrics(mesh);
        if (!report.valid) FAIL("mesh metrics: %s", report.message.c_str());

        CfdSolver solver;
        if (!solver.load_mesh(mesh)) FAIL("load_mesh failed");

        CfdConfig cfg;
        cfg.max_iter = 10;
        cfg.cfl = 0.1f;
        cfg.convergence_tol = 1e-12f;
        cfg.viscous = true;
        cfg.Re = 1e5f;
        cfg.turbulence_model = TurbulenceModel::SST;
        cfg.amr.enabled = true;
        cfg.amr.interval = 5;
        cfg.amr.max_level = 2;
        cfg.amr.yplus_target = 1.0f;

        FreestreamCondition freestream;
        freestream.mach = 0.5f;

        auto summary = solver.solve(freestream, cfg);
        if (summary.failed) {
            std::printf("  (fail: iter=%d cell=%d reason=%s)",
                summary.diagnostics.failure.iteration,
                summary.diagnostics.failure.cell,
                summary.diagnostics.failure.reason.c_str());
            FAIL("SST CPU solver with AMR failed");
        }

        if (!std::isfinite(summary.forces.CX) || !std::isfinite(summary.forces.CY) ||
            !std::isfinite(summary.forces.CZ) || !std::isfinite(summary.forces.CD) ||
            !std::isfinite(summary.forces.CL))
            FAIL("non-finite forces from SST CPU AMR");

        std::printf("  (cells=%d->%d res=%.2e CX=%.4f CD=%.4f CL=%.4f)",
            static_cast<int>(mesh.cells.size()),
            static_cast<int>(summary.final_state.size()),
            summary.residual_history.empty() ? -1.0 : summary.residual_history.back(),
            summary.forces.CX, summary.forces.CD, summary.forces.CL);

        CfdMesh mesh_after = solver.mesh();
        auto qr = compute_mesh_metrics(mesh_after);
        if (qr.min_volume <= 0.0f) FAIL("min volume=%g", qr.min_volume);
        if (qr.negative_jacobian_count != 0) FAIL("negative Jacobians=%d", qr.negative_jacobian_count);

        // SST without AMR: same config but amr disabled
        CfdSolver solver2;
        if (!solver2.load_mesh(mesh)) FAIL("load_mesh failed for ref");

        CfdConfig cfg2 = cfg;
        cfg2.amr.enabled = false;

        auto summary2 = solver2.solve(freestream, cfg2);
        if (summary2.failed) FAIL("SST CPU solver without AMR failed");

        if (!std::isfinite(summary2.forces.CX) || !std::isfinite(summary2.forces.CD))
            FAIL("non-finite forces from SST CPU no-AMR");

        std::printf(" (ref CD=%.4f CL=%.4f)", summary2.forces.CD, summary2.forces.CL);
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_cube_mesh();
    result |= test_flat_plate_mesh();
    result |= test_su2_round_trip();
    result |= test_su2_invalid_file();
    result |= test_cgns_fallback();
    result |= test_mesh_quality_detail();
    result |= test_tet4_refinement();
    result |= test_hex8_refinement();
    result |= test_gradient_sensor();
    result |= test_yplus_sensor();
    result |= test_solution_interpolation();
    result |= test_hanging_faces();
    result |= test_amr_disabled_regression();
    result |= test_amr_max_level();
    result |= test_amr_order2_hanging();
    result |= test_compact_mesh_nodes();
    result |= test_turbulence_amr();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}


