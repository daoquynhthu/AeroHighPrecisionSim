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
    result |= test_solution_interpolation();
    result |= test_hanging_faces();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}


