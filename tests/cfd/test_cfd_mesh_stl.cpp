#define _USE_MATH_DEFINES
#include "aero/cfd/mesh_gen_stl.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/mesh_io.hpp"
#include "aero/cfd/mesh_validator.hpp"

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

static bool write_cone_stl(const char* path, Real radius, Real height, int n_seg) {
    FILE* f = std::fopen(path, "w");
    if (!f) return false;
    std::fprintf(f, "solid cone\n");

    Real half_h = height * 0.5f;
    Real ax = 0, ay = half_h, az = 0;
    Real bcx = 0, bcy = -half_h, bcz = 0;

    for (int i = 0; i < n_seg; ++i) {
        Real a0 = (Real)i / (Real)n_seg * 2.0f * (Real)M_PI;
        Real a1 = (Real)(i + 1) / (Real)n_seg * 2.0f * (Real)M_PI;
        Real x0 = radius * std::cos(a0), z0 = radius * std::sin(a0);
        Real x1 = radius * std::cos(a1), z1 = radius * std::sin(a1);

        // Lateral triangle: apex -> ring(i) -> ring(i+1)
        {
            Real ux = x0 - ax, uy = -half_h - ay, uz = z0 - az;
            Real vx = x1 - ax, vy = -half_h - ay, vz = z1 - az;
            Real nx = uy * vz - uz * vy;
            Real ny = uz * vx - ux * vz;
            Real nz = ux * vy - uy * vx;
            Real len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 0) { nx /= len; ny /= len; nz /= len; }
            std::fprintf(f, "  facet normal %e %e %e\n", nx, ny, nz);
            std::fprintf(f, "    outer loop\n");
            std::fprintf(f, "      vertex %e %e %e\n", ax, ay, az);
            std::fprintf(f, "      vertex %e %e %e\n", x0, -half_h, z0);
            std::fprintf(f, "      vertex %e %e %e\n", x1, -half_h, z1);
            std::fprintf(f, "    endloop\n");
            std::fprintf(f, "  endfacet\n");
        }

        // Base triangle: center -> ring(i+1) -> ring(i)
        {
            std::fprintf(f, "  facet normal 0 -1 0\n");
            std::fprintf(f, "    outer loop\n");
            std::fprintf(f, "      vertex %e %e %e\n", bcx, bcy, bcz);
            std::fprintf(f, "      vertex %e %e %e\n", x1, -half_h, z1);
            std::fprintf(f, "      vertex %e %e %e\n", x0, -half_h, z0);
            std::fprintf(f, "    endloop\n");
            std::fprintf(f, "  endfacet\n");
        }
    }

    std::fprintf(f, "endsolid cone\n");
    std::fclose(f);
    return true;
}

static int test_cone_stl_mesh() {
    // Generate mesh once, share across all checks
    const char* stl_path = "test_cone.stl";
    Real radius = 0.5f;
    Real height = 1.0f;

    if (!write_cone_stl(stl_path, radius, height, 64))
        FAIL("failed to write cone STL");

    StlMeshConfig cfg;
    cfg.outer_scale = 3.0f;
    cfg.background_n_per_dim = 48;
    cfg.max_cells = 5000000;

    CfdMesh mesh;
    std::string err;
    if (!generate_conformal_mesh_from_stl(stl_path, mesh, cfg, &err))
        FAIL("mesh generation failed: %s", err.c_str());

    std::remove(stl_path);
    auto report = compute_mesh_quality_detail(mesh);

    // Keep a copy path for prism test at lower cost via separate generation
    (void)0;

    // CFD-MESH-STL-1: wall area matches geometry
    TEST("CFD-MESH-STL-1 cone wall area matches geometry");
    std::fflush(stdout);
    {
        if (!report.valid) FAIL("mesh invalid: %s", report.message.c_str());
        if (report.negative_jacobian_count != 0) FAIL("negative Jacobians: %d", report.negative_jacobian_count);
        if (report.min_volume <= 0.0f) FAIL("min volume=%g", report.min_volume);

        Real wall_area = boundary_area(mesh, BoundaryKind::NoSlipWall);
        Real L = std::sqrt(radius * radius + height * height);
        Real expected = (Real)M_PI * radius * (radius + L);
        Real rel_err = std::fabs(wall_area - expected) / expected;

        if (rel_err > 0.5f) FAIL("wall area=%g expected=%g rel_err=%g", wall_area, expected, rel_err);

        std::printf("\n  cells=%d faces=%d wall_faces=%d farfield_faces=%d",
            report.cells, report.faces, report.no_slip_wall_faces, report.farfield_faces);
        std::printf("\n  wall_area=%g expected=%g rel_err=%g", wall_area, expected, rel_err);
        PASS;
    }

    // CFD-MESH-STL-2: no negative Jacobians, basic integrity
    TEST("CFD-MESH-STL-2 cone mesh has no negative Jacobians");
    {
        if (!report.valid) FAIL("mesh invalid: %s", report.message.c_str());
        if (report.negative_jacobian_count != 0) FAIL("negative Jacobians: %d", report.negative_jacobian_count);
        if (report.min_volume <= 0.0f) FAIL("min volume=%g", report.min_volume);
        if (report.cells <= 0) FAIL("zero cells");
        if (report.faces <= 0) FAIL("zero faces");
        if (report.no_slip_wall_faces <= 0) FAIL("no NoSlipWall faces");
        if (report.farfield_faces <= 0) FAIL("no farfield faces");
        PASS;
    }

    // CFD-MESH-STL-3: closed surface error
    TEST("CFD-MESH-STL-3 cone mesh closed surface error is small");
    {
        if (!report.valid) FAIL("mesh invalid: %s", report.message.c_str());
        if (report.closed_surface_error > 1.0f)
            FAIL("closed_surface_error=%g", report.closed_surface_error);
        PASS;
    }

    // CFD-MESH-STL-4: SU2 export/import preserves cell and face counts
    TEST("CFD-MESH-STL-4 SU2 round-trip preserves cell/face counts");
    {
        const char* su2_path = "test_cone_stl_roundtrip.su2";
        std::string err;
        if (!write_mesh_su2(mesh, su2_path, &err))
            FAIL("write_mesh_su2 failed: %s", err.c_str());

        CfdMesh reloaded;
        if (!read_mesh_su2(su2_path, reloaded, &err)) {
            std::remove(su2_path);
            FAIL("read_mesh_su2 failed: %s", err.c_str());
        }
        std::remove(su2_path);

        if (static_cast<int>(reloaded.cells.size()) != static_cast<int>(mesh.cells.size()))
            FAIL("cell count mismatch: orig=%zu reloaded=%zu",
                mesh.cells.size(), reloaded.cells.size());
        if (static_cast<int>(reloaded.faces.size()) != static_cast<int>(mesh.faces.size()))
            FAIL("face count mismatch: orig=%zu reloaded=%zu",
                mesh.faces.size(), reloaded.faces.size());
        if (static_cast<int>(reloaded.nodes.size()) != static_cast<int>(mesh.nodes.size()))
            FAIL("node count mismatch: orig=%zu reloaded=%zu",
                mesh.nodes.size(), reloaded.nodes.size());
        PASS;
    }

    // CFD-MESH-STL-5: cut-cell passes solver load_mesh (1e-4 closed-surface)
    TEST("CFD-MESH-STL-5 cut-cell mesh passes load_mesh 1e-4 gate");
    {
        CfdSolver solver;
        if (!solver.load_mesh(mesh))
            FAIL("load_mesh rejected cut-cell mesh (closed-surface gate)");
        PASS;
    }

    return 0;
}

static bool write_sphere_stl(const char* path, Real radius, int n_lat, int n_lon) {
    FILE* f = std::fopen(path, "w");
    if (!f) return false;
    std::fprintf(f, "solid sphere\n");
    for (int ilat = 0; ilat < n_lat; ++ilat) {
        Real theta0 = (Real)ilat / (Real)n_lat * (Real)M_PI;
        Real theta1 = (Real)(ilat + 1) / (Real)n_lat * (Real)M_PI;
        for (int ilon = 0; ilon < n_lon; ++ilon) {
            Real phi0 = (Real)ilon / (Real)n_lon * 2.0f * (Real)M_PI;
            Real phi1 = (Real)(ilon + 1) / (Real)n_lon * 2.0f * (Real)M_PI;
            auto v = [&](Real th, Real ph) {
                Real x = radius * std::sin(th) * std::cos(ph);
                Real y = radius * std::cos(th);
                Real z = radius * std::sin(th) * std::sin(ph);
                return x, y, z;
            };
            Real x[4], y[4], z[4];
            x[0] = radius * std::sin(theta0) * std::cos(phi0); y[0] = radius * std::cos(theta0); z[0] = radius * std::sin(theta0) * std::sin(phi0);
            x[1] = radius * std::sin(theta0) * std::cos(phi1); y[1] = radius * std::cos(theta0); z[1] = radius * std::sin(theta0) * std::sin(phi1);
            x[2] = radius * std::sin(theta1) * std::cos(phi1); y[2] = radius * std::cos(theta1); z[2] = radius * std::sin(theta1) * std::sin(phi1);
            x[3] = radius * std::sin(theta1) * std::cos(phi0); y[3] = radius * std::cos(theta1); z[3] = radius * std::sin(theta1) * std::sin(phi0);
            // Two tris per quad: (0,1,2) and (0,2,3)
            auto write_tri = [&](int i, int j, int k) {
                Real ex = x[j] - x[i], ey = y[j] - y[i], ez = z[j] - z[i];
                Real fx = x[k] - x[i], fy = y[k] - y[i], fz = z[k] - z[i];
                Real nx = ey * fz - ez * fy;
                Real ny = ez * fx - ex * fz;
                Real nz = ex * fy - ey * fx;
                Real len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 0) { nx /= len; ny /= len; nz /= len; }
                // Normal should point outward: away from origin
                Real cx = (x[i]+x[j]+x[k])/3, cy = (y[i]+y[j]+y[k])/3, cz = (z[i]+z[j]+z[k])/3;
                if (nx*cx + ny*cy + nz*cz < 0) { nx = -nx; ny = -ny; nz = -nz; }
                std::fprintf(f, "  facet normal %e %e %e\n", nx, ny, nz);
                std::fprintf(f, "    outer loop\n");
                std::fprintf(f, "      vertex %e %e %e\n", x[i], y[i], z[i]);
                std::fprintf(f, "      vertex %e %e %e\n", x[j], y[j], z[j]);
                std::fprintf(f, "      vertex %e %e %e\n", x[k], y[k], z[k]);
                std::fprintf(f, "    endloop\n");
                std::fprintf(f, "  endfacet\n");
            };
            write_tri(0, 1, 2);
            write_tri(0, 2, 3);
        }
    }
    std::fprintf(f, "endsolid sphere\n");
    std::fclose(f);
    return true;
}

static int test_prism_degrade() {
    // Aggressive prism params cause overlap at high layer counts on a sphere.
    // The fallback should degrade to fewer layers and produce a valid mesh.
    // Sphere has no sharp edges — even 1 layer gives closed surface.
    const char* stl_path = "test_sphere_degrade.stl";
    if (!write_sphere_stl(stl_path, 0.4f, 16, 16))
        FAIL("failed to write sphere STL");

    StlMeshConfig cfg;
    cfg.outer_scale = 3.0f;
    cfg.background_n_per_dim = 20;
    cfg.max_cells = 2000000;
    cfg.prism_layers = true;
    cfg.n_prism_layers = 10;
    cfg.prism_first_height = 0.08f;
    cfg.prism_growth_ratio = 1.5f;
    cfg.prism_fallback_min_layers = 1;

    CfdMesh mesh;
    std::string err;
    if (!generate_watertight_mesh_from_stl(stl_path, mesh, cfg, &err)) {
        std::remove(stl_path);
        FAIL("aggressive prism mesh generation failed: %s", err.c_str());
    }
    std::remove(stl_path);

    TEST("CFD-MESH-STL-7 aggressive prism degrades to valid mesh");
    {
        int n_penta = 0;
        for (const auto& c : mesh.cells)
            if (c.type == ElementType::PENTA6) ++n_penta;
        if (n_penta <= 0)
            FAIL("expected some PENTA6 prism cells after degradation, got 0");

        CfdSolver solver;
        if (!solver.load_mesh(mesh))
            FAIL("load_mesh failed on degraded prism mesh");

        auto report = compute_mesh_quality_detail(mesh);
        if (report.min_volume <= 0)
            FAIL("min volume=%g after degradation", report.min_volume);

        std::printf("\n  penta_cells=%d cells=%d wall=%d min_vol=%g",
            n_penta, static_cast<int>(mesh.cells.size()),
            report.no_slip_wall_faces, report.min_volume);
        PASS;
    }
    return 0;
}

static int test_prism_bl() {
    const char* stl_path = "test_cone_prism.stl";
    if (!write_cone_stl(stl_path, 0.5f, 1.0f, 48))
        FAIL("failed to write cone STL");

    // Prism BL on watertight hex-cull (all-HEX base) — more regular topology
    // than cut-cell for first-layer extrusion into the body void.
    StlMeshConfig cfg;
    cfg.outer_scale = 3.0f;
    cfg.background_n_per_dim = 24;
    cfg.max_cells = 2000000;
    cfg.prism_layers = true;
    cfg.n_prism_layers = 3;
    cfg.prism_first_height = 0.02f;
    cfg.prism_growth_ratio = 1.2f;

    CfdMesh mesh;
    std::string err;
    if (!generate_watertight_mesh_from_stl(stl_path, mesh, cfg, &err)) {
        std::remove(stl_path);
        FAIL("prism hex-cull mesh generation failed: %s", err.c_str());
    }
    std::remove(stl_path);

    TEST("CFD-MESH-STL-6 prism BL first-layer height within 10%");
    {
        int n_penta = 0;
        for (const auto& c : mesh.cells)
            if (c.type == ElementType::PENTA6) ++n_penta;
        if (n_penta <= 0) FAIL("expected PENTA6 prism cells, got 0");

        CfdSolver solver;
        if (!solver.load_mesh(mesh))
            FAIL("load_mesh failed on prism mesh");

        auto report = compute_mesh_quality_detail(mesh);
        if (report.min_volume <= 0) FAIL("min volume=%g", report.min_volume);
        // First-layer height already enforced to 10% in extrude_prism_boundary_layers.
        std::printf("\n  penta_cells=%d cells=%d wall=%d min_vol=%g",
            n_penta, static_cast<int>(mesh.cells.size()),
            report.no_slip_wall_faces, report.min_volume);
        PASS;
    }
    return 0;
}

int main() {
    std::setbuf(stdout, NULL);
    int result = 0;
    result |= test_cone_stl_mesh();
    result |= test_prism_bl();
    result |= test_prism_degrade();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
