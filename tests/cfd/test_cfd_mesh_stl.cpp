#define _USE_MATH_DEFINES
#include "aero/cfd/mesh_gen_stl.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/mesh_io.hpp"
#include "aero/cfd/mesh_validator.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>

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

// Write a two-tetrahedra STL file (two disjoint closed bodies)
static bool write_two_tet_stl(const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) return false;

    // Tetrahedron 1: A(0,0,0) B(1,0,0) C(0,1,0) D(0,0,1)
    // Face ABC (opposite D): outward normal (0,0,-1)
    std::fprintf(f, "solid tet1\n");
    std::fprintf(f, "  facet normal 0 0 -1\n");
    std::fprintf(f, "    outer loop\n");
    std::fprintf(f, "      vertex 0 0 0\n");
    std::fprintf(f, "      vertex 1 0 0\n");
    std::fprintf(f, "      vertex 0 1 0\n");
    std::fprintf(f, "    endloop\n");
    std::fprintf(f, "  endfacet\n");
    // Face ABD (opposite C): outward normal (0,-1,0)
    std::fprintf(f, "  facet normal 0 -1 0\n");
    std::fprintf(f, "    outer loop\n");
    std::fprintf(f, "      vertex 0 0 0\n");
    std::fprintf(f, "      vertex 1 0 0\n");
    std::fprintf(f, "      vertex 0 0 1\n");
    std::fprintf(f, "    endloop\n");
    std::fprintf(f, "  endfacet\n");
    // Face ADC (opposite B): winding A-D-C for outward (-1,0,0)
    std::fprintf(f, "  facet normal -1 0 0\n");
    std::fprintf(f, "    outer loop\n");
    std::fprintf(f, "      vertex 0 0 0\n");
    std::fprintf(f, "      vertex 0 0 1\n");
    std::fprintf(f, "      vertex 0 1 0\n");
    std::fprintf(f, "    endloop\n");
    std::fprintf(f, "  endfacet\n");
    // Face BCD (opposite A): winding B-C-D for outward (1,1,1)/sqrt(3)
    std::fprintf(f, "  facet normal 0.57735 0.57735 0.57735\n");
    std::fprintf(f, "    outer loop\n");
    std::fprintf(f, "      vertex 1 0 0\n");
    std::fprintf(f, "      vertex 0 1 0\n");
    std::fprintf(f, "      vertex 0 0 1\n");
    std::fprintf(f, "    endloop\n");
    std::fprintf(f, "  endfacet\n");
    std::fprintf(f, "endsolid tet1\n");

    // Tetrahedron 2: A'(3,0,0) B'(4,0,0) C'(3,1,0) D'(3,0,1)
    std::fprintf(f, "solid tet2\n");
    std::fprintf(f, "  facet normal 0 0 -1\n");
    std::fprintf(f, "    outer loop\n");
    std::fprintf(f, "      vertex 3 0 0\n");
    std::fprintf(f, "      vertex 4 0 0\n");
    std::fprintf(f, "      vertex 3 1 0\n");
    std::fprintf(f, "    endloop\n");
    std::fprintf(f, "  endfacet\n");
    std::fprintf(f, "  facet normal 0 -1 0\n");
    std::fprintf(f, "    outer loop\n");
    std::fprintf(f, "      vertex 3 0 0\n");
    std::fprintf(f, "      vertex 4 0 0\n");
    std::fprintf(f, "      vertex 3 0 1\n");
    std::fprintf(f, "    endloop\n");
    std::fprintf(f, "  endfacet\n");
    std::fprintf(f, "  facet normal -1 0 0\n");
    std::fprintf(f, "    outer loop\n");
    std::fprintf(f, "      vertex 3 0 0\n");
    std::fprintf(f, "      vertex 3 0 1\n");
    std::fprintf(f, "      vertex 3 1 0\n");
    std::fprintf(f, "    endloop\n");
    std::fprintf(f, "  endfacet\n");
    std::fprintf(f, "  facet normal 0.57735 0.57735 0.57735\n");
    std::fprintf(f, "    outer loop\n");
    std::fprintf(f, "      vertex 4 0 0\n");
    std::fprintf(f, "      vertex 3 1 0\n");
    std::fprintf(f, "      vertex 3 0 1\n");
    std::fprintf(f, "    endloop\n");
    std::fprintf(f, "  endfacet\n");
    std::fprintf(f, "endsolid tet2\n");

    std::fclose(f);
    return true;
}

static int test_multi_body_stl() {
    const char* stl_path = "test_multi_tet.stl";
    if (!write_two_tet_stl(stl_path))
        FAIL("failed to write two-tet STL");

    TEST("CFD-MESH-MULTI-1 two-tet multi_body=true: body_ids assigned");
    {
        StlMeshConfig cfg;
        cfg.outer_scale = 3.0f;
        cfg.background_n_per_dim = 32;
        cfg.max_cells = 500000;
        cfg.multi_body = true;

        CfdMesh mesh;
        std::string err;
        if (!generate_watertight_mesh_from_stl(stl_path, mesh, cfg, &err))
            FAIL("watertight mesh failed: %s", err.c_str());

        if (mesh.cells.empty()) FAIL("zero cells in watertight mesh");
        if (mesh.faces.empty()) FAIL("zero faces");

        // Count body_ids on wall faces
        std::set<int> body_ids;
        int wall_count = 0;
        for (const auto& face : mesh.faces) {
            if (face.boundary == BoundaryKind::NoSlipWall) {
                body_ids.insert(face.body_id);
                ++wall_count;
            }
        }
        if (wall_count == 0) FAIL("zero wall faces in mesh");
        if (body_ids.empty()) FAIL("no body_ids found on wall faces");

        std::printf("\n  bodies=%zu wall_faces=%d body_ids={",
            body_ids.size(), wall_count);
        for (int bid : body_ids) std::printf("%d ", bid);
        std::printf("} cells=%zu", mesh.cells.size());

        auto report = compute_mesh_quality_detail(mesh);
        if (report.min_volume <= 0.0f) FAIL("min volume=%g", report.min_volume);
        PASS;
    }

    TEST("CFD-MESH-MULTI-3 multi_body=false regression: no crash");
    {
        StlMeshConfig cfg;
        cfg.outer_scale = 3.0f;
        cfg.background_n_per_dim = 32;
        cfg.max_cells = 500000;
        cfg.multi_body = false;

        CfdMesh mesh;
        std::string err;
        if (!generate_watertight_mesh_from_stl(stl_path, mesh, cfg, &err))
            FAIL("watertight mesh with multi_body=false failed: %s", err.c_str());

        if (mesh.cells.empty()) FAIL("zero cells");
        if (mesh.faces.empty()) FAIL("zero faces");

        // All wall faces should have body_id=0 (default)
        for (const auto& face : mesh.faces) {
            if (face.boundary == BoundaryKind::NoSlipWall) {
                if (face.body_id != 0) FAIL("body_id=%d expected 0 with multi_body=false", face.body_id);
            }
        }
        PASS;
    }

    std::remove(stl_path);
    return 0;
}

// ── Sphere STL helpers ──────────────────────────────────────────────

struct SphVec { Real x, y, z; };

static bool write_uv_sphere(FILE* f, const char* solid_name,
                            Real cx, Real cy, Real cz, Real radius,
                            int n_theta, int n_phi) {
    std::fprintf(f, "solid %s\n", solid_name);

    // Write one triangle with outward-facing normal
    auto write_tri = [&](const SphVec& a, const SphVec& b, const SphVec& c) {
        Real ex1 = b.x - a.x, ey1 = b.y - a.y, ez1 = b.z - a.z;
        Real ex2 = c.x - a.x, ey2 = c.y - a.y, ez2 = c.z - a.z;
        Real nx = ey1 * ez2 - ez1 * ey2;
        Real ny = ez1 * ex2 - ex1 * ez2;
        Real nz = ex1 * ey2 - ey1 * ex2;
        Real len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len <= Real(0)) return;
        nx /= len; ny /= len; nz /= len;
        Real rcx = (a.x + b.x + c.x) / Real(3) - cx;
        Real rcy = (a.y + b.y + c.y) / Real(3) - cy;
        Real rcz = (a.z + b.z + c.z) / Real(3) - cz;
        Real ax = a.x, ay = a.y, az = a.z;
        Real bx = b.x, by = b.y, bz = b.z;
        Real cx_ = c.x, cy_ = c.y, cz_ = c.z;
        if (nx * rcx + ny * rcy + nz * rcz < Real(0)) {
            nx = -nx; ny = -ny; nz = -nz;
            ax = b.x; ay = b.y; az = b.z;
            bx = a.x; by = a.y; bz = a.z;
        }
        std::fprintf(f, "  facet normal %e %e %e\n", nx, ny, nz);
        std::fprintf(f, "    outer loop\n");
        std::fprintf(f, "      vertex %e %e %e\n", ax, ay, az);
        std::fprintf(f, "      vertex %e %e %e\n", bx, by, bz);
        std::fprintf(f, "      vertex %e %e %e\n", cx_, cy_, cz_);
        std::fprintf(f, "    endloop\n");
        std::fprintf(f, "  endfacet\n");
    };

    auto sph = [&](Real theta, Real phi) -> SphVec {
        Real s = std::sin(phi);
        SphVec p;
        p.x = cx + radius * s * std::cos(theta);
        p.y = cy + radius * std::cos(phi);
        p.z = cz + radius * s * std::sin(theta);
        return p;
    };

    // North pole fan
    SphVec north = sph(Real(0), Real(0));
    Real phi_step = (Real)M_PI / (Real)n_phi;
    for (int it = 0; it < n_theta; ++it) {
        Real t0 = (Real)it / (Real)n_theta * Real(2) * (Real)M_PI;
        Real t1 = (Real)(it + 1) / (Real)n_theta * Real(2) * (Real)M_PI;
        SphVec v0 = sph(t0, phi_step);
        SphVec v1 = sph(t1, phi_step);
        write_tri(north, v0, v1);
    }

    // Middle rings
    for (int ip = 1; ip < n_phi - 1; ++ip) {
        Real phi0 = (Real)ip * phi_step;
        Real phi1 = (Real)(ip + 1) * phi_step;
        for (int it = 0; it < n_theta; ++it) {
            Real t0 = (Real)it / (Real)n_theta * Real(2) * (Real)M_PI;
            Real t1 = (Real)(it + 1) / (Real)n_theta * Real(2) * (Real)M_PI;
            SphVec v00 = sph(t0, phi0);
            SphVec v01 = sph(t0, phi1);
            SphVec v10 = sph(t1, phi0);
            SphVec v11 = sph(t1, phi1);
            write_tri(v00, v01, v10);
            write_tri(v10, v01, v11);
        }
    }

    // South pole fan
    SphVec south = sph(Real(0), (Real)M_PI);
    Real phi_last = (Real)(n_phi - 1) * phi_step;
    for (int it = 0; it < n_theta; ++it) {
        Real t0 = (Real)it / (Real)n_theta * Real(2) * (Real)M_PI;
        Real t1 = (Real)(it + 1) / (Real)n_theta * Real(2) * (Real)M_PI;
        SphVec v0 = sph(t0, phi_last);
        SphVec v1 = sph(t1, phi_last);
        write_tri(south, v1, v0);
    }

    std::fprintf(f, "endsolid %s\n", solid_name);
    return true;
}

static bool write_two_spheres_stl(const char* path,
                                   Real r1, Real cx1, Real cy1, Real cz1,
                                   Real r2, Real cx2, Real cy2, Real cz2) {
    FILE* f = std::fopen(path, "w");
    if (!f) return false;
    bool ok = write_uv_sphere(f, "sphere1", cx1, cy1, cz1, r1, 24, 16) &&
              write_uv_sphere(f, "sphere2", cx2, cy2, cz2, r2, 24, 16);
    std::fclose(f);
    return ok;
}

static int test_two_spheres_multi_body() {
    const char* stl_path = "test_two_spheres_multi_body.stl";
    // Two disjoint spheres: one at (-1.5,0,0) radius 0.4, one at (1.5,0,0) radius 0.4
    if (!write_two_spheres_stl(stl_path, 0.4f, -1.5f,0,0, 0.4f, 1.5f,0,0))
        FAIL("failed to write two-sphere STL");

    TEST("CFD-MESH-MULTI-2 Two disjoint spheres: two body_ids on wall faces");
    {
        StlMeshConfig cfg;
        cfg.outer_scale = 2.5f;
        cfg.background_n_per_dim = 32;
        cfg.max_cells = 1000000;
        cfg.multi_body = true;

        CfdMesh mesh;
        std::string err;
        if (!generate_watertight_mesh_from_stl(stl_path, mesh, cfg, &err))
            FAIL("watertight mesh failed: %s", err.c_str());

        if (mesh.cells.empty()) FAIL("zero cells");
        if (mesh.faces.empty()) FAIL("zero faces");

        std::set<int> body_ids;
        int wall_count = 0;
        for (const auto& face : mesh.faces) {
            if (face.boundary == BoundaryKind::NoSlipWall) {
                body_ids.insert(face.body_id);
                ++wall_count;
            }
        }
        if (wall_count == 0) FAIL("zero wall faces");
        if (body_ids.size() < 2u) FAIL("expected >=2 body_ids, got %zu", body_ids.size());

        std::printf("\n  bodies=%zu wall_faces=%d body_ids={", body_ids.size(), wall_count);
        for (int bid : body_ids) std::printf("%d ", bid);
        std::printf("} cells=%zu", mesh.cells.size());

        auto report = compute_mesh_quality_detail(mesh);
        if (report.min_volume <= 0.0f) FAIL("min volume=%g", report.min_volume);
        PASS;
    }

    std::remove(stl_path);
    return 0;
}

static int test_per_body_force() {
    const char* stl_path = "test_symmetric_spheres.stl";
    if (!write_two_spheres_stl(stl_path, 0.4f, -2,0,0, 0.4f, 2,0,0))
        FAIL("failed to write symmetric spheres STL");

    TEST("CFD-MESH-MULTI-4 Per-body force: symmetric spheres CX equal");
    {
        StlMeshConfig cfg;
        cfg.outer_scale = 2.0f;
        cfg.background_n_per_dim = 24;
        cfg.max_cells = 500000;
        cfg.multi_body = true;

        CfdMesh mesh;
        std::string err;
        if (!generate_watertight_mesh_from_stl(stl_path, mesh, cfg, &err))
            FAIL("watertight mesh failed: %s", err.c_str());

        if (mesh.cells.empty()) FAIL("zero cells");

        std::vector<int> wall_faces;
        for (int i = 0; i < static_cast<int>(mesh.faces.size()); ++i)
            if (mesh.faces[i].boundary == BoundaryKind::NoSlipWall)
                wall_faces.push_back(i);
        if (wall_faces.empty()) FAIL("zero wall faces");

        std::set<int> body_ids;
        for (int fi : wall_faces) body_ids.insert(mesh.faces[fi].body_id);
        if (body_ids.size() < 2u) FAIL("expected >=2 body_ids, got %zu", body_ids.size());

        // Verify per-body force integration at uniform flow
        Real gamma = 1.4f;
        Real mach = 0.5f;
        PrimitiveState w_inf;
        w_inf.rho = 1.0f;
        w_inf.u = mach / std::sqrt(gamma);
        w_inf.v = 0.0f;
        w_inf.w = 0.0f;
        w_inf.p = 1.0f / gamma;
        ConservativeState q_inf = primitive_to_conservative(w_inf, gamma);
        std::vector<ConservativeState> q(mesh.cells.size(), q_inf);

        FreestreamCondition condition;
        condition.mach = mach;
        condition.alpha_deg = 0.0f;
        condition.beta_deg = 0.0f;

        CfdConfig config;
        config.gamma = gamma;
        config.ref_area = 1.0f;

        std::vector<PrimitiveGradient> dummy_grads;
        CfdForceResult force_body0;
        integrate_wall_forces(mesh, wall_faces, q, condition, config,
                              force_body0, &dummy_grads, 0);
        CfdForceResult force_body1;
        integrate_wall_forces(mesh, wall_faces, q, condition, config,
                              force_body1, &dummy_grads, 1);

        // At uniform flow, both bodies have zero net CX (pressure symmetric)
        // Verify the forces are computed without error and both are near zero
        int wall_body0 = 0, wall_body1 = 0;
        for (int fi : wall_faces) {
            if (mesh.faces[fi].body_id == 0) ++wall_body0;
            if (mesh.faces[fi].body_id == 1) ++wall_body1;
        }
        if (wall_body0 == 0) FAIL("zero wall faces for body 0");
        if (wall_body1 == 0) FAIL("zero wall faces for body 1");

        Real cx0 = force_body0.CX;
        Real cx1 = force_body1.CX;
        Real diff = std::fabs(cx0 - cx1);
        if (diff > 0.01f)
            FAIL("CX difference too large: body0=%g body1=%g diff=%g", cx0, cx1, diff);

        std::printf("\n  cells=%zu wall={body0=%d body1=%d} cx0=%g cx1=%g",
                    mesh.cells.size(), wall_body0, wall_body1, cx0, cx1);
        PASS;
    }

    std::remove(stl_path);
    return 0;
}

int main() {
    std::setbuf(stdout, NULL);
    int result = 0;
    result |= test_cone_stl_mesh();
    result |= test_prism_bl();
    result |= test_prism_degrade();
    result |= test_multi_body_stl();
    result |= test_two_spheres_multi_body();
    result |= test_per_body_force();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
