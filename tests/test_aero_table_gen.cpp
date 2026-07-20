#include "aero/panel/aero_solver.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_result.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

using namespace aerosp::aero::panel;
namespace cfd = aerosp::aero::cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { \
    ++test_count; \
    std::cout << "[" << (name) << "]\n"; \
} while(0)

#define FAIL(...) do { \
    char buf[512]; snprintf(buf, sizeof(buf), __VA_ARGS__); \
    std::cerr << "FAIL: " << buf << "\n"; \
    return 1; \
} while(0)

#define PASS do { ++pass_count; } while(0)

static bool is_cfd_fidelity(const std::string& f) {
    return f == "cfd-gpu" || f == "cfd-cpu";
}

// RAII temp file guard
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& p) : path(p) { std::remove(path.c_str()); }
    ~TempFile() { std::remove(path.c_str()); }
};

// ─── Helper: read CSV into rows ─────────────────────────────────────────
struct CsvRow {
    double mach, alpha, beta;
    double CX, CY, CZ, CL, CD, L_D, Cl, Cm, Cn;
    std::string fidelity;
};

static bool read_csv(const std::string& path, std::vector<CsvRow>& rows,
                     bool has_fidelity, std::string* error) {
    std::ifstream f(path);
    if (!f.is_open()) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::string line;
    if (!std::getline(f, line)) {
        if (error) *error = "empty file";
        return false;
    }
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        CsvRow r;
        auto next = [&](double& v) {
            if (!std::getline(ss, tok, ',')) return false;
            v = std::stod(tok);
            return true;
        };
        // CSV format: Mach,Alpha,Beta,CX,CY,CZ,CL,CD,L_D,Cl,Cm,Cn[,Fidelity]
        if (!next(r.mach) || !next(r.alpha) || !next(r.beta) ||
            !next(r.CX) || !next(r.CY) || !next(r.CZ) ||
            !next(r.CL) || !next(r.CD) || !next(r.L_D) ||
            !next(r.Cl) || !next(r.Cm) || !next(r.Cn))
            return false;
        if (has_fidelity) {
            std::getline(ss, r.fidelity, ',');
            // trim whitespace
            if (!r.fidelity.empty() && r.fidelity.back() == '\r') r.fidelity.pop_back();
        } else {
            r.fidelity = "newtonian";
        }
        rows.push_back(r);
    }
    return true;
}

// ─── Test: CFD table in-range ────────────────────────────────────────────
static int test_cfd_gpu_table_in_range() {
    TEST("TABLE-CFD-1 CFD table generation with 3x3 in-range grid");
    TempFile csv("test_table_cfd_gpu.csv");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;
    cfg.mesh_outer_scale = 3.0f;
    cfg.mesh_subdivisions = 1000;

    std::vector<double> mach   = {2.0, 4.0, 6.0};
    std::vector<double> alpha  = {0.0, 5.0, 10.0};
    std::vector<double> beta   = {0.0};

    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            csv.path, mach, alpha, beta, cfg))
        FAIL("generate_aero_table returned false for 3x3 in-range grid");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, true, &err))
        FAIL("read_csv failed: %s", err.c_str());
    if (rows.size() != 9)
        FAIL("expected 9 rows, got %zu", rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        auto& r = rows[i];
        if (!is_cfd_fidelity(r.fidelity))
            FAIL("row %zu: fidelity='%s' expected cfd-gpu|cfd-cpu", i, r.fidelity.c_str());
        if (!std::isfinite(r.CX) || !std::isfinite(r.CY) || !std::isfinite(r.CZ) ||
            !std::isfinite(r.CL) || !std::isfinite(r.CD) ||
            !std::isfinite(r.Cl) || !std::isfinite(r.Cm) || !std::isfinite(r.Cn))
            FAIL("row %zu: non-finite force (CX=%g CY=%g CZ=%g)", i, r.CX, r.CY, r.CZ);
        // beta=0 symmetry: expect CY, Cl, Cn near zero
        if (std::abs(r.CY) > 1e-2 || std::abs(r.Cl) > 1e-2 || std::abs(r.Cn) > 1e-2)
            FAIL("row %zu: symmetry violation beta=0: CY=%g Cl=%g Cn=%g (tol=1e-2)", i, r.CY, r.Cl, r.Cn);
    }

    std::cout << "PASS: " << rows.size() << " rows, all finite, symmetry holds\n";
    PASS;
    return 0;
}

// ─── Test: out-of-range → rejection ──────────────────────────────────────
static int test_cfd_gpu_out_of_range() {
    TEST("TABLE-CFD-2 CFD table rejects out-of-range conditions");
    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;

    // Test each boundary: Mach < 1.2, |alpha| > 30, |beta| > 10, Mach > 30

    // (a) Mach=0.5 below minimum
    {
        TempFile csv("test_cfd_out_mach_low.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {0.5}, {0.0}, {0.0}, cfg);
        if (ok) FAIL("Mach=0.5 should be rejected (below 1.2)");
    }

    // (b) alpha=31 above maximum
    {
        TempFile csv("test_cfd_out_alpha_high.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {4.0}, {31.0}, {0.0}, cfg);
        if (ok) FAIL("alpha=31 should be rejected (above 30)");
    }

    // (c) alpha=-31 below minimum
    {
        TempFile csv("test_cfd_out_alpha_low.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {4.0}, {-31.0}, {0.0}, cfg);
        if (ok) FAIL("alpha=-31 should be rejected (below -30)");
    }

    // (d) beta=11 above maximum
    {
        TempFile csv("test_cfd_out_beta_high.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {4.0}, {0.0}, {11.0}, cfg);
        if (ok) FAIL("beta=11 should be rejected (above 10)");
    }

    // (e) beta=-11 below minimum
    {
        TempFile csv("test_cfd_out_beta_low.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {4.0}, {0.0}, {-11.0}, cfg);
        if (ok) FAIL("beta=-11 should be rejected (below -10)");
    }

    // (f) Mach=31 above maximum
    {
        TempFile csv("test_cfd_out_mach_high.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {31.0}, {0.0}, {0.0}, cfg);
        if (ok) FAIL("Mach=31 should be rejected (above 30)");
    }

    std::cout << "PASS: all 6 out-of-range conditions correctly rejected\n";
    PASS;
    return 0;
}

// ─── Test: Newtonian baseline unchanged ──────────────────────────────────
static int test_newtonian_baseline() {
    TEST("TABLE-CFD-3 Newtonian baseline still works (use_fvm=false)");
    TempFile csv("test_table_newtonian.csv");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = false;

    std::vector<double> mach   = {2.0, 4.0};
    std::vector<double> alpha  = {0.0, 5.0};
    std::vector<double> beta   = {0.0};

    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            csv.path, mach, alpha, beta, cfg))
        FAIL("Newtonian: generate_aero_table returned false");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, false, &err))
        FAIL("Newtonian: read_csv failed: %s", err.c_str());
    if (rows.size() != 4)
        FAIL("Newtonian: expected 4 rows, got %zu", rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        if (!std::isfinite(rows[i].CX))
            FAIL("Newtonian row %zu: CX=%g not finite", i, rows[i].CX);
    }

    std::cout << "PASS: " << rows.size() << " Newtonian rows\n";
    PASS;
    return 0;
}

// ─── Test: CFD results differ from Newtonian ─────────────────────────────
static int test_cfd_differs_from_newtonian() {
    TEST("TABLE-CFD-4 CFD forces differ from Newtonian at non-zero alpha");
    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.mesh_outer_scale = 3.0f;
    cfg.mesh_subdivisions = 1000;

    std::vector<double> mach   = {4.0};
    std::vector<double> alpha  = {10.0};  // non-zero alpha for larger difference
    std::vector<double> beta   = {0.0};

    // Newtonian baseline
    TempFile nt_csv("test_nt.csv");
    cfg.use_fvm = false;
    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            nt_csv.path, mach, alpha, beta, cfg))
        FAIL("Newtonian table failed");

    // CFD GPU
    TempFile cfd_csv("test_cfd.csv");
    cfg.use_fvm = true;
    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            cfd_csv.path, mach, alpha, beta, cfg))
        FAIL("CFD table failed");

    std::vector<CsvRow> nrows, crows;
    std::string err;
    if (!read_csv(nt_csv.path,  nrows, false, &err)) FAIL("read Newtonian: %s", err.c_str());
    if (!read_csv(cfd_csv.path, crows, true,  &err)) FAIL("read CFD: %s", err.c_str());

    // Compare L/D ratio - more robust discriminator than single force component
    double newtonian_ld = std::abs(nrows[0].CL) / (std::abs(nrows[0].CD) + 1e-12);
    double cfd_ld       = std::abs(crows[0].CL) / (std::abs(crows[0].CD) + 1e-12);
    double diff = std::abs(cfd_ld - newtonian_ld) / (newtonian_ld + 1e-12);

    if (diff < 0.02)
        FAIL("L/D too similar: Newtonian L/D=%g CFD L/D=%g rel_diff=%g (need >=0.02)",
             newtonian_ld, cfd_ld, diff);

    std::cout << "PASS: Newtonian L/D=" << newtonian_ld << " CFD L/D=" << cfd_ld
              << " rel_diff=" << diff << "\n";
    PASS;
    return 0;
}

// ─── Test: single beta=0 ─────────────────────────────────────────────────
static int test_cfd_gpu_single_beta() {
    TEST("TABLE-CFD-5 CFD table with single beta=0 works");
    TempFile csv("test_cfd_beta0.csv");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;
    cfg.mesh_outer_scale = 3.0f;
    cfg.mesh_subdivisions = 1000;

    std::vector<double> mach   = {3.0, 5.0};
    std::vector<double> alpha  = {2.0};
    std::vector<double> beta   = {0.0};

    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            csv.path, mach, alpha, beta, cfg))
        FAIL("table gen failed");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, true, &err))
        FAIL("read_csv: %s", err.c_str());
    if (rows.size() != 2)
        FAIL("expected 2 rows, got %zu", rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        if (!is_cfd_fidelity(rows[i].fidelity))
            FAIL("row %zu: fidelity='%s' expected cfd-gpu|cfd-cpu", i, rows[i].fidelity.c_str());
        if (std::abs(rows[i].CY) > 1e-2)
            FAIL("row %zu: beta=0 symmetry CY=%g > 1e-2", i, rows[i].CY);
    }

    std::cout << "PASS: " << rows.size() << " rows, fidelity=" << rows[0].fidelity << "\n";
    PASS;
    return 0;
}

// ─── Test: non-zero beta grid ────────────────────────────────────────────
static int test_cfd_gpu_nonzero_beta() {
    TEST("TABLE-CFD-6 CFD table with multidimensional beta grid");
    TempFile csv("test_cfd_beta_multi.csv");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;
    cfg.mesh_outer_scale = 3.0f;
    cfg.mesh_subdivisions = 1000;

    std::vector<double> mach   = {3.0};
    std::vector<double> alpha  = {0.0, 5.0};
    std::vector<double> beta   = {-5.0, 0.0, 5.0};

    if (!generate_aero_table(
            "data/missile/hgv_model_optimized.stl",
            csv.path, mach, alpha, beta, cfg))
        FAIL("table gen failed for multi-beta grid");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, true, &err))
        FAIL("read_csv: %s", err.c_str());
    // 1 mach x 2 alpha x 3 beta = 6 rows
    if (rows.size() != 6)
        FAIL("expected 6 rows (1x2x3), got %zu", rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        auto& r = rows[i];
        if (!is_cfd_fidelity(r.fidelity))
            FAIL("row %zu: fidelity='%s' expected cfd-gpu|cfd-cpu", i, r.fidelity.c_str());
        if (!std::isfinite(r.CX))
            FAIL("row %zu: CX=%g not finite", i, r.CX);
    }

    // Check beta symmetry: rows i (beta=-5) and i+2 (beta=+5) should have near-opposite CY
    // Row order: alpha=0 beta=-5, alpha=0 beta=0, alpha=0 beta=+5,
    //            alpha=5 beta=-5, alpha=5 beta=0, alpha=5 beta=+5
    double cy_neg = rows[0].CY;  // alpha=0, beta=-5
    double cy_pos = rows[2].CY;  // alpha=0, beta=+5
    // CY should be antisymmetric with beta
    double cy_sum = cy_neg + cy_pos;
    double cy_avg = 0.5 * (std::abs(cy_neg) + std::abs(cy_pos));
    if (cy_avg > 1e-3 && std::abs(cy_sum) / cy_avg > 0.5)
        FAIL("beta symmetry violation: alpha=0 CY(beta=-5)=%g CY(beta=+5)=%g sum/avg=%g",
             cy_neg, cy_pos, std::abs(cy_sum)/cy_avg);

    std::cout << "PASS: " << rows.size() << " rows, beta symmetry holds\n";
    PASS;
    return 0;
}

// ─── Test: empty input vectors ───────────────────────────────────────────
static int test_cfd_gpu_empty_input() {
    TEST("TABLE-CFD-7 Empty input vectors rejected");
    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;

    // All empty
    {
        TempFile csv("test_cfd_empty.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {}, {}, {}, cfg);
        if (ok) FAIL("empty input should return false");
    }

    // Partial empty
    {
        TempFile csv("test_cfd_empty_part.csv");
        bool ok = generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {3.0}, {}, {0.0}, cfg);
        if (ok) FAIL("partially empty input (alpha empty) should return false");
    }

    std::cout << "PASS: empty input correctly rejected\n";
    PASS;
    return 0;
}

// Write a closed solid cone STL for watertight hex-cull (ray-cast SDF needs
// a closed surface; some production STLs are open shells).
static bool write_test_cone_stl(const char* path, double radius, double height, int n_seg) {
    FILE* f = std::fopen(path, "w");
    if (!f) return false;
    std::fprintf(f, "solid cone\n");
    double half_h = height * 0.5;
    for (int i = 0; i < n_seg; ++i) {
        double a0 = 2.0 * 3.14159265358979323846 * i / n_seg;
        double a1 = 2.0 * 3.14159265358979323846 * (i + 1) / n_seg;
        double x0 = radius * std::cos(a0), z0 = radius * std::sin(a0);
        double x1 = radius * std::cos(a1), z1 = radius * std::sin(a1);
        std::fprintf(f, "  facet normal 0 0 0\n    outer loop\n");
        std::fprintf(f, "      vertex 0 %g 0\n", half_h);
        std::fprintf(f, "      vertex %g %g %g\n", x0, -half_h, z0);
        std::fprintf(f, "      vertex %g %g %g\n", x1, -half_h, z1);
        std::fprintf(f, "    endloop\n  endfacet\n");
        std::fprintf(f, "  facet normal 0 -1 0\n    outer loop\n");
        std::fprintf(f, "      vertex 0 %g 0\n", -half_h);
        std::fprintf(f, "      vertex %g %g %g\n", x1, -half_h, z1);
        std::fprintf(f, "      vertex %g %g %g\n", x0, -half_h, z0);
        std::fprintf(f, "    endloop\n  endfacet\n");
    }
    std::fprintf(f, "endsolid cone\n");
    std::fclose(f);
    return true;
}

// ─── Test: STL volume mesh forces differ from cube embedding ─────────────
static int test_stl_volume_mesh_differs_from_cube() {
    TEST("TABLE-STL-1 use_fvm + stl_volume_mesh forces differ from cube-embedding");

    const char* stl_path = "test_table_cone.stl";
    if (!write_test_cone_stl(stl_path, 0.5, 1.0, 64))
        FAIL("failed to write cone STL");

    std::vector<double> mach  = {3.0};
    std::vector<double> alpha = {5.0};
    std::vector<double> beta  = {0.0};

    AeroTableConfig cfg;
    cfg.ref_area   = 1.0f;
    cfg.ref_length = 1.0f;
    cfg.ref_span   = 1.0f;
    cfg.com_x      = 0.0f;
    cfg.use_fvm    = true;
    cfg.mesh_subdivisions = 1000;
    cfg.mesh_outer_scale  = 3.0f;

    TempFile cube_csv("test_table_cube_embed.csv");
    cfg.stl_volume_mesh = false;
    if (!generate_aero_table(stl_path, cube_csv.path, mach, alpha, beta, cfg)) {
        std::remove(stl_path);
        FAIL("cube-embedding table generation failed");
    }

    TempFile stl_csv("test_table_stl_volume.csv");
    cfg.stl_volume_mesh = true;
    cfg.mesh_outer_scale = 3.0f;
    cfg.stl_background_n_per_dim = 24;
    cfg.stl_max_cells = 500000;
    if (!generate_aero_table(stl_path, stl_csv.path, mach, alpha, beta, cfg)) {
        std::remove(stl_path);
        FAIL("watertight STL table generation failed");
    }
    std::remove(stl_path);

    std::vector<CsvRow> cube_rows, stl_rows;
    std::string err;
    if (!read_csv(cube_csv.path, cube_rows, true, &err))
        FAIL("read cube CSV: %s", err.c_str());
    if (!read_csv(stl_csv.path, stl_rows, true, &err))
        FAIL("read STL CSV: %s", err.c_str());
    if (cube_rows.size() != 1 || stl_rows.size() != 1)
        FAIL("expected 1 row each, got cube=%zu stl=%zu",
             cube_rows.size(), stl_rows.size());

    if (!is_cfd_fidelity(cube_rows[0].fidelity) || !is_cfd_fidelity(stl_rows[0].fidelity))
        FAIL("fidelity expected cfd-*, got cube='%s' stl='%s'",
             cube_rows[0].fidelity.c_str(), stl_rows[0].fidelity.c_str());

    if (!std::isfinite(stl_rows[0].CX) || !std::isfinite(stl_rows[0].CD))
        FAIL("STL conformal forces non-finite: CX=%g CD=%g",
             stl_rows[0].CX, stl_rows[0].CD);

    double d_cx = std::abs(cube_rows[0].CX - stl_rows[0].CX);
    double d_cd = std::abs(cube_rows[0].CD - stl_rows[0].CD);
    double scale = 1.0 + std::max({std::abs(cube_rows[0].CX), std::abs(stl_rows[0].CX),
                                   std::abs(cube_rows[0].CD), std::abs(stl_rows[0].CD)});
    if (d_cx / scale < 1e-4 && d_cd / scale < 1e-4)
        FAIL("conformal and cube forces too similar: cube CX=%g CD=%g stl CX=%g CD=%g",
             cube_rows[0].CX, cube_rows[0].CD, stl_rows[0].CX, stl_rows[0].CD);

    std::cout << "PASS: cube CX=" << cube_rows[0].CX << " CD=" << cube_rows[0].CD
              << " | stl CX=" << stl_rows[0].CX << " CD=" << stl_rows[0].CD << "\n";
    PASS;
    return 0;
}

// ─── Test: stl_volume_mesh=false keeps cube path (regression) ────────────
static int test_stl_volume_mesh_false_regression() {
    TEST("TABLE-STL-2 stl_volume_mesh=false retains cube-embedding path");
    TempFile csv("test_table_stl_flag_off.csv");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;
    cfg.stl_volume_mesh = false;
    cfg.mesh_outer_scale = 3.0f;
    cfg.mesh_subdivisions = 1000;

    if (!generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {2.0}, {0.0}, {0.0}, cfg))
        FAIL("cube path with stl_volume_mesh=false failed");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, true, &err))
        FAIL("read_csv: %s", err.c_str());
    if (rows.size() != 1)
        FAIL("expected 1 row, got %zu", rows.size());
    if (!is_cfd_fidelity(rows[0].fidelity))
        FAIL("fidelity='%s' expected cfd-gpu|cfd-cpu", rows[0].fidelity.c_str());
    if (!std::isfinite(rows[0].CX))
        FAIL("CX not finite");

    std::cout << "PASS: cube-embedding regression OK fidelity=" << rows[0].fidelity << "\n";
    PASS;
    return 0;
}

// ─── Test: production HGV STL watertight mesh + CFD forces ───────────────
static int test_stl_volume_mesh_hgv_production() {
    TEST("TABLE-STL-3 HGV production STL hex-cull mesh load_mesh + finite CFD");

    AeroTableConfig cfg;
    cfg.ref_area   = 1.131f;
    cfg.ref_length = 12.0f;
    cfg.ref_span   = 3.0f;
    cfg.com_x      = 6.0f;
    cfg.use_fvm    = true;
    cfg.stl_volume_mesh = true;
    cfg.mesh_outer_scale = 3.0f;
    cfg.stl_background_n_per_dim = 20;
    cfg.stl_max_cells = 2000000;

    TempFile csv("test_table_hgv_stl_volume.csv");
    if (!generate_aero_table("data/missile/hgv_model_optimized.stl",
            csv.path, {3.0}, {0.0}, {0.0}, cfg))
        FAIL("HGV stl_volume_mesh table generation failed");

    std::vector<CsvRow> rows;
    std::string err;
    if (!read_csv(csv.path, rows, true, &err))
        FAIL("read_csv: %s", err.c_str());
    if (rows.size() != 1)
        FAIL("expected 1 row, got %zu", rows.size());
    if (!is_cfd_fidelity(rows[0].fidelity))
        FAIL("fidelity='%s'", rows[0].fidelity.c_str());
    if (!std::isfinite(rows[0].CX) || !std::isfinite(rows[0].CD))
        FAIL("non-finite forces CX=%g CD=%g", rows[0].CX, rows[0].CD);
    // Closed body at alpha=0 should keep lateral coefficients small.
    if (std::abs(rows[0].CY) > 1e-2 || std::abs(rows[0].Cl) > 1e-2 ||
        std::abs(rows[0].Cn) > 1e-2)
        FAIL("alpha=0 lateral forces CY=%g Cl=%g Cn=%g", rows[0].CY, rows[0].Cl, rows[0].Cn);

    std::cout << "PASS: HGV STL CX=" << rows[0].CX << " CD=" << rows[0].CD
              << " fidelity=" << rows[0].fidelity << "\n";
    PASS;
    return 0;
}

int main() {
    int failures = 0;
    failures += test_cfd_gpu_table_in_range();
    failures += test_cfd_gpu_out_of_range();
    failures += test_newtonian_baseline();
    failures += test_cfd_differs_from_newtonian();
    failures += test_cfd_gpu_single_beta();
    failures += test_cfd_gpu_nonzero_beta();
    failures += test_cfd_gpu_empty_input();
    failures += test_stl_volume_mesh_differs_from_cube();
    failures += test_stl_volume_mesh_false_regression();
    failures += test_stl_volume_mesh_hgv_production();

    std::cout << "\n[" << pass_count << "/" << test_count << " tests passed]\n";
    if (failures) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cout << "All tests PASS\n";
    return 0;
}
