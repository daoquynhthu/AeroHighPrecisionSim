#define _USE_MATH_DEFINES
#include "aero/cfd/mesh_gen_stl.hpp"
#include "aero/cfd/mesh_gen_stl_internal.hpp"
#include "aero/cfd/mesh_validator.hpp"
#include "aero/cfd/mesh_io.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace aerosp;
using namespace aerosp::aero::cfd;
using namespace aerosp::aero::cfd::stl_internal;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static int test_vec3_ops() {
    TEST("vec3_ops");
    Vec3 a{1,2,3}, b{4,5,6};

    Vec3 sum = a + b;
    if (sum.x != 5 || sum.y != 7 || sum.z != 9) FAIL("a+b failed");

    Vec3 diff = a - b;
    if (diff.x != -3 || diff.y != -3 || diff.z != -3) FAIL("a-b failed");

    Vec3 scaled = a * 2.0f;
    if (scaled.x != 2 || scaled.y != 4 || scaled.z != 6) FAIL("a*2 failed");

    Vec3 divd = scaled / 2.0f;
    if (divd.x != 1 || divd.y != 2 || divd.z != 3) FAIL("a/2 failed");

    Real d = dot(a, b);
    if (std::fabs(d - 32.0f) > 1e-7f) FAIL("dot=32 expected, got %g", d);

    Real n = norm(a);
    Real expected_n = std::sqrt(14.0f);
    if (std::fabs(n - expected_n) > 1e-7f) FAIL("norm=%g expected %g", n, expected_n);

    Vec3 c = cross(a, b);
    if (std::fabs(c.x + 3) > 1e-7f || std::fabs(c.y - 6) > 1e-7f || std::fabs(c.z + 3) > 1e-7f)
        FAIL("cross failed: (%g,%g,%g) expected (-3,6,-3)", c.x, c.y, c.z);

    Vec3 normed = normalize(a);
    Real len = norm(normed);
    if (std::fabs(len - 1.0f) > 1e-7f) FAIL("normalize length=%g expected 1", len);

    Vec3 zero{0,0,0};
    Vec3 nz = normalize(zero);
    if (nz.x != 0 || nz.y != 0 || nz.z != 0) FAIL("normalize zero should be zero");

    PASS;
    return 0;
}

static int test_volume_tet_signed() {
    TEST("volume_tet_signed");
    Vec3 a{0,0,0}, b{1,0,0}, c{0,1,0}, d{0,0,1};
    Real v = volume_tet_signed(a, b, c, d);
    if (std::fabs(v - 1.0f/6.0f) > 1e-7f)
        FAIL("unit tet volume=%g expected=%g", v, 1.0f/6.0f);
    PASS;
    return 0;
}

static int test_volume_tet_signed_negative() {
    TEST("volume_tet_signed_negative");
    Vec3 a{0,0,0}, b{1,0,0}, c{0,1,0}, d{0,0,1};
    Real v_pos = volume_tet_signed(a, b, c, d);
    Real v_neg = volume_tet_signed(a, c, b, d);
    if (v_neg >= 0) FAIL("expected negative volume, got %g", v_neg);
    if (std::fabs(v_neg + v_pos) > 1e-7f) FAIL("negated volume mismatch: %g vs %g", v_neg, -v_pos);
    PASS;
    return 0;
}

static int test_volume_tet_signed_zero_volume() {
    TEST("volume_tet_signed_zero_volume");
    Vec3 a{0,0,0}, b{1,0,0}, c{0,1,0}, d{2,3,0};
    Real v = volume_tet_signed(a, b, c, d);
    if (std::fabs(v) > 1e-7f) FAIL("coplanar volume=%g expected 0", v);
    PASS;
    return 0;
}

static int test_hex_to_6_tets_unit_cube() {
    TEST("hex_to_6_tets_unit_cube");
    Vec3 hex[8] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
    };
    Vec3 tets[6][4];
    hex_to_6_tets(hex, tets);

    Real total_vol = 0;
    for (int t = 0; t < 6; ++t) {
        Real v = volume_tet_signed(tets[t][0], tets[t][1], tets[t][2], tets[t][3]);
        if (v <= 0) FAIL("tet %d has non-positive volume %g", t, v);
        total_vol += v;
    }

    if (std::fabs(total_vol - 1.0f) > 1e-7f)
        FAIL("total volume=%g expected 1.0", total_vol);

    PASS;
    return 0;
}

static int test_hex_to_6_tets_non_cuboid() {
    TEST("hex_to_6_tets_non_cuboid");
    Vec3 hex[8] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0.5f,0,1}, {1.5f,0,1}, {1.5f,1,1}, {0.5f,1,1}
    };
    Vec3 tets[6][4];
    hex_to_6_tets(hex, tets);

    Real total_vol = 0;
    for (int t = 0; t < 6; ++t) {
        Real v = volume_tet_signed(tets[t][0], tets[t][1], tets[t][2], tets[t][3]);
        if (v <= 0) FAIL("tet %d has non-positive volume %g", t, v);
        total_vol += v;
    }

    if (std::fabs(total_vol - 1.0f) > 1e-7f)
        FAIL("sheared hex total volume=%g expected 1.0", total_vol);

    PASS;
    return 0;
}

static int test_hex_to_6_tets_all_vertices_from_input() {
    TEST("hex_to_6_tets_all_vertices_from_input");
    Vec3 hex[8] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
    };
    Vec3 tets[6][4];
    hex_to_6_tets(hex, tets);

    for (int t = 0; t < 6; ++t) {
        for (int v = 0; v < 4; ++v) {
            bool found = false;
            for (int h = 0; h < 8; ++h) {
                if (std::fabs(tets[t][v].x - hex[h].x) < 1e-10f &&
                    std::fabs(tets[t][v].y - hex[h].y) < 1e-10f &&
                    std::fabs(tets[t][v].z - hex[h].z) < 1e-10f) {
                    found = true; break;
                }
            }
            if (!found) FAIL("tet %d vertex %d not in hex input", t, v);
        }
    }

    PASS;
    return 0;
}

static int test_detect_stl_format_ascii() {
    TEST("detect_stl_format_ascii");
    const char* path = "test_detect_ascii.stl";
    {
        std::FILE* f = std::fopen(path, "w");
        if (!f) FAIL("cannot create ASCII test file");
        std::fprintf(f, "solid test\n");
        std::fprintf(f, "  facet normal 0 0 1\n");
        std::fprintf(f, "    outer loop\n");
        std::fprintf(f, "      vertex 0 0 0\n");
        std::fprintf(f, "      vertex 1 0 0\n");
        std::fprintf(f, "      vertex 0 1 0\n");
        std::fprintf(f, "    endloop\n");
        std::fprintf(f, "  endfacet\n");
        std::fprintf(f, "endsolid test\n");
        std::fclose(f);
    }

    StlHeader hdr;
    int fmt = detect_stl_format(path, hdr);
    std::remove(path);
    if (fmt != 1) FAIL("expected ASCII (1) got %d", fmt);
    PASS;
    return 0;
}

static int test_detect_stl_format_binary() {
    TEST("detect_stl_format_binary");
    const char* path = "test_detect_binary.stl";
    {
        std::FILE* f = std::fopen(path, "wb");
        if (!f) FAIL("cannot create binary test file");
        char hdr[80] = {0};
        std::fwrite(hdr, 1, 80, f);
        unsigned int nt = 1;
        std::fwrite(&nt, 4, 1, f);
        float tri[12] = {0,0,1, 0,0,0, 1,0,0, 0,1,0};
        std::fwrite(tri, sizeof(float), 12, f);
        unsigned short attr = 0;
        std::fwrite(&attr, 2, 1, f);
        std::fclose(f);
    }

    StlHeader hdr;
    int fmt = detect_stl_format(path, hdr);
    std::remove(path);
    if (fmt != 2) FAIL("expected binary (2) got %d", fmt);
    PASS;
    return 0;
}

static int test_detect_stl_format_not_found() {
    TEST("detect_stl_format_not_found");
    StlHeader hdr;
    int fmt = detect_stl_format("nonexistent_file.stl", hdr);
    if (fmt != 0) FAIL("expected 0 for nonexistent file, got %d", fmt);
    PASS;
    return 0;
}

static int test_parse_stl_ascii_known_tri_count() {
    TEST("parse_stl_ascii_known_tri_count");
    const char* path = "test_parse_ascii.stl";
    {
        std::FILE* f = std::fopen(path, "w");
        std::fprintf(f, "solid test\n");
        std::fprintf(f, "  facet normal 0 0 1\n    outer loop\n");
        std::fprintf(f, "      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\n");
        std::fprintf(f, "  facet normal 0 0 1\n    outer loop\n");
        std::fprintf(f, "      vertex 1 0 0\n      vertex 1 1 0\n      vertex 0 1 0\n    endloop\n  endfacet\n");
        std::fprintf(f, "endsolid test\n");
        std::fclose(f);
    }

    std::string err;
    auto tris = parse_stl(path, &err);
    std::remove(path);
    if (tris.empty()) FAIL("parse returned empty: %s", err.c_str());
    if (static_cast<int>(tris.size()) != 2) FAIL("expected 2 triangles, got %zu", tris.size());
    PASS;
    return 0;
}

static int test_parse_stl_binary_known_tri_count() {
    TEST("parse_stl_binary_known_tri_count");
    const char* path = "test_parse_binary.stl";
    {
        std::FILE* f = std::fopen(path, "wb");
        char hdr[80] = {0};
        std::fwrite(hdr, 1, 80, f);
        unsigned int nt = 2;
        std::fwrite(&nt, 4, 1, f);
        float tri1[12] = {0,0,1, 0,0,0, 1,0,0, 0,1,0};
        std::fwrite(tri1, sizeof(float), 12, f);
        unsigned short attr1 = 0;
        std::fwrite(&attr1, 2, 1, f);
        float tri2[12] = {0,0,1, 1,0,0, 1,1,0, 0,1,0};
        std::fwrite(tri2, sizeof(float), 12, f);
        unsigned short attr2 = 0;
        std::fwrite(&attr2, 2, 1, f);
        std::fclose(f);
    }

    std::string err;
    auto tris = parse_stl(path, &err);
    std::remove(path);
    if (tris.empty()) FAIL("parse returned empty: %s", err.c_str());
    if (static_cast<int>(tris.size()) != 2) FAIL("expected 2 triangles, got %zu", tris.size());
    PASS;
    return 0;
}

int main() {
    std::setbuf(stdout, NULL);
    int result = 0;
#define RUN(test) do { result |= test(); } while(0)
    RUN(test_vec3_ops);
    RUN(test_volume_tet_signed);
    RUN(test_volume_tet_signed_negative);
    RUN(test_volume_tet_signed_zero_volume);
    RUN(test_hex_to_6_tets_unit_cube);
    RUN(test_hex_to_6_tets_non_cuboid);
    RUN(test_hex_to_6_tets_all_vertices_from_input);
    RUN(test_detect_stl_format_ascii);
    RUN(test_detect_stl_format_binary);
    RUN(test_detect_stl_format_not_found);
    RUN(test_parse_stl_ascii_known_tri_count);
    RUN(test_parse_stl_binary_known_tri_count);
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && test_count == pass_count ? 0 : 1;
}
