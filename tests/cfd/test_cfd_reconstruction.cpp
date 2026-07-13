#include "aero/cfd/reconstruction.hpp"
#include "aero/cfd/real.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace aerosp;
using namespace aerosp::aero::cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static int test_green_gauss() {
    TEST("CFD-RECON-1 Green-Gauss gradient is zero for constant primitive state");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        auto gradients = compute_green_gauss_gradients(mesh, q, 1.4f);
        if (gradients.size() != mesh.cells.size()) FAIL("gradient size=%zu cells=%zu", gradients.size(), mesh.cells.size());
        for (const auto& g : gradients) {
            if (std::fabs(g.drho_dx) > 1e-5f || std::fabs(g.drho_dy) > 1e-5f || std::fabs(g.drho_dz) > 1e-5f) {
                FAIL("rho gradient=[%g,%g,%g]", g.drho_dx, g.drho_dy, g.drho_dz);
            }
            if (std::fabs(g.dp_dx) > 1e-5f || std::fabs(g.dp_dy) > 1e-5f || std::fabs(g.dp_dz) > 1e-5f) {
                FAIL("p gradient=[%g,%g,%g]", g.dp_dx, g.dp_dy, g.dp_dz);
            }
        }
        PASS;
    }

    TEST("CFD-RECON-2 invalid state makes Green-Gauss fail closed");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        q[0].rho = -1.0f;
        auto gradients = compute_green_gauss_gradients(mesh, q, 1.4f);
        if (!gradients.empty()) FAIL("expected empty gradients");
        PASS;
    }
    return 0;
}

static int test_least_squares() {
    TEST("CFD-RECON-3 least-squares recovers linear pressure gradient");
    {
        CfdMesh mesh;
        mesh.cells.resize(4);
        mesh.cells[0].cx = 0.0f;
        mesh.cells[0].cy = 0.0f;
        mesh.cells[0].cz = 0.0f;
        mesh.cells[1].cx = 1.0f;
        mesh.cells[1].cy = 0.0f;
        mesh.cells[1].cz = 0.0f;
        mesh.cells[2].cx = 0.0f;
        mesh.cells[2].cy = 1.0f;
        mesh.cells[2].cz = 0.0f;
        mesh.cells[3].cx = 0.0f;
        mesh.cells[3].cy = 0.0f;
        mesh.cells[3].cz = 1.0f;

        for (int i = 1; i < 4; ++i) {
            CfdFace face;
            face.left_cell = 0;
            face.right_cell = i;
            face.boundary = BoundaryKind::Interior;
            mesh.faces.push_back(face);
        }

        std::vector<ConservativeState> q;
        for (const auto& cell : mesh.cells) {
            PrimitiveState w;
            w.rho = 1.0f;
            w.u = 0.0f;
            w.v = 0.0f;
            w.w = 0.0f;
            w.p = 1.0f + 0.2f*cell.cx - 0.1f*cell.cy + 0.3f*cell.cz;
            q.push_back(primitive_to_conservative(w, 1.4f));
        }

        auto gradients = compute_least_squares_gradients(mesh, q, 1.4f);
        if (gradients.size() != mesh.cells.size()) FAIL("gradient size=%zu", gradients.size());
        if (std::fabs(gradients[0].dp_dx - 0.2f) > 1e-6f) FAIL("dp_dx=%g", gradients[0].dp_dx);
        if (std::fabs(gradients[0].dp_dy + 0.1f) > 1e-6f) FAIL("dp_dy=%g", gradients[0].dp_dy);
        if (std::fabs(gradients[0].dp_dz - 0.3f) > 1e-6f) FAIL("dp_dz=%g", gradients[0].dp_dz);
        PASS;
    }
    return 0;
}

static int test_positive_guard() {
    TEST("CFD-RECON-4 positive guard limits density and pressure");
    {
        PrimitiveState center;
        center.rho = 1.0f;
        center.u = 0.0f;
        center.v = 0.0f;
        center.w = 0.0f;
        center.p = 1.0f;

        PrimitiveGradient g;
        g.drho_dx = -10.0f;
        g.dp_dx = -20.0f;
        Real theta = 1.0f;
        auto out = reconstruct_primitive_positive(center, g, 0.1f, 0.0f, 0.0f, 0.2f, 0.2f, &theta);
        if (theta >= 1.0f || theta <= 0.0f) FAIL("theta=%g", theta);
        if (out.rho < 0.2f - 1e-6f) FAIL("rho=%g", out.rho);
        if (out.p < 0.2f - 1e-6f) FAIL("p=%g", out.p);
        PASS;
    }

    TEST("CFD-RECON-5 positive guard preserves safe reconstruction");
    {
        PrimitiveState center;
        center.rho = 1.0f;
        center.u = 0.0f;
        center.v = 0.0f;
        center.w = 0.0f;
        center.p = 1.0f;

        PrimitiveGradient g;
        g.drho_dx = 0.5f;
        g.dp_dx = -0.5f;
        Real theta = 0.0f;
        auto guarded = reconstruct_primitive_positive(center, g, 0.1f, 0.0f, 0.0f, 0.2f, 0.2f, &theta);
        auto raw = reconstruct_primitive(center, g, 0.1f, 0.0f, 0.0f);
        if (std::fabs(theta - 1.0f) > 1e-6f) FAIL("theta=%g", theta);
        if (std::fabs(guarded.rho - raw.rho) > 1e-6f) FAIL("rho=%g raw=%g", guarded.rho, raw.rho);
        if (std::fabs(guarded.p - raw.p) > 1e-6f) FAIL("p=%g raw=%g", guarded.p, raw.p);
        PASS;
    }
    return 0;
}

static int test_limiter() {
    TEST("CFD-RECON-6 limiter is inactive for zero gradients");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        std::vector<PrimitiveGradient> gradients(mesh.cells.size());
        auto limiters = compute_barth_jespersen_limiters(mesh, q, gradients, 1.4f);
        if (limiters.size() != mesh.cells.size()) FAIL("limiter size=%zu", limiters.size());
        for (const auto& limiter : limiters) {
            if (std::fabs(limiter.rho - 1.0f) > 1e-6f) FAIL("rho limiter=%g", limiter.rho);
            if (std::fabs(limiter.p - 1.0f) > 1e-6f) FAIL("p limiter=%g", limiter.p);
        }
        PASS;
    }

    TEST("CFD-RECON-7 limiter suppresses new pressure extrema");
    {
        CfdMesh mesh = generate_flat_plate_mesh(0.5f, 0.05f, 0.1f, 1e-5f, 1.12f, 5, 3, 6);
        auto w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        std::vector<ConservativeState> q(mesh.cells.size(), primitive_to_conservative(w, 1.4f));
        std::vector<PrimitiveGradient> gradients(mesh.cells.size());
        for (auto& gradient : gradients) {
            gradient.dp_dx = 100.0f;
            gradient.dp_dy = 100.0f;
            gradient.dp_dz = 100.0f;
        }
        auto limiters = compute_barth_jespersen_limiters(mesh, q, gradients, 1.4f);
        if (limiters.size() != mesh.cells.size()) FAIL("limiter size=%zu", limiters.size());

        Real min_p_limiter = 1.0f;
        for (const auto& limiter : limiters) min_p_limiter = std::min(min_p_limiter, limiter.p);
        if (min_p_limiter >= 1.0f) FAIL("min pressure limiter=%g", min_p_limiter);

        auto limited = apply_limiter(gradients[0], limiters[0]);
        if (std::fabs(limited.dp_dx) > std::fabs(gradients[0].dp_dx) + 1e-6f) FAIL("limited dp_dx=%g", limited.dp_dx);
        PASS;
    }
    return 0;
}

static int test_reconstruct_primitive() {
    TEST("CFD-RECON-8 reconstruct_primitive zero gradient returns center");
    {
        PrimitiveState center;
        center.rho = 1.0f; center.u = 2.0f; center.v = 3.0f; center.w = 4.0f; center.p = 5.0f; center.nu_tilde = 0.1f;
        PrimitiveGradient g;
        PrimitiveState r = reconstruct_primitive(center, g, 1.0f, 2.0f, 3.0f);
        if (std::fabs(r.rho - center.rho) > 1e-6f) FAIL("rho=%g", r.rho);
        if (std::fabs(r.u - center.u) > 1e-6f) FAIL("u=%g", r.u);
        if (std::fabs(r.v - center.v) > 1e-6f) FAIL("v=%g", r.v);
        if (std::fabs(r.w - center.w) > 1e-6f) FAIL("w=%g", r.w);
        if (std::fabs(r.p - center.p) > 1e-6f) FAIL("p=%g", r.p);
        if (std::fabs(r.nu_tilde - center.nu_tilde) > 1e-6f) FAIL("nu_tilde=%g", r.nu_tilde);
        PASS;
    }

    TEST("CFD-RECON-9 reconstruct_primitive linear reconstruction with known gradient");
    {
        PrimitiveState center;
        center.rho = 1.0f; center.u = 2.0f; center.v = 3.0f; center.w = 4.0f; center.p = 5.0f; center.nu_tilde = 0.1f;
        PrimitiveGradient g;
        g.drho_dx = 10.0f; g.du_dy = 20.0f; g.dv_dz = 30.0f; g.dw_dx = 1.0f; g.dp_dy = 2.0f; g.dnu_tilde_dz = 5.0f;
        PrimitiveState r = reconstruct_primitive(center, g, 0.5f, 0.25f, 0.125f);
        if (std::fabs(r.rho - (1.0f + 10.0f * 0.5f)) > 1e-6f) FAIL("rho=%g", r.rho);
        if (std::fabs(r.u - (2.0f + 20.0f * 0.25f)) > 1e-6f) FAIL("u=%g", r.u);
        if (std::fabs(r.v - (3.0f + 30.0f * 0.125f)) > 1e-6f) FAIL("v=%g", r.v);
        if (std::fabs(r.w - (4.0f + 1.0f * 0.5f)) > 1e-6f) FAIL("w=%g", r.w);
        if (std::fabs(r.p - (5.0f + 2.0f * 0.25f)) > 1e-6f) FAIL("p=%g", r.p);
        if (std::fabs(r.nu_tilde - (0.1f + 5.0f * 0.125f)) > 1e-6f) FAIL("nu_tilde=%g", r.nu_tilde);
        PASS;
    }
    return 0;
}

static int test_reconstruct_positive() {
    TEST("CFD-RECON-10 reconstruct_primitive_positive clamps negative rho");
    {
        PrimitiveState center;
        center.rho = 1.0f; center.u = 0.0f; center.p = 1.0f;
        PrimitiveGradient g;
        g.drho_dx = -100.0f;
        PrimitiveState r = reconstruct_primitive_positive(center, g, 0.5f, 0.0f, 0.0f, 1e-3f, 1e-3f);
        if (r.rho < 1e-3f - 1e-4f) FAIL("rho=%g should be >= floor", r.rho);
        PASS;
    }

    TEST("CFD-RECON-11 reconstruct_primitive_positive clamps negative p");
    {
        PrimitiveState center;
        center.rho = 1.0f; center.u = 0.0f; center.p = 1.0f;
        PrimitiveGradient g;
        g.dp_dx = -100.0f;
        PrimitiveState r = reconstruct_primitive_positive(center, g, 0.5f, 0.0f, 0.0f, 1e-3f, 1e-3f);
        // p_floor=1e-3 is chosen so that (center - floor) is distinguishable from center in float32
        if (r.p < 1e-3f - 1e-4f) FAIL("p=%g should be >= floor", r.p);
        PASS;
    }

    TEST("CFD-RECON-12 reconstruct_primitive_positive returns theta");
    {
        PrimitiveState center;
        center.rho = 1.0f; center.u = 0.0f; center.p = 1.0f;
        PrimitiveGradient g;
        g.drho_dx = -100.0f;
        Real theta = -1.0f;
        reconstruct_primitive_positive(center, g, 0.5f, 0.0f, 0.0f, 1e-3f, 1e-3f, &theta);
        if (theta < 0.0f || theta > 1.0f) FAIL("theta=%g not in [0,1]", theta);
        if (theta >= 1.0f) FAIL("theta=%g should be < 1 for clamping case", theta);
        PASS;
    }

    TEST("CFD-RECON-13 reconstruct_primitive_positive no clamping gives theta=1");
    {
        PrimitiveState center;
        center.rho = 1.0f; center.u = 0.0f; center.p = 1.0f;
        PrimitiveGradient g;
        g.drho_dx = 0.5f; g.dp_dx = 0.5f;
        Real theta = -1.0f;
        reconstruct_primitive_positive(center, g, 0.5f, 0.0f, 0.0f, 1e-3f, 1e-3f, &theta);
        if (std::fabs(theta - 1.0f) > 1e-6f) FAIL("theta=%g expected 1.0", theta);
        PASS;
    }
    return 0;
}

static int test_apply_limiter() {
    TEST("CFD-RECON-14 apply_limiter identity returns same gradient");
    {
        PrimitiveGradient g;
        g.drho_dx = 1.0f; g.drho_dy = 2.0f; g.drho_dz = 3.0f;
        g.du_dx = 4.0f; g.du_dy = 5.0f; g.du_dz = 6.0f;
        g.dv_dx = 7.0f; g.dv_dy = 8.0f; g.dv_dz = 9.0f;
        g.dw_dx = 10.0f; g.dw_dy = 11.0f; g.dw_dz = 12.0f;
        g.dp_dx = 13.0f; g.dp_dy = 14.0f; g.dp_dz = 15.0f;
        g.dnu_tilde_dx = 16.0f; g.dnu_tilde_dy = 17.0f; g.dnu_tilde_dz = 18.0f;
        PrimitiveLimiter lim;
        PrimitiveGradient l = apply_limiter(g, lim);
        if (std::fabs(l.drho_dx - 1.0f) > 1e-6f) FAIL("drho_dx=%g", l.drho_dx);
        if (std::fabs(l.dp_dy - 14.0f) > 1e-6f) FAIL("dp_dy=%g", l.dp_dy);
        if (std::fabs(l.dnu_tilde_dz - 18.0f) > 1e-6f) FAIL("dnu_tilde_dz=%g", l.dnu_tilde_dz);
        PASS;
    }

    TEST("CFD-RECON-15 apply_limiter zero returns zero gradient");
    {
        PrimitiveGradient g;
        g.drho_dx = 1.0f; g.dv_dz = 9.0f; g.dw_dy = 11.0f; g.dp_dx = 13.0f; g.dnu_tilde_dx = 16.0f;
        PrimitiveLimiter lim;
        lim.rho = 0.0f; lim.u = 0.0f; lim.v = 0.0f; lim.w = 0.0f; lim.p = 0.0f; lim.nu_tilde = 0.0f;
        PrimitiveGradient l = apply_limiter(g, lim);
        if (l.drho_dx != 0.0f) FAIL("drho_dx=%g", l.drho_dx);
        if (l.dv_dz != 0.0f) FAIL("dv_dz=%g", l.dv_dz);
        if (l.dw_dy != 0.0f) FAIL("dw_dy=%g", l.dw_dy);
        if (l.dp_dx != 0.0f) FAIL("dp_dx=%g", l.dp_dx);
        if (l.dnu_tilde_dx != 0.0f) FAIL("dnu_tilde_dx=%g", l.dnu_tilde_dx);
        PASS;
    }

    TEST("CFD-RECON-16 apply_limiter partial scale each component");
    {
        PrimitiveGradient g;
        g.drho_dx = 10.0f; g.du_dy = 20.0f; g.dv_dz = 30.0f;
        g.dw_dx = 40.0f; g.dp_dy = 50.0f; g.dnu_tilde_dz = 60.0f;
        PrimitiveLimiter lim;
        lim.rho = 0.5f; lim.u = 0.25f; lim.v = 0.75f; lim.w = 0.1f; lim.p = 0.9f; lim.nu_tilde = 0.5f;
        PrimitiveGradient l = apply_limiter(g, lim);
        if (std::fabs(l.drho_dx - 5.0f) > 2e-6f) FAIL("drho_dx=%g", l.drho_dx);
        if (std::fabs(l.du_dy - 5.0f) > 2e-6f) FAIL("du_dy=%g", l.du_dy);
        if (std::fabs(l.dv_dz - 22.5f) > 2e-6f) FAIL("dv_dz=%g", l.dv_dz);
        if (std::fabs(l.dw_dx - 4.0f) > 2e-6f) FAIL("dw_dx=%g", l.dw_dx);
        if (std::fabs(l.dp_dy - 45.0f) > 2e-6f) FAIL("dp_dy=%g", l.dp_dy);
        if (std::fabs(l.dnu_tilde_dz - 30.0f) > 2e-6f) FAIL("dnu_tilde_dz=%g", l.dnu_tilde_dz);
        PASS;
    }
    return 0;
}

static int test_linear_solver() {
    TEST("CFD-RECON-17 solve_3x3 identity system returns input");
    {
        Real a[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
        Real b[3] = {3,5,7};
        Real x[3];
        if (!solve_3x3(a, b, x)) FAIL("solve_3x3 returned false on identity");
        if (std::fabs(x[0]-3) > 1e-7f) FAIL("x[0]=%g", x[0]);
        if (std::fabs(x[1]-5) > 1e-7f) FAIL("x[1]=%g", x[1]);
        if (std::fabs(x[2]-7) > 1e-7f) FAIL("x[2]=%g", x[2]);
        PASS;
    }

    TEST("CFD-RECON-18 solve_3x3 general system");
    {
        Real a[3][3] = {{2,1,1},{1,3,2},{1,0,1}};
        Real b[3] = {9,19,6};
        Real x[3];
        if (!solve_3x3(a, b, x)) FAIL("solve_3x3 returned false");
        Real r0 = a[0][0]*x[0] + a[0][1]*x[1] + a[0][2]*x[2] - b[0];
        Real r1 = a[1][0]*x[0] + a[1][1]*x[1] + a[1][2]*x[2] - b[1];
        Real r2 = a[2][0]*x[0] + a[2][1]*x[1] + a[2][2]*x[2] - b[2];
        if (std::fabs(r0) > 1e-6f) FAIL("residual[0]=%g", r0);
        if (std::fabs(r1) > 1e-6f) FAIL("residual[1]=%g", r1);
        if (std::fabs(r2) > 1e-6f) FAIL("residual[2]=%g", r2);
        PASS;
    }

    TEST("CFD-RECON-19 solve_3x3 singular returns false");
    {
        Real a[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        Real b[3] = {1,2,3};
        Real x[3] = {};
        if (solve_3x3(a, b, x)) FAIL("should have returned false for singular");
        PASS;
    }

    TEST("CFD-RECON-20 lu_factor_3x3 + lu_solve_3x3 round trip");
    {
        Real a[3][3] = {{3,1,2},{1,4,1},{2,1,5}};
        Real b[3] = {11,16,21};
        Real lu[3][3];
        int pivot[3];
        if (!lu_factor_3x3(a, lu, pivot)) FAIL("lu_factor failed");
        Real x[3];
        lu_solve_3x3(lu, pivot, b, x);
        Real r0 = a[0][0]*x[0] + a[0][1]*x[1] + a[0][2]*x[2] - b[0];
        Real r1 = a[1][0]*x[0] + a[1][1]*x[1] + a[1][2]*x[2] - b[1];
        Real r2 = a[2][0]*x[0] + a[2][1]*x[1] + a[2][2]*x[2] - b[2];
        if (std::fabs(r0) > 1e-6f) FAIL("lu residual[0]=%g", r0);
        if (std::fabs(r1) > 1e-6f) FAIL("lu residual[1]=%g", r1);
        if (std::fabs(r2) > 1e-6f) FAIL("lu residual[2]=%g", r2);
        PASS;
    }

    TEST("CFD-RECON-21 lu_factor_3x3 singular returns false");
    {
        Real a[3][3] = {{1,2,3},{2,4,6},{3,6,9}};
        Real lu[3][3];
        int pivot[3];
        if (lu_factor_3x3(a, lu, pivot)) FAIL("should have returned false for singular");
        PASS;
    }

    return 0;
}

int main() {
    int result = 0;
    result |= test_green_gauss();
    result |= test_least_squares();
    result |= test_positive_guard();
    result |= test_limiter();
    result |= test_reconstruct_primitive();
    result |= test_reconstruct_positive();
    result |= test_apply_limiter();
    result |= test_linear_solver();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}


