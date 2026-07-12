#include "aero/cfd/rans.hpp"
#include "aero/cfd/cfd_mesh.hpp"
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

static int test_sa_vorticity() {
    TEST("CFD-RANS-1 sa_vorticity zero for zero gradient");
    {
        PrimitiveGradient g;
        Real v = sa_vorticity(g);
        if (v != 0.0f) FAIL("vorticity=%g", v);
        PASS;
    }

    TEST("CFD-RANS-2 sa_vorticity known value for linear shear");
    {
        PrimitiveGradient g;
        g.du_dz = 3.0f;
        g.dw_dx = 1.0f;
        Real v = sa_vorticity(g);
        // vort_y = du_dz - dw_dx = 3 - 1 = 2, others zero
        Real expected = 2.0f;
        if (std::fabs(v - expected) > 1e-6f) FAIL("vorticity=%g expected=%g", v, expected);
        PASS;
    }

    TEST("CFD-RANS-3 sa_vorticity all three components");
    {
        PrimitiveGradient g;
        g.dw_dy = 2.0f;
        g.dv_dz = 1.0f;
        g.du_dz = 3.0f;
        g.dw_dx = 0.5f;
        g.dv_dx = 4.0f;
        g.du_dy = 1.5f;
        Real v = sa_vorticity(g);
        Real vx = 2.0f - 1.0f;
        Real vy = 3.0f - 0.5f;
        Real vz = 4.0f - 1.5f;
        Real expected = std::sqrt(vx*vx + vy*vy + vz*vz);
        if (std::fabs(v - expected) > 1e-6f) FAIL("vorticity=%g expected=%g", v, expected);
        PASS;
    }
    return 0;
}

static int test_rans_source_positive_chi() {
    TEST("CFD-RANS-4 rans_source positive chi produces finite production/destruction");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 100.0f; w.p = 1.0f; w.nu_tilde = 0.1f;
        PrimitiveGradient grad;
        grad.du_dz = 50.0f;
        Real wall_distance = 0.01f;
        Real mu = 1.0f;
        Real rho = 1.0f;
        Real Re = 1e6f;
        RansSource s = compute_rans_source(w, grad, wall_distance, mu, rho, Re);
        if (!std::isfinite(s.total_source)) FAIL("total_source not finite");
        if (!std::isfinite(s.production)) FAIL("production not finite");
        if (!std::isfinite(s.destruction)) FAIL("destruction not finite");
        if (!std::isfinite(s.diffusion)) FAIL("diffusion not finite");
        if (s.production < 0.0f) FAIL("negative production=%g", s.production);
        PASS;
    }

    TEST("CFD-RANS-5 rans_source positive chi production > destruction far from wall");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 100.0f; w.p = 1.0f; w.nu_tilde = 0.1f;
        PrimitiveGradient grad;
        grad.du_dz = 5000.0f;
        // Large wall distance so destruction (proportional to 1/d^2) is small
        RansSource s = compute_rans_source(w, grad, 1.0f, 1.0f, 1.0f, 1e6f);
        if (!std::isfinite(s.total_source)) FAIL("total_source not finite");
        if (s.production <= s.destruction) FAIL("production=%g should exceed destruction=%g far from wall", s.production, s.destruction);
        PASS;
    }

    TEST("CFD-RANS-6 rans_source positive chi destruction > production at low vorticity");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = 1.0f; w.nu_tilde = 0.1f;
        PrimitiveGradient grad;
        Real wall_distance = 0.001f;
        RansSource s = compute_rans_source(w, grad, wall_distance, 1.0f, 1.0f, 1e6f);
        if (!std::isfinite(s.total_source)) FAIL("total_source not finite");
        PASS;
    }

    TEST("CFD-RANS-7 rans_source zero wall_distance clamped safely");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = 1.0f; w.nu_tilde = 0.1f;
        PrimitiveGradient grad;
        RansSource s = compute_rans_source(w, grad, 0.0f, 1.0f, 1.0f, 1e6f);
        if (!std::isfinite(s.total_source)) FAIL("total_source not finite with zero wall dist");
        PASS;
    }

    TEST("CFD-RANS-8 rans_source negative wall_distance clamped safely");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = 1.0f; w.nu_tilde = 0.1f;
        PrimitiveGradient grad;
        RansSource s = compute_rans_source(w, grad, -1.0f, 1.0f, 1.0f, 1e6f);
        if (!std::isfinite(s.total_source)) FAIL("total_source not finite with negative wall dist");
        PASS;
    }

    TEST("CFD-RANS-9 rans_source NaN wall_distance clamped safely");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = 1.0f; w.nu_tilde = 0.1f;
        PrimitiveGradient grad;
        RansSource s = compute_rans_source(w, grad, std::numeric_limits<Real>::quiet_NaN(), 1.0f, 1.0f, 1e6f);
        if (!std::isfinite(s.total_source)) FAIL("total_source not finite with NaN wall dist");
        PASS;
    }

    TEST("CFD-RANS-10 rans_source zero nu_tilde produces zero production");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 100.0f; w.p = 1.0f; w.nu_tilde = 0.0f;
        PrimitiveGradient grad;
        grad.du_dz = 5000.0f;
        RansSource s = compute_rans_source(w, grad, 0.01f, 1.0f, 1.0f, 1e6f);
        if (!std::isfinite(s.total_source)) FAIL("total_source not finite");
        if (s.production != 0.0f) FAIL("production=%g expected 0 for nu_tilde=0", s.production);
        PASS;
    }
    return 0;
}

static int test_rans_source_negative_chi() {
    TEST("CFD-RANS-11 rans_source negative chi (ft2 branch) is finite");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = 1.0f; w.nu_tilde = -0.1f;
        PrimitiveGradient grad;
        grad.du_dz = 100.0f;
        RansSource s = compute_rans_source(w, grad, 0.01f, 1.0f, 1.0f, 1e6f);
        if (!std::isfinite(s.total_source)) FAIL("total_source=%g", s.total_source);
        if (!std::isfinite(s.production)) FAIL("production=%g", s.production);
        if (!std::isfinite(s.destruction)) FAIL("destruction=%g", s.destruction);
        if (!std::isfinite(s.diffusion)) FAIL("diffusion=%g", s.diffusion);
        PASS;
    }

    TEST("CFD-RANS-12 rans_source negative chi production ~ cb1*(1-ct3)*vort*nu_tilde (positive)");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = 1.0f; w.nu_tilde = -0.1f;
        PrimitiveGradient grad;
        grad.du_dz = 100.0f;
        RansSource s = compute_rans_source(w, grad, 0.01f, 1.0f, 1.0f, 1e6f);
        // (1-ct3) = -0.2 < 0, nu_tilde = -0.1 < 0, product cb1*(-0.2)*(-0.1) > 0
        if (s.production < 0.0f) FAIL("production=%g should be positive (fix PHYS-2)", s.production);
        PASS;
    }
    return 0;
}

static int test_rans_neg_destruction_sign() {
    TEST("CFD-RANS-13 rans_source negative chi destruction is positive (PHYS-2 fix)");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = 1.0f; w.nu_tilde = -0.1f;
        PrimitiveGradient grad;
        grad.du_dz = 100.0f;
        RansSource s = compute_rans_source(w, grad, 0.01f, 1.0f, 1.0f, 1e6f);
        // For nu_tilde < 0, (nu_tilde/d)^2 > 0, destruction = cw1*(nu_tilde/d)^2 > 0
        if (s.destruction < 0.0f) FAIL("destruction=%g should be positive (fix PHYS-2)", s.destruction);
        PASS;
    }

    TEST("CFD-RANS-14 rans_source negative chi total source less negative than production alone");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = 1.0f; w.nu_tilde = -0.1f;
        PrimitiveGradient grad;
        grad.du_dz = 100.0f;
        RansSource s = compute_rans_source(w, grad, 0.01f, 1.0f, 1.0f, 1e6f);
        // With fix: source = production + destruction + diffusion
        // production = cb1*(1-ct3)*vort*nu_tilde = 0.1355*(-0.2)*100*(-0.1) = +0.271
        // destruction = cw1*(nu_tilde/d)^2 = 3.239*(0.1/0.01)^2 = 3.239*100 = 323.9
        // source = +0.271 + 323.9 + diffusion > 0 (strongly positive, pushes nu_tilde up)
        if (s.total_source <= 0.0f) FAIL("total_source=%g should be positive post-fix (pushes nu_tilde toward zero)", s.total_source);
        if (s.destruction <= std::fabs(s.production))
            FAIL("destruction=%g should dominate production=%g for negative chi", s.destruction, s.production);
        PASS;
    }
    return 0;
}

static int test_rans_sources_mesh() {
    TEST("CFD-RANS-COV2-1 compute_rans_sources output size matches n_cells");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        auto freestream = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        auto q = std::vector<ConservativeState>(mesh.cells.size());
        for (auto& c : q) c = primitive_to_conservative(freestream, 1.4f);
        auto gradients = std::vector<PrimitiveGradient>(mesh.cells.size());

        auto sources = compute_rans_sources(mesh, q, gradients, 1.4f, 1e6f);
        if (sources.size() != mesh.cells.size())
            FAIL("sources.size=%zu cells=%zu", sources.size(), mesh.cells.size());
        PASS;
    }

    TEST("CFD-RANS-COV2-2 uniform state gives finite source terms");
    {
        auto mesh = generate_structured_cube_mesh(5.0f, 9);
        compute_mesh_metrics(mesh);
        auto freestream = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        auto q = std::vector<ConservativeState>(mesh.cells.size());
        for (auto& c : q) c = primitive_to_conservative(freestream, 1.4f);
        auto gradients = std::vector<PrimitiveGradient>(mesh.cells.size());

        auto sources = compute_rans_sources(mesh, q, gradients, 1.4f, 1e6f);
        for (std::size_t i = 0; i < sources.size(); ++i) {
            if (!std::isfinite(sources[i].total_source))
                FAIL("cell %zu total_source=%g not finite", i, sources[i].total_source);
            if (!std::isfinite(sources[i].production))
                FAIL("cell %zu production=%g not finite", i, sources[i].production);
        }
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_sa_vorticity();
    result |= test_rans_source_positive_chi();
    result |= test_rans_source_negative_chi();
    result |= test_rans_neg_destruction_sign();
    result |= test_rans_sources_mesh();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
