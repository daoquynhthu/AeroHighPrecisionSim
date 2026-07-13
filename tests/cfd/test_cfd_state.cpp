#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/real_fwd.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

using namespace aerosp;
using namespace aerosp::aero::cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static int test_is_valid_primitive() {
    TEST("CFD-STATE-1 valid primitive returns true");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.v = 0.0f; w.w = 0.0f; w.p = 1.0f; w.nu_tilde = 0.0f;
        if (!is_valid_primitive(w)) FAIL("valid state rejected");
        PASS;
    }

    TEST("CFD-STATE-2 rho=0 edge returns false");
    {
        PrimitiveState w;
        w.rho = 0.0f; w.u = 0.0f; w.p = 1.0f;
        if (is_valid_primitive(w)) FAIL("rho=0 accepted");
        PASS;
    }

    TEST("CFD-STATE-3 rho<0 returns false");
    {
        PrimitiveState w;
        w.rho = -1.0f; w.u = 0.0f; w.p = 1.0f;
        if (is_valid_primitive(w)) FAIL("rho<0 accepted");
        PASS;
    }

    TEST("CFD-STATE-4 p=0 returns false");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = 0.0f;
        if (is_valid_primitive(w)) FAIL("p=0 accepted");
        PASS;
    }

    TEST("CFD-STATE-5 p<0 returns false");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.p = -1.0f;
        if (is_valid_primitive(w)) FAIL("p<0 accepted");
        PASS;
    }

    TEST("CFD-STATE-6 NaN rho returns false");
    {
        PrimitiveState w;
        w.rho = std::numeric_limits<Real>::quiet_NaN();
        w.u = 0.0f; w.p = 1.0f;
        if (is_valid_primitive(w)) FAIL("NaN rho accepted");
        PASS;
    }

    TEST("CFD-STATE-7 NaN u returns false");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = std::numeric_limits<Real>::quiet_NaN(); w.p = 1.0f;
        if (is_valid_primitive(w)) FAIL("NaN u accepted");
        PASS;
    }

    TEST("CFD-STATE-8 Inf rho returns false");
    {
        PrimitiveState w;
        w.rho = std::numeric_limits<Real>::infinity();
        w.u = 0.0f; w.p = 1.0f;
        if (is_valid_primitive(w)) FAIL("Inf rho accepted");
        PASS;
    }

    TEST("CFD-STATE-9 mixed NaN/valid returns false");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.v = 0.0f; w.w = 0.0f;
        w.p = std::numeric_limits<Real>::quiet_NaN();
        w.nu_tilde = 0.0f;
        if (is_valid_primitive(w)) FAIL("NaN p among valid fields accepted");
        PASS;
    }
    return 0;
}

static int test_speed_of_sound() {
    TEST("CFD-STATE-10 speed_of_sound gamma=1.4");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.p = 1.0f;
        Real a = speed_of_sound(w, 1.4f);
        Real expected = std::sqrt(1.4f);
        if (std::fabs(a - expected) > 1e-6f) FAIL("a=%g expected=%g", a, expected);
        PASS;
    }

    TEST("CFD-STATE-11 speed_of_sound gamma=1.0 (isothermal)");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.p = 1.0f;
        Real a = speed_of_sound(w, 1.0f);
        Real expected = 1.0f;
        if (std::fabs(a - expected) > 1e-6f) FAIL("a=%g expected=%g", a, expected);
        PASS;
    }

    TEST("CFD-STATE-12 speed_of_sound gamma=1.66 (monatomic)");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.p = 2.0f;
        Real a = speed_of_sound(w, 1.66f);
        Real expected = std::sqrt(1.66f * 2.0f);
        if (std::fabs(a - expected) > 1e-6f) FAIL("a=%g expected=%g", a, expected);
        PASS;
    }

    TEST("CFD-STATE-13 speed_of_sound rho near zero returns finite");
    {
        PrimitiveState w;
        w.rho = 1e-30f; w.p = 1.0f;
        Real a = speed_of_sound(w, 1.4f);
        if (!std::isfinite(a)) FAIL("a not finite for rho near zero");
        PASS;
    }

    TEST("CFD-STATE-14 speed_of_sound p=0 returns 0");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.p = 0.0f;
        Real a = speed_of_sound(w, 1.4f);
        if (a != 0.0f) FAIL("a=%g expected 0", a);
        PASS;
    }

    TEST("CFD-STATE-15 speed_of_sound negative p returns NaN");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.p = -1.0f;
        Real a = speed_of_sound(w, 1.4f);
        if (!std::isfinite(a)) { PASS; } else FAIL("negative p produced finite a=%g", a);
    }

    TEST("CFD-STATE-16 speed_of_sound NaN p returns NaN");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.p = std::numeric_limits<Real>::quiet_NaN();
        Real a = speed_of_sound(w, 1.4f);
        if (std::isnan(a)) { PASS; } else FAIL("NaN p produced non-NaN a=%g", a);
    }

    TEST("CFD-STATE-17 speed_of_sound NaN rho returns NaN");
    {
        PrimitiveState w;
        w.rho = std::numeric_limits<Real>::quiet_NaN(); w.p = 1.0f;
        Real a = speed_of_sound(w, 1.4f);
        if (std::isnan(a)) { PASS; } else FAIL("NaN rho produced non-NaN a=%g", a);
    }
    return 0;
}

static int test_conservative_to_primitive() {
    TEST("CFD-STATE-18 conservative_to_primitive round-trip");
    {
        PrimitiveState w;
        w.rho = 1.2f; w.u = 0.5f; w.v = -0.3f; w.w = 0.1f; w.p = 1.4f; w.nu_tilde = 0.01f;
        ConservativeState q = primitive_to_conservative(w, 1.4f);
        PrimitiveState w2;
        if (!conservative_to_primitive(q, 1.4f, w2)) FAIL("round-trip returned false");
        if (std::fabs(w2.rho - w.rho) > 1e-6f) FAIL("rho: %g vs %g", w2.rho, w.rho);
        if (std::fabs(w2.u - w.u) > 1e-6f) FAIL("u: %g vs %g", w2.u, w.u);
        if (std::fabs(w2.v - w.v) > 1e-6f) FAIL("v: %g vs %g", w2.v, w.v);
        if (std::fabs(w2.w - w.w) > 1e-6f) FAIL("w: %g vs %g", w2.w, w.w);
        if (std::fabs(w2.p - w.p) > 1e-6f) FAIL("p: %g vs %g", w2.p, w.p);
        if (std::fabs(w2.nu_tilde - w.nu_tilde) > 1e-6f) FAIL("nu_tilde: %g vs %g", w2.nu_tilde, w.nu_tilde);
        PASS;
    }

    TEST("CFD-STATE-19 conservative_to_primitive NaN rho_E returns false");
    {
        ConservativeState q;
        q.rho = 1.0f; q.rho_u = 0.0f; q.rho_v = 0.0f; q.rho_w = 0.0f;
        q.rho_E = std::numeric_limits<Real>::quiet_NaN();
        q.rho_nu_tilde = 0.0f;
        PrimitiveState w;
        if (conservative_to_primitive(q, 1.4f, w)) FAIL("NaN rho_E accepted");
        PASS;
    }

    TEST("CFD-STATE-20 conservative_to_primitive negative rho_E returns false");
    {
        ConservativeState q;
        q.rho = 1.0f; q.rho_u = 0.0f; q.rho_v = 0.0f; q.rho_w = 0.0f;
        q.rho_E = -1.0f;
        q.rho_nu_tilde = 0.0f;
        PrimitiveState w;
        if (conservative_to_primitive(q, 1.4f, w)) FAIL("negative rho_E accepted");
        PASS;
    }
    return 0;
}

static int test_hllc_flux() {
    const Real nx = 1.0f, ny = 0.0f, nz = 0.0f;
    const Real gamma = 1.4f;

    TEST("CFD-STATE-21 hllc_flux symmetric states yields finite flux");
    {
        PrimitiveState w;
        w.rho = 1.0f; w.u = 0.0f; w.v = 0.0f; w.w = 0.0f; w.p = 1.0f; w.nu_tilde = 0.0f;
        EulerFlux f = hllc_flux(w, w, gamma, nx, ny, nz);
        if (!std::isfinite(f.mass)) FAIL("mass not finite");
        if (!std::isfinite(f.mom_x)) FAIL("mom_x not finite");
        if (!std::isfinite(f.mom_y)) FAIL("mom_y not finite");
        if (!std::isfinite(f.mom_z)) FAIL("mom_z not finite");
        if (!std::isfinite(f.energy)) FAIL("energy not finite");
        if (!std::isfinite(f.turbulence)) FAIL("turbulence not finite");
        PASS;
    }

    TEST("CFD-STATE-22 hllc_flux near-vacuum left is finite");
    {
        PrimitiveState low, high;
        low.rho = 1e-10f; low.u = 0.0f; low.p = 1e-10f; low.nu_tilde = 0.0f;
        high.rho = 1.0f; high.u = 100.0f; high.p = 1.0f; high.nu_tilde = 0.0f;
        EulerFlux f = hllc_flux(high, low, gamma, nx, ny, nz);
        if (!std::isfinite(f.mass)) FAIL("mass not finite");
        if (!std::isfinite(f.energy)) FAIL("energy not finite");
        PASS;
    }

    TEST("CFD-STATE-23 hllc_flux near-sonic state is finite");
    {
        PrimitiveState w1, w2;
        w1.rho = 1.0f; w1.u = -0.1f; w1.p = 1.0f / gamma; w1.nu_tilde = 0.0f;
        w2.rho = 1.0f; w2.u = 0.1f; w2.p = 1.0f / gamma; w2.nu_tilde = 0.0f;
        EulerFlux f = hllc_flux(w1, w2, gamma, nx, ny, nz);
        if (!std::isfinite(f.mass)) FAIL("mass not finite at near-sonic state");
        if (!std::isfinite(f.energy)) FAIL("energy not finite at near-sonic state");
        PASS;
    }

    TEST("CFD-STATE-24 hllc_flux strong shock M=10 is finite");
    {
        PrimitiveState w1, w2;
        w1.rho = 1.0f; w1.u = 10.0f; w1.p = 1.0f; w1.nu_tilde = 0.0f;
        w2.rho = 1.0f; w2.u = 0.0f; w2.p = 1.0f; w2.nu_tilde = 0.0f;
        EulerFlux f = hllc_flux(w1, w2, gamma, nx, ny, nz);
        if (!std::isfinite(f.mass)) FAIL("mass not finite");
        if (!std::isfinite(f.mom_x)) FAIL("mom_x not finite");
        if (!std::isfinite(f.energy)) FAIL("energy not finite");
        if (f.mass <= 0.0f) FAIL("mass=%g should be positive for M=10 shock", f.mass);
        PASS;
    }

    TEST("CFD-STATE-25 hllc_flux subsonic is finite");
    {
        PrimitiveState w1, w2;
        w1.rho = 1.0f; w1.u = 0.3f; w1.p = 1.0f; w1.nu_tilde = 0.0f;
        w2.rho = 1.0f; w2.u = 0.1f; w2.p = 1.0f; w2.nu_tilde = 0.0f;
        EulerFlux f = hllc_flux(w1, w2, gamma, nx, ny, nz);
        if (!std::isfinite(f.mass)) FAIL("mass not finite");
        if (!std::isfinite(f.energy)) FAIL("energy not finite");
        PASS;
    }

    TEST("CFD-STATE-26 hllc_flux both supersonic returns physical flux");
    {
        // Both states moving right at M > 1 → all waves travel right → s_l >= 0
        PrimitiveState w;
        w.rho = 1.0f; w.u = 5.0f; w.v = 0.0f; w.w = 0.0f; w.p = 1.0f; w.nu_tilde = 0.0f;
        EulerFlux f = hllc_flux(w, w, gamma, nx, ny, nz);
        EulerFlux f_exact = physical_flux(w, gamma, nx, ny, nz);
        if (std::fabs(f.mass - f_exact.mass) > 1e-6f) FAIL("mass flux mismatch: %g vs %g", f.mass, f_exact.mass);
        if (std::fabs(f.mom_x - f_exact.mom_x) > 1e-6f) FAIL("mom_x flux mismatch: %g vs %g", f.mom_x, f_exact.mom_x);
        PASS;
    }
    return 0;
}

static int test_make_freestream() {
    TEST("CFD-STATE-27 make_freestream Mach=0 has zero velocity");
    {
        PrimitiveState w = make_freestream(0.0f, 0.0f, 0.0f, 1.4f);
        if (std::fabs(w.u) > 1e-6f) FAIL("u=%g", w.u);
        if (std::fabs(w.v) > 1e-6f) FAIL("v=%g", w.v);
        if (std::fabs(w.w) > 1e-6f) FAIL("w=%g", w.w);
        if (std::fabs(w.rho - 1.0f) > 1e-6f) FAIL("rho=%g", w.rho);
        if (std::fabs(w.p - 1.0f / 1.4f) > 1e-6f) FAIL("p=%g", w.p);
        PASS;
    }

    TEST("CFD-STATE-28 make_freestream Mach=2 alpha=0 velocity in -x");
    {
        PrimitiveState w = make_freestream(2.0f, 0.0f, 0.0f, 1.4f);
        if (std::fabs(w.u + 2.0f) > 1e-6f) FAIL("u=%g expected -2", w.u);
        if (std::fabs(w.v) > 1e-6f) FAIL("v=%g", w.v);
        if (std::fabs(w.w) > 1e-6f) FAIL("w=%g", w.w);
        PASS;
    }

    TEST("CFD-STATE-29 make_freestream alpha=45 decomposes u/w correctly");
    {
        Real mach = 2.0f;
        Real alpha = 45.0f;
        PrimitiveState w = make_freestream(mach, alpha, 0.0f, 1.4f);
        Real speed = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
        if (std::fabs(speed - mach) > 1e-5f) FAIL("speed=%g expected %g", speed, mach);
        if (std::fabs(w.u - (-mach * std::cos(alpha * 3.14159265358979323846f / 180.0f))) > 1e-6f) FAIL("u=%g", w.u);
        if (std::fabs(w.w - (-mach * std::sin(alpha * 3.14159265358979323846f / 180.0f))) > 1e-6f) FAIL("w=%g", w.w);
        PASS;
    }

    TEST("CFD-STATE-30 make_freestream beta=30 puts velocity in y");
    {
        Real mach = 1.5f;
        Real beta = 30.0f;
        PrimitiveState w = make_freestream(mach, 0.0f, beta, 1.4f);
        Real speed = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
        if (std::fabs(speed - mach) > 1e-5f) FAIL("speed=%g expected %g", speed, mach);
        Real expected_v = -mach * std::sin(beta * 3.14159265358979323846f / 180.0f);
        if (std::fabs(w.v - expected_v) > 1e-6f) FAIL("v=%g expected %g", w.v, expected_v);
        PASS;
    }

    TEST("CFD-STATE-31 make_freestream all finite for extreme alpha");
    {
        PrimitiveState w = make_freestream(10.0f, 89.999f, 89.999f, 1.4f);
        if (!std::isfinite(w.u)) FAIL("u not finite");
        if (!std::isfinite(w.v)) FAIL("v not finite");
        if (!std::isfinite(w.w)) FAIL("w not finite");
        Real speed = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
        if (std::fabs(speed - 10.0f) > 1e-5f) FAIL("speed=%g", speed);
        PASS;
    }

    TEST("CFD-STATE-35 make_freestream Mach>30 hypersonic is finite");
    {
        PrimitiveState w = make_freestream(40.0f, 10.0f, 5.0f, 1.4f);
        if (!std::isfinite(w.rho)) FAIL("rho not finite");
        if (!std::isfinite(w.u)) FAIL("u not finite");
        if (!std::isfinite(w.v)) FAIL("v not finite");
        if (!std::isfinite(w.w)) FAIL("w not finite");
        if (!std::isfinite(w.p)) FAIL("p not finite");
        Real speed = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
        if (std::fabs(speed - 40.0f) > 1e-5f) FAIL("speed=%g", speed);
        PASS;
    }

    TEST("CFD-STATE-36 make_freestream very low Mach is finite");
    {
        PrimitiveState w = make_freestream(0.005f, 0.0f, 0.0f, 1.4f);
        if (!std::isfinite(w.rho)) FAIL("rho not finite");
        if (!std::isfinite(w.u)) FAIL("u not finite");
        if (!std::isfinite(w.v)) FAIL("v not finite");
        if (!std::isfinite(w.w)) FAIL("w not finite");
        if (!std::isfinite(w.p)) FAIL("p not finite");
        Real speed = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
        if (std::fabs(speed - 0.005f) > 1e-7f) FAIL("speed=%g expected 0.005", speed);
        PASS;
    }
    return 0;
}

static int test_farfield_ghost() {
    TEST("CFD-STATE-32 farfield_ghost_state supersonic inflow returns left");
    {
        PrimitiveState left;
        left.rho = 1.0f; left.u = 0.5f; left.p = 1.0f;
        PrimitiveState freestream;
        freestream.rho = 2.0f; freestream.u = -5.0f; freestream.p = 2.0f;
        Real gamma = 1.4f;
        PrimitiveState ghost = farfield_ghost_state(left, freestream, gamma, -1.0f, 0.0f, 0.0f);
        if (std::fabs(ghost.rho - left.rho) > 1e-6f) FAIL("rho=%g expected left=%g", ghost.rho, left.rho);
        PASS;
    }

    TEST("CFD-STATE-33 farfield_ghost_state subsonic returns freestream");
    {
        PrimitiveState left;
        left.rho = 1.0f; left.u = 0.5f; left.p = 1.0f;
        PrimitiveState freestream;
        freestream.rho = 2.0f; freestream.u = -0.3f; freestream.p = 2.0f;
        PrimitiveState ghost = farfield_ghost_state(left, freestream, 1.4f, -1.0f, 0.0f, 0.0f);
        if (std::fabs(ghost.rho - freestream.rho) > 1e-6f) FAIL("rho=%g expected freestream=%g", ghost.rho, freestream.rho);
        PASS;
    }

    TEST("CFD-STATE-34 farfield_ghost_state output always finite");
    {
        PrimitiveState left;
        left.rho = 1.0f; left.u = 0.5f; left.p = 1.0f;
        PrimitiveState freestream;
        freestream.rho = 2.0f; freestream.u = -0.3f; freestream.p = 2.0f;
        PrimitiveState ghost = farfield_ghost_state(left, freestream, 1.4f, 1.0f, 0.0f, 0.0f);
        if (!std::isfinite(ghost.rho)) FAIL("rho not finite");
        if (!std::isfinite(ghost.u)) FAIL("u not finite");
        if (!std::isfinite(ghost.p)) FAIL("p not finite");
        PASS;
    }
    return 0;
}

int main() {
    int result = 0;
    result |= test_is_valid_primitive();
    result |= test_speed_of_sound();
    result |= test_conservative_to_primitive();
    result |= test_hllc_flux();
    result |= test_make_freestream();
    result |= test_farfield_ghost();
    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return result == 0 && pass_count == test_count ? 0 : 1;
}
