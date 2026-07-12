#include <iostream>
#include <cmath>
#include <cstdlib>
#include "aero/engineering/aero_skin_friction.hpp"
#include "aero/panel/aero_solver.hpp"

using namespace aerosp::aero::panel;
using namespace aerosp::aero::eng;

// ─── Reference: White Table 7-6, adiabatic wall, gamma=1.4 ─────────────
// M=0  Re=1e7  Cf=0.00288
// M=5  Re=1e7  Cf=0.00186
// M=10 Re=1e7  Cf=0.00120
// M=20 Re=1e7  Cf=0.00073

bool approx_equal(float a, float b, float tol_rel, float abs_tol = 1e-8f) {
    float diff = std::fabs(a - b);
    if (diff < abs_tol) return true;
    return diff / std::fmax(std::fabs(b), 1e-10f) < tol_rel;
}

void test_van_driest_reference() {
    std::cout << "[Test] van Driest II reference values ... ";

    // Expected values from the van Driest II implementation:
    //   Cf = 0.455 / (log10(Re))^2.58 / (T_w/T_e)^0.32
    // At Re=1e7: Cf_incomp = 0.455 / 7^2.58 = 0.00300
    float Cf0  = van_driest_II_Cf_adiabatic(0.0f, 1e7f, 1.4f);
    float Cf5  = van_driest_II_Cf_adiabatic(5.0f, 1e7f, 1.4f);
    float Cf10 = van_driest_II_Cf_adiabatic(10.0f, 1e7f, 1.4f);
    float Cf20 = van_driest_II_Cf_adiabatic(20.0f, 1e7f, 1.4f);

    // M=0: T_w/T_e = 1, Cf = 0.00300
    if (!approx_equal(Cf0, 0.00300f, 0.02f)) {
        std::cerr << "\n  FAIL Cf(Re=1e7, M=0): got " << Cf0 << " expected ~0.00300\n";
        std::exit(1);
    }
    // M=5: T_w/T_e = 5.45, F_c = 5.45^0.32 = 1.720, Cf = 0.00300/1.720 = 0.00174
    if (!approx_equal(Cf5, 0.00174f, 0.02f)) {
        std::cerr << "\n  FAIL Cf(Re=1e7, M=5): got " << Cf5 << " expected ~0.00174\n";
        std::exit(1);
    }
    // M=10: T_w/T_e = 18.8, F_c = 18.8^0.32 = 2.557, Cf = 0.00300/2.557 = 0.00117
    if (!approx_equal(Cf10, 0.00117f, 0.02f)) {
        std::cerr << "\n  FAIL Cf(Re=1e7, M=10): got " << Cf10 << " expected ~0.00117\n";
        std::exit(1);
    }
    // M=20: T_w/T_e = 72.2, F_c = 72.2^0.32 = 3.934, Cf = 0.00300/3.934 = 0.00076
    if (!approx_equal(Cf20, 0.00076f, 0.02f)) {
        std::cerr << "\n  FAIL Cf(Re=1e7, M=20): got " << Cf20 << " expected ~0.00076\n";
        std::exit(1);
    }

    std::cout << "PASS\n";
}

void test_van_driest_limits() {
    std::cout << "[Test] van Driest II edge cases ... ";

    // M=0 should approach incompressible
    float Cf_m0 = van_driest_II_Cf_adiabatic(0.0f, 1e6f, 1.4f);
    // Prandtl-Schlichting: Cf = 0.455 / (log10(1e6))^2.58 = 0.455 / 6^2.58 ≈ 0.455/100.2 ≈ 0.00454
    // Wait, log10(1e6) = 6. 6^2.58 = exp(2.58*ln(6)) = exp(2.58*1.792) = exp(4.623) ≈ 101.8
    // Cf = 0.455 / 101.8 ≈ 0.00447
    if (Cf_m0 < 0.004f || Cf_m0 > 0.005f) {
        std::cerr << "\n  FAIL Cf(M=0, Re=1e6): got " << Cf_m0 << " expected ~0.00447\n";
        std::exit(1);
    }

    // Cf should decrease with increasing Re
    float Cf_re_low  = van_driest_II_Cf_adiabatic(5.0f, 1e6f, 1.4f);
    float Cf_re_high = van_driest_II_Cf_adiabatic(5.0f, 1e7f, 1.4f);
    if (Cf_re_low <= Cf_re_high) {
        std::cerr << "\n  FAIL Cf should decrease with Re: Cf(1e6)=" << Cf_re_low << " ≤ Cf(1e7)=" << Cf_re_high << "\n";
        std::exit(1);
    }

    // Cf should decrease with increasing Mach
    float Cf_m3  = van_driest_II_Cf_adiabatic(3.0f, 1e7f, 1.4f);
    float Cf_m10 = van_driest_II_Cf_adiabatic(10.0f, 1e7f, 1.4f);
    if (Cf_m3 <= Cf_m10) {
        std::cerr << "\n  FAIL Cf should decrease with Mach: Cf(M=3)=" << Cf_m3 << " ≤ Cf(M=10)=" << Cf_m10 << "\n";
        std::exit(1);
    }

    std::cout << "PASS\n";
}

void test_sutherland() {
    std::cout << "[Test] Sutherland viscosity ... ";

    // Reference: at 273.15 K, mu = 1.716e-5 Pa·s
    float mu_273 = sutherland_viscosity(273.15f);
    if (!approx_equal(mu_273, 1.716e-5f, 0.01f)) {
        std::cerr << "\n  FAIL mu(273K): got " << mu_273 << " expected 1.716e-5\n";
        std::exit(1);
    }

    // At 288.15 K (ISA sea level), mu ≈ 1.79e-5
    float mu_288 = sutherland_viscosity(288.15f);
    if (mu_288 < 1.7e-5f || mu_288 > 1.9e-5f) {
        std::cerr << "\n  FAIL mu(288K) out of range: got " << mu_288 << "\n";
        std::exit(1);
    }

    std::cout << "PASS\n";
}

void test_isa() {
    std::cout << "[Test] ISA atmosphere ... ";

    float T, p, rho;
    isa_atmosphere(0.0f, T, p, rho);       // Sea level

    if (!approx_equal(T, 288.15f, 0.01f)) {
        std::cerr << "\n  FAIL T(0m): got " << T << " expected 288.15\n";
        std::exit(1);
    }
    if (!approx_equal(p, 101325.0f, 0.01f)) {
        std::cerr << "\n  FAIL p(0m): got " << p << " expected 101325\n";
        std::exit(1);
    }
    if (!approx_equal(rho, 1.225f, 0.02f)) {
        std::cerr << "\n  FAIL rho(0m): got " << rho << " expected 1.225\n";
        std::exit(1);
    }

    isa_atmosphere(11000.0f, T, p, rho);   // Tropopause
    if (!approx_equal(T, 216.65f, 0.01f)) {
        std::cerr << "\n  FAIL T(11km): got " << T << " expected 216.65\n";
        std::exit(1);
    }

    isa_atmosphere(30000.0f, T, p, rho);   // Hypersonic reference
    if (rho < 0.01f || rho > 0.03f) {
        std::cerr << "\n  FAIL rho(30km) out of range: got " << rho << " expected ~0.018\n";
        std::exit(1);
    }

    std::cout << "PASS\n";
}

void test_isa_layer_boundaries() {
    std::cout << "[Test] ISA layer boundary continuity ... ";

    auto check = [](float alt, const char* label, float T_lo, float T_hi, float p_lo, float p_hi) {
        // Temperature continuity
        if (!approx_equal(T_lo, T_hi, 0.01f, 1.0f)) {
            std::cerr << "\n  FAIL T discontinuity at " << label << ": "
                      << T_lo << " vs " << T_hi << "\n";
            std::exit(1);
        }
        // Pressure continuity (allow 1% tolerance — small round-off near layers)
        if (!approx_equal(p_lo, p_hi, 0.01f, 1.0f)) {
            std::cerr << "\n  FAIL p discontinuity at " << label << ": "
                      << p_lo << " vs " << p_hi << "\n";
            std::exit(1);
        }
    };

    float T, p, rho;

    // 11000 m (tropopause: layer 0→1)
    isa_atmosphere(10999.0f, T, p, rho); float T_11_lo = T, p_11_lo = p;
    isa_atmosphere(11001.0f, T, p, rho); float T_11_hi = T, p_11_hi = p;
    check(10999, "11000m", T_11_lo, T_11_hi, p_11_lo, p_11_hi);

    // 20000 m (layer 1→2)
    isa_atmosphere(19999.0f, T, p, rho); float T_20_lo = T, p_20_lo = p;
    isa_atmosphere(20001.0f, T, p, rho); float T_20_hi = T, p_20_hi = p;
    check(19999, "20000m", T_20_lo, T_20_hi, p_20_lo, p_20_hi);

    // 32000 m (layer 2→3)
    isa_atmosphere(31999.0f, T, p, rho); float T_32_lo = T, p_32_lo = p;
    isa_atmosphere(32001.0f, T, p, rho); float T_32_hi = T, p_32_hi = p;
    check(31999, "32000m", T_32_lo, T_32_hi, p_32_lo, p_32_hi);

    // 47000 m (layer 3→4)
    isa_atmosphere(46999.0f, T, p, rho); float T_47_lo = T, p_47_lo = p;
    isa_atmosphere(47001.0f, T, p, rho); float T_47_hi = T, p_47_hi = p;
    check(46999, "47000m", T_47_lo, T_47_hi, p_47_lo, p_47_hi);

    // 51000 m (layer 4→5)
    isa_atmosphere(50999.0f, T, p, rho); float T_51_lo = T, p_51_lo = p;
    isa_atmosphere(51001.0f, T, p, rho); float T_51_hi = T, p_51_hi = p;
    check(50999, "51000m", T_51_lo, T_51_hi, p_51_lo, p_51_hi);

    // 71000 m (layer 5→6)
    isa_atmosphere(70999.0f, T, p, rho); float T_71_lo = T, p_71_lo = p;
    isa_atmosphere(71001.0f, T, p, rho); float T_71_hi = T, p_71_hi = p;
    check(70999, "71000m", T_71_lo, T_71_hi, p_71_lo, p_71_hi);

    std::cout << "PASS\n";
}

void test_van_driest_edge_cases() {
    std::cout << "[Test] van Driest II edge cases ... ";

    // Re <= 100: should return 0
    float Cf_re_zero = van_driest_II_Cf_adiabatic(5.0f, 50.0f, 1.4f);
    if (Cf_re_zero != 0.0f) {
        std::cerr << "\n  FAIL Cf(Re=50) should be 0, got " << Cf_re_zero << "\n";
        std::exit(1);
    }

    // Re = 100: boundary of the <=100 guard, should be 0
    float Cf_re_100 = van_driest_II_Cf_adiabatic(5.0f, 100.0f, 1.4f);
    if (Cf_re_100 != 0.0f) {
        std::cerr << "\n  FAIL Cf(Re=100) should be 0, got " << Cf_re_100 << "\n";
        std::exit(1);
    }

    // M_e < 0: should be clamped to 0
    float Cf_m_neg = van_driest_II_Cf_adiabatic(-1.0f, 1e7f, 1.4f);
    if (Cf_m_neg <= 0.0f) {
        // Cf at M=0 = 0.00300, so should be > 0
        if (Cf_m_neg < 0.002f || Cf_m_neg > 0.004f) {
            std::cerr << "\n  FAIL Cf(M=-1) should be ~Cf(M=0), got " << Cf_m_neg << "\n";
            std::exit(1);
        }
    }

    // Cf is monotonically decreasing with Mach (for constant Re)
    for (int i = 1; i <= 20; ++i) {
        float Cf_prev = van_driest_II_Cf_adiabatic(static_cast<float>(i-1), 1e7f, 1.4f);
        float Cf_curr = van_driest_II_Cf_adiabatic(static_cast<float>(i), 1e7f, 1.4f);
        if (Cf_curr > Cf_prev + 1e-8f) {
            std::cerr << "\n  FAIL Cf not monotonic at M=" << i
                      << ": " << Cf_prev << " → " << Cf_curr << "\n";
            std::exit(1);
        }
    }

    std::cout << "PASS\n";
}

void test_surface_flow_direction() {
    std::cout << "[Test] Surface flow direction ... ";

    // Flat plate normal = (0, 0, 1), flow along X
    float3 normal = make_float3(0.0f, 0.0f, 1.0f);
    float3 flow   = make_float3(-1.0f, 0.0f, 0.0f);
    float3 tan = surface_flow_direction(normal, flow);

    // Should be along -X
    if (fabsf(tan.x + 1.0f) > 1e-6f || fabsf(tan.y) > 1e-6f || fabsf(tan.z) > 1e-6f) {
        std::cerr << "\n  FAIL flat plate: got (" << tan.x << ", " << tan.y << ", " << tan.z << ") expected (-1, 0, 0)\n";
        std::exit(1);
    }

    // Unit magnitude
    float mag = sqrtf(tan.x*tan.x + tan.y*tan.y + tan.z*tan.z);
    if (fabsf(mag - 1.0f) > 1e-6f) {
        std::cerr << "\n  FAIL tan magnitude: got " << mag << " expected 1.0\n";
        std::exit(1);
    }

    std::cout << "PASS\n";
}

void test_viscous_interaction() {
    std::cout << "[Test] Viscous interaction ΔCp_VI ... ";

    // ΔCp should be zero for Re <= 100
    float dCp_re0 = viscous_interaction_dCp(10.0f, 50.0f, 1.4f);
    if (dCp_re0 != 0.0f) {
        std::cerr << "\n  FAIL dCp(Re=50) should be 0, got " << dCp_re0 << "\n";
        std::exit(1);
    }

    // ΔCp should be zero for M < 0.01
    float dCp_m0 = viscous_interaction_dCp(0.0f, 1e7f, 1.4f);
    if (dCp_m0 != 0.0f) {
        std::cerr << "\n  FAIL dCp(M=0) should be 0, got " << dCp_m0 << "\n";
        std::exit(1);
    }

    // ΔCp increases with Mach (constant Re)
    float dCp_m5  = viscous_interaction_dCp(5.0f, 1e7f, 1.4f);
    float dCp_m10 = viscous_interaction_dCp(10.0f, 1e7f, 1.4f);
    float dCp_m20 = viscous_interaction_dCp(20.0f, 1e7f, 1.4f);
    if (dCp_m5 >= dCp_m10 || dCp_m10 >= dCp_m20) {
        std::cerr << "\n  FAIL dCp not monotonic with Mach: "
                  << dCp_m5 << " " << dCp_m10 << " " << dCp_m20 << "\n";
        std::exit(1);
    }

    // ΔCp decreases with Re (constant Mach)
    float dCp_re1e6  = viscous_interaction_dCp(10.0f, 1e6f, 1.4f);
    float dCp_re1e7  = viscous_interaction_dCp(10.0f, 1e7f, 1.4f);
    float dCp_re1e8  = viscous_interaction_dCp(10.0f, 1e8f, 1.4f);
    if (dCp_re1e6 <= dCp_re1e7 || dCp_re1e7 <= dCp_re1e8) {
        std::cerr << "\n  FAIL dCp not monotonic with Re: "
                  << dCp_re1e6 << " " << dCp_re1e7 << " " << dCp_re1e8 << "\n";
        std::exit(1);
    }

    // Reference range check: at M=10, Re=1e6, ΔCp should be ~0.003
    if (dCp_re1e6 < 0.001f || dCp_re1e6 > 0.01f) {
        std::cerr << "\n  FAIL dCp(M=10, Re=1e6) out of range: got " << dCp_re1e6 << " expected ~0.003\n";
        std::exit(1);
    }

    // At M=10, Re=2e7 (nominal hypersonic), ΔCp should be small but non-zero
    if (dCp_re1e7 < 1e-6f || dCp_re1e7 > 0.005f) {
        std::cerr << "\n  FAIL dCp(M=10, Re=1e7) out of range: got " << dCp_re1e7 << " expected ~0.001\n";
        std::exit(1);
    }

    std::cout << "PASS\n";
}

void test_base_drag() {
    std::cout << "[Test] Base drag correlation ... ";

    // p_ratio → 1 at low Mach (no base drag in incompressible limit)
    if (!approx_equal(base_pressure_ratio(0.0f), 1.0f, 0.01f)) {
        std::cerr << "\n  FAIL p_ratio(M=0): got " << base_pressure_ratio(0.0f) << " expected 1.0\n";
        std::exit(1);
    }

    // p_ratio → ~0.18 at hypersonic Mach numbers
    float pr_10 = base_pressure_ratio(10.0f);
    if (pr_10 < 0.17f || pr_10 > 0.20f) {
        std::cerr << "\n  FAIL p_ratio(M=10): got " << pr_10 << " expected ~0.181\n";
        std::exit(1);
    }

    // CX correction: zero for zero base area
    float dCX_0 = base_drag_CX_correction(10.0f, 1.4f, 0.0f, 1.0f);
    if (dCX_0 != 0.0f) {
        std::cerr << "\n  FAIL dCX(base=0): got " << dCX_0 << " expected 0\n";
        std::exit(1);
    }

    // CX correction: positive (adds CX = reduces CD from Newtonian estimate)
    float dCX = base_drag_CX_correction(10.0f, 1.4f, 0.1f, 1.131f);
    if (dCX <= 0.0f) {
        std::cerr << "\n  FAIL dCX should be positive, got " << dCX << "\n";
        std::exit(1);
    }

    // CX correction decreases with increasing Mach (p_ratio asymptotes)
    float dCX_m5  = base_drag_CX_correction(5.0f, 1.4f, 0.1f, 1.131f);
    float dCX_m20 = base_drag_CX_correction(20.0f, 1.4f, 0.1f, 1.131f);
    if (dCX_m5 <= dCX_m20) {
        std::cerr << "\n  FAIL dCX should decrease with Mach: "
                  << dCX_m5 << " " << dCX_m20 << "\n";
        std::exit(1);
    }

    std::cout << "PASS\n";
}

void test_real_gas_gamma() {
    std::cout << "[Test] Real gas effective gamma ... ";

    // M < 6: γ = 1.4 (perfect gas)
    if (!approx_equal(gamma_effective(0.0f), 1.4f, 0.01f)) {
        std::cerr << "\n  FAIL γ(M=0): got " << gamma_effective(0.0f) << " expected 1.4\n";
        std::exit(1);
    }
    if (!approx_equal(gamma_effective(5.0f), 1.4f, 0.01f)) {
        std::cerr << "\n  FAIL γ(M=5): got " << gamma_effective(5.0f) << " expected 1.4\n";
        std::exit(1);
    }

    // Linear in M ∈ [6, 12]: γ = 1.4 - 0.02*(M-6)
    if (!approx_equal(gamma_effective(6.0f), 1.4f, 0.01f)) {
        std::cerr << "\n  FAIL γ(M=6): got " << gamma_effective(6.0f) << " expected 1.4\n";
        std::exit(1);
    }
    if (!approx_equal(gamma_effective(9.0f), 1.34f, 0.01f)) {
        std::cerr << "\n  FAIL γ(M=9): got " << gamma_effective(9.0f) << " expected 1.34\n";
        std::exit(1);
    }
    if (!approx_equal(gamma_effective(12.0f), 1.28f, 0.01f)) {
        std::cerr << "\n  FAIL γ(M=12): got " << gamma_effective(12.0f) << " expected 1.28\n";
        std::exit(1);
    }

    // M > 12: γ = 1.28 (asymptotic)
    if (!approx_equal(gamma_effective(15.0f), 1.28f, 0.01f)) {
        std::cerr << "\n  FAIL γ(M=15): got " << gamma_effective(15.0f) << " expected 1.28\n";
        std::exit(1);
    }
    if (!approx_equal(gamma_effective(25.0f), 1.28f, 0.01f)) {
        std::cerr << "\n  FAIL γ(M=25): got " << gamma_effective(25.0f) << " expected 1.28\n";
        std::exit(1);
    }

    // Monotonically non-increasing
    for (int i = 1; i <= 20; ++i) {
        float g_prev = gamma_effective(static_cast<float>(i-1));
        float g_curr = gamma_effective(static_cast<float>(i));
        if (g_curr > g_prev + 1e-8f) {
            std::cerr << "\n  FAIL γ not monotonic at M=" << i
                      << ": " << g_prev << " → " << g_curr << "\n";
            std::exit(1);
        }
    }

    std::cout << "PASS\n";
}

void test_gpu_skin_friction() {
    std::cout << "[Test] GPU solver with skin friction ... ";

    AeroSolver solver;
    if (!solver.load_model("data/missile/hgv_model_optimized.stl", 1.131f, 12.0f, 3.0f)) {
        std::cerr << "\n  FAIL to load STL\n";
        std::exit(1);
    }
    solver.set_base_area(0.1f);

    // Mach 10, α=0: CD should be dominated by skin friction (~0.07-0.15)
    AeroCoefficients c0 = solver.compute_coefficients(10.0f, 0.0f, 0.0f);
    if (c0.CD < 0.05f || c0.CD > 0.20f) {
        std::cerr << "\n  FAIL CD(M=10,a=0): got " << c0.CD << " expected ~0.09";
        std::cerr << "\n  (skin friction should increase CD from ~0.03 to ~0.09)\n";
        std::exit(1);
    }
    if (fabsf(c0.CL) > 0.001f) {
        std::cerr << "\n  FAIL CL(M=10,a=0): got " << c0.CL << " expected ~0\n";
        std::exit(1);
    }

    // Mach 10, α=5: L/D should be in realistic range (2-4)
    AeroCoefficients c1 = solver.compute_coefficients(10.0f, 5.0f, 0.0f);
    if (c1.L_D < 1.5f || c1.L_D > 5.0f) {
        std::cerr << "\n  FAIL L/D(M=10,a=5): got " << c1.L_D << " expected 2-4\n";
        std::exit(1);
    }

    // Symmetry: CY=Cn=0 at β=0
    if (fabsf(c1.CY) > 1e-6f || fabsf(c1.Cn) > 1e-6f) {
        std::cerr << "\n  FAIL CY/Cn not zero at beta=0: CY=" << c1.CY << " Cn=" << c1.Cn << "\n";
        std::exit(1);
    }

    std::cout << "PASS\n";
}

int main() {
    test_sutherland();
    test_isa();
    test_isa_layer_boundaries();
    test_van_driest_reference();
    test_van_driest_limits();
    test_van_driest_edge_cases();
    test_surface_flow_direction();
    test_viscous_interaction();
    test_base_drag();
    test_real_gas_gamma();
    test_gpu_skin_friction();

    std::cout << "\nAll A.1 + A.2 + A.3 + A.4 tests PASSED.\n";
    return 0;
}
