#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cstring>

#include "aero/panel/aero_solver.hpp"
#include "aero/engineering/engineering_aero.hpp"

struct Config {
    std::string stl_path;
    double mach = 0.0;
    double alpha_deg = 0.0;
    double beta_deg = 0.0;
    double com_x = 0.0;
    double com_y = 0.0;
    double com_z = 0.0;
    double ref_area = 1.0;
    double ref_length = 1.0;
    double ref_span = 1.0;
    double wet_area = -1.0;
    double planform_area = -1.0;
    double base_area = 0.0;
    double body_fineness = 10.0;
    double nose_fineness = 3.0;
    std::string mode = "auto";
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --stl <path>           STL file path (required for GPU mode)\n"
              << "  --mach <m>             Mach number\n"
              << "  --alpha <deg>          Angle of attack (degrees)\n"
              << "  --beta <deg>           Sideslip angle (degrees)\n"
              << "  --com-x <m>            CoM X offset from nose (moment ref)\n"
              << "  --com-y <m>            CoM Y\n"
              << "  --com-z <m>            CoM Z\n"
              << "  --ref-area <m2>        Reference area\n"
              << "  --ref-length <m>       Reference length\n"
              << "  --ref-span <m>         Reference span\n"
              << "  --wet-area <m2>        Wetted area (engineering mode)\n"
              << "  --planform-area <m2>   Planform area (engineering mode)\n"
              << "  --base-area <m2>       Base area (engineering mode)\n"
              << "  --nose-fineness <n>    Nose fineness ratio (engineering mode)\n"
              << "  --mode <auto|gpu|eng>  Computation mode (default: auto)\n"
              << "  --help                 Show this help\n";
}

int main(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") { print_usage(argv[0]); return 0; }
        if (i + 1 >= argc) { std::cerr << "Missing value for " << arg << "\n"; return 1; }
        auto next = [&]() { return argv[++i]; };
        if (arg == "--stl")            cfg.stl_path = next();
        else if (arg == "--mach")      cfg.mach = std::stod(next());
        else if (arg == "--alpha")     cfg.alpha_deg = std::stod(next());
        else if (arg == "--beta")      cfg.beta_deg = std::stod(next());
        else if (arg == "--com-x")     cfg.com_x = std::stod(next());
        else if (arg == "--com-y")     cfg.com_y = std::stod(next());
        else if (arg == "--com-z")     cfg.com_z = std::stod(next());
        else if (arg == "--ref-area")  cfg.ref_area = std::stod(next());
        else if (arg == "--ref-length") cfg.ref_length = std::stod(next());
        else if (arg == "--ref-span")  cfg.ref_span = std::stod(next());
        else if (arg == "--wet-area")  cfg.wet_area = std::stod(next());
        else if (arg == "--planform-area") cfg.planform_area = std::stod(next());
        else if (arg == "--base-area") cfg.base_area = std::stod(next());
        else if (arg == "--nose-fineness") cfg.nose_fineness = std::stod(next());
        else if (arg == "--mode")      cfg.mode = next();
    }

    bool use_gpu = false;
    bool use_eng = false;

    if (cfg.mode == "gpu") {
        use_gpu = true;
    } else if (cfg.mode == "eng") {
        use_eng = true;
    } else {
        use_gpu = (cfg.mach >= 5.0);
        use_eng = (cfg.mach < 5.0);
        if (cfg.mach >= 4.0 && cfg.mach < 6.0) {
            use_gpu = true;
            use_eng = true;
        }
    }

    double CX=0, CY=0, CZ=0, Cl=0, Cm=0, Cn=0, CL=0, CD=0;

    // --- GPU Newtonian Solver (Mach >= 5) ---
    if (use_gpu) {
        if (cfg.stl_path.empty()) {
            std::cerr << "GPU mode requires --stl path\n";
            return 1;
        }
        aerosp::aero::panel::AeroSolver solver;
        if (!solver.load_model(cfg.stl_path, static_cast<float>(cfg.ref_area),
                               static_cast<float>(cfg.ref_length),
                               static_cast<float>(cfg.ref_span))) {
            std::cerr << "Failed to load STL: " << cfg.stl_path << "\n";
            return 2;
        }
        solver.set_moment_ref_point(static_cast<float>(cfg.com_x),
                                     static_cast<float>(cfg.com_y),
                                     static_cast<float>(cfg.com_z));

        auto c = solver.compute_coefficients(static_cast<float>(cfg.mach),
                                              static_cast<float>(cfg.alpha_deg),
                                              static_cast<float>(cfg.beta_deg));
        CX = c.CX; CY = c.CY; CZ = c.CZ;
        Cl = c.Cl; Cm = c.Cm; Cn = c.Cn;
        CL = c.CL; CD = c.CD;
    }

    // --- Engineering Estimate (Mach < 5) ---
    if (use_eng) {
        aerosp::aero::panel::AeroGeometry geo;
        geo.ref_area = cfg.ref_area;
        geo.ref_length = cfg.ref_length;
        geo.ref_span = cfg.ref_span;
        geo.base_area = cfg.base_area;
        geo.nose_fineness = cfg.nose_fineness;
        if (cfg.wet_area > 0) geo.wet_area = cfg.wet_area;
        else geo.wet_area = cfg.ref_area * 4.0;

        if (cfg.planform_area > 0) geo.planform_area = cfg.planform_area;
        else geo.planform_area = cfg.ref_area;

        double alpha_rad = cfg.alpha_deg * 3.141592653589793 / 180.0;
        double beta_rad = cfg.beta_deg * 3.141592653589793 / 180.0;

        auto c = aerosp::aero::eng::compute_engineering_coeffs(geo, cfg.mach, alpha_rad, beta_rad);

        // Engineering Cm is computed via static_margin about CG; no additional offset needed.
        // GPU solver computes moments about moment_ref_point (= com), so both methods
        // produce CG-referenced moments. No cross-method offset adjustment required.

        // Blend with GPU result in transition region (Mach 4-6)
        if (use_gpu && use_eng) {
            double t = (cfg.mach - 4.0) / 2.0;
            t = std::max(0.0, std::min(1.0, t));
            auto blend = [t](double a, double b) { return (1.0 - t) * a + t * b; };
            CX = blend(c.CX, CX); CY = blend(c.CY, CY); CZ = blend(c.CZ, CZ);
            Cl = blend(c.Cl, Cl); Cm = blend(c.Cm, Cm); Cn = blend(c.Cn, Cn);
            CL = blend(c.CL, CL); CD = blend(c.CD, CD);
        } else {
            CX = c.CX; CY = c.CY; CZ = c.CZ;
            Cl = c.Cl; Cm = c.Cm; Cn = c.Cn;
            CL = c.CL; CD = c.CD;
        }
    }

    double L_D = (std::abs(CD) > 1e-12) ? CL / CD : 0.0;

    std::cout << "{"
              << "\"mach\": " << cfg.mach << ", "
              << "\"alpha\": " << cfg.alpha_deg << ", "
              << "\"beta\": " << cfg.beta_deg << ", "
              << "\"CX\": " << CX << ", "
              << "\"CY\": " << CY << ", "
              << "\"CZ\": " << CZ << ", "
              << "\"CL\": " << CL << ", "
              << "\"CD\": " << CD << ", "
              << "\"L_D\": " << L_D << ", "
              << "\"Cl\": " << Cl << ", "
              << "\"Cm\": " << Cm << ", "
              << "\"Cn\": " << Cn
              << "}" << std::endl;

    return 0;
}
