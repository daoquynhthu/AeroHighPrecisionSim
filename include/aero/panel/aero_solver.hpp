#pragma once

#include <vector>
#include <string>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include "infra/common.hpp"
static inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
#endif

namespace aerosp {
namespace aero {
namespace panel {

    struct Triangle {
        float3 v0, v1, v2;
        float3 center;       // Centroid for moment calculation
        float3 normal;       // Unit outward normal
        float  area;         // Triangle area
        float  body_axis_x;  // Axial coordinate from nose tip (for running length)
    };

    struct AeroCoefficients {
        float CX, CY, CZ; // Body frame force coefficients
        float CL, CD;     // Wind frame lift/drag coefficients
        float Cl, Cm, Cn; // Moment coefficients (Roll, Pitch, Yaw)
        float L_D;        // Lift-to-Drag ratio
    };

    struct BatchCondition {
        float mach;
        float alpha_deg;
        float beta_deg;
        float com_x, com_y, com_z;
        float T_ref;    // Freestream temperature (K)
        float rho_ref;  // Freestream density (kg/m³)
        float mu_ref;   // Freestream viscosity (Pa·s)
    };

    struct BatchResult {
        float CX, CY, CZ;
        float Cl, Cm, Cn;
        float CL, CD;
    };

    struct AeroGeometry {
        float ref_area;
        float ref_length;
        float ref_span;
        float wet_area;
        float planform_area;
        float base_area;
        float nose_fineness;
    };

    class AeroSolver {
    public:
        AeroSolver();
        ~AeroSolver();

        bool load_model(const std::string& stl_path, float ref_area = 1.0f, float ref_length = 1.0f, float ref_span = 1.0f);
        bool load_mesh(const std::vector<Triangle>& mesh, float ref_area = 1.0f, float ref_length = 1.0f, float ref_span = 1.0f);

        AeroCoefficients compute_coefficients(float mach, float alpha_deg, float beta_deg = 0.0f);

        // Batch compute all conditions in a single GPU pass.
        // For each condition: if mach >= 5, uses GPU Newtonian panel method;
        // if mach < 5, uses device-side engineering estimate.
        // Returns num_conditions results.
        std::vector<BatchResult> compute_batch(
            const std::vector<BatchCondition>& conditions,
            const AeroGeometry& eng_geo);

        void set_moment_ref_point(float x, float y, float z) {
            moment_ref_point = make_float3(x, y, z);
        }

        void set_gamma(float g) { gamma = g; }
        void set_base_area(float a) { base_area = a; }

    private:
        Triangle* d_triangles = nullptr;
        float3* d_forces = nullptr;
        float3* d_moments = nullptr;
        int num_triangles = 0;
        
        float ref_area = 1.0f;
        float ref_length = 1.0f;
        float ref_span = 1.0f;
        float base_area = 0.0f;
        float3 moment_ref_point = {0.0f, 0.0f, 0.0f};
        float gamma = 1.4f;

        std::vector<Triangle> parse_stl(const std::string& path);
    };

    // ─── High-level API ───────────────────────────────────────────────

    struct AeroTableConfig {
        float ref_area = 1.131f;
        float ref_length = 12.0f;
        float ref_span = 3.0f;
        float com_x = 6.0f, com_y = 0.0f, com_z = 0.0f;
        float wet_area = 40.0f;
        float planform_area = 3.0f;
        float base_area = 0.1f;
        float nose_fineness = 3.0f;

        // When true, uses GPU CFD solver for conditions in
        // Mach [1.2, 30], |alpha| <= 30, |beta| <= 10.
        // NOTE: CFD uses a structured cube mesh embeding a unit-cube slip-wall
        // body, NOT the loaded STL geometry. Forces correspond to the cube body,
        // not the actual vehicle shape. See comment in generate_aero_table.
        bool   use_fvm = false;
        int    mesh_subdivisions = 5000;
        float  mesh_outer_scale = 10.0f;

        // Viscous NS parameters (only used when use_fvm=true).
        bool   viscous = false;
        float  Re = 1e6f;
        float  prandtl = 0.72f;
        float  wall_temperature = 300.0f;
    };

    // Single-GPU-pass generation of complete aerodynamics CSV table.
    // Uses Newtonian panel (Mach >= 5) + engineering estimate (Mach < 5)
    // with smooth blending in Mach 4-6 transition.
    //
    // When cfg.use_fvm=true, replaces results with GPU CFD solver for
    // in-range conditions (Mach [1.2,30], |alpha|<=30, |beta|<=10).
    //
    // LIMITATION: The CFD solver runs on a structured cube mesh with an
    // embedded unit-cube slip-wall body, NOT the loaded STL geometry.
    // The resulting force coefficients correspond to the cube body and
    // are NOT representative of the actual vehicle shape. This is a
    // pipeline demonstration — a conformal volume mesh generator is
    // required for production use.
    bool generate_aero_table(
        const std::string& stl_path,
        const std::string& csv_path,
        const std::vector<double>& mach_grid,
        const std::vector<double>& alpha_grid,
        const std::vector<double>& beta_grid,
        const AeroTableConfig& cfg);

}
}
}
