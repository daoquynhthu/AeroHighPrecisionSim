#include "aero/panel/aero_solver.hpp"
#include "aero/cfd/cfd_solver.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/mesh_gen_stl.hpp"
#if defined(AEROSP_HAS_CUDA) && AEROSP_HAS_CUDA
#include "aero/cfd/cfd_solver_gpu.hpp"
#endif

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <string>

namespace aerosp {
namespace aero {
namespace panel {

bool generate_aero_table(
    const std::string& stl_path,
    const std::string& csv_path,
    const std::vector<double>& mach_grid,
    const std::vector<double>& alpha_grid,
    const std::vector<double>& beta_grid,
    const AeroTableConfig& cfg)
{
    if (mach_grid.empty() || alpha_grid.empty() || beta_grid.empty()) {
        std::cerr << "[aero_table_gen] Input grids must be non-empty (got mach="
                  << mach_grid.size() << " alpha=" << alpha_grid.size()
                  << " beta=" << beta_grid.size() << ")\n";
        return false;
    }

    AeroGeometry eng_geo;
    eng_geo.ref_area = cfg.ref_area;
    eng_geo.ref_length = cfg.ref_length;
    eng_geo.ref_span = cfg.ref_span;
    eng_geo.wet_area = cfg.wet_area;
    eng_geo.planform_area = cfg.planform_area;
    eng_geo.base_area = cfg.base_area;
    eng_geo.nose_fineness = cfg.nose_fineness;

    std::vector<BatchCondition> conditions;
    for (double m : mach_grid) {
        for (double a : alpha_grid) {
            for (double b : beta_grid) {
                conditions.push_back({
                    static_cast<float>(m),
                    static_cast<float>(a),
                    static_cast<float>(b),
                    cfg.com_x, cfg.com_y, cfg.com_z,
                    288.15f,
                    0.0f,
                    0.0f
                });
            }
        }
    }

    std::cout << "[aero_table_gen] Computing " << conditions.size()
              << " conditions (" << mach_grid.size() << " Mach x "
              << alpha_grid.size() << " Alpha x " << beta_grid.size()
              << " Beta)...\n";

    bool use_cfd = cfg.use_fvm;
    std::vector<aerosp::aero::cfd::CfdForceResult> cfd_results;
    std::vector<BatchResult> newtonian_results;

    if (use_cfd) {
        const double MACH_MIN  = 1.2;
        const double MACH_MAX  = 30.0;
        const double ALPHA_LIM = 30.0;
        const double BETA_LIM  = 10.0;

        for (auto& c : conditions) {
            if (c.mach < MACH_MIN || c.mach > MACH_MAX ||
                std::abs(c.alpha_deg) > ALPHA_LIM ||
                std::abs(c.beta_deg) > BETA_LIM) {
                std::cerr << "[aero_table_gen] CFD range: Mach [" << MACH_MIN << ", " << MACH_MAX
                          << "], |alpha| <= " << ALPHA_LIM << ", |beta| <= " << BETA_LIM
                          << ". Got Mach=" << c.mach << " alpha=" << c.alpha_deg
                          << " beta=" << c.beta_deg << "\n";
                return false;
            }
        }

        aerosp::aero::cfd::CfdMesh mesh;
        if (cfg.stl_volume_mesh) {
            aerosp::aero::cfd::StlMeshConfig stl_cfg;
            stl_cfg.outer_scale = static_cast<Real>(
                cfg.mesh_outer_scale > 0.0f ? cfg.mesh_outer_scale : 5.0f);
            stl_cfg.max_cells = cfg.stl_max_cells > 0 ? cfg.stl_max_cells : 5000000;

            if (cfg.stl_background_n_per_dim > 0) {
                stl_cfg.background_n_per_dim = cfg.stl_background_n_per_dim;
            } else {
                if (cfg.mesh_subdivisions < 0) {
                    std::cerr << "[aero_table_gen] Warning: mesh_subdivisions="
                              << cfg.mesh_subdivisions
                              << " is negative, using absolute value\n";
                }
                double effective_sub = std::max(
                    1.0, static_cast<double>(std::abs(cfg.mesh_subdivisions)));
                int n = std::max(8, static_cast<int>(
                    std::ceil(std::pow(effective_sub, 1.0 / 3.0))));
                stl_cfg.background_n_per_dim = std::min(n, 80);
            }

            std::cout << "[aero_table_gen] Generating watertight STL hex-cull mesh: path="
                      << stl_path << " outer_scale=" << stl_cfg.outer_scale
                      << " n_per_dim=" << stl_cfg.background_n_per_dim << "\n";

            std::string mesh_err;
            // Production path requires load_mesh 1e-4 closed-surface. Cut-cell
            // conformal meshes do not yet meet that gate; hex-cull is watertight.
            if (!aerosp::aero::cfd::generate_watertight_mesh_from_stl(
                    stl_path, mesh, stl_cfg, &mesh_err)) {
                std::cerr << "[aero_table_gen] Watertight STL mesh failed: "
                          << mesh_err << "\n";
                return false;
            }
        } else {
            if (cfg.mesh_subdivisions < 0) {
                std::cerr << "[aero_table_gen] Warning: mesh_subdivisions="
                          << cfg.mesh_subdivisions
                          << " is negative, using absolute value\n";
            }
            double effective_sub = std::max(
                1.0, static_cast<double>(std::abs(cfg.mesh_subdivisions)));
            int n = std::max(5, static_cast<int>(
                std::ceil(std::pow(effective_sub / 5.0, 1.0 / 3.0)) + 1.0));
            // Ensure at least ~2 hex cells across the unit-cube body on each
            // axis: delta = 2*outer/(n-1) <= 1  =>  n >= 2*outer + 1.
            double outer = cfg.mesh_outer_scale > 0.0f
                ? static_cast<double>(cfg.mesh_outer_scale) : 10.0;
            int n_body = static_cast<int>(std::ceil(2.0 * outer + 1.0));
            n = std::max(n, n_body);
            n = std::min(n, 100);

            std::cout << "[aero_table_gen] Generating cube mesh: outer_scale="
                      << cfg.mesh_outer_scale << " n_per_dim=" << n << "\n";
            mesh = aerosp::aero::cfd::generate_structured_cube_mesh(
                static_cast<Real>(cfg.mesh_outer_scale), n);
        }

        aerosp::aero::cfd::MeshQualityReport qr =
            aerosp::aero::cfd::compute_mesh_metrics(mesh);
        if (!qr.valid || qr.cells == 0) {
            std::cerr << "[aero_table_gen] Mesh invalid: " << qr.message
                      << " (cells=" << qr.cells << ")\n";
            return false;
        }

        std::cout << "[aero_table_gen] Mesh ready: cells=" << qr.cells
                  << " faces=" << qr.faces
                  << (cfg.stl_volume_mesh ? " (conformal STL)" : " (cube)")
                  << "\n";

        aerosp::aero::cfd::CfdConfig cfd_cfg;
        cfd_cfg.use_gpu = true;
        cfd_cfg.cfl = 0.25f;
        cfd_cfg.max_iter = 2000;
        cfd_cfg.reconstruction_order = 1;
        cfd_cfg.ref_area  = static_cast<Real>(cfg.ref_area);
        cfd_cfg.ref_length = static_cast<Real>(cfg.ref_length);
        cfd_cfg.ref_span  = static_cast<Real>(cfg.ref_span);
        cfd_cfg.viscous   = cfg.viscous;
        cfd_cfg.Re        = static_cast<Real>(cfg.Re);
        cfd_cfg.prandtl   = static_cast<Real>(cfg.prandtl);
        cfd_cfg.wall_temperature = static_cast<Real>(cfg.wall_temperature);

        // Validate closed surface / wall set before solving (same checks as CfdSolver::load_mesh).
        {
            aerosp::aero::cfd::CfdSolver probe;
            if (!probe.load_mesh(mesh)) {
                Real sx = 0, sy = 0, sz = 0, ta = 0;
                int n_wall = 0;
                for (const auto& face : mesh.faces) {
                    if (face.boundary == aerosp::aero::cfd::BoundaryKind::Interior) continue;
                    if (face.boundary == aerosp::aero::cfd::BoundaryKind::SlipWall ||
                        face.boundary == aerosp::aero::cfd::BoundaryKind::NoSlipWall)
                        ++n_wall;
                    sx += face.area * face.nx;
                    sy += face.area * face.ny;
                    sz += face.area * face.nz;
                    ta += face.area;
                }
                Real ce = std::sqrt(static_cast<double>(sx * sx + sy * sy + sz * sz));
                std::cerr << "[aero_table_gen] Failed to load mesh into CFD solver"
                          << " (cells=" << mesh.cells.size()
                          << " faces=" << mesh.faces.size()
                          << " wall_faces=" << n_wall
                          << " closure=" << ce
                          << " total_bnd_area=" << ta
                          << " rel=" << (ce / (static_cast<double>(ta) + 1e-30))
                          << ")\n";
                return false;
            }
        }

        // Mesh is cached once; freestream (Mach/alpha/beta) varies per condition.
        cfd_results.resize(conditions.size());
        for (size_t i = 0; i < conditions.size(); ++i) {
            auto& c = conditions[i];
            aerosp::aero::cfd::FreestreamCondition fc;
            fc.mach      = static_cast<Real>(c.mach);
            fc.alpha_deg = static_cast<Real>(c.alpha_deg);
            fc.beta_deg  = static_cast<Real>(c.beta_deg);

            std::cout << "[aero_table_gen] CFD solve [" << (i + 1) << "/"
                      << conditions.size() << "] Mach=" << fc.mach
                      << " alpha=" << fc.alpha_deg
                      << " beta=" << fc.beta_deg << "\n";

            aerosp::aero::cfd::CfdSolveSummary summary;
            bool used_gpu = false;
#if defined(AEROSP_HAS_CUDA) && AEROSP_HAS_CUDA
            if (cfd_cfg.use_gpu) {
                std::string gpu_err;
                summary = aerosp::aero::cfd::solve_gpu_dispatch(
                    mesh, fc, cfd_cfg, &gpu_err);
                if (!summary.failed) {
                    used_gpu = true;
                } else {
                    std::cerr << "[aero_table_gen] GPU CFD failed ("
                              << (gpu_err.empty() ? "unknown" : gpu_err)
                              << "), falling back to CPU oracle path\n";
                }
            }
#endif
            if (!used_gpu) {
                aerosp::aero::cfd::CfdConfig cpu_cfg = cfd_cfg;
                cpu_cfg.use_gpu = false;
                aerosp::aero::cfd::CfdSolver cfd_solver;
                if (!cfd_solver.load_mesh(mesh)) {
                    std::cerr << "[aero_table_gen] CPU load_mesh failed\n";
                    return false;
                }
                summary = cfd_solver.solve(fc, cpu_cfg);
            }

            if (summary.failed) {
                std::cerr << "[aero_table_gen] CFD failed Mach=" << fc.mach
                          << " alpha=" << fc.alpha_deg
                          << " beta=" << fc.beta_deg << "\n";
                return false;
            }

            summary.forces.fidelity = used_gpu ? "cfd-gpu" : "cfd-cpu";
            cfd_results[i] = summary.forces;
        }
    } else {
        AeroSolver solver;
        if (!solver.load_model(stl_path, cfg.ref_area, cfg.ref_length, cfg.ref_span)) {
            std::cerr << "[aero_table_gen] Failed to load STL: " << stl_path << "\n";
            return false;
        }
        solver.set_moment_ref_point(cfg.com_x, cfg.com_y, cfg.com_z);

        newtonian_results = solver.compute_batch(conditions, eng_geo);
    }

    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "[aero_table_gen] Failed to write: " << csv_path << "\n";
        return false;
    }

    if (use_cfd) {
        csv << "Mach,Alpha,Beta,CX,CY,CZ,CL,CD,L_D,Cl,Cm,Cn,Fidelity,TurbulenceModel\n";
        for (size_t i = 0; i < conditions.size(); ++i) {
            auto& c = conditions[i];
            auto& r = cfd_results[i];
            double ld = (std::abs(r.CD) > Real(1e-12)) ? static_cast<double>(r.CL / r.CD) : 0.0;
            csv << c.mach << "," << c.alpha_deg << "," << c.beta_deg << ","
                << r.CX << "," << r.CY << "," << r.CZ << ","
                << r.CL << "," << r.CD << "," << ld << ","
                << r.Cl << "," << r.Cm << "," << r.Cn << ","
                << r.fidelity << ","
                << r.turbulence_model << "\n";
        }
    } else {
        csv << "Mach,Alpha,Beta,CX,CY,CZ,CL,CD,L_D,Cl,Cm,Cn\n";
        for (size_t i = 0; i < newtonian_results.size(); ++i) {
            auto& r = newtonian_results[i];
            auto& c = conditions[i];
            double ld = (std::abs(r.CD) > 1e-12f) ? static_cast<double>(r.CL / r.CD) : 0.0;
            csv << c.mach << "," << c.alpha_deg << "," << c.beta_deg << ","
                << r.CX << "," << r.CY << "," << r.CZ << ","
                << r.CL << "," << r.CD << "," << ld << ","
                << r.Cl << "," << r.Cm << "," << r.Cn << "\n";
        }
    }

    csv.close();
    std::cout << "[aero_table_gen] Done. Wrote " << conditions.size()
              << " rows to " << csv_path << "\n";
    return true;
}

} // namespace panel
} // namespace aero
} // namespace aerosp
