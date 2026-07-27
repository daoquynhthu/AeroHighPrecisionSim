#pragma once

#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/diagnostics.hpp"
#include "aero/cfd/turbulence_model.hpp"

namespace aerosp {
namespace aero {
namespace cfd {

constexpr int CFD_NVAR = 6;

// Forward decl for MMS-compatible farfield BC.
struct MmsSolutionEulerBC;

struct CfdConfig {
    int nvar = CFD_NVAR;
    Real cfl = 0.5f;
    int max_iter = 1000;
    Real convergence_tol = 1e-8f;
    Real gamma = 1.4f;
    Real ref_area = 1.0f;
    Real ref_length = 1.0f;
    Real ref_span = 1.0f;
    int reconstruction_order = 1;
    bool use_gpu = true;
    bool cpu_oracle = false;
    DiagnosticLevel diagnostic_level = DiagnosticLevel::Off;

    // Viscous NS parameters
    bool viscous = false;
    Real Re = 1e6f;
    Real prandtl = 0.72f;
    Real mu_ref = 1.0f;
    Real T_ref = 288.15f;
    Real sutherland_T = 110.4f;
    Real wall_temperature = 300.0f;

    // RANS turbulence model (LAMINAR=0, SA=1, SA_DDES=2, SST=3)
    TurbulenceModel turbulence_model = TurbulenceModel::LAMINAR;

    // MMS source term (empty = disabled)
    std::vector<EulerFlux> mms_source;

    // MMS-compatible farfield BC (null = use characteristic BC for production)
    // When non-null, farfield boundaries impose q_exact at the face center.
    const MmsSolutionEulerBC* mms_solution = nullptr;

    // AMR (h-refinement)
    AmrConfig amr;

    // Thermochemistry (Phase 15+)
    int gas_model_kind = 0;         // 0=PerfectGas, 1=Frozen, 2=EquilibriumAir, 3=ChemNonEq
    std::string config_path;        // path to YAML config (empty = PerfectGas default)

    // Implicit solver
    bool implicit = false;
    Real cfl_start = 1.0f;
    Real cfl_end = 1e6f;
    int cfl_ramp_steps = 100;
    // When true (or implicit=true), CFL ramps cfl_start → cfl_end over cfl_ramp_steps.
    // Explicit steady RANS should set cfl_ramp=true with moderate cfl_end (e.g. 5–50).
    bool cfl_ramp = false;
    bool local_time_stepping = true;
    int fgmres_restart = 30;
    int fgmres_max_iter = 100;
    Real fgmres_tol = 1e-2f;
    int newton_max_iter = 3;
    Real newton_sufficient_decrease = 0.5f;

    // SA: re-linearize point-implicit source after residual assembly (0 = off).
    // Default 2 removes turb residual lag on coarse BL meshes without lowering CFL.
    int sa_sub_iters = 2;
    // Cap local/global dt ratio for LTS coupling stability (max_dt <= ratio * min_dt).
    Real lts_dt_ratio_max = 1.0e4f;
    // Point-implicit residual scaling for mean-flow eqs via local spectral radius.
    // When true, prefer larger CFL (cfl_ramp) because PI already damps by ~1/(1+CFL).
    // Default false: spectral CFL alone is the stability mechanism (avoids double damping).
    bool mean_flow_point_implicit = false;
};

struct FreestreamCondition {
    Real mach = 2.0f;
    Real alpha_deg = 0.0f;
    Real beta_deg = 0.0f;
    Real nu_tilde = 0.0f;
    Real nu_tilde_ratio = 0.1f;  // nu_tilde_inf / nu_inf, 0 = disabled
};

} // namespace cfd
} // namespace aero
} // namespace aerosp
