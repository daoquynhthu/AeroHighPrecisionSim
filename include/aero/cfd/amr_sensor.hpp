#pragma once

#include "aero/cfd/amr_types.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/turbulence_model.hpp"
#include "aero/cfd/real_fwd.hpp"

#include <cstdint>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// Sensor type for turbulence-aware AMR refinement criteria
enum class SensorType : std::uint8_t {
    GRADIENT,      // density/pressure/velocity-gradient jump sensor
    CURVATURE,     // curvature-based refinement (future)
    YPLUS,         // wall y+ constraint
    Q_CRITERION,   // Q-criterion for vortex detection
    TKE_RATIO,     // k / (0.5 * U²) for wake/mixing-layer refinement
    SHEAR_LAYER    // resolved_k / (resolved_k + modeled_k) for DDES under-resolved detection
};

// Compute refinement requests from density-gradient sensor.
// Each interior face contributes |density jump|/|avg density| to both adjacent cells.
// Cells with max error > refine_tol get Refine; cells with max error < coarsen_tol
// AND refinement_level>0 get Coarsen.
std::vector<RefinementRequest> compute_gradient_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma = 1.4f);

// Compute refinement requests from y+ sensor for wall-adjacent cells.
// Estimates y+ using the wall-adjacent cell state and wall distance:
//   mu = sutherland(T), chi = rho*nu_tilde/mu, fv1 = chi^3/(chi^3+cv1^3)
//   mu_eff = mu + rho*nu_tilde*fv1 (SA/SA_DDES), or just mu (LAMINAR/SST)
//   y+ = sqrt(Re * d * rho * |u|) * sqrt(mu_eff) / mu
// Cells with y+ > yplus_target -> Refine; y+ < 0.5*yplus_target and level>0 -> Coarsen.
// For SST: no k/omega in CPU state, uses mu_eff=mu (conservative over-refinement).
std::vector<RefinementRequest> compute_yplus_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma,
    Real Re,
    Real mu_ref,
    Real T_ref,
    Real sutherland_T,
    TurbulenceModel turbulence_model);

// Compute refinement requests from Q-criterion sensor for wake, shear layer, and vortex cores.
// Computes per-cell velocity gradients via Green-Gauss, then Q = -0.5*tr(S²) - Σ(cross terms).
// Positive Q > refine_tol*|Q|_max -> Refine (vortex-dominated).
// Negative Q with |Q| < coarsen_tol*|Q|_max and level>0 -> Coarsen (strain-dominated, smooth).
// Thresholds are normalized by max |Q| across all cells for flow-independent tuning.
std::vector<RefinementRequest> compute_qcriterion_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma);

// Compute refinement requests from geometric wake cone region.
// Cells whose centroids fall within a user-defined cone downstream of the body
// are flagged for refinement. Useful for wake refinement behind bluff bodies.
std::vector<RefinementRequest> compute_wake_cone_sensor(
    const CfdMesh& mesh,
    const AmrConfig& config,
    const WakeConeConfig& cone);

// Compute refinement requests from TKE-to-mean-flow kinetic energy ratio.
// Flags cells where k / (0.5 * U²) > tke_ratio_threshold (wake, mixing layer).
// SST: reads k from sst_k vector (one per cell). Non-SST models: no-op.
// Threshold interpreted as: e.g. 0.05 = 5% turbulent vs. mean-flow KE.
std::vector<RefinementRequest> compute_tke_ratio_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma,
    TurbulenceModel turbulence_model,
    const std::vector<Real>* sst_k = nullptr,
    Real tke_ratio_threshold = 0.05f);

// Compute refinement requests from shear-layer under-resolution sensor (DDES).
// Estimates resolved TKE from velocity gradients × cell size (spatial high-pass):
//   resolved_k = 0.125 * h_cell² * (|∇u|² + |∇v|² + |∇w|²)
// Modeled TKE (SST: k directly; SA/SA-DDES: nu_tilde * S_mag / a1).
// Ratio = resolved_k / (resolved_k + modeled_k).
// Ratio < threshold → under-resolved → refine.
// For LAMINAR: no-op (no turbulence model to under-resolve).
std::vector<RefinementRequest> compute_shear_layer_sensor(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const AmrConfig& config,
    Real gamma,
    TurbulenceModel turbulence_model,
    const std::vector<Real>* sst_k = nullptr,
    Real shear_layer_threshold = 0.3f);

// Merge multiple refinement request vectors using OR logic:
// - If any sensor marks a cell Refine → Refine
// - Only if ALL non-unchanged sensors agree on Coarsen → Coarsen
// - Otherwise → Unchanged
// Empty sensors vector returns all Unchanged.
std::vector<RefinementRequest> merge_refinement_requests(
    const std::vector<std::vector<RefinementRequest>>& sensors);

} // namespace cfd
} // namespace aero
} // namespace aerosp
