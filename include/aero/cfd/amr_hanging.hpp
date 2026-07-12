#pragma once

#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/reconstruction.hpp"
#include "aero/cfd/real_fwd.hpp"

#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

struct HangingFaceInfo {
    int face_id;
    int coarse_cell_id;
    int fine_cell_id;
    int coarse_level;
    int fine_level;
};

// Per-cell gradient vector (one per conserved variable component).
struct CellGradient3 {
    Real gx = 0.0f, gy = 0.0f, gz = 0.0f;
};

// Detect interior faces at refinement boundaries
// (left_cell and right_cell have different refinement_level).
std::vector<HangingFaceInfo> detect_hanging_faces(const CfdMesh& mesh);

// Interpolate coarse cell state to each hanging face center using linear
// extrapolation: q_face = q_cell + grad_q . (face_center - cell_center).
// Only modifies states on the coarse side of hanging faces.
// Arrays must be sized [n_cells] and [n_faces] respectively.
void apply_hanging_interpolation(
    const CfdMesh& mesh,
    const std::vector<HangingFaceInfo>& hanging_faces,
    const std::vector<ConservativeState>& q_cell,
    const std::vector<CellGradient3>& grad_rho,
    const std::vector<CellGradient3>& grad_rho_u,
    const std::vector<CellGradient3>& grad_rho_v,
    const std::vector<CellGradient3>& grad_rho_w,
    const std::vector<CellGradient3>& grad_rho_E,
    const std::vector<CellGradient3>& grad_rho_nu,
    std::vector<ConservativeState>& q_face_left,
    std::vector<ConservativeState>& q_face_right);

// Primitive-space hanging face flux correction.
// Reconstructs the coarse-side state at each hanging face center using
// primitive gradients (positive-preserving by limiter design), then
// computes the HLLC flux and adjusts the residual.
//
// For each hanging face:
//   1. QL_old = reconstruct_primitive(w[coarse], grad[coarse], dr_coarse)
//      QR_old = reconstruct_primitive(w[fine], grad[fine], dr_fine)
//   2. QL_new = reconstruct_primitive_positive(w[coarse], grad[coarse], dr_coarse, ...)
//   3. flux_old = hllc(QL_old, QR_old)
//   4. flux_new = hllc(QL_new, QR_old)   (only coarse side changes)
//   5. residual += (flux_old - flux_new) * area
//
// The residual adjustment accounts for the fact that the main residual
// kernel already added flux_old. This function replaces it with flux_new.
void apply_hanging_flux_correction_primitive(
    const CfdMesh& mesh,
    const std::vector<HangingFaceInfo>& hanging_faces,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveState>& w,
    const std::vector<PrimitiveGradient>& grads,
    Real gamma,
    std::vector<EulerFlux>& residual);

} // namespace cfd
} // namespace aero
} // namespace aerosp
