#pragma once

#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/real.hpp"

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

} // namespace cfd
} // namespace aero
} // namespace aerosp
