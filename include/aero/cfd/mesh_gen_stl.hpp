#pragma once

#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/element_types.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

struct StlMeshConfig {
    Real outer_scale = 5.0f;
    int background_n_per_dim = 20;
    int max_cells = 500000;

    bool prism_layers = false;
    int n_prism_layers = 5;
    Real prism_first_height = 1e-4f;
    Real prism_growth_ratio = 1.2f;
    int prism_fallback_min_layers = 1;
};

// Generate a body-fitted conformal volume mesh from an STL surface mesh.
// The pipeline:
//   1. Parse STL triangle soup
//   2. Build signed-distance field on a background hex grid
//   3. Hex-cull: remove cells inside the body, cut boundary cells at SDF=0
//   4. Extract SDF=0 iso-surface as wall boundary faces
//   5. Assign farfield faces from outer bounding box
//   6. (Optional) Extrude prism layers from wall surface
//   7. Rebuild cell-face connectivity and compute metrics
//   8. Validate mesh (zero negative Jacobians, closed surface)
//
// Parameters:
//   stl_path   - Path to STL file (ASCII or binary)
//   mesh       - Output mesh (overwritten on success)
//   cfg        - Generation parameters
//   error      - Optional error message output
//
// Returns true on success, false on failure.
bool generate_conformal_mesh_from_stl(
    const std::string& stl_path,
    CfdMesh& mesh,
    const StlMeshConfig& cfg,
    std::string* error = nullptr);

// Watertight hex-cull mesh for solver production path (Phase 9-B.5).
// Removes background hex cells whose centers lie inside the STL (SDF < 0).
// No cut-cell clipping — closed surface matches CfdSolver::load_mesh (1e-4).
bool generate_watertight_mesh_from_stl(
    const std::string& stl_path,
    CfdMesh& mesh,
    const StlMeshConfig& cfg,
    std::string* error = nullptr);

} // namespace cfd
} // namespace aero
} // namespace aerosp
