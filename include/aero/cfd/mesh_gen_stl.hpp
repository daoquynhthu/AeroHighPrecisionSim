#pragma once

#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/element_types.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// Unified volume-mesh backend for CFD / aero-table production.
// Production default = StlWatertight (hex-cull, load_mesh 1e-4).
enum class VolumeMeshBackend : int {
    // Prefer cut-cell quality mesh; if load_mesh closed-surface fails, fall back
    // to StlWatertight. Never falls back to cube.
    Auto = 0,
    // Hex-cull watertight body-fitted mesh (production strong path).
    StlWatertight = 1,
    // Cut-cell conformal mesh (quality); may fail closed-surface gate.
    StlCutCell = 2,
    // Legacy structured cube with body cavity (regression only).
    CubeLegacy = 3,
};

struct StlMeshConfig {
    Real outer_scale = 5.0f;
    int background_n_per_dim = 20;
    int max_cells = 500000;

    bool prism_layers = false;
    int n_prism_layers = 5;
    Real prism_first_height = 1e-4f;
    Real prism_growth_ratio = 1.2f;
    int prism_fallback_min_layers = 1;

    bool multi_body = false;
    Real gap_cell_threshold = 0.0f;

    // Backend selection (production: Auto → watertight with optional cut-cell try).
    VolumeMeshBackend backend = VolumeMeshBackend::StlWatertight;
    // When backend==Auto: attempt cut-cell first if true (default false for speed).
    bool auto_try_cut_cell = false;
};

// Generate a body-fitted conformal volume mesh from an STL surface mesh.
// Cut-cell pipeline (quality). Closed-surface may not meet load_mesh 1e-4.
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

// Unified production entry: dispatches on cfg.backend.
// - StlWatertight / Auto: watertight hex-cull (Auto may try cut-cell first)
// - StlCutCell: conformal only
// - CubeLegacy: structured cube (stl_path ignored; uses outer_scale / n_per_dim)
// On success mesh has metrics computed and is load_mesh-ready (except pure
// StlCutCell which may fail closed-surface — caller must validate).
bool generate_volume_mesh(
    const std::string& stl_path,
    CfdMesh& mesh,
    const StlMeshConfig& cfg,
    std::string* error = nullptr);

// Human-readable backend name for logs / fidelity tags.
const char* volume_mesh_backend_name(VolumeMeshBackend b);

} // namespace cfd
} // namespace aero
} // namespace aerosp
