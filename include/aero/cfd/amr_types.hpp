#pragma once

#include "aero/cfd/element_types.hpp"
#include "aero/cfd/real_fwd.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

enum class RefinementFlag : int8_t {
    Coarsen = -1,
    Unchanged = 0,
    Refine = 1
};

enum class AnisotropicDir : int8_t {
    NONE = 0,
    DIR_X,
    DIR_Y,
    DIR_Z,
    WALL_NORMAL,
    STREAMWISE
};

struct RefinementRequest {
    int cell_id = -1;
    RefinementFlag flag = RefinementFlag::Unchanged;
    AnisotropicDir dir = AnisotropicDir::NONE;
};

struct RefinementRecord {
    int parent_cell_id = -1;
    int child_cell_ids[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int n_children = 0;
    int parent_level = 0;
    ElementType parent_type = ElementType::TET4;
    int parent_node[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    int parent_face_count = 0;
};

struct CoarsenInfo {
    int new_parent_id = -1;
    int old_parent_id = -1;
    int old_child_ids[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int n_children = 0;
};

// Geometrically-defined wake refinement cone
struct WakeConeConfig {
    Real origin_x = 0.0f;
    Real origin_y = 0.0f;
    Real origin_z = 0.0f;
    Real axis_x = 1.0f;
    Real axis_y = 0.0f;
    Real axis_z = 0.0f;
    Real half_angle_deg = 10.0f;
    Real length = 1.0f;
};

struct AmrConfig {
    bool enabled = false;
    int interval = 50;
    int max_level = 5;
    Real refine_tol = 0.1f;
    Real coarsen_tol = 0.01f;
    Real yplus_target = 1.0f;           // target y+ for wall-adjacent cells (y+ sensor)
    Real tke_ratio_threshold = 0.05f;   // k / (0.5*U^2) threshold for TKE ratio sensor
    Real shear_layer_threshold = 0.3f;  // resolved_k/(resolved_k+modeled_k) threshold for shear-layer sensor
    WakeConeConfig wake_cone;           // wake refinement region (empty = disabled)
    int anisotropic_layers = 0;         // levels of anisotropic split before switching to isotropic
};

struct CfdMesh;

bool refine_cells(CfdMesh& mesh,
                  const std::vector<RefinementRequest>& requests,
                  std::vector<RefinementRecord>* records_out = nullptr,
                  std::string* error = nullptr,
                  const std::vector<RefinementRecord>* prev_records = nullptr,
                  std::vector<CoarsenInfo>* coarsen_info = nullptr,
                  int max_level = 5);

} // namespace cfd
} // namespace aero
} // namespace aerosp
