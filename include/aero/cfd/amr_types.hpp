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

struct RefinementRequest {
    int cell_id = -1;
    RefinementFlag flag = RefinementFlag::Unchanged;
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

struct AmrConfig {
    bool enabled = false;
    int interval = 50;
    int max_level = 5;
    Real refine_tol = 0.1f;
    Real coarsen_tol = 0.01f;
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
