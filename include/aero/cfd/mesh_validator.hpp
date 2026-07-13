#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/cfd_mesh.hpp"

namespace aerosp {
namespace aero {
namespace cfd {

MeshQualityReport compute_mesh_quality_detail(const CfdMesh& mesh);

bool tet_jacobian_sign(const CfdNode* nodes, int& neg_count);
bool hex_jacobian_sign(const CfdNode* nodes, int& neg_count);
bool penta_jacobian_sign(const CfdNode* nodes, int& neg_count);
bool pyramid_jacobian_sign(const CfdNode* nodes, int& neg_count);

} // namespace cfd
} // namespace aero
} // namespace aerosp
