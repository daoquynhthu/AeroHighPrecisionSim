#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/cfd_mesh.hpp"

namespace aerosp {
namespace aero {
namespace cfd {

MeshQualityReport compute_mesh_quality_detail(const CfdMesh& mesh);

} // namespace cfd
} // namespace aero
} // namespace aerosp
