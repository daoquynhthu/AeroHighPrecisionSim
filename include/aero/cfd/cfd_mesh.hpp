#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/element_types.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

enum class BoundaryKind : int {
    Interior = 0,
    Farfield = 1,
    SlipWall = 2,
    NoSlipWall = 3,
    Symmetry = 4
};

struct CfdNode {
    Real x = 0.0f;
    Real y = 0.0f;
    Real z = 0.0f;
};

struct CfdCell {
    ElementType type = ElementType::TET4;
    int node[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int first_face = 0;
    int face_count = 0;
    Real volume = 0.0f;
    Real cx = 0.0f;
    Real cy = 0.0f;
    Real cz = 0.0f;
    Real h_min = 0.0f;
    Real wall_distance = 0.0f;
    int refinement_level = 0;
    int parent_id = -1;
};

struct CfdFace {
    int left_cell = -1;
    int right_cell = -1;
    int node_count = 3;
    int node[4] = {-1, -1, -1, -1};
    BoundaryKind boundary = BoundaryKind::Interior;
    Real area = 0.0f;
    Real nx = 0.0f;
    Real ny = 0.0f;
    Real nz = 0.0f;
    Real cx = 0.0f;
    Real cy = 0.0f;
    Real cz = 0.0f;
};

struct CfdMesh {
    std::vector<CfdNode> nodes;
    std::vector<CfdCell> cells;
    std::vector<CfdFace> faces;
};

struct MeshQualityReport {
    int nodes = 0;
    int cells = 0;
    int faces = 0;
    int interior_faces = 0;
    int farfield_faces = 0;
    int slip_wall_faces = 0;
    int no_slip_wall_faces = 0;
    int symmetry_faces = 0;
    Real min_volume = 0.0f;
    Real max_volume = 0.0f;
    Real min_h = 0.0f;
    Real max_h = 0.0f;
    Real min_wall_distance = 0.0f;
    Real min_aspect_ratio = 0.0f;
    Real max_aspect_ratio = 0.0f;
    Real avg_aspect_ratio = 0.0f;
    Real min_skewness = 0.0f;
    Real max_skewness = 0.0f;
    Real avg_skewness = 0.0f;
    Real min_orthogonality = 0.0f;
    Real max_orthogonality = 0.0f;
    Real avg_orthogonality = 0.0f;
    Real closed_surface_error = 0.0f;
    int negative_jacobian_count = 0;
    int high_aspect_ratio_count = 0;
    int high_skew_count = 0;
    bool valid = false;
    std::string message;
};

CfdMesh generate_structured_cube_mesh(Real outer_scale = 5.0f, int n_nodes_per_dim = 13);

CfdMesh generate_flat_plate_mesh(
    Real length = 0.5f,
    Real width = 0.05f,
    Real height = 0.1f,
    Real first_height = 1e-5f,
    Real growth_ratio = 1.12f,
    int nx = 30,
    int ny = 3,
    int nz = 50);

CfdMesh generate_structured_hex_mesh(int n_per_dim = 10);

CfdMesh generate_prism_boundary_layer_mesh(
    int nx = 10, int ny = 5, int nz = 10,
    Real length = 0.5f, Real width = 0.05f,
    Real first_height = 1e-5f, Real growth_ratio = 1.12f);

void rebuild_mesh_faces(CfdMesh& mesh);
MeshQualityReport compute_mesh_metrics(CfdMesh& mesh, bool recompute_cells = true);

bool validate_mesh(const CfdMesh& mesh, MeshQualityReport* report = nullptr);

Real boundary_area(const CfdMesh& mesh, BoundaryKind boundary);

} // namespace cfd
} // namespace aero
} // namespace aerosp



