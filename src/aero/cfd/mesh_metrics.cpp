#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

struct Vec3 {
    Real x;
    Real y;
    Real z;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, Real s) { return {a.x * s, a.y * s, a.z * s}; }

Vec3 to_vec(const CfdNode& n) { return {n.x, n.y, n.z}; }

Real dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}

Real norm(Vec3 a) { return std::sqrt(dot(a, a)); }

Real volume_tet(Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    return std::fabs(dot(b - a, cross(c - a, d - a))) / 6.0f;
}

Real volume_hex(const Vec3 v[8]) {
    // Decompose hex into 5 or 6 tets. 6-tet decomposition (stable):
    // split along diagonal (0,2,5,7)
    Real vol = 0.0f;
    vol += volume_tet(v[0], v[1], v[2], v[6]);
    vol += volume_tet(v[0], v[2], v[3], v[6]);
    vol += volume_tet(v[0], v[3], v[7], v[6]);
    vol += volume_tet(v[0], v[7], v[4], v[6]);
    vol += volume_tet(v[0], v[4], v[5], v[6]);
    vol += volume_tet(v[0], v[5], v[1], v[6]);
    return vol;
}

Real volume_prism(const Vec3 v[6]) {
    // Decompose prism into 3 tets
    Real vol = 0.0f;
    vol += volume_tet(v[0], v[1], v[2], v[4]);
    vol += volume_tet(v[0], v[2], v[5], v[4]);
    vol += volume_tet(v[0], v[3], v[4], v[5]);
    return vol;
}

Real volume_pyramid(const Vec3 v[5]) {
    // Split into 2 tets along diagonal v[0]-v[2]
    Real vol = 0.0f;
    vol += volume_tet(v[0], v[1], v[2], v[4]);
    vol += volume_tet(v[0], v[3], v[4], v[2]);
    return vol;
}

Real area_tri(Vec3 a, Vec3 b, Vec3 c) {
    return 0.5f * norm(cross(b - a, c - a));
}

Real area_quad(Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    // Split into 2 triangles: (a,b,c) + (a,c,d)
    return area_tri(a, b, c) + area_tri(a, c, d);
}

Vec3 centroid_tet(Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    return (a + b + c + d) * (1.0f / 4.0f);
}

Vec3 centroid_hex(const Vec3 v[8]) {
    Vec3 sum = v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
    return sum * (1.0f / 8.0f);
}

Vec3 centroid_prism(const Vec3 v[6]) {
    Vec3 sum = v[0] + v[1] + v[2] + v[3] + v[4] + v[5];
    return sum * (1.0f / 6.0f);
}

Vec3 centroid_pyramid(const Vec3 v[5]) {
    Vec3 base_sum = v[0] + v[1] + v[2] + v[3];
    return (base_sum * 3.0f + v[4] * 4.0f) * (1.0f / 16.0f);
}

struct FaceKey {
    int n;
    int v[4];

    FaceKey(int count, int a, int b, int c, int d = -1) : n(count) {
        v[0] = a; v[1] = b; v[2] = c; v[3] = d;
        if (n == 3) {
            if (v[0] > v[1]) std::swap(v[0], v[1]);
            if (v[1] > v[2]) std::swap(v[1], v[2]);
            if (v[0] > v[1]) std::swap(v[0], v[1]);
            v[3] = -1;
        } else {
            std::sort(v, v + 4);
        }
    }

    bool operator==(const FaceKey& o) const {
        if (n != o.n) return false;
        return v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2] && v[3] == o.v[3];
    }
};

struct FaceKeyHash {
    std::size_t operator()(const FaceKey& key) const {
        std::uint64_t h = static_cast<std::uint64_t>(key.n);
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<std::uint64_t>(key.v[i]) << (i * 11);
        }
        return static_cast<std::size_t>(h);
    }
};

struct PendingFace {
    int cell;
    int local_face;
};

// Dispatch cell_face_nodes per element type
// Returns face node indices into cell.node[], fills n_out with node count
void cell_face_nodes(const CfdCell& cell, int local_face, int out[4], int& n_out) {
    const int* fn = get_face_nodes(cell.type, local_face);
    n_out = FACE_NODES_PER_ELEMENT[static_cast<int>(cell.type)][local_face];
    for (int i = 0; i < n_out; ++i) {
        out[i] = cell.node[fn[i]];
    }
    for (int i = n_out; i < 4; ++i) {
        out[i] = -1;
    }
}

// Get face node positions for a given element type and local face
void cell_face_positions(const CfdMesh& mesh, const CfdCell& cell, int local_face, Vec3 pos[4], int& n_out) {
    int nodes[4];
    int n;
    cell_face_nodes(cell, local_face, nodes, n);
    n_out = n;
    for (int i = 0; i < n; ++i) {
        pos[i] = to_vec(mesh.nodes[nodes[i]]);
    }
}

int grid_index(int i, int j, int k, int n) {
    return i + j*n + k*n*n;
}

int hex_index(int i, int j, int k, int n_hex) {
    return i + j*n_hex + k*n_hex*n_hex;
}

void add_tet(CfdMesh& mesh, int a, int b, int c, int d) {
    CfdCell cell;
    cell.type = ElementType::TET4;
    cell.node[0] = a;
    cell.node[1] = b;
    cell.node[2] = c;
    cell.node[3] = d;
    mesh.cells.push_back(cell);
}

void add_hex_tets(CfdMesh& mesh, int p0, int p1, int p2, int p3, int p4, int p5, int p6, int p7) {
    add_tet(mesh, p0, p1, p2, p6);
    add_tet(mesh, p0, p2, p3, p6);
    add_tet(mesh, p0, p3, p7, p6);
    add_tet(mesh, p0, p7, p4, p6);
    add_tet(mesh, p0, p4, p5, p6);
    add_tet(mesh, p0, p5, p1, p6);
}

void add_hex(CfdMesh& mesh, int p0, int p1, int p2, int p3, int p4, int p5, int p6, int p7) {
    CfdCell cell;
    cell.type = ElementType::HEX8;
    cell.node[0] = p0;
    cell.node[1] = p1;
    cell.node[2] = p2;
    cell.node[3] = p3;
    cell.node[4] = p4;
    cell.node[5] = p5;
    cell.node[6] = p6;
    cell.node[7] = p7;
    mesh.cells.push_back(cell);
}

BoundaryKind classify_cube_boundary(Vec3 fc, Vec3 outward, Real outer_scale, Real delta, const std::vector<bool>& is_body, int n_hex) {
    const Real outer_tol = delta * 0.25f;
    if (std::fabs(std::fabs(fc.x) - outer_scale) < outer_tol ||
        std::fabs(std::fabs(fc.y) - outer_scale) < outer_tol ||
        std::fabs(std::fabs(fc.z) - outer_scale) < outer_tol) {
        return BoundaryKind::Farfield;
    }

    int di = (outward.x > 0.25f) ? 1 : ((outward.x < -0.25f) ? -1 : 0);
    int dj = (outward.y > 0.25f) ? 1 : ((outward.y < -0.25f) ? -1 : 0);
    int dk = (outward.z > 0.25f) ? 1 : ((outward.z < -0.25f) ? -1 : 0);

    Real x0 = -outer_scale;
    int hi = static_cast<int>(std::floor((fc.x - x0) / delta));
    int hj = static_cast<int>(std::floor((fc.y - x0) / delta));
    int hk = static_cast<int>(std::floor((fc.z - x0) / delta));
    hi = std::max(0, std::min(n_hex - 1, hi));
    hj = std::max(0, std::min(n_hex - 1, hj));
    hk = std::max(0, std::min(n_hex - 1, hk));

    int bi = hi + di;
    int bj = hj + dj;
    int bk = hk + dk;
    if (bi >= 0 && bi < n_hex && bj >= 0 && bj < n_hex && bk >= 0 && bk < n_hex) {
        if (is_body[hex_index(bi, bj, bk, n_hex)]) return BoundaryKind::SlipWall;
    }

    return BoundaryKind::Farfield;
}

void rebuild_faces(CfdMesh& mesh, bool classify_cube = false, Real outer_scale = 0.0f, Real delta = 0.0f, const std::vector<bool>& is_body = {}, int n_hex = 0) {
    mesh.faces.clear();
    for (auto& cell : mesh.cells) {
        cell.first_face = 0;
        cell.face_count = 0;
    }

    std::unordered_map<FaceKey, PendingFace, FaceKeyHash> face_map;
    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        const auto& cell = mesh.cells[ci];
        int nfaces = ELEMENT_FACES[static_cast<int>(cell.type)];
        for (int lf = 0; lf < nfaces; ++lf) {
            int nodes[4];
            int nn;
            cell_face_nodes(cell, lf, nodes, nn);
            FaceKey key(nn, nodes[0], nodes[1], nodes[2], nn == 4 ? nodes[3] : -1);
            auto it = face_map.find(key);
            if (it == face_map.end()) {
                face_map[key] = {ci, lf};
            } else {
                CfdFace face;
                face.left_cell = it->second.cell;
                face.right_cell = ci;
                face.boundary = BoundaryKind::Interior;
                face.node_count = nn;
                int left_nodes[4];
                int left_nn;
                cell_face_nodes(mesh.cells[face.left_cell], it->second.local_face, left_nodes, left_nn);
                for (int i = 0; i < left_nn; ++i) face.node[i] = left_nodes[i];
                mesh.faces.push_back(face);
                face_map.erase(it);
            }
        }
    }

    for (const auto& item : face_map) {
        CfdFace face;
        face.left_cell = item.second.cell;
        face.right_cell = -1;
        int fn[4];
        int fn_n;
        cell_face_nodes(mesh.cells[face.left_cell], item.second.local_face, fn, fn_n);
        face.node_count = fn_n;
        for (int i = 0; i < fn_n; ++i) face.node[i] = fn[i];
        face.boundary = BoundaryKind::Farfield;
        mesh.faces.push_back(face);
    }

    compute_mesh_metrics(mesh);

    for (auto& face : mesh.faces) {
        if (face.boundary == BoundaryKind::Interior) continue;
        Vec3 fc{face.cx, face.cy, face.cz};
        Vec3 n{face.nx, face.ny, face.nz};
        if (classify_cube) {
            face.boundary = classify_cube_boundary(fc, n, outer_scale, delta, is_body, n_hex);
        } else if (std::fabs(fc.z) < 1e-7f) {
            face.boundary = BoundaryKind::NoSlipWall;
        } else {
            face.boundary = BoundaryKind::Farfield;
        }
    }

    compute_mesh_metrics(mesh, false);
}

// Compute volume, centroid, h_min for any element type
void compute_cell_metrics(CfdMesh& mesh, CfdCell& cell) {
    Vec3 v[8];
    int nnodes = ELEMENT_NODES[static_cast<int>(cell.type)];
    for (int i = 0; i < nnodes; ++i) {
        v[i] = to_vec(mesh.nodes[cell.node[i]]);
    }

    switch (cell.type) {
        case ElementType::TET4: {
            cell.volume = volume_tet(v[0], v[1], v[2], v[3]);
            if (!std::isfinite(cell.volume) || cell.volume <= 0.0f) cell.volume = 1e-30f;
            Vec3 c = centroid_tet(v[0], v[1], v[2], v[3]);
            cell.cx = c.x; cell.cy = c.y; cell.cz = c.z;
            // h_min = 3*V / max_face_area
            Real max_area = 0.0f;
            for (int lf = 0; lf < 4; ++lf) {
                Vec3 fp[4]; int fn;
                cell_face_positions(mesh, cell, lf, fp, fn);
                Real area = area_tri(fp[0], fp[1], fp[2]);
                max_area = std::max(max_area, area);
            }
            cell.h_min = 3.0f * cell.volume / std::max(max_area, Real(1e-30));
            break;
        }
        case ElementType::HEX8: {
            cell.volume = volume_hex(v);
            if (!std::isfinite(cell.volume) || cell.volume <= 0.0f) cell.volume = 1e-30f;
            Vec3 c = centroid_hex(v);
            cell.cx = c.x; cell.cy = c.y; cell.cz = c.z;
            // h_min = V^(1/3) for hex
            cell.h_min = std::pow(cell.volume, 1.0f / 3.0f);
            break;
        }
        case ElementType::PENTA6: {
            cell.volume = volume_prism(v);
            if (!std::isfinite(cell.volume) || cell.volume <= 0.0f) cell.volume = 1e-30f;
            Vec3 c = centroid_prism(v);
            cell.cx = c.x; cell.cy = c.y; cell.cz = c.z;
            cell.h_min = std::pow(cell.volume, 1.0f / 3.0f);
            break;
        }
        case ElementType::PYRAMID5: {
            cell.volume = volume_pyramid(v);
            if (!std::isfinite(cell.volume) || cell.volume <= 0.0f) cell.volume = 1e-30f;
            Vec3 c = centroid_pyramid(v);
            cell.cx = c.x; cell.cy = c.y; cell.cz = c.z;
            cell.h_min = std::pow(cell.volume, 1.0f / 3.0f);
            break;
        }
    }
}

} // namespace

CfdMesh generate_structured_cube_mesh(Real outer_scale, int n_nodes_per_dim) {
    CfdMesh mesh;
    int n = n_nodes_per_dim;
    if (outer_scale <= 1.0f || n < 3) return mesh;

    Real delta = 2.0f * outer_scale / static_cast<Real>(n - 1);
    mesh.nodes.reserve(static_cast<std::size_t>(n) * n * n);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                mesh.nodes.push_back({
                    -outer_scale + i * delta,
                    -outer_scale + j * delta,
                    -outer_scale + k * delta
                });
            }
        }
    }

    int n_hex = n - 1;
    std::vector<bool> is_body(static_cast<std::size_t>(n_hex) * n_hex * n_hex, false);
    // Unit-cube body [-1,1]^3: mark hex if AABB intersects body
    // (centroid-only tests miss the body when delta > 2 on coarse grids).
    const Real body_lo = -1.0f - 1e-6f;
    const Real body_hi = 1.0f + 1e-6f;
    for (int k = 0; k < n_hex; ++k) {
        for (int j = 0; j < n_hex; ++j) {
            for (int i = 0; i < n_hex; ++i) {
                Real x0 = -outer_scale + i * delta;
                Real y0 = -outer_scale + j * delta;
                Real z0 = -outer_scale + k * delta;
                Real x1 = x0 + delta;
                Real y1 = y0 + delta;
                Real z1 = z0 + delta;
                bool body = (x0 < body_hi && x1 > body_lo) &&
                            (y0 < body_hi && y1 > body_lo) &&
                            (z0 < body_hi && z1 > body_lo);
                is_body[hex_index(i, j, k, n_hex)] = body;
            }
        }
    }

    for (int k = 0; k < n_hex; ++k) {
        for (int j = 0; j < n_hex; ++j) {
            for (int i = 0; i < n_hex; ++i) {
                if (is_body[hex_index(i, j, k, n_hex)]) continue;
                int p0 = grid_index(i,   j,   k, n);
                int p1 = grid_index(i+1, j,   k, n);
                int p2 = grid_index(i+1, j+1, k, n);
                int p3 = grid_index(i,   j+1, k, n);
                int p4 = grid_index(i,   j,   k+1, n);
                int p5 = grid_index(i+1, j,   k+1, n);
                int p6 = grid_index(i+1, j+1, k+1, n);
                int p7 = grid_index(i,   j+1, k+1, n);
                // HEX8 keeps coordinate-plane reflection symmetry (tet fans do not).
                add_hex(mesh, p0, p1, p2, p3, p4, p5, p6, p7);
            }
        }
    }

    rebuild_faces(mesh, true, outer_scale, delta, is_body, n_hex);
    return mesh;
}

CfdMesh generate_flat_plate_mesh(Real length, Real width, Real height, Real first_height, Real growth_ratio, int nx, int ny, int nz) {
    CfdMesh mesh;
    if (length <= 0.0f || width <= 0.0f || height <= 0.0f || first_height <= 0.0f ||
        nx < 2 || ny < 2 || nz < 2) {
        return mesh;
    }

    std::vector<Real> xs(nx);
    std::vector<Real> ys(ny);
    std::vector<Real> zs(nz);
    for (int i = 0; i < nx; ++i) xs[i] = length * static_cast<Real>(i) / static_cast<Real>(nx - 1);
    for (int j = 0; j < ny; ++j) ys[j] = -0.5f * width + width * static_cast<Real>(j) / static_cast<Real>(ny - 1);
    zs[0] = 0.0f;
    for (int k = 1; k < nz - 1; ++k) {
        if (growth_ratio > 1.001f) {
            zs[k] = first_height * (std::pow(growth_ratio, static_cast<Real>(k)) - 1.0f) / (growth_ratio - 1.0f);
        } else {
            zs[k] = first_height * static_cast<Real>(k);
        }
    }
    zs[nz - 1] = height;
    if (nz > 2 && zs[nz - 2] >= height) return CfdMesh{};

    mesh.nodes.reserve(static_cast<std::size_t>(nx) * ny * nz);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                mesh.nodes.push_back({xs[i], ys[j], zs[k]});
            }
        }
    }

    auto idx = [nx, ny](int i, int j, int k) {
        return (k * ny + j) * nx + i;
    };

    for (int k = 0; k < nz - 1; ++k) {
        for (int j = 0; j < ny - 1; ++j) {
            for (int i = 0; i < nx - 1; ++i) {
                int p0 = idx(i,   j,   k);
                int p1 = idx(i+1, j,   k);
                int p2 = idx(i+1, j+1, k);
                int p3 = idx(i,   j+1, k);
                int p4 = idx(i,   j,   k+1);
                int p5 = idx(i+1, j,   k+1);
                int p6 = idx(i+1, j+1, k+1);
                int p7 = idx(i,   j+1, k+1);
                add_hex_tets(mesh, p0, p1, p2, p3, p4, p5, p6, p7);
            }
        }
    }

    std::vector<bool> empty;
    rebuild_faces(mesh, false, 0.0f, 0.0f, empty, 0);
    return mesh;
}

CfdMesh generate_structured_cube_mesh_no_body(Real outer_scale, int n_per_dim) {
    // Solid cube mesh without any body cavity — all hex cells are retained.
    // Unlike generate_structured_cube_mesh, no cells are removed for a body
    // at the origin, making this mesh suitable for freestream-preservation tests.
    CfdMesh mesh;
    int n = n_per_dim;
    if (outer_scale <= 0.0f || n < 2) return mesh;
    Real delta = 2.0f * outer_scale / static_cast<Real>(n - 1);
    mesh.nodes.reserve(static_cast<std::size_t>(n) * n * n);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                mesh.nodes.push_back({
                    -outer_scale + i * delta,
                    -outer_scale + j * delta,
                    -outer_scale + k * delta
                });
            }
        }
    }
    auto idx = [n](int i, int j, int k) { return i + j * n + k * n * n; };
    int n_hex = n - 1;
    for (int k = 0; k < n_hex; ++k) {
        for (int j = 0; j < n_hex; ++j) {
            for (int i = 0; i < n_hex; ++i) {
                int p0 = idx(i,   j,   k);
                int p1 = idx(i+1, j,   k);
                int p2 = idx(i+1, j+1, k);
                int p3 = idx(i,   j+1, k);
                int p4 = idx(i,   j,   k+1);
                int p5 = idx(i+1, j,   k+1);
                int p6 = idx(i+1, j+1, k+1);
                int p7 = idx(i,   j+1, k+1);
                add_hex(mesh, p0, p1, p2, p3, p4, p5, p6, p7);
            }
        }
    }
    rebuild_faces(mesh);
    return mesh;
}

CfdMesh generate_structured_hex_mesh(int n_per_dim) {
    CfdMesh mesh;
    int n = n_per_dim;
    if (n < 2) return mesh;

    Real delta = 1.0f / static_cast<Real>(n - 1);
    mesh.nodes.reserve(static_cast<std::size_t>(n) * n * n);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                mesh.nodes.push_back({
                    -0.5f + i * delta,
                    -0.5f + j * delta,
                    -0.5f + k * delta
                });
            }
        }
    }

    auto idx = [n](int i, int j, int k) { return i + j * n + k * n * n; };
    int n_hex = n - 1;
    for (int k = 0; k < n_hex; ++k) {
        for (int j = 0; j < n_hex; ++j) {
            for (int i = 0; i < n_hex; ++i) {
                int p0 = idx(i,   j,   k);
                int p1 = idx(i+1, j,   k);
                int p2 = idx(i+1, j+1, k);
                int p3 = idx(i,   j+1, k);
                int p4 = idx(i,   j,   k+1);
                int p5 = idx(i+1, j,   k+1);
                int p6 = idx(i+1, j+1, k+1);
                int p7 = idx(i,   j+1, k+1);
                add_hex(mesh, p0, p1, p2, p3, p4, p5, p6, p7);
            }
        }
    }

    rebuild_faces(mesh);
    return mesh;
}

CfdMesh generate_prism_boundary_layer_mesh(
    int nx, int ny, int nz,
    Real length, Real width,
    Real first_height, Real growth_ratio)
{
    CfdMesh mesh;
    if (nx < 2 || ny < 2 || nz < 2 || length <= 0.0f || width <= 0.0f || first_height <= 0.0f) {
        return mesh;
    }

    // Surface: nx × ny grid of quads, each quad split into 2 triangles
    // then each triangle extruded into a prism
    int layer_stride = nx * ny;

    // Surface nodes (z=0)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            Real x = length * static_cast<Real>(i) / static_cast<Real>(nx - 1);
            Real y = -0.5f * width + width * static_cast<Real>(j) / static_cast<Real>(ny - 1);
            mesh.nodes.push_back({x, y, 0.0f});
        }
    }

    // Extrude: for each layer k=1..nz-1, create nodes at height z[k]
    for (int k = 1; k < nz; ++k) {
        Real z;
        if (growth_ratio > 1.001f) {
            z = first_height * (std::pow(growth_ratio, static_cast<Real>(k)) - 1.0f) / (growth_ratio - 1.0f);
        } else {
            z = first_height * static_cast<Real>(k);
        }
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                Real x = length * static_cast<Real>(i) / static_cast<Real>(nx - 1);
                Real y = -0.5f * width + width * static_cast<Real>(j) / static_cast<Real>(ny - 1);
                mesh.nodes.push_back({x, y, z});
            }
        }
    }

    // For each quad cell on the surface, split into 2 triangles, extrude each into a prism
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            int t0 = j * nx + i;
            int t1 = j * nx + (i + 1);
            int t2 = (j + 1) * nx + i;
            int t3 = (j + 1) * nx + (i + 1);

            for (int tri = 0; tri < 2; ++tri) {
                int a, b, c;
                if (tri == 0) { a = t0; b = t1; c = t2; }
                else { a = t1; b = t3; c = t2; }

                for (int k = 0; k < nz - 1; ++k) {
                    CfdCell cell;
                    cell.type = ElementType::PENTA6;
                    cell.node[0] = layer_stride * k + a;
                    cell.node[1] = layer_stride * k + b;
                    cell.node[2] = layer_stride * k + c;
                    cell.node[3] = layer_stride * (k + 1) + a;
                    cell.node[4] = layer_stride * (k + 1) + b;
                    cell.node[5] = layer_stride * (k + 1) + c;
                    mesh.cells.push_back(cell);
                }
            }
        }
    }

    rebuild_faces(mesh);
    return mesh;
}

void rebuild_mesh_faces(CfdMesh& mesh) {
    std::unordered_map<FaceKey, BoundaryKind, FaceKeyHash> old_boundaries;
    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::Interior) {
            FaceKey key(face.node_count, face.node[0], face.node[1], face.node[2],
                        face.node_count == 4 ? face.node[3] : -1);
            old_boundaries[key] = face.boundary;
        }
    }
    rebuild_faces(mesh);
    for (auto& face : mesh.faces) {
        if (face.boundary == BoundaryKind::Interior) continue;
        FaceKey key(face.node_count, face.node[0], face.node[1], face.node[2],
                    face.node_count == 4 ? face.node[3] : -1);
        auto it = old_boundaries.find(key);
        if (it != old_boundaries.end())
            face.boundary = it->second;
    }
}

MeshQualityReport compute_mesh_metrics(CfdMesh& mesh, bool recompute_cells) {
    MeshQualityReport report;
    report.nodes = static_cast<int>(mesh.nodes.size());
    report.cells = static_cast<int>(mesh.cells.size());
    report.faces = static_cast<int>(mesh.faces.size());
    report.min_volume = std::numeric_limits<Real>::max();
    report.max_volume = 0.0f;
    report.min_h = std::numeric_limits<Real>::max();
    report.max_h = 0.0f;
    report.min_wall_distance = std::numeric_limits<Real>::max();

    if (recompute_cells) {
        for (auto& cell : mesh.cells) {
            compute_cell_metrics(mesh, cell);
            cell.wall_distance = 0.0f;

            report.min_volume = std::min(report.min_volume, cell.volume);
            report.max_volume = std::max(report.max_volume, cell.volume);
            report.min_h = std::min(report.min_h, cell.h_min);
            report.max_h = std::max(report.max_h, cell.h_min);
        }
    }

    std::vector<std::vector<int>> cell_faces(mesh.cells.size());
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
        CfdFace& face = mesh.faces[fi];
        Vec3 a = to_vec(mesh.nodes[face.node[0]]);
        Vec3 b = to_vec(mesh.nodes[face.node[1]]);
        Vec3 c = to_vec(mesh.nodes[face.node[2]]);
        Real area_tri_val = 0.5f * norm(cross(b - a, c - a));
        face.area = area_tri_val;
        if (face.node_count == 4) {
            Vec3 d = to_vec(mesh.nodes[face.node[3]]);
            face.area = area_quad(a, b, c, d);
        }

        if (face.node_count == 3) {
            face.cx = (a.x + b.x + c.x) / 3.0f;
            face.cy = (a.y + b.y + c.y) / 3.0f;
            face.cz = (a.z + b.z + c.z) / 3.0f;
        } else {
            Vec3 d = to_vec(mesh.nodes[face.node[3]]);
            face.cx = (a.x + b.x + c.x + d.x) / 4.0f;
            face.cy = (a.y + b.y + c.y + d.y) / 4.0f;
            face.cz = (a.z + b.z + c.z + d.z) / 4.0f;
        }

        if (face.area > 0.0f) {
            // Face normal via cross product of first two edges
            Vec3 nf = cross(b - a, c - a);
            Real nlen = norm(nf);
            if (nlen > 0.0f) {
                face.nx = nf.x / nlen;
                face.ny = nf.y / nlen;
                face.nz = nf.z / nlen;
            }
        }

        // Orient normal outward from left cell
        Vec3 fc{face.cx, face.cy, face.cz};
        const CfdCell& left = mesh.cells[face.left_cell];
        Vec3 lc{left.cx, left.cy, left.cz};
        if (dot(fc - lc, {face.nx, face.ny, face.nz}) < 0.0f) {
            face.nx = -face.nx;
            face.ny = -face.ny;
            face.nz = -face.nz;
        }

        if (face.left_cell >= 0) cell_faces[face.left_cell].push_back(fi);
        if (face.right_cell >= 0) cell_faces[face.right_cell].push_back(fi);

        switch (face.boundary) {
            case BoundaryKind::Interior: report.interior_faces++; break;
            case BoundaryKind::Farfield: report.farfield_faces++; break;
            case BoundaryKind::SlipWall: report.slip_wall_faces++; break;
            case BoundaryKind::NoSlipWall: report.no_slip_wall_faces++; break;
            case BoundaryKind::Symmetry: report.symmetry_faces++; break;
        }
    }

    std::vector<int> wall_face_ids;
    wall_face_ids.reserve(mesh.faces.size() / 8 + 8);
    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
        const CfdFace& face = mesh.faces[fi];
        if (face.boundary == BoundaryKind::SlipWall || face.boundary == BoundaryKind::NoSlipWall)
            wall_face_ids.push_back(fi);
    }

    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        auto& cell = mesh.cells[ci];
        cell.first_face = cell_faces[ci].empty() ? 0 : cell_faces[ci][0];
        cell.face_count = static_cast<int>(cell_faces[ci].size());
        // Distance to EVERY wall face (not only cell-adjacent). Interior BL cells
        // must see d>0 or SA destruction vanishes and residual cannot drop.
        Real min_wall = std::numeric_limits<Real>::max();
        Vec3 cc{cell.cx, cell.cy, cell.cz};
        for (int fi : wall_face_ids) {
            const CfdFace& face = mesh.faces[fi];
            Vec3 fc{face.cx, face.cy, face.cz};
            Vec3 n{face.nx, face.ny, face.nz};
            Real distance = std::fabs(dot(cc - fc, n));
            if (distance < min_wall) min_wall = distance;
        }
        if (min_wall < std::numeric_limits<Real>::max()) {
            cell.wall_distance = min_wall;
            report.min_wall_distance = std::min(report.min_wall_distance, min_wall);
        }
    }

    if (report.cells == 0) {
        report.min_volume = 0.0f;
        report.min_h = 0.0f;
    }
    if (report.min_wall_distance == std::numeric_limits<Real>::max()) report.min_wall_distance = 0.0f;

    report.valid = validate_mesh(mesh, &report);
    return report;
}

bool validate_mesh(const CfdMesh& mesh, MeshQualityReport* report) {
    if (mesh.nodes.empty() || mesh.cells.empty() || mesh.faces.empty()) {
        if (report) {
            report->valid = false;
            report->message = "mesh is empty";
        }
        return false;
    }

    for (const auto& cell : mesh.cells) {
        if (cell.volume <= 0.0f || !std::isfinite(cell.volume)) {
            if (report) {
                report->valid = false;
                report->message = "cell has non-positive volume";
            }
            return false;
        }
        if (cell.h_min <= 0.0f || !std::isfinite(cell.h_min)) {
            if (report) {
                report->valid = false;
                report->message = "cell has invalid h_min";
            }
            return false;
        }
    }

    for (const auto& face : mesh.faces) {
        if (face.area <= 0.0f || !std::isfinite(face.area)) {
            if (report) {
                report->valid = false;
                report->message = "face has invalid area";
            }
            return false;
        }
        if (face.boundary == BoundaryKind::Interior && face.right_cell < 0) {
            if (report) {
                report->valid = false;
                report->message = "interior face has no right cell";
            }
            return false;
        }
        if (face.boundary != BoundaryKind::Interior && face.right_cell >= 0) {
            if (report) {
                report->valid = false;
                report->message = "boundary face has right cell";
            }
            return false;
        }
    }

    if (report) {
        report->valid = true;
        report->message.clear();
    }
    return true;
}

Real boundary_area(const CfdMesh& mesh, BoundaryKind boundary) {
    Real area = 0.0f;
    for (const auto& face : mesh.faces) {
        if (face.boundary == boundary) area += face.area;
    }
    return area;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp

