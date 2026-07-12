#include "aero/cfd/mesh_gen_stl.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/element_types.hpp"
#include "aero/cfd/mesh_validator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

struct Vec3 {
    Real x, y, z;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, Real s) { return {a.x * s, a.y * s, a.z * s}; }
Vec3 operator/(Vec3 a, Real s) { return {a.x / s, a.y / s, a.z / s}; }
Real dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Real norm(Vec3 a) { return std::sqrt(dot(a, a)); }

Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

Vec3 normalize(Vec3 a) {
    Real n = norm(a);
    if (n < Real(1e-30)) return {0, 0, 0};
    return a * (1.0f / n);
}

Real volume_tet_signed(Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    return dot(b - a, cross(c - a, d - a)) / 6.0f;
}

// ---------------------------------------------------------------------------
// Triangle representation from STL
// ---------------------------------------------------------------------------
struct Tri {
    Vec3 v0, v1, v2;
    Vec3 n;
};

struct AABB {
    Vec3 bmin, bmax;
};

AABB tri_aabb(const Tri& t) {
    return {
        {std::min({t.v0.x, t.v1.x, t.v2.x}),
         std::min({t.v0.y, t.v1.y, t.v2.y}),
         std::min({t.v0.z, t.v1.z, t.v2.z})},
        {std::max({t.v0.x, t.v1.x, t.v2.x}),
         std::max({t.v0.y, t.v1.y, t.v2.y}),
         std::max({t.v0.z, t.v1.z, t.v2.z})}
    };
}

bool aabb_contains(const AABB& b, Vec3 p) {
    return p.x >= b.bmin.x && p.x <= b.bmax.x &&
           p.y >= b.bmin.y && p.y <= b.bmax.y &&
           p.z >= b.bmin.z && p.z <= b.bmax.z;
}

AABB aabb_union(const AABB& a, const AABB& b) {
    return {
        {std::min(a.bmin.x, b.bmin.x), std::min(a.bmin.y, b.bmin.y), std::min(a.bmin.z, b.bmin.z)},
        {std::max(a.bmax.x, b.bmax.x), std::max(a.bmax.y, b.bmax.y), std::max(a.bmax.z, b.bmax.z)}
    };
}

Real aabb_surface_area(const AABB& b) {
    Vec3 d = b.bmax - b.bmin;
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

int aabb_longest_axis(const AABB& b) {
    Vec3 d = b.bmax - b.bmin;
    if (d.x >= d.y && d.x >= d.z) return 0;
    if (d.y >= d.z) return 1;
    return 2;
}

// ---------------------------------------------------------------------------
// BVH: simple median-split bounding volume hierarchy
// ---------------------------------------------------------------------------
struct BVHNode {
    AABB box;
    int left = -1;
    int right = -1;
    int tri_start = -1;
    int tri_count = 0;
};

class BVH {
public:
    void build(const std::vector<Tri>& tris) {
        tris_ = &tris;
        indices_.resize(tris.size());
        for (size_t i = 0; i < tris.size(); ++i)
            indices_[i] = static_cast<int>(i);
        nodes_.reserve(tris.size() * 2);
        build_node(0, static_cast<int>(tris.size()), 0);
    }

    // Closest unsigned distance from p to any triangle surface
    Real closest_distance(Vec3 p) const {
        Real best = std::numeric_limits<Real>::max();
        closest_dist_node(0, p, best);
        return best;
    }

    // Ray-triangle intersection: returns true if ray from origin hits any triangle before t_max
    bool ray_intersect(Vec3 origin, Vec3 dir, Real t_max = std::numeric_limits<Real>::max()) const {
        Real dummy;
        return ray_intersect_node(0, origin, dir, t_max, dummy);
    }

    // Signed distance: closest distance with sign from ray-casting parity
    Real signed_distance(Vec3 p, int* ray_dir = nullptr) const {
        Real d = closest_distance(p);

        Vec3 dir;
        if (ray_dir) {
            int a = *ray_dir % 3;
            if (a == 0) dir = {1, 0, 0};
            else if (a == 1) dir = {0, 1, 0};
            else dir = {0, 0, 1};
        } else {
            dir = {1, 0, 0};
        }

        int hits = 0;
        Real t_max = 1e10f;
        Real t_hit;
        // Shoot ray and count intersections
        // We do a simple traversal
        count_ray_hits_node(0, p, dir, t_max, hits);

        // If odd number of intersections, point is inside
        if (hits % 2 == 1)
            d = -d;
        return d;
    }

private:
    const std::vector<Tri>* tris_ = nullptr;
    std::vector<int> indices_;
    std::vector<BVHNode> nodes_;

    int build_node(int start, int count, int depth) {
        BVHNode node;
        node.tri_start = start;
        node.tri_count = count;

        AABB box;
        bool first = true;
        for (int i = 0; i < count; ++i) {
            const auto& t = (*tris_)[indices_[start + i]];
            AABB tb = tri_aabb(t);
            if (first) { box = tb; first = false; }
            else { box = aabb_union(box, tb); }
        }
        node.box = box;

        int idx = static_cast<int>(nodes_.size());
        nodes_.push_back(node);

        if (count <= 4) {
            nodes_[idx].left = -1;
            nodes_[idx].right = -1;
            return idx;
        }

        int axis = aabb_longest_axis(box);
        Real mid = ((&box.bmin.x)[axis] + (&box.bmax.x)[axis]) * 0.5f;

        // Partition by midpoint
        int split = start;
        for (int i = start; i < start + count; ++i) {
            const auto& t = (*tris_)[indices_[i]];
            Vec3 c = (t.v0 + t.v1 + t.v2) / 3.0f;
            if ((&c.x)[axis] < mid) {
                std::swap(indices_[split], indices_[i]);
                ++split;
            }
        }

        // If partition failed (all on one side), split in half
        if (split == start || split == start + count)
            split = start + count / 2;

        nodes_[idx].left = build_node(start, split - start, depth + 1);
        nodes_[idx].right = build_node(split, start + count - split, depth + 1);
        return idx;
    }

    void closest_dist_node(int idx, Vec3 p, Real& best) const {
        const auto& node = nodes_[idx];
        Real d_box = box_dist(node.box, p);
        if (d_box >= best) return;

        if (node.left < 0) {
            for (int i = 0; i < node.tri_count; ++i) {
                const auto& t = (*tris_)[indices_[node.tri_start + i]];
                Real d = point_tri_distance(p, t.v0, t.v1, t.v2);
                if (d < best) best = d;
            }
            return;
        }

        Real dl = box_dist(nodes_[node.left].box, p);
        Real dr = box_dist(nodes_[node.right].box, p);
        int first = (dl <= dr) ? node.left : node.right;
        int second = (dl <= dr) ? node.right : node.left;
        closest_dist_node(first, p, best);
        closest_dist_node(second, p, best);
    }

    bool ray_intersect_node(int idx, Vec3 origin, Vec3 dir, Real t_max, Real& t_hit) const {
        const auto& node = nodes_[idx];
        if (!ray_aabb_intersect(origin, dir, node.box, 0, t_max))
            return false;

        if (node.left < 0) {
            bool hit = false;
            for (int i = 0; i < node.tri_count; ++i) {
                const auto& t = (*tris_)[indices_[node.tri_start + i]];
                Real t_tri;
                if (ray_tri_intersect(origin, dir, t.v0, t.v1, t.v2, t_tri) && t_tri < t_max) {
                    t_max = t_tri;
                    t_hit = t_tri;
                    hit = true;
                }
            }
            return hit;
        }

        bool hit_left = ray_intersect_node(node.left, origin, dir, t_max, t_hit);
        if (hit_left) t_max = t_hit;
        bool hit_right = ray_intersect_node(node.right, origin, dir, t_max, t_hit);
        if (hit_right) return true;
        return hit_left;
    }

    void count_ray_hits_node(int idx, Vec3 origin, Vec3 dir, Real t_max, int& hits) const {
        const auto& node = nodes_[idx];
        if (!ray_aabb_intersect(origin, dir, node.box, 0, t_max))
            return;

        if (node.left < 0) {
            for (int i = 0; i < node.tri_count; ++i) {
                const auto& t = (*tris_)[indices_[node.tri_start + i]];
                Real t_tri;
                if (ray_tri_intersect(origin, dir, t.v0, t.v1, t.v2, t_tri)) {
                    // Count intersection if it's in front and not at the origin
                    if (t_tri > 1e-10f)
                        ++hits;
                }
            }
            return;
        }

        count_ray_hits_node(node.left, origin, dir, t_max, hits);
        count_ray_hits_node(node.right, origin, dir, t_max, hits);
    }

    static Real box_dist(const AABB& b, Vec3 p) {
        Real dx = std::max({b.bmin.x - p.x, p.x - b.bmax.x, Real(0)});
        Real dy = std::max({b.bmin.y - p.y, p.y - b.bmax.y, Real(0)});
        Real dz = std::max({b.bmin.z - p.z, p.z - b.bmax.z, Real(0)});
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    static bool ray_aabb_intersect(Vec3 origin, Vec3 dir, const AABB& box, Real t_min, Real t_max) {
        for (int a = 0; a < 3; ++a) {
            Real dir_comp = (&dir.x)[a];
            if (std::fabs(dir_comp) < Real(1e-30)) {
                Real o = (&origin.x)[a];
                if (o < (&box.bmin.x)[a] || o > (&box.bmax.x)[a]) return false;
                continue;
            }
            Real inv_d = 1.0f / dir_comp;
            Real t0 = ((&box.bmin.x)[a] - (&origin.x)[a]) * inv_d;
            Real t1 = ((&box.bmax.x)[a] - (&origin.x)[a]) * inv_d;
            if (inv_d < 0) std::swap(t0, t1);
            t_min = std::max(t_min, t0);
            t_max = std::min(t_max, t1);
            if (t_max < t_min) return false;
        }
        return true;
    }

    // Moller-Trumbore ray-triangle intersection
    static bool ray_tri_intersect(Vec3 origin, Vec3 dir, Vec3 v0, Vec3 v1, Vec3 v2, Real& t) {
        Vec3 e1 = v1 - v0;
        Vec3 e2 = v2 - v0;
        Vec3 pvec = cross(dir, e2);
        Real det = dot(e1, pvec);
        if (std::fabs(det) < std::numeric_limits<Real>::epsilon()) return false;
        Real inv_det = 1.0f / det;
        Vec3 tvec = origin - v0;
        Real u = dot(tvec, pvec) * inv_det;
        if (u < 0 || u > 1) return false;
        Vec3 qvec = cross(tvec, e1);
        Real v = dot(dir, qvec) * inv_det;
        if (v < 0 || u + v > 1) return false;
        t = dot(e2, qvec) * inv_det;
        if (t < 0) return false;
        return true;
    }

    // Distance from point p to triangle (v0,v1,v2)
    static Real point_tri_distance(Vec3 p, Vec3 v0, Vec3 v1, Vec3 v2) {
        Vec3 e0 = v1 - v0;
        Vec3 e1 = v2 - v0;
        Vec3 n = cross(e0, e1);
        Real n_len = norm(n);
        if (n_len < std::numeric_limits<Real>::min())
            return norm(p - v0); // degenerate triangle

        n = n / n_len;

        // Project p onto triangle plane
        Real d = dot(p - v0, n);
        Vec3 proj = p - n * d;

        // Barycentric coordinates of projected point
        Vec3 vp = proj - v0;
        Real dot00 = dot(e0, e0);
        Real dot01 = dot(e0, e1);
        Real dot11 = dot(e1, e1);
        Real dot0p = dot(e0, vp);
        Real dot1p = dot(e1, vp);

        Real denom = dot00 * dot11 - dot01 * dot01;
        if (std::fabs(denom) < std::numeric_limits<Real>::min()) {
            // Near-degenerate triangle — distance to closest vertex/edge
            Real d0 = norm(p - v0);
            Real d1 = norm(p - v1);
            Real d2 = norm(p - v2);
            return std::min({d0, d1, d2});
        }

        Real u = (dot11 * dot0p - dot01 * dot1p) / denom;
        Real v = (dot00 * dot1p - dot01 * dot0p) / denom;
        Real w = 1 - u - v;

        if (u >= 0 && v >= 0 && w >= 0) {
            // Inside triangle
            return std::fabs(d);
        }

        // Outside triangle — distance to closest edge or vertex
        Real d0 = norm(p - v0);
        Real d1 = norm(p - v1);
        Real d2 = norm(p - v2);
        return std::min({d0, d1, d2});
    }
};

// ---------------------------------------------------------------------------
// STL parsing (binary and ASCII)
// ---------------------------------------------------------------------------
struct StlHeader {
    char header[80];
    uint32_t num_triangles;
};

#pragma pack(push, 1)
struct StlTriangle {
    float n[3];
    float v0[3];
    float v1[3];
    float v2[3];
    uint16_t attr;
};
#pragma pack(pop)

// Detect STL format without consuming the header:
// Returns: 0 = cannot open, 1 = ASCII, 2 = binary
int detect_stl_format(const std::string& path, StlHeader& hdr_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    f.read(reinterpret_cast<char*>(&hdr_out), sizeof(hdr_out));
    if (!f) return 1; // short file = probably ASCII

    // Check for ASCII STL: starts with "solid " (6 bytes including trailing space)
    if (std::memcmp(hdr_out.header, "solid ", 6) == 0) {
        return 1; // unequivocally ASCII
    }
    // Binary header may also start with "solid" (no trailing space) — check file size
    if (std::memcmp(hdr_out.header, "solid", 5) == 0) {
        f.seekg(0, std::ios::end);
        std::streampos size = f.tellg();
        uint64_t expected = static_cast<uint64_t>(84) + static_cast<uint64_t>(hdr_out.num_triangles) * 50;
        uint64_t usz = static_cast<uint64_t>(size);
        if (usz != expected)
            return 1; // size doesn't match binary formula — treat as ASCII
        // Size matches binary formula exactly; still verify num_triangles is sane
        if (hdr_out.num_triangles > 0 && hdr_out.num_triangles < 100000000)
            return 2; // binary
        return 1; // treat as ASCII
    }
    return 2; // binary
}

std::vector<Tri> parse_stl_ascii(const std::string& path, std::string* error) {
    std::ifstream f(path);
    if (!f) {
        if (error) *error = "Cannot open STL file: " + path;
        return {};
    }

    std::vector<Tri> tris;
    std::string line;
    Vec3 v0, v1, v2, n;
    bool reading_vertex = false;
    int vertex_count = 0;

    auto parse_vec3 = [](const std::string& s, size_t pos) -> Vec3 {
        // Skip keyword before coordinates
        while (pos < s.size() && s[pos] != ' ' && s[pos] != '\t') ++pos;
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) ++pos;
        Vec3 r;
        r.x = static_cast<Real>(std::stod(s.substr(pos)));
        while (pos < s.size() && s[pos] != ' ' && s[pos] != '\t') ++pos;
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) ++pos;
        r.y = static_cast<Real>(std::stod(s.substr(pos)));
        while (pos < s.size() && s[pos] != ' ' && s[pos] != '\t') ++pos;
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) ++pos;
        r.z = static_cast<Real>(std::stod(s.substr(pos)));
        return r;
    };

    while (std::getline(f, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        if (line.compare(0, 6, "facet ") == 0 && line.find("normal") != std::string::npos) {
            // If we were accumulating a facet and didn't get 3 vertices, it's malformed
            if (vertex_count != 0 && vertex_count != 3) {
                if (error) *error = "Malformed ASCII STL: facet with " + std::to_string(vertex_count) + " vertices";
                return {};
            }
            size_t pos = line.find("normal");
            if (pos != std::string::npos) {
                n = parse_vec3(line, pos + 6);
            }
            vertex_count = 0;
        } else if (line.compare(0, 7, "vertex ") == 0) {
            Vec3 v = parse_vec3(line, 0);
            if (vertex_count == 0) v0 = v;
            else if (vertex_count == 1) v1 = v;
            else if (vertex_count == 2) v2 = v;
            ++vertex_count;

            if (vertex_count == 3) {
                Vec3 geom_n = normalize(cross(v1 - v0, v2 - v0));
                Real ndot = dot(n, geom_n);
                // Use recomputed normal with sign check; flip if file normal is backward
                if (ndot < 0) geom_n = geom_n * Real(-1);
                tris.push_back({v0, v1, v2, geom_n});
                vertex_count = 0;
            }
        } else if (line.compare(0, 5, "solid") == 0) {
            // Start of a solid — reset
            if (vertex_count != 0 && vertex_count != 3) {
                if (error) *error = "Malformed ASCII STL: solid with " + std::to_string(vertex_count) + " vertices";
                return {};
            }
            vertex_count = 0;
        }
    }

    if (tris.empty()) {
        if (error) *error = "No triangles found in ASCII STL: " + path;
    }
    return tris;
}

std::vector<Tri> parse_stl_binary(const std::string& path, std::string* error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (error) *error = "Cannot open binary STL: " + path;
        return {};
    }

    StlHeader h;
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f) {
        if (error) *error = "Failed to read STL header";
        return {};
    }

    uint32_t n = h.num_triangles;
    if (n > 100000000) {
        if (error) *error = "Suspiciously large triangle count in STL: " + std::to_string(n);
        return {};
    }

    std::vector<Tri> tris;
    tris.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        StlTriangle st;
        f.read(reinterpret_cast<char*>(&st), sizeof(st));
        if (!f) {
            if (error) *error = "Failed to read triangle " + std::to_string(i) + " from binary STL";
            return {};
        }

        Tri t;
        t.v0 = {static_cast<Real>(st.v0[0]), static_cast<Real>(st.v0[1]), static_cast<Real>(st.v0[2])};
        t.v1 = {static_cast<Real>(st.v1[0]), static_cast<Real>(st.v1[1]), static_cast<Real>(st.v1[2])};
        t.v2 = {static_cast<Real>(st.v2[0]), static_cast<Real>(st.v2[1]), static_cast<Real>(st.v2[2])};

        // Recompute normal from geometry for robustness
        Vec3 e1 = t.v1 - t.v0;
        Vec3 e2 = t.v2 - t.v0;
        t.n = normalize(cross(e1, e2));

        tris.push_back(t);
    }

    return tris;
}

std::vector<Tri> parse_stl(const std::string& path, std::string* error) {
    StlHeader hdr;
    int fmt = detect_stl_format(path, hdr);
    if (fmt == 0) {
        if (error) *error = "Cannot open STL file: " + path;
        return {};
    }
    if (fmt == 1)
        return parse_stl_ascii(path, error);
    return parse_stl_binary(path, error);
}

// ---------------------------------------------------------------------------
// SDF grid on background hex mesh
// ---------------------------------------------------------------------------
struct SDFGrid {
    int nx, ny, nz;
    Vec3 origin;
    Real dx, dy, dz;
    std::vector<Real> sdf; // (nx+1)*(ny+1)*(nz+1)

    int stride_y() const { return nx + 1; }
    int stride_z() const { return stride_y() * (ny + 1); }

    Real& at(int i, int j, int k) {
        return sdf[k * stride_z() + j * stride_y() + i];
    }

    const Real& at(int i, int j, int k) const {
        return sdf[k * stride_z() + j * stride_y() + i];
    }

    Vec3 grid_point(int i, int j, int k) const {
        return {origin.x + i * dx, origin.y + j * dy, origin.z + k * dz};
    }

    Vec3 cell_center(int i, int j, int k) const {
        return {origin.x + (i + 0.5f) * dx,
                origin.y + (j + 0.5f) * dy,
                origin.z + (k + 0.5f) * dz};
    }

    // 8 corners of hex cell (i,j,k)
    void cell_corners(int i, int j, int k, Vec3 c[8]) const {
        c[0] = grid_point(i, j, k);
        c[1] = grid_point(i+1, j, k);
        c[2] = grid_point(i+1, j+1, k);
        c[3] = grid_point(i, j+1, k);
        c[4] = grid_point(i, j, k+1);
        c[5] = grid_point(i+1, j, k+1);
        c[6] = grid_point(i+1, j+1, k+1);
        c[7] = grid_point(i, j+1, k+1);
    }
};

void compute_sdf_grid(SDFGrid& grid, const BVH& bvh) {
    int npts = (grid.nx + 1) * (grid.ny + 1) * (grid.nz + 1);
    grid.sdf.resize(npts);

    for (int k = 0; k <= grid.nz; ++k) {
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                Vec3 p = grid.grid_point(i, j, k);
                int ray_axis = (i + j + k) % 3;
                Real d = bvh.signed_distance(p, &ray_axis);
                grid.at(i, j, k) = d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Marching Tetrahedra: clip a tet against the SDF=0 iso-surface
// ---------------------------------------------------------------------------
struct ClipOutput {
    std::vector<Vec3> cell_verts;  // output tet vertices (groups of 4)
    std::vector<Vec3> wall_verts;  // wall triangle vertices
    std::vector<int> wall_indices; // wall triangle indices (groups of 3)
};

ClipOutput clip_tet(Vec3 v[4], Real sdf[4]) {
    ClipOutput out;

    int outside[4], n_out = 0;
    int inside[4], n_in = 0;
    for (int i = 0; i < 4; ++i) {
        if (sdf[i] >= Real(0)) outside[n_out++] = i;
        else inside[n_in++] = i;
    }

    if (n_out == 0) return out;
    if (n_out == 4) {
        for (int i = 0; i < 4; ++i) out.cell_verts.push_back(v[i]);
        return out;
    }

    auto interp = [&](int a, int b) -> Vec3 {
        // a must be outside (SDF >= 0), b inside (SDF < 0)
        Real sa = sdf[a], sb = sdf[b];
        Real denom = sa - sb;
        if (std::fabs(denom) < Real(1e-30)) return (v[a] + v[b]) * 0.5f;
        Real t = sa / denom;
        return v[a] + (v[b] - v[a]) * t;
    };

    // Collect all outside vertices and edge-interpolated vertices.
    // Then tessellate by fanning from the centroid of all vertices.
    std::vector<Vec3> all_verts;
    std::vector<std::vector<int>> bounding_tris; // indices into all_verts

    for (int i = 0; i < n_out; ++i) {
        all_verts.push_back(v[outside[i]]);
    }

    if (n_out == 1) {
        int a = outside[0];
        for (int i = 0; i < n_in; ++i) {
            all_verts.push_back(interp(a, inside[i]));
        }
        // 3 bounding tris from clipped original faces + 1 iso tri = 4 tris
        // Clip of face (a,b,c): tri (a, E_ab, E_ac)
        // Clip of face (a,b,d): tri (a, E_ab, E_ad)
        // Clip of face (a,c,d): tri (a, E_ac, E_ad)
        // Iso tri: (E_ab, E_ac, E_ad)
        // all_verts = [a, E_ab, E_ac, E_ad]
        // Indices: 0=a, 1=E_ab, 2=E_ac, 3=E_ad
        // Bounding tris:
        //   outer tri (a, E_ab, E_ac): (0,1,2)
        //   outer tri (a, E_ab, E_ad): (0,1,3)
        //   outer tri (a, E_ac, E_ad): (0,2,3)
        //   iso tri (E_ab, E_ac, E_ad): (1,2,3) — WALL
        bounding_tris = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};

    } else if (n_out == 2) {
        int a = outside[0], b = outside[1];
        // For each inside vertex, interpolate edges from both outside vertices
        for (int i = 0; i < n_in; ++i) {
            all_verts.push_back(interp(a, inside[i]));
            all_verts.push_back(interp(b, inside[i]));
        }
        // all_verts = [a, b, E_ac, E_ad, E_bc, E_bd]
        // Indices: 0=a, 1=b, 2=E_ac, 3=E_ad, 4=E_bc, 5=E_bd
        // Outer quad (a, b, E_bc, E_ac) → 2 tris: (0,1,4), (0,4,2)
        // Outer quad (a, b, E_bd, E_ad) → 2 tris: (0,1,5), (0,5,3)
        // Inner tri (a, E_ac, E_ad): (0,2,3)
        // Inner tri (b, E_bc, E_bd): (1,4,5)
        // Iso quad (E_ac, E_bc, E_bd, E_ad) → 2 tris: (2,4,5), (2,5,3) — WALL
        bounding_tris = {
            {0, 1, 4}, {0, 4, 2},  // quad (a,b,E_bc,E_ac)
            {0, 1, 5}, {0, 5, 3},  // quad (a,b,E_bd,E_ad)
            {0, 2, 3},             // tri (a,E_ac,E_ad)
            {1, 4, 5},             // tri (b,E_bc,E_bd)
            {2, 4, 5}, {2, 5, 3}  // iso quad (E_ac,E_bc,E_bd,E_ad) — WALL
        };

    } else { // n_out == 3
        int in = inside[0];
        // Interpolate from inside vertex to each outside vertex
        for (int i = 0; i < n_out; ++i) {
            all_verts.push_back(interp(in, outside[i]));
        }
        // all_verts = [a, b, c, E_ia, E_ib, E_ic]
        // where a,b,c are outside vertices and E_ia etc are edge-interpolated
        // But wait, in this case the outside vertices are first.
        // Let me re-index: outside[0..2] are first, then E_interp[0..2]
        // Actually all_verts is: [out[0], out[1], out[2], E(in,out[0]), E(in,out[1]), E(in,out[2])]
        // Indices: 0=out[0], 1=out[1], 2=out[2], 3=E_i0, 4=E_i1, 5=E_i2
        //
        // Outer tri (out[0], out[1], out[2]): (0,1,2)
        // Side quads:
        //   (out[0], out[1], E_i1, E_i0) → 2 tris: (0,1,4), (0,4,3)
        //   (out[1], out[2], E_i2, E_i1) → 2 tris: (1,2,5), (1,5,4)
        //   (out[0], out[2], E_i2, E_i0) → 2 tris: (0,2,5), (0,5,3)
        // Iso tri (E_i0, E_i1, E_i2): (3,4,5) — WALL
        bounding_tris = {
            {0, 1, 2},              // outer cap
            {0, 1, 4}, {0, 4, 3},  // quad side 1
            {1, 2, 5}, {1, 5, 4},  // quad side 2
            {0, 2, 5}, {0, 5, 3},  // quad side 3
            {3, 4, 5}              // iso tri — WALL
        };
    }

    // Identify which triangles are on the iso-surface
    // They are those formed entirely from interpolated vertices (index >= n_out)
    int n_iso_tris = 0;
    for (size_t ti = 0; ti < bounding_tris.size(); ++ti) {
        const auto& tri = bounding_tris[ti];
        bool all_interp = true;
        for (int k = 0; k < 3; ++k) {
            if (tri[k] < n_out) { all_interp = false; break; }
        }
        if (all_interp) ++n_iso_tris;
    }

    // Compute centroid of all vertices
    Vec3 cen{0, 0, 0};
    for (const auto& pv : all_verts) cen = cen + pv;
    cen = cen / static_cast<Real>(all_verts.size());
    int cen_idx = static_cast<int>(all_verts.size());
    all_verts.push_back(cen);

    // Generate tets from centroid to each bounding triangle
    for (size_t ti = 0; ti < bounding_tris.size(); ++ti) {
        const auto& tri = bounding_tris[ti];
        bool all_interp = true;
        for (int k = 0; k < 3; ++k) {
            if (tri[k] < n_out) { all_interp = false; break; }
        }
        if (all_interp) {
            // Wall face — record it, but still add the tet so the volume is filled
            int wb = static_cast<int>(out.wall_verts.size());
            out.wall_verts.push_back(all_verts[tri[0]]);
            out.wall_verts.push_back(all_verts[tri[1]]);
            out.wall_verts.push_back(all_verts[tri[2]]);
            out.wall_indices.push_back(wb);
            out.wall_indices.push_back(wb + 1);
            out.wall_indices.push_back(wb + 2);
        }
        // Volume tet: (cen, tri0, tri1, tri2)
        out.cell_verts.push_back(all_verts[cen_idx]);
        out.cell_verts.push_back(all_verts[tri[0]]);
        out.cell_verts.push_back(all_verts[tri[1]]);
        out.cell_verts.push_back(all_verts[tri[2]]);
    }

    // Fix negative volumes: swap last two vertices if volume is negative
    for (size_t ci = 0; ci + 3 < out.cell_verts.size(); ci += 4) {
        Real vol = volume_tet_signed(
            out.cell_verts[ci], out.cell_verts[ci+1],
            out.cell_verts[ci+2], out.cell_verts[ci+3]);
        if (vol < 0) {
            std::swap(out.cell_verts[ci+2], out.cell_verts[ci+3]);
        }
    }

    return out;
}

// Decompose a hex into 6 tets sharing body diagonal (0,6)
// Hex nodes: [0..7] as per CfdCell convention
// Opposite corners: 0 and 6
void hex_to_6_tets(Vec3 hex_v[8], Vec3 tet_v[6][4]) {
    static const int tets[6][4] = {
        {0, 1, 2, 6},
        {0, 2, 3, 6},
        {0, 3, 7, 6},
        {0, 7, 4, 6},
        {0, 4, 5, 6},
        {0, 5, 1, 6}
    };
    for (int t = 0; t < 6; ++t) {
        for (int i = 0; i < 4; ++i) {
            tet_v[t][i] = hex_v[tets[t][i]];
        }
    }
}

// ---------------------------------------------------------------------------
// Main mesh generation
// ---------------------------------------------------------------------------
struct GeneratedCell {
    Vec3 nodes[4]; // tet vertices (4 per tet)
    int n_nodes = 0;
};

struct WallTri {
    int i0, i1, i2;
};

} // anonymous namespace

bool generate_conformal_mesh_from_stl(
    const std::string& stl_path,
    CfdMesh& mesh,
    const StlMeshConfig& cfg,
    std::string* error)
{
    // 1. Parse STL
    std::vector<Tri> tris = parse_stl(stl_path, error);
    if (tris.empty()) {
        if (error && error->empty()) *error = "No triangles loaded from STL";
        return false;
    }

    // 2. Build BVH
    BVH bvh;
    bvh.build(tris);

    // 3. Compute STL bounding box
    AABB stl_box = tri_aabb(tris[0]);
    for (size_t i = 1; i < tris.size(); ++i) {
        stl_box = aabb_union(stl_box, tri_aabb(tris[i]));
    }

    Vec3 stl_size = stl_box.bmax - stl_box.bmin;
    Real max_dim = std::max({stl_size.x, stl_size.y, stl_size.z});

    // 4. Build background grid
    SDFGrid grid;
    grid.nx = cfg.background_n_per_dim;
    grid.ny = cfg.background_n_per_dim;
    grid.nz = cfg.background_n_per_dim;

    Real padding = max_dim * (cfg.outer_scale - 1.0f) * 0.5f;
    grid.origin = {
        stl_box.bmin.x - padding,
        stl_box.bmin.y - padding,
        stl_box.bmin.z - padding
    };
    Vec3 grid_size = stl_size + Vec3{padding * 2, padding * 2, padding * 2};
    grid.dx = grid_size.x / static_cast<Real>(grid.nx);
    grid.dy = grid_size.y / static_cast<Real>(grid.ny);
    grid.dz = grid_size.z / static_cast<Real>(grid.nz);

    // 5. Compute SDF on grid
    compute_sdf_grid(grid, bvh);

    // 6. Classify background cells and generate output
    std::vector<GeneratedCell> cells;
    std::vector<Vec3> wall_verts;
    std::vector<WallTri> wall_tris;

    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Real crn_sdf[8];
                Vec3 crn[8];
                grid.cell_corners(i, j, k, crn);
                for (int ci = 0; ci < 8; ++ci) {
                    crn_sdf[ci] = grid.at(
                        i + (ci == 0 || ci == 3 || ci == 4 || ci == 7 ? 0 : 1),
                        j + (ci == 0 || ci == 1 || ci == 4 || ci == 5 ? 0 : 1),
                        k + (ci == 0 || ci == 1 || ci == 2 || ci == 3 ? 0 : 1));
                }

                // Check if any corner is outside
                bool any_outside = false;
                bool any_inside = false;
                for (int ci = 0; ci < 8; ++ci) {
                    if (crn_sdf[ci] >= 0) any_outside = true;
                    else any_inside = true;
                }

                if (!any_outside) continue; // fully inside — discard

                if (!any_inside) {
                    // Fully outside — keep as hex
                    // Decompose hex into 6 tets
                    Vec3 tet_v[6][4];
                    hex_to_6_tets(crn, tet_v);
                    for (int t = 0; t < 6; ++t) {
                        GeneratedCell gc;
                        for (int nv = 0; nv < 4; ++nv) gc.nodes[nv] = tet_v[t][nv];
                        gc.n_nodes = 4;
                        // Fix negative volumes like clip_tet does
                        Real vol = volume_tet_signed(
                            gc.nodes[0], gc.nodes[1], gc.nodes[2], gc.nodes[3]);
                        if (vol < 0) {
                            std::swap(gc.nodes[2], gc.nodes[3]);
                        }
                        cells.push_back(gc);
                    }
                } else {
                    // Boundary cell — clip each tet
                    Vec3 tet_v[6][4];
                    hex_to_6_tets(crn, tet_v);

                    for (int t = 0; t < 6; ++t) {
                        Real tet_sdf[4];
                        // Map tet vertices back to SDF grid values
                        // Tet vertices are a subset of the 8 hex corners
                        // We need to find which grid indices each tet vertex corresponds to
                        for (int nv = 0; nv < 4; ++nv) {
                            // Find the original corner index by matching positions
                            Vec3 tv = tet_v[t][nv];
                            // Determine grid coordinates from the position
                            int gi = static_cast<int>((tv.x - grid.origin.x) / grid.dx + 0.5f);
                            int gj = static_cast<int>((tv.y - grid.origin.y) / grid.dy + 0.5f);
                            int gk = static_cast<int>((tv.z - grid.origin.z) / grid.dz + 0.5f);
                            gi = std::clamp(gi, 0, grid.nx);
                            gj = std::clamp(gj, 0, grid.ny);
                            gk = std::clamp(gk, 0, grid.nz);
                            tet_sdf[nv] = grid.at(gi, gj, gk);
                        }

                        ClipOutput clip = clip_tet(tet_v[t], tet_sdf);

                        // Add clipped cells
                        for (size_t ci = 0; ci < clip.cell_verts.size(); ci += 4) {
                            if (ci + 3 < clip.cell_verts.size()) {
                                GeneratedCell gc;
                                gc.nodes[0] = clip.cell_verts[ci];
                                gc.nodes[1] = clip.cell_verts[ci + 1];
                                gc.nodes[2] = clip.cell_verts[ci + 2];
                                gc.nodes[3] = clip.cell_verts[ci + 3];
                                gc.n_nodes = 4;
                                cells.push_back(gc);
                            }
                        }

                        // Add wall triangles, filtering degenerates
                        for (size_t ti = 0; ti + 2 < clip.wall_indices.size(); ti += 3) {
                            int wi0 = static_cast<int>(wall_verts.size());
                            wall_verts.push_back(clip.wall_verts[clip.wall_indices[ti]]);
                            wall_verts.push_back(clip.wall_verts[clip.wall_indices[ti + 1]]);
                            wall_verts.push_back(clip.wall_verts[clip.wall_indices[ti + 2]]);
                            Vec3 wv0 = wall_verts[wi0], wv1 = wall_verts[wi0+1], wv2 = wall_verts[wi0+2];
                            Vec3 e01 = wv1 - wv0, e02 = wv2 - wv0;
                            if (norm(cross(e01, e02)) < Real(1e-20)) {
                                wall_verts.resize(wi0);
                                continue;
                            }
                            WallTri wt;
                            wt.i0 = wi0; wt.i1 = wi0 + 1; wt.i2 = wi0 + 2;
                            wall_tris.push_back(wt);
                        }
                    }
                }

                if (cells.size() > cfg.max_cells) {
                    if (error) *error = "Exceeded max_cells limit (" + std::to_string(cfg.max_cells) + ")";
                    return false;
                }
            }
        }
    }

    if (cells.empty()) {
        if (error) *error = "No cells generated — STL may cover entire background grid";
        return false;
    }

    // 7. Build CfdMesh from generated cells
    // Build node list from unique positions
    struct NodeRef {
        int idx;
        Vec3 pos;
    };
    std::vector<NodeRef> unique_nodes;
    auto find_or_add_node = [&](Vec3 p) -> int {
        Real scale = std::max({std::fabs(p.x), std::fabs(p.y), std::fabs(p.z), Real(1)});
        Real eps = 1e-8f * scale;
        for (size_t i = 0; i < unique_nodes.size(); ++i) {
            Vec3 d = unique_nodes[i].pos - p;
            if (std::fabs(d.x) < eps && std::fabs(d.y) < eps && std::fabs(d.z) < eps)
                return unique_nodes[i].idx;
        }
        int idx = static_cast<int>(unique_nodes.size());
        unique_nodes.push_back({idx, p});
        return idx;
    };

    mesh.nodes.clear();
    mesh.cells.clear();
    mesh.faces.clear();

    for (const auto& gc : cells) {
        CfdCell cell;
        cell.type = ElementType::TET4;
        for (int i = 0; i < 4; ++i) {
            cell.node[i] = find_or_add_node(gc.nodes[i]);
        }
        mesh.cells.push_back(cell);
    }

    // Add nodes to mesh
    for (const auto& nr : unique_nodes) {
        CfdNode n;
        n.x = nr.pos.x;
        n.y = nr.pos.y;
        n.z = nr.pos.z;
        mesh.nodes.push_back(n);
    }

    // Rebuild faces from cell connectivity
    rebuild_mesh_faces(mesh);

    // Classify boundary faces: wall faces are those on the iso-surface
    // Use a very tight threshold based on grid spacing to avoid mis-classifying
    // cap/side faces of the clipped region as wall faces.
    Real min_spacing = std::min({grid.dx, grid.dy, grid.dz});
    Real wall_dist_threshold = std::max(min_spacing * 0.01f, Real(1e-12f));

    for (auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::Interior) {
            Vec3 fc{face.cx, face.cy, face.cz};
            Real d = bvh.closest_distance(fc);
            if (d < wall_dist_threshold) {
                face.boundary = BoundaryKind::NoSlipWall;
            }
        }
    }

    // Recompute metrics with correct boundary classification (skip cell recompute — quality_detail handles it)
    compute_mesh_metrics(mesh, false);

    // Full quality check including negative Jacobians
    MeshQualityReport report = compute_mesh_quality_detail(mesh);

    if (report.negative_jacobian_count > 0) {
        if (error) {
            *error = "Generated mesh has " + std::to_string(report.negative_jacobian_count)
                   + " negative Jacobian cells";
        }
        return false;
    }

    if (report.min_volume <= 0) {
        if (error) {
            *error = "Generated mesh has non-positive minimum volume";
        }
        return false;
    }

    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
