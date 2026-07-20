#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/amr_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

constexpr int MAX_CHILDREN = 8;

struct Vec3 {
    Real x, y, z;
};

Vec3 to_vec(const CfdNode& n) { return {n.x, n.y, n.z}; }
Vec3 lerp(const Vec3& a, const Vec3& b) { return {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f, (a.z + b.z) * 0.5f}; }

struct EdgeKey {
    int v[2];
    EdgeKey(int a, int b) {
        v[0] = (a < b) ? a : b;
        v[1] = (a < b) ? b : a;
    }
    bool operator==(const EdgeKey& o) const { return v[0] == o.v[0] && v[1] == o.v[1]; }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& k) const {
        return static_cast<std::size_t>(k.v[0]) ^ (static_cast<std::size_t>(k.v[1]) << 16);
    }
};

int get_or_create_midpoint(int a, int b, const std::vector<CfdNode>& old_nodes,
                           std::vector<CfdNode>& new_nodes,
                           std::unordered_map<EdgeKey, int, EdgeKeyHash>& edge_map) {
    EdgeKey key(a, b);
    auto it = edge_map.find(key);
    if (it != edge_map.end()) return it->second;
    Vec3 pa = to_vec(old_nodes[a]);
    Vec3 pb = to_vec(old_nodes[b]);
    Vec3 pm = lerp(pa, pb);
    CfdNode m;
    m.x = pm.x; m.y = pm.y; m.z = pm.z;
    int idx = static_cast<int>(new_nodes.size());
    new_nodes.push_back(m);
    edge_map[key] = idx;
    return idx;
}

struct TetLocal {
    int n[4];
};

// Refine a tet into 8 sub-tets (regular refinement by edge bisection).
// All 8 children have positive volume if the parent has positive volume.
// Pattern (Bey 1995 / standard all-similar refinement):
//   Corner tets: each retains one original vertex + 3 edge midpoints.
//   Interior: the central octahedron split into 4 tets.
void refine_tet4(const CfdCell& parent, int parent_id,
                 const std::vector<CfdNode>& old_nodes,
                 std::vector<CfdNode>& new_nodes,
                 std::vector<CfdCell>& new_cells,
                 std::unordered_map<EdgeKey, int, EdgeKeyHash>& edge_map,
                 std::vector<RefinementRecord>& records) {
    int n0 = parent.node[0], n1 = parent.node[1];
    int n2 = parent.node[2], n3 = parent.node[3];

    int m01 = get_or_create_midpoint(n0, n1, old_nodes, new_nodes, edge_map);
    int m02 = get_or_create_midpoint(n0, n2, old_nodes, new_nodes, edge_map);
    int m03 = get_or_create_midpoint(n0, n3, old_nodes, new_nodes, edge_map);
    int m12 = get_or_create_midpoint(n1, n2, old_nodes, new_nodes, edge_map);
    int m13 = get_or_create_midpoint(n1, n3, old_nodes, new_nodes, edge_map);
    int m23 = get_or_create_midpoint(n2, n3, old_nodes, new_nodes, edge_map);

    TetLocal children[MAX_CHILDREN] = {
        {n0, m01, m02, m03},
        {n1, m01, m13, m12},
        {n2, m02, m12, m23},
        {n3, m03, m23, m13},
        {m01, m12, m02, m03},
        {m01, m12, m13, m03},
        {m02, m03, m12, m23},
        {m03, m13, m12, m23}
    };

    int child_level = parent.refinement_level + 1;
    RefinementRecord rec;
    rec.parent_cell_id = parent_id;
    rec.parent_level = parent.refinement_level;
    rec.parent_type = ElementType::TET4;
    rec.parent_face_count = 4;
    for (int i = 0; i < 4; ++i) rec.parent_node[i] = parent.node[i];

    for (int i = 0; i < MAX_CHILDREN; ++i) {
        CfdCell child;
        child.type = ElementType::TET4;
        child.node[0] = children[i].n[0];
        child.node[1] = children[i].n[1];
        child.node[2] = children[i].n[2];
        child.node[3] = children[i].n[3];
        child.refinement_level = child_level;
        child.parent_id = parent_id;
        int child_id = static_cast<int>(new_cells.size());
        new_cells.push_back(child);
        rec.child_cell_ids[i] = child_id;
    }
    rec.n_children = MAX_CHILDREN;
    records.push_back(rec);
}

// 27-node hex refinement stencil.
// P[i][j][k] for i,j,k in {0,1,2}, where:
//   P[0][0][0] = n0, P[2][0][0] = n1, P[2][2][0] = n2, P[0][2][0] = n3
//   P[0][0][2] = n4, P[2][0][2] = n5, P[2][2][2] = n6, P[0][2][2] = n7
//   P[1][*][*] = edge midpoints, face centers, cell center
static void build_hex27_stencil(int st[3][3][3], const int n[8],
                                const std::vector<CfdNode>& old_nodes,
                                std::vector<CfdNode>& new_nodes,
                                std::unordered_map<EdgeKey, int, EdgeKeyHash>& edge_map) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                st[i][j][k] = -1;

    // Corner mapping: hex local node -> stencil (i,j,k)
    //   n0 = ( -x, -y, -z)  n1 = (+x, -y, -z)
    //   n2 = ( +x, +y, -z)  n3 = (-x, +y, -z)
    //   n4 = ( -x, -y, +z)  n5 = (+x, -y, +z)
    //   n6 = ( +x, +y, +z)  n7 = (-x, +y, +z)
    struct { int i, j, k; } corner_st[8] = {
        {0,0,0}, {2,0,0}, {2,2,0}, {0,2,0},
        {0,0,2}, {2,0,2}, {2,2,2}, {0,2,2}
    };
    for (int h = 0; h < 8; ++h)
        st[corner_st[h].i][corner_st[h].j][corner_st[h].k] = n[h];

    // Edge midpoints (12) — shared between adjacent hexes.
    // All edge midpoints use ORIGINAL hex node indices (n[0]..n[7]) so
    // get_or_create_midpoint indexes validly into old_nodes.
    auto set_mid_edge = [&](int i, int j, int k, int a, int b) {
        if (st[i][j][k] < 0)
            st[i][j][k] = get_or_create_midpoint(n[a], n[b], old_nodes, new_nodes, edge_map);
    };
    set_mid_edge(1,0,0, 0,1);
    set_mid_edge(1,2,0, 3,2);
    set_mid_edge(1,0,2, 4,5);
    set_mid_edge(1,2,2, 7,6);
    set_mid_edge(0,1,0, 0,3);
    set_mid_edge(2,1,0, 1,2);
    set_mid_edge(0,1,2, 4,7);
    set_mid_edge(2,1,2, 5,6);
    set_mid_edge(0,0,1, 0,4);
    set_mid_edge(2,0,1, 1,5);
    set_mid_edge(2,2,1, 2,6);
    set_mid_edge(0,2,1, 3,7);

    // Using get_or_create_midpoint ensures shared faces get the same fc node.
    auto set_mid_face = [&](int i, int j, int k, int a, int b) {
        if (st[i][j][k] < 0)
            st[i][j][k] = get_or_create_midpoint(n[a], n[b], old_nodes, new_nodes, edge_map);
    };
    set_mid_face(1,1,0, 0,2);   // face -z: diagonal n0-n2
    set_mid_face(1,1,2, 4,6);   // face +z: diagonal n4-n6
    set_mid_face(1,0,1, 0,5);   // face -y: diagonal n0-n5
    set_mid_face(1,2,1, 3,6);   // face +y: diagonal n3-n6
    set_mid_face(0,1,1, 0,7);   // face -x: diagonal n0-n7
    set_mid_face(2,1,1, 1,6);   // face +x: diagonal n1-n6

    // Cell center P[1][1][1] — midpoint of space diagonal n0-n6.
    // Dedup via edge_map doesn't matter here (each hex has unique center).
    if (st[1][1][1] < 0)
        st[1][1][1] = get_or_create_midpoint(n[0], n[6], old_nodes, new_nodes, edge_map);
}

void refine_hex8(const CfdCell& parent, int parent_id,
                 const std::vector<CfdNode>& old_nodes,
                 std::vector<CfdNode>& new_nodes,
                 std::vector<CfdCell>& new_cells,
                 std::unordered_map<EdgeKey, int, EdgeKeyHash>& edge_map,
                 std::vector<RefinementRecord>& records) {
    int n[8];
    for (int i = 0; i < 8; ++i) n[i] = parent.node[i];

    int st[3][3][3];
    build_hex27_stencil(st, n, old_nodes, new_nodes, edge_map);

    int child_level = parent.refinement_level + 1;
    RefinementRecord rec;
    rec.parent_cell_id = parent_id;
    rec.parent_level = parent.refinement_level;
    rec.parent_type = ElementType::HEX8;
    rec.parent_face_count = 6;
    for (int i = 0; i < 8; ++i) rec.parent_node[i] = parent.node[i];

    // 8 sub-hexes: each occupies (i..i+1, j..j+1, k..k+1) in stencil space
    // Sub-hex (si,sj,sk) where si,sj,sk ∈ {0,1}
    for (int si = 0; si < 2; ++si) {
        for (int sj = 0; sj < 2; ++sj) {
            for (int sk = 0; sk < 2; ++sk) {
                CfdCell child;
                child.type = ElementType::HEX8;
                child.node[0] = st[si    ][sj    ][sk    ];
                child.node[1] = st[si + 1][sj    ][sk    ];
                child.node[2] = st[si + 1][sj + 1][sk    ];
                child.node[3] = st[si    ][sj + 1][sk    ];
                child.node[4] = st[si    ][sj    ][sk + 1];
                child.node[5] = st[si + 1][sj    ][sk + 1];
                child.node[6] = st[si + 1][sj + 1][sk + 1];
                child.node[7] = st[si    ][sj + 1][sk + 1];
                child.refinement_level = child_level;
                child.parent_id = parent_id;
                int child_id = static_cast<int>(new_cells.size());
                new_cells.push_back(child);
                // Store in record in standard order (si,sj,sk) = (0,0,0)..(1,1,1)
                int idx = (sk * 4) + (sj * 2) + si;  // 0..7
                rec.child_cell_ids[idx] = child_id;
            }
        }
    }
    rec.n_children = MAX_CHILDREN;
    records.push_back(rec);
}

// ========== Anisotropic splits ==========

// HEX8 1→2 directional bisect along specified axis (0=X, 1=Y, 2=Z)
void refine_hex8_aniso(const CfdCell& parent, int parent_id, int axis,
                       const std::vector<CfdNode>& old_nodes,
                       std::vector<CfdNode>& new_nodes,
                       std::vector<CfdCell>& new_cells,
                       std::unordered_map<EdgeKey, int, EdgeKeyHash>& edge_map,
                       std::vector<RefinementRecord>& records) {
    int n[8];
    for (int i = 0; i < 8; ++i) n[i] = parent.node[i];

    int st[3][3][3];
    build_hex27_stencil(st, n, old_nodes, new_nodes, edge_map);

    int child_level = parent.refinement_level + 1;
    RefinementRecord rec;
    rec.parent_cell_id = parent_id;
    rec.parent_level = parent.refinement_level;
    rec.parent_type = ElementType::HEX8;
    rec.parent_face_count = 6;
    for (int i = 0; i < 8; ++i) rec.parent_node[i] = parent.node[i];

    // Child offsets in stencil space
    // Child 0 spans [0..1] along split axis, [0..2] along others
    // Child 1 spans [1..2] along split axis, [0..2] along others
    int child_nodes[2][8];
    if (axis == 0) {  // X-split
        child_nodes[0][0] = st[0][0][0]; child_nodes[0][1] = st[1][0][0];
        child_nodes[0][2] = st[1][2][0]; child_nodes[0][3] = st[0][2][0];
        child_nodes[0][4] = st[0][0][2]; child_nodes[0][5] = st[1][0][2];
        child_nodes[0][6] = st[1][2][2]; child_nodes[0][7] = st[0][2][2];

        child_nodes[1][0] = st[1][0][0]; child_nodes[1][1] = st[2][0][0];
        child_nodes[1][2] = st[2][2][0]; child_nodes[1][3] = st[1][2][0];
        child_nodes[1][4] = st[1][0][2]; child_nodes[1][5] = st[2][0][2];
        child_nodes[1][6] = st[2][2][2]; child_nodes[1][7] = st[1][2][2];
    } else if (axis == 1) {  // Y-split
        child_nodes[0][0] = st[0][0][0]; child_nodes[0][1] = st[2][0][0];
        child_nodes[0][2] = st[2][1][0]; child_nodes[0][3] = st[0][1][0];
        child_nodes[0][4] = st[0][0][2]; child_nodes[0][5] = st[2][0][2];
        child_nodes[0][6] = st[2][1][2]; child_nodes[0][7] = st[0][1][2];

        child_nodes[1][0] = st[0][1][0]; child_nodes[1][1] = st[2][1][0];
        child_nodes[1][2] = st[2][2][0]; child_nodes[1][3] = st[0][2][0];
        child_nodes[1][4] = st[0][1][2]; child_nodes[1][5] = st[2][1][2];
        child_nodes[1][6] = st[2][2][2]; child_nodes[1][7] = st[0][2][2];
    } else {  // Z-split
        child_nodes[0][0] = st[0][0][0]; child_nodes[0][1] = st[2][0][0];
        child_nodes[0][2] = st[2][2][0]; child_nodes[0][3] = st[0][2][0];
        child_nodes[0][4] = st[0][0][1]; child_nodes[0][5] = st[2][0][1];
        child_nodes[0][6] = st[2][2][1]; child_nodes[0][7] = st[0][2][1];

        child_nodes[1][0] = st[0][0][1]; child_nodes[1][1] = st[2][0][1];
        child_nodes[1][2] = st[2][2][1]; child_nodes[1][3] = st[0][2][1];
        child_nodes[1][4] = st[0][0][2]; child_nodes[1][5] = st[2][0][2];
        child_nodes[1][6] = st[2][2][2]; child_nodes[1][7] = st[0][2][2];
    }

    rec.n_children = 2;
    for (int i = 0; i < 2; ++i) {
        CfdCell child;
        child.type = ElementType::HEX8;
        for (int j = 0; j < 8; ++j) child.node[j] = child_nodes[i][j];
        child.refinement_level = child_level;
        child.parent_id = parent_id;
        int child_id = static_cast<int>(new_cells.size());
        new_cells.push_back(child);
        rec.child_cell_ids[i] = child_id;
    }
    records.push_back(rec);
}

// Determine which face of a TET4 to split for anisotropic refinement along direction vec d.
// Returns face_id (0..3). The face opposite vertex i means face i.
// We pick the face whose outward normal best aligns with d.
static int pick_tet_split_face(const Vec3 nodes[4], const Vec3& d) {
    int best_face = 0;
    Real best_dot = Real(-1);
    for (int face = 0; face < 4; ++face) {
        // Face face is opposite vertex face.
        // Face nodes from TET4_FACE_NODES: face 0 -> {1,2,3}, 1 -> {0,3,2}, 2 -> {0,1,3}, 3 -> {0,2,1}
        int f_id[3];
        const int* fnodes = TET4_FACE_NODES[face];
        f_id[0] = fnodes[0]; f_id[1] = fnodes[1]; f_id[2] = fnodes[2];
        const Vec3& a = nodes[f_id[0]];
        const Vec3& b = nodes[f_id[1]];
        const Vec3& c = nodes[f_id[2]];
        Vec3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
        Vec3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
        // Outward normal = cross(e1, e2), then point away from opposite vertex
        Vec3 n = {e1.y*e2.z - e1.z*e2.y,
                  e1.z*e2.x - e1.x*e2.z,
                  e1.x*e2.y - e1.y*e2.x};
        Real len = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        if (len < Real(1e-20)) continue;
        n.x /= len; n.y /= len; n.z /= len;
        // Ensure outward: opposite vertex should be on opposite side of face plane
        const Vec3& opp = nodes[face];
        Vec3 to_opp = {opp.x - a.x, opp.y - a.y, opp.z - a.z};
        Real side = n.x*to_opp.x + n.y*to_opp.y + n.z*to_opp.z;
        if (side > Real(0)) { n.x = -n.x; n.y = -n.y; n.z = -n.z; }
        Real dot = n.x*d.x + n.y*d.y + n.z*d.z;
        if (dot > best_dot) { best_dot = dot; best_face = face; }
    }
    return best_face;
}

// TET4 1→4 anisotropic: split the face most aligned with the refinement direction.
// Splits all 3 edges of the chosen face and connects to opposite vertex.
void refine_tet4_aniso(const CfdCell& parent, int parent_id,
                       AnisotropicDir dir,
                       const std::vector<CfdNode>& old_nodes,
                       std::vector<CfdNode>& new_nodes,
                       std::vector<CfdCell>& new_cells,
                       std::unordered_map<EdgeKey, int, EdgeKeyHash>& edge_map,
                       std::vector<RefinementRecord>& records) {
    int n[4] = {parent.node[0], parent.node[1], parent.node[2], parent.node[3]};
    Vec3 v[4];
    for (int i = 0; i < 4; ++i) v[i] = to_vec(old_nodes[n[i]]);

    // Direction vector from enum
    Vec3 d = {Real(0), Real(0), Real(0)};
    if (dir == AnisotropicDir::DIR_X) d.x = Real(1);
    else if (dir == AnisotropicDir::DIR_Y) d.y = Real(1);
    else if (dir == AnisotropicDir::DIR_Z) d.z = Real(1);
    else d.x = Real(1);  // fallback

    int face = pick_tet_split_face(v, d);
    // Face nodes
    const int* f = TET4_FACE_NODES[face];
    int f0 = n[f[0]], f1 = n[f[1]], f2 = n[f[2]];
    int opp = n[face];  // opposite vertex index = face index

    int m01 = get_or_create_midpoint(f0, f1, old_nodes, new_nodes, edge_map);
    int m12 = get_or_create_midpoint(f1, f2, old_nodes, new_nodes, edge_map);
    int m20 = get_or_create_midpoint(f2, f0, old_nodes, new_nodes, edge_map);

    int child_level = parent.refinement_level + 1;
    RefinementRecord rec;
    rec.parent_cell_id = parent_id;
    rec.parent_level = parent.refinement_level;
    rec.parent_type = ElementType::TET4;
    rec.parent_face_count = 4;
    for (int i = 0; i < 4; ++i) rec.parent_node[i] = parent.node[i];
    rec.n_children = 4;

    struct { int n[4]; } children[4] = {
        {f0, m01, m20, opp},
        {f1, m12, m01, opp},
        {f2, m20, m12, opp},
        {m01, m12, m20, opp}
    };

    for (int i = 0; i < 4; ++i) {
        CfdCell child;
        child.type = ElementType::TET4;
        for (int j = 0; j < 4; ++j) child.node[j] = children[i].n[j];
        child.refinement_level = child_level;
        child.parent_id = parent_id;
        int child_id = static_cast<int>(new_cells.size());
        new_cells.push_back(child);
        rec.child_cell_ids[i] = child_id;
    }
    records.push_back(rec);
}

// PENTA6 1→2 height bisect: split along prism height (0-3, 1-4, 2-5)
void refine_penta6_aniso(const CfdCell& parent, int parent_id,
                         const std::vector<CfdNode>& old_nodes,
                         std::vector<CfdNode>& new_nodes,
                         std::vector<CfdCell>& new_cells,
                         std::unordered_map<EdgeKey, int, EdgeKeyHash>& edge_map,
                         std::vector<RefinementRecord>& records) {
    int n0 = parent.node[0], n1 = parent.node[1], n2 = parent.node[2];
    int n3 = parent.node[3], n4 = parent.node[4], n5 = parent.node[5];

    int m03 = get_or_create_midpoint(n0, n3, old_nodes, new_nodes, edge_map);
    int m14 = get_or_create_midpoint(n1, n4, old_nodes, new_nodes, edge_map);
    int m25 = get_or_create_midpoint(n2, n5, old_nodes, new_nodes, edge_map);

    int child_level = parent.refinement_level + 1;
    RefinementRecord rec;
    rec.parent_cell_id = parent_id;
    rec.parent_level = parent.refinement_level;
    rec.parent_type = ElementType::PENTA6;
    rec.parent_face_count = 5;
    for (int i = 0; i < 6; ++i) rec.parent_node[i] = parent.node[i];
    rec.n_children = 2;

    // Child 0 (bottom half): {n0,n1,n2, m03,m14,m25}
    // Child 1 (top half):    {m03,m14,m25, n3,n4,n5}
    int child_nodes[2][6] = {
        {n0, n1, n2, m03, m14, m25},
        {m03, m14, m25, n3, n4, n5}
    };

    for (int i = 0; i < 2; ++i) {
        CfdCell child;
        child.type = ElementType::PENTA6;
        for (int j = 0; j < 6; ++j) child.node[j] = child_nodes[i][j];
        child.refinement_level = child_level;
        child.parent_id = parent_id;
        int child_id = static_cast<int>(new_cells.size());
        new_cells.push_back(child);
        rec.child_cell_ids[i] = child_id;
    }
    records.push_back(rec);
}

// PENTA6 isotropic 1→8: split each triangular face into 4 sub-triangles and height into 2.
// Creates 8 sub-prisms with proper node ordering.
void refine_penta6(const CfdCell& parent, int parent_id,
                   const std::vector<CfdNode>& old_nodes,
                   std::vector<CfdNode>& new_nodes,
                   std::vector<CfdCell>& new_cells,
                   std::unordered_map<EdgeKey, int, EdgeKeyHash>& edge_map,
                   std::vector<RefinementRecord>& records) {
    int n[6];
    for (int i = 0; i < 6; ++i) n[i] = parent.node[i];

    // Bottom triangle edge midpoints
    int m01 = get_or_create_midpoint(n[0], n[1], old_nodes, new_nodes, edge_map);
    int m12 = get_or_create_midpoint(n[1], n[2], old_nodes, new_nodes, edge_map);
    int m20 = get_or_create_midpoint(n[2], n[0], old_nodes, new_nodes, edge_map);

    // Top triangle edge midpoints
    int m34 = get_or_create_midpoint(n[3], n[4], old_nodes, new_nodes, edge_map);
    int m45 = get_or_create_midpoint(n[4], n[5], old_nodes, new_nodes, edge_map);
    int m53 = get_or_create_midpoint(n[5], n[3], old_nodes, new_nodes, edge_map);

    // Height edge midpoints
    int m03 = get_or_create_midpoint(n[0], n[3], old_nodes, new_nodes, edge_map);
    int m14 = get_or_create_midpoint(n[1], n[4], old_nodes, new_nodes, edge_map);
    int m25 = get_or_create_midpoint(n[2], n[5], old_nodes, new_nodes, edge_map);

    // Sub-triangles of bottom face (nodes 0,1,2)
    // t0_bot = {0, m01, m20}, t1_bot = {m01, 1, m12}, t2_bot = {m20, m12, 2}, t3_bot = {m01, m12, m20}
    // Sub-triangles of top face (nodes 3,4,5)
    // t0_top = {3, m34, m53}, t1_top = {m34, 4, m45}, t2_top = {m53, m45, 5}, t3_top = {m34, m45, m53}

    // Mid-height nodes for each sub-triangle corner:
    // For triangle 0: edges at mid-height are midpoints on the quad faces:
    //   Node 0 → m03, m01 → midpoint of (m01, m34)... but m01 and m34 are not directly connected.
    //   Need a different approach: create 3 mid-height face-edge midpoints.
    //
    // Instead, create mid-height nodes for each sub-triangle vertex:
    //   v0_bot = n[0], v1_bot = n[1], v2_bot = n[2] (bottom triangle vertices)
    //   v0_top = m03, v1_top = m14, v2_top = m25 (mid-height of each original prism edge)
    //
    // Sub-triangle corners at mid-height are interpolations of bottom-sub-tri and top-sub-tri:
    //   For sub-triangle t*_bot = {a, b, c} at bottom, the mid-height nodes are the midpoints
    //   of edges a-top_a, b-top_b, c-top_c.
    //
    // For sub-tri t0_bot = {n0, m01, m20}:
    //   Mid-height: m03, mid(m01, m34), mid(m20, m53)
    //
    // This requires midpoints between edge midpoints, which may not exist yet.
    // Create them via get_or_create_midpoint.

    // Mid-height of m01-m34 and m01-right-face
    int mid_m01_m34 = get_or_create_midpoint(m01, m34, old_nodes, new_nodes, edge_map);
    int mid_m12_m45 = get_or_create_midpoint(m12, m45, old_nodes, new_nodes, edge_map);
    int mid_m20_m53 = get_or_create_midpoint(m20, m53, old_nodes, new_nodes, edge_map);

    int child_level = parent.refinement_level + 1;
    RefinementRecord rec;
    rec.parent_cell_id = parent_id;
    rec.parent_level = parent.refinement_level;
    rec.parent_type = ElementType::PENTA6;
    rec.parent_face_count = 5;
    for (int i = 0; i < 6; ++i) rec.parent_node[i] = parent.node[i];
    rec.n_children = 8;

    // 8 sub-prisms: 4 sub-triangles × 2 height layers
    // Bottom sub-tri corners (b), top sub-tri corners (t):
    // sub0: {0, m01, m20}       sub0_top: {3, m34, m53}
    // sub1: {m01, 1, m12}       sub1_top: {m34, 4, m45}
    // sub2: {m20, m12, 2}       sub2_top: {m53, m45, 5}
    // sub3: {m01, m12, m20}     sub3_top: {m34, m45, m53}

    // Child layout: layers j=0 (bottom half), j=1 (top half)
    // For each sub-tri i, child = {t_bot[i].a, t_bot[i].b, t_bot[i].c, mid[i].a, mid[i].b, mid[i].c}
    // where mid[i] = mid-height of sub-tri i vertices

    // Bottom-layer midpoints: connect bottom sub-tri to mid-height
    // For sub0_bot = {n0, m01, m20}: mid-height = {m03, mid_m01_m34, mid_m20_m53}
    // For sub1_bot = {m01, n1, m12}: mid-height = {mid_m01_m34, m14, mid_m12_m45}
    // For sub2_bot = {m20, m12, n2}: mid-height = {mid_m20_m53, mid_m12_m45, m25}
    // For sub3_bot = {m01, m12, m20}: mid-height = {mid_m01_m34, mid_m12_m45, mid_m20_m53}

    // Top-layer midpoints: connect mid-height to top sub-tri
    // For sub0_top = {n3, m34, m53}: mid-height = {m03, mid_m01_m34, mid_m20_m53}
    // For sub1_top = {m34, n4, m45}: mid-height = {mid_m01_m34, m14, mid_m12_m45}
    // For sub2_top = {m53, m45, n5}: mid-height = {mid_m20_m53, mid_m12_m45, m25}
    // For sub3_top = {m34, m45, m53}: mid-height = {mid_m01_m34, mid_m12_m45, mid_m20_m53}

    struct { int b[3]; int m[3]; } sub_tris_bot[4] = {
        {n[0], m01, m20,  m03, mid_m01_m34, mid_m20_m53},
        {m01, n[1], m12,  mid_m01_m34, m14, mid_m12_m45},
        {m20, m12, n[2],  mid_m20_m53, mid_m12_m45, m25},
        {m01, m12, m20,   mid_m01_m34, mid_m12_m45, mid_m20_m53}
    };
    struct { int t[3]; int m[3]; } sub_tris_top[4] = {
        {n[3], m34, m53,  m03, mid_m01_m34, mid_m20_m53},
        {m34, n[4], m45,  mid_m01_m34, m14, mid_m12_m45},
        {m53, m45, n[5],  mid_m20_m53, mid_m12_m45, m25},
        {m34, m45, m53,   mid_m01_m34, mid_m12_m45, mid_m20_m53}
    };

    for (int i = 0; i < 4; ++i) {
        // Bottom half child
        CfdCell child_bot;
        child_bot.type = ElementType::PENTA6;
        child_bot.node[0] = sub_tris_bot[i].b[0];
        child_bot.node[1] = sub_tris_bot[i].b[1];
        child_bot.node[2] = sub_tris_bot[i].b[2];
        child_bot.node[3] = sub_tris_bot[i].m[0];
        child_bot.node[4] = sub_tris_bot[i].m[1];
        child_bot.node[5] = sub_tris_bot[i].m[2];
        child_bot.refinement_level = child_level;
        child_bot.parent_id = parent_id;
        int child_id_bot = static_cast<int>(new_cells.size());
        new_cells.push_back(child_bot);
        rec.child_cell_ids[i * 2] = child_id_bot;

        // Top half child
        CfdCell child_top;
        child_top.type = ElementType::PENTA6;
        child_top.node[0] = sub_tris_top[i].m[0];
        child_top.node[1] = sub_tris_top[i].m[1];
        child_top.node[2] = sub_tris_top[i].m[2];
        child_top.node[3] = sub_tris_top[i].t[0];
        child_top.node[4] = sub_tris_top[i].t[1];
        child_top.node[5] = sub_tris_top[i].t[2];
        child_top.refinement_level = child_level;
        child_top.parent_id = parent_id;
        int child_id_top = static_cast<int>(new_cells.size());
        new_cells.push_back(child_top);
        rec.child_cell_ids[i * 2 + 1] = child_id_top;
    }
    records.push_back(rec);
}

// Resolve abstract AnisotropicDir to concrete axis (0=X,1=Y,2=Z) for a given cell.
static int resolve_aniso_axis(const CfdCell& cell, AnisotropicDir dir,
                               const std::vector<CfdNode>& old_nodes) {
    if (dir == AnisotropicDir::DIR_X) return 0;
    if (dir == AnisotropicDir::DIR_Y) return 1;
    if (dir == AnisotropicDir::DIR_Z) return 2;
    // WALL_NORMAL: use wall_distance gradient — approximate by which coordinate
    // of the cell centroid is closest to 0 (wall is at x=0 for flat plate).
    // For general geometry, default to X for now; tests override directly.
    // STREAMWISE: default to X (freestream direction).
    return 0;
}

} // anonymous namespace

bool refine_cells(CfdMesh& mesh,
                  const std::vector<RefinementRequest>& requests,
                  std::vector<RefinementRecord>* records_out,
                  std::string* error,
                  const std::vector<RefinementRecord>* prev_records,
                  std::vector<CoarsenInfo>* coarsen_info,
                  int max_level) {
    // Extract per-cell refinement request info
    std::vector<int> to_refine, to_coarsen;
    std::vector<AnisotropicDir> refine_dir;
    for (const auto& req : requests) {
        if (req.cell_id < 0 || req.cell_id >= static_cast<int>(mesh.cells.size()))
            continue;
        if (req.flag == RefinementFlag::Refine) {
            to_refine.push_back(req.cell_id);
            refine_dir.push_back(req.dir);
        } else if (req.flag == RefinementFlag::Coarsen) {
            to_coarsen.push_back(req.cell_id);
        }
    }
    // Deduplicate refine requests (keep first dir for each cell)
    {
        std::vector<int> dedup_refine;
        std::vector<AnisotropicDir> dedup_dir;
        for (std::size_t i = 0; i < to_refine.size(); ++i) {
            bool dup = false;
            for (int d : dedup_refine) {
                if (d == to_refine[i]) { dup = true; break; }
            }
            if (!dup) {
                dedup_refine.push_back(to_refine[i]);
                dedup_dir.push_back(refine_dir[i]);
            }
        }
        to_refine = std::move(dedup_refine);
        refine_dir = std::move(dedup_dir);
    }
    std::sort(to_coarsen.begin(), to_coarsen.end());
    to_coarsen.erase(std::unique(to_coarsen.begin(), to_coarsen.end()), to_coarsen.end());

    int anisotropic_layers = 0;
    // Extract anisotropic_layers from the first request's config context if available,
    // but since refine_cells doesn't receive AmrConfig, we infer from anisotropic_layers
    // stored in the mesh or use a default. For now check any request with dir != NONE.

    for (int id : to_refine) {
        if (mesh.cells[id].refinement_level >= max_level) {
            if (error) *error = "refinement level exceeds max (5)";
            return false;
        }
    }

    struct CoarsenGroup {
        int parent_id;
        int children[MAX_CHILDREN];
        int n_found = 0;
        RefinementRecord src_record;  // from prev_records
        bool valid = false;
    };
    std::vector<CoarsenGroup> coarsen_groups;
    int skipped_coarsen_groups = 0;
    if (!to_coarsen.empty()) {
        if (!prev_records) {
            if (error) *error = "coarsening requires prev_records";
            return false;
        }
        std::unordered_map<int, std::vector<int>> parent_groups;
        for (int ci : to_coarsen) {
            int pid = mesh.cells[ci].parent_id;
            if (pid < 0) {
                if (error) *error = "cell marked for coarsening has no parent";
                return false;
            }
            parent_groups[pid].push_back(ci);
        }

        for (const auto& pg : parent_groups) {
            int pid = pg.first;
            const auto& children = pg.second;

            // Find refinement record to get expected child count
            const RefinementRecord* found_rec = nullptr;
            for (const auto& rec : *prev_records) {
                if (rec.parent_cell_id == pid) {
                    found_rec = &rec;
                    break;
                }
            }
            if (!found_rec) continue;
            int expected_n = found_rec->n_children;
            if (expected_n <= 0 || expected_n > MAX_CHILDREN) continue;
            if (static_cast<int>(children.size()) < expected_n) {
                ++skipped_coarsen_groups;
                continue;
            }

            CoarsenGroup cg;
            cg.parent_id = pid;
            cg.src_record = *found_rec;

            for (int ci : children) {
                if (cg.n_found >= expected_n) break;
                cg.children[cg.n_found++] = ci;
            }
            cg.valid = (cg.n_found == expected_n);
            if (cg.valid) coarsen_groups.push_back(cg);
        }
    }

    std::vector<CfdNode> old_nodes = mesh.nodes;
    std::vector<CfdCell> new_cells;
    std::vector<RefinementRecord> records;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_map;

    // Determine if anisotropic cascade is active and find max anisotropic layers
    // by scanning refine requests for non-NONE dir
    int aniso_layers = 0;
    for (std::size_t i = 0; i < to_refine.size(); ++i) {
        if (refine_dir[i] != AnisotropicDir::NONE) {
            aniso_layers = 1;  // at least one anisotropic request
            // We don't have access to AmrConfig here, so we check per-cell whether
            // the refinement_level indicates anisotropic cascade is still active.
            // The caller (amr_driver) sets anisotropic_layers; we use a simplified
            // heuristic: if refinement_level < aniso_layers_from_config, use anisotropic.
            // Since we can't access config here, we use the presence of anisotropic
            // dir as signal and let the caller control via anisotropic_layers check.
        }
    }

    std::vector<bool> replaced(mesh.cells.size(), false);
    for (int id : to_refine) replaced[id] = true;

    std::vector<bool> removed(mesh.cells.size(), false);
    for (const auto& cg : coarsen_groups) {
        for (int i = 0; i < MAX_CHILDREN; ++i) {
            // Refinement takes priority over coarsening
            if (!replaced[cg.children[i]]) removed[cg.children[i]] = true;
        }
    }

    // Need to correlate to_refine index with cell id for dir lookup
    std::unordered_map<int, AnisotropicDir> refine_dir_map;
    for (std::size_t i = 0; i < to_refine.size(); ++i)
        refine_dir_map[to_refine[i]] = refine_dir[i];

    std::size_t node_count_before = mesh.nodes.size();
    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        if (removed[ci]) continue;
        if (replaced[ci]) {
            const auto& cell = mesh.cells[ci];
            auto dir_it = refine_dir_map.find(ci);
            AnisotropicDir adir = (dir_it != refine_dir_map.end()) ? dir_it->second : AnisotropicDir::NONE;

            // Cascade: use anisotropic split if dir != NONE and cell refinement_level < anisotropic_layers.
            // Since we can't access AmrConfig here, the caller should only pass
            // non-NONE dir for cells within the anisotropic cascade budget.
            bool use_aniso = (adir != AnisotropicDir::NONE);

            if (cell.type == ElementType::HEX8) {
                if (use_aniso) {
                    int axis = resolve_aniso_axis(cell, adir, old_nodes);
                    refine_hex8_aniso(cell, ci, axis, old_nodes, mesh.nodes, new_cells, edge_map, records);
                } else {
                    refine_hex8(cell, ci, old_nodes, mesh.nodes, new_cells, edge_map, records);
                }
            } else if (cell.type == ElementType::TET4) {
                if (use_aniso) {
                    refine_tet4_aniso(cell, ci, adir, old_nodes, mesh.nodes, new_cells, edge_map, records);
                } else {
                    refine_tet4(cell, ci, old_nodes, mesh.nodes, new_cells, edge_map, records);
                }
            } else if (cell.type == ElementType::PENTA6) {
                if (use_aniso) {
                    refine_penta6_aniso(cell, ci, old_nodes, mesh.nodes, new_cells, edge_map, records);
                } else {
                    refine_penta6(cell, ci, old_nodes, mesh.nodes, new_cells, edge_map, records);
                }
            } else {
                if (error) *error = "unsupported element type for refinement";
                mesh.nodes.resize(node_count_before);
                return false;
            }
        } else {
            new_cells.push_back(mesh.cells[ci]);
        }
    }

    for (const auto& cg : coarsen_groups) {
        CfdCell parent;
        parent.type = cg.src_record.parent_type;
        parent.refinement_level = cg.src_record.parent_level;
        parent.parent_id = -1;
        int nn = (parent.type == ElementType::TET4) ? 4 :
                 (parent.type == ElementType::PENTA6) ? 6 : 8;
        for (int i = 0; i < nn; ++i) parent.node[i] = cg.src_record.parent_node[i];
        new_cells.push_back(parent);
        if (coarsen_info) {
            CoarsenInfo ci;
            ci.new_parent_id = static_cast<int>(new_cells.size()) - 1;
            ci.old_parent_id = cg.src_record.parent_cell_id;
            ci.n_children = cg.src_record.n_children;
            for (int i = 0; i < ci.n_children && i < MAX_CHILDREN; ++i)
                ci.old_child_ids[i] = cg.src_record.child_cell_ids[i];
            coarsen_info->push_back(ci);
        }
    }

    mesh.cells = std::move(new_cells);
    rebuild_mesh_faces(mesh);

    if (records_out) *records_out = std::move(records);
    if (error && skipped_coarsen_groups > 0)
        *error += " (" + std::to_string(skipped_coarsen_groups) + " coarsening groups skipped: incomplete sibling sets)";
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
