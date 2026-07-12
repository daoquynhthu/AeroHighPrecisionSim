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

// Add midpoint for an edge if not already in the map, return node index.
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

    // 8 child tets
    TetLocal children[8] = {
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

    for (int i = 0; i < 8; ++i) {
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
    rec.n_children = 8;
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
    // Initialize all to -1
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
    // Edges along x
    set_mid_edge(1,0,0, 0,1);
    set_mid_edge(1,2,0, 3,2);
    set_mid_edge(1,0,2, 4,5);
    set_mid_edge(1,2,2, 7,6);
    // Edges along y
    set_mid_edge(0,1,0, 0,3);
    set_mid_edge(2,1,0, 1,2);
    set_mid_edge(0,1,2, 4,7);
    set_mid_edge(2,1,2, 5,6);
    // Edges along z
    set_mid_edge(0,0,1, 0,4);
    set_mid_edge(2,0,1, 1,5);
    set_mid_edge(2,2,1, 2,6);
    set_mid_edge(0,2,1, 3,7);

    // Face centers (6) — midpoint of face diagonal (original hex nodes).
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
    rec.n_children = 8;
    records.push_back(rec);
}

} // anonymous namespace

bool refine_cells(CfdMesh& mesh,
                  const std::vector<RefinementRequest>& requests,
                  std::vector<RefinementRecord>* records_out,
                  std::string* error,
                  const std::vector<RefinementRecord>* prev_records,
                  std::vector<CoarsenInfo>* coarsen_info,
                  int max_level) {
    // ----- 1. Build sets for refine and coarsen -----
    std::vector<int> to_refine, to_coarsen;
    for (const auto& req : requests) {
        if (req.cell_id < 0 || req.cell_id >= static_cast<int>(mesh.cells.size()))
            continue;
        if (req.flag == RefinementFlag::Refine) to_refine.push_back(req.cell_id);
        else if (req.flag == RefinementFlag::Coarsen) to_coarsen.push_back(req.cell_id);
    }
    std::sort(to_refine.begin(), to_refine.end());
    to_refine.erase(std::unique(to_refine.begin(), to_refine.end()), to_refine.end());
    std::sort(to_coarsen.begin(), to_coarsen.end());
    to_coarsen.erase(std::unique(to_coarsen.begin(), to_coarsen.end()), to_coarsen.end());

    // Level cap check
    for (int id : to_refine) {
        if (mesh.cells[id].refinement_level >= max_level) {
            if (error) *error = "refinement level exceeds max (5)";
            return false;
        }
    }

    // ----- 2. Prepare coarsening groups (children of same parent) -----
    struct CoarsenGroup {
        int parent_id;
        int children[8];
        int n_found = 0;
        RefinementRecord src_record;  // from prev_records
        bool valid = false;
    };
    std::vector<CoarsenGroup> coarsen_groups;
    if (!to_coarsen.empty()) {
        // Group cells marked for coarsening by parent_id
        std::unordered_map<int, std::vector<int>> parent_groups;
        for (int ci : to_coarsen) {
            int pid = mesh.cells[ci].parent_id;
            if (pid < 0) {
                if (error) *error = "cell marked for coarsening has no parent";
                return false;
            }
            parent_groups[pid].push_back(ci);
        }

        // Build CoarsenGroup from each complete set of 8 siblings
        for (const auto& pg : parent_groups) {
            int pid = pg.first;
            const auto& children = pg.second;
            if (static_cast<int>(children.size()) < 8) continue;  // incomplete group — skip

            CoarsenGroup cg;
            cg.parent_id = pid;

            // Find the matching prev_record
            bool found_record = false;
            if (prev_records) {
                for (const auto& rec : *prev_records) {
                    if (rec.parent_cell_id == pid) {
                        cg.src_record = rec;
                        found_record = true;
                        break;
                    }
                }
            }
            if (!found_record) continue;  // can't reconstruct parent without record
            if (cg.src_record.n_children != 8) continue;

            // Mark the 8 children
            for (int ci : children) {
                if (cg.n_found >= 8) break;
                cg.children[cg.n_found++] = ci;
            }
            cg.valid = (cg.n_found == 8);
            if (cg.valid) coarsen_groups.push_back(cg);
        }
    }

    // ----- 3. Build new cell array (refine + coarsen in one pass) -----
    std::vector<CfdNode> old_nodes = mesh.nodes;
    std::vector<CfdCell> new_cells;
    std::vector<RefinementRecord> records;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_map;

    // Mark cells to be replaced by children
    std::vector<bool> replaced(mesh.cells.size(), false);
    for (int id : to_refine) replaced[id] = true;

    // Mark cells to be removed (coarsened — absorbed into parent)
    std::vector<bool> removed(mesh.cells.size(), false);
    for (const auto& cg : coarsen_groups) {
        for (int i = 0; i < 8; ++i) {
            // Refinement takes priority over coarsening
            if (!replaced[cg.children[i]]) removed[cg.children[i]] = true;
        }
    }

    std::size_t node_count_before = mesh.nodes.size();
    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        if (removed[ci]) continue;
        if (replaced[ci]) {
            const auto& cell = mesh.cells[ci];
            if (cell.type == ElementType::TET4) {
                refine_tet4(cell, ci, old_nodes, mesh.nodes, new_cells, edge_map, records);
            } else if (cell.type == ElementType::HEX8) {
                refine_hex8(cell, ci, old_nodes, mesh.nodes, new_cells, edge_map, records);
            } else {
                if (error) *error = "unsupported element type for refinement";
                mesh.nodes.resize(node_count_before);
                return false;
            }
        } else {
            new_cells.push_back(mesh.cells[ci]);
        }
    }

    // ----- 4. Insert coarsened parent cells -----
    for (const auto& cg : coarsen_groups) {
        CfdCell parent;
        parent.type = cg.src_record.parent_type;
        parent.refinement_level = cg.src_record.parent_level;
        parent.parent_id = -1;
        int nn = (parent.type == ElementType::TET4) ? 4 : 8;
        for (int i = 0; i < nn; ++i) parent.node[i] = cg.src_record.parent_node[i];
        new_cells.push_back(parent);
        if (coarsen_info) {
            CoarsenInfo ci;
            ci.new_parent_id = static_cast<int>(new_cells.size()) - 1;
            ci.old_parent_id = cg.src_record.parent_cell_id;
            ci.n_children = cg.src_record.n_children;
            for (int i = 0; i < ci.n_children && i < 8; ++i)
                ci.old_child_ids[i] = cg.src_record.child_cell_ids[i];
            coarsen_info->push_back(ci);
        }
    }

    // ----- 5. Replace mesh -----
    mesh.cells = std::move(new_cells);
    rebuild_mesh_faces(mesh);

    if (records_out) *records_out = std::move(records);
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
