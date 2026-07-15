#include "aero/cfd/mesh_io.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/element_types.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {
namespace {

// SU2 element type → our ElementType
// SU2: 3=TETRA, 5=TRI, 9=HEXA, 12=WEDGE, 13=QUAD, 14=PYRAMID
struct Su2ElemInfo {
    ElementType type;
    int n_nodes;
};

Su2ElemInfo su2_to_elem(int su2_type) {
    switch (su2_type) {
        case 3:  return {ElementType::TET4, 4};
        case 5:  return {ElementType::TET4, 3};  // TRI face element
        case 9:  return {ElementType::HEX8, 8};
        case 12: return {ElementType::PENTA6, 6};
        case 13: return {ElementType::TET4, 4};  // QUAD face element
        case 14: return {ElementType::PYRAMID5, 5};
        default: return {ElementType::TET4, 0};   // unsupported
    }
}

int elem_to_su2(ElementType type) {
    switch (type) {
        case ElementType::TET4:     return 3;
        case ElementType::HEX8:     return 9;
        case ElementType::PENTA6:   return 12;
        case ElementType::PYRAMID5: return 14;
        default: return 0;
    }
}

BoundaryKind tag_to_boundary(const std::string& tag) {
    std::string low = tag;
    for (auto& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (low == "wall" || low == "noslipwall")    return BoundaryKind::NoSlipWall;
    if (low == "farfield")                        return BoundaryKind::Farfield;
    if (low == "symmetry")                        return BoundaryKind::Symmetry;
    if (low == "inflow" || low == "inlet")        return BoundaryKind::Farfield;
    if (low == "outflow" || low == "outlet")      return BoundaryKind::Farfield;
    // Unknown tag — return Farfield as fallback, err set by caller if needed
    return BoundaryKind::Farfield;
}

bool is_known_boundary_tag(const std::string& tag) {
    std::string low = tag;
    for (auto& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return low == "wall" || low == "noslipwall" || low == "farfield" ||
           low == "symmetry" || low == "inflow" || low == "inlet" ||
           low == "outflow" || low == "outlet";
}

std::string boundary_to_tag(BoundaryKind b) {
    switch (b) {
        case BoundaryKind::NoSlipWall: return "wall";
        case BoundaryKind::SlipWall:   return "slipwall";
        case BoundaryKind::Farfield:   return "farfield";
        case BoundaryKind::Symmetry:   return "symmetry";
        default:                       return "farfield";
    }
}

// Trim whitespace from both ends
std::string trim(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

// Parse a line into tokens, handling both space and tab separators
// Returns string_views pointing into the original line — caller must ensure
// the line outlives the returned vector.
std::vector<std::string_view> tokenize(const std::string& line) {
    std::vector<std::string_view> tokens;
    std::size_t tok_start = std::string::npos;
    for (std::size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '%') break;  // comment
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (tok_start != std::string::npos) {
                tokens.push_back(
                    std::string_view(line.data() + tok_start, i - tok_start));
                tok_start = std::string::npos;
            }
        } else {
            if (tok_start == std::string::npos) tok_start = i;
        }
    }
    if (tok_start != std::string::npos) {
        tokens.push_back(
            std::string_view(line.data() + tok_start, line.size() - tok_start));
    }
    return tokens;
}

struct MarkerFace {
    BoundaryKind boundary;
    std::vector<int> sorted_nodes;
};

struct SortedNodeHash {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t h = 0;
        for (int x : v) {
            h ^= static_cast<std::size_t>(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// Face building helpers (mirrors mesh_metrics.cpp)
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
        std::size_t h = static_cast<std::size_t>(key.n);
        for (int i = 0; i < 4; ++i)
            h ^= static_cast<std::size_t>(key.v[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
struct PendingFace {
    int cell;
    int local_face;
};

void cell_face_nodes_local(const CfdCell& cell, int local_face, int out[4], int& n_out) {
    const int* fn = get_face_nodes(cell.type, local_face);
    n_out = FACE_NODES_PER_ELEMENT[static_cast<int>(cell.type)][local_face];
    for (int i = 0; i < n_out; ++i) out[i] = cell.node[fn[i]];
    for (int i = n_out; i < 4; ++i) out[i] = -1;
}

void build_faces_from_cells(CfdMesh& mesh) {
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
            cell_face_nodes_local(cell, lf, nodes, nn);
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
                cell_face_nodes_local(mesh.cells[face.left_cell], it->second.local_face, left_nodes, left_nn);
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
        cell_face_nodes_local(mesh.cells[face.left_cell], item.second.local_face, fn, fn_n);
        face.node_count = fn_n;
        for (int i = 0; i < fn_n; ++i) face.node[i] = fn[i];
        face.boundary = BoundaryKind::Farfield;
        mesh.faces.push_back(face);
    }
    compute_mesh_metrics(mesh);
}

} // anonymous namespace

bool read_mesh_su2(const std::string& path, CfdMesh& mesh, std::string* err) {
    std::ifstream file(path);
    if (!file.is_open()) {
        if (err) *err = "cannot open file: " + path;
        return false;
    }

    mesh = CfdMesh{};
    int ndime = 0, npoin = 0, nelem = 0, nmark = 0;
    std::vector<MarkerFace> marker_faces;

    std::string line;
    int section = 0;  // 0=none, 1=points, 2=elements, 3=markers
    std::string current_marker_tag;
    int current_marker_count = 0;
    int expected_marker_count = 0;

    while (std::getline(file, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty()) continue;

        auto tokens = tokenize(line);
        if (tokens.empty()) continue;

        try {
        // Handle key = value (SU2 format: "NDIME= 3" or "NDIME=3" or "NDIME 3")
        auto split_kv = [](std::string_view s) -> std::pair<std::string_view, std::string_view> {
            auto eq = s.find('=');
            if (eq != std::string_view::npos)
                return {s.substr(0, eq), s.substr(eq + 1)};
            return {s, std::string_view{}};
        };
        auto [kw, val] = split_kv(tokens[0]);
        if (tokens.size() > 1 && val.empty()) val = tokens[1];

        if (kw == "NDIME") { ndime = std::stoi(std::string(val)); continue; }
        if (kw == "NPOIN") {
            if (section >= 1) { if (err) *err = "duplicate NPOIN"; return false; }
            npoin = std::stoi(std::string(val));
            mesh.nodes.reserve(static_cast<std::size_t>(npoin));
            section = 1;
            continue;
        }
        if (kw == "NELEM") {
            if (section >= 2) { if (err) *err = "duplicate NELEM"; return false; }
            if (section < 1) { if (err) *err = "NELEM before NPOIN"; return false; }
            nelem = std::stoi(std::string(val));
            mesh.cells.reserve(static_cast<std::size_t>(nelem));
            section = 2;
            continue;
        }
        if (kw == "NMARK") {
            if (section >= 3) { if (err) *err = "duplicate NMARK"; return false; }
            if (section < 2) { if (err) *err = "NMARK before NELEM"; return false; }
            nmark = std::stoi(std::string(val));
            section = 3;
            continue;
        }
        if (kw == "MARKER_TAG") {
            current_marker_tag = val;
            if (current_marker_tag.empty() && tokens.size() > 1) current_marker_tag = tokens[1];
            if (current_marker_tag.empty()) {
                if (err) *err = "MARKER_TAG without value";
                return false;
            }
            if (!is_known_boundary_tag(current_marker_tag)) {
                if (err) *err = "unknown boundary tag: " + current_marker_tag;
                return false;
            }
            expected_marker_count = 0;
            continue;
        }
        if (kw == "MARKER_ELEMS") {
            if (val.empty() && tokens.size() > 1) val = tokens[1];
            if (val.empty()) {
                if (err) *err = "MARKER_ELEMS without value";
                return false;
            }
            expected_marker_count = std::stoi(std::string(val));
            current_marker_count = 0;
            continue;
        }

        // Parse data lines based on section
        if (section == 1) {
            // NPOIN: <id> <x> <y> [<z>]
            if (tokens.size() < 4) continue;
            CfdNode node;
            node.x = static_cast<Real>(std::stod(std::string(tokens[1])));
            node.y = static_cast<Real>(std::stod(std::string(tokens[2])));
            node.z = static_cast<Real>(std::stod(std::string(tokens[3])));
            if (!std::isfinite(node.x) || !std::isfinite(node.y) || !std::isfinite(node.z)) {
                if (err) *err = "non-finite coordinate at node " + std::to_string(mesh.nodes.size());
                return false;
            }
            mesh.nodes.push_back(node);
        } else if (section == 2) {
            // NELEM: <su2_type> <tag> <node1> <node2> ...
            if (tokens.size() < 3) continue;
            int su2_type = std::stoi(std::string(tokens[0]));
            if (su2_type == 5 || su2_type == 13) {
                if (err) *err = std::string("face element type ") + std::to_string(su2_type) + " in volume element section";
                return false;
            }
            auto info = su2_to_elem(su2_type);
            if (info.n_nodes == 0) {
                if (err) *err = "unsupported SU2 element type: " + std::to_string(su2_type);
                return false;
            }
            if (static_cast<int>(tokens.size()) < 2 + info.n_nodes) {
                if (err) *err = "truncated element line";
                return false;
            }
            CfdCell cell;
            cell.type = info.type;
            for (int i = 0; i < info.n_nodes; ++i) {
                int ni = std::stoi(std::string(tokens[2 + i]));
                if (ni < 0 || static_cast<std::size_t>(ni) >= mesh.nodes.size()) {
                    if (err) *err = "node index " + std::to_string(ni) + " out of range [0," + std::to_string(mesh.nodes.size()) + ")";
                    return false;
                }
                cell.node[i] = ni;
            }
            mesh.cells.push_back(cell);
        } else if (section == 3) {
            // NMARK data: each line is a boundary face
            // <su2_face_type> <tag> <n1> <n2> [<n3> [<n4>]]
            if (tokens.size() < 4) continue;
            int face_type = std::stoi(std::string(tokens[0]));
            int fn = (face_type == 5) ? 3 : (face_type == 13) ? 4 : 0;
            if (fn == 0) continue;
            if (static_cast<int>(tokens.size()) < 2 + fn) continue;

            MarkerFace mf;
            mf.boundary = tag_to_boundary(current_marker_tag);
            for (int i = 0; i < fn; ++i) {
                mf.sorted_nodes.push_back(std::stoi(std::string(tokens[2 + i])));
            }
            std::sort(mf.sorted_nodes.begin(), mf.sorted_nodes.end());
            marker_faces.push_back(mf);
            current_marker_count++;
        }
        } // try
        catch (const std::exception& e) {
            if (err) *err = std::string("parse error: ") + e.what();
            return false;
        }
    }

    if (expected_marker_count > 0 && current_marker_count != expected_marker_count) {
        if (err) *err = "marker element count mismatch: expected " + std::to_string(expected_marker_count) +
                        ", got " + std::to_string(current_marker_count);
        return false;
    }
    if (static_cast<int>(mesh.nodes.size()) != npoin) {
        if (err) *err = "node count mismatch: declared " + std::to_string(npoin) +
                        ", read " + std::to_string(mesh.nodes.size());
        return false;
    }
    if (static_cast<int>(mesh.cells.size()) != nelem) {
        if (err) *err = "element count mismatch: declared " + std::to_string(nelem) +
                        ", read " + std::to_string(mesh.cells.size());
        return false;
    }
    if (ndime != 3) {
        if (err) *err = "only 3D meshes supported (NDIME=" + std::to_string(ndime) + ")";
        return false;
    }
    if (mesh.nodes.empty() || mesh.cells.empty()) {
        if (err) *err = "no nodes or cells read";
        return false;
    }

    // Build faces from volume elements
    build_faces_from_cells(mesh);

    // Validate positive cell volumes
    for (std::size_t ci = 0; ci < mesh.cells.size(); ++ci) {
        if (mesh.cells[ci].volume <= Real(0)) {
            if (err) *err = "non-positive volume at cell " + std::to_string(ci);
            return false;
        }
    }

    // Build O(1) marker lookup: sorted-node tuple → marker index
    std::unordered_map<std::vector<int>, int, SortedNodeHash> marker_map;
    for (int mi = 0; mi < static_cast<int>(marker_faces.size()); ++mi) {
        marker_map.emplace(marker_faces[mi].sorted_nodes, mi);
    }

    // Match boundary faces to marker faces
    for (auto& face : mesh.faces) {
        if (face.right_cell >= 0) continue;  // interior face
        std::vector<int> key(face.node, face.node + face.node_count);
        std::sort(key.begin(), key.end());
        auto it = marker_map.find(key);
        if (it != marker_map.end()) {
            face.boundary = marker_faces[it->second].boundary;
        }
    }

    return true;
}

static bool checked_fprintf(std::FILE* f, std::string* err, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (std::vfprintf(f, fmt, args) < 0) {
        va_end(args);
        if (err) *err = "write error";
        return false;
    }
    va_end(args);
    return true;
}

bool write_mesh_su2(const CfdMesh& mesh, const std::string& path, std::string* err) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        if (err) *err = "cannot open file for writing: " + path;
        return false;
    }

#define CHECKED_PRINTF(...) do { if (!checked_fprintf(f, err, __VA_ARGS__)) { std::fclose(f); return false; } } while(0)

    CHECKED_PRINTF("NDIME= 3\n");
    CHECKED_PRINTF("NPOIN= %zu\n", mesh.nodes.size());
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
        CHECKED_PRINTF("%zu  %.15g  %.15g  %.15g\n", i,
                     static_cast<double>(mesh.nodes[i].x),
                     static_cast<double>(mesh.nodes[i].y),
                     static_cast<double>(mesh.nodes[i].z));
    }

    CHECKED_PRINTF("NELEM= %zu\n", mesh.cells.size());
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        const auto& cell = mesh.cells[i];
        int su2_type = elem_to_su2(cell.type);
        if (su2_type == 0) {
            std::fclose(f);
            if (err) *err = "unsupported cell type at cell " + std::to_string(i);
            return false;
        }
        int et = static_cast<int>(cell.type);
        if (et < 0 || et >= 4) {
            std::fclose(f);
            if (err) *err = "invalid cell type at cell " + std::to_string(i);
            return false;
        }
        int nn = ELEMENT_NODES[et];
        CHECKED_PRINTF("%d %zu", su2_type, i);
        for (int j = 0; j < nn; ++j) {
            CHECKED_PRINTF(" %d", cell.node[j]);
        }
        CHECKED_PRINTF("\n");
    }

    // Collect boundary faces by kind
    std::unordered_map<int, std::vector<std::size_t>> boundary_groups;
    for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
        const auto& face = mesh.faces[i];
        if (face.boundary == BoundaryKind::Interior) continue;
        int bk = static_cast<int>(face.boundary);
        boundary_groups[bk].push_back(i);
    }

    CHECKED_PRINTF("NMARK= %zu\n", boundary_groups.size());
    for (const auto& [bk_int, face_indices] : boundary_groups) {
        auto bk = static_cast<BoundaryKind>(bk_int);
        CHECKED_PRINTF("MARKER_TAG= %s\n", boundary_to_tag(bk).c_str());
        CHECKED_PRINTF("MARKER_ELEMS= %zu\n", face_indices.size());
        for (std::size_t fi : face_indices) {
            const auto& face = mesh.faces[fi];
            int face_type = (face.node_count == 4) ? 13 : 5;  // QUAD or TRI
            CHECKED_PRINTF("%d 0", face_type);
            for (int j = 0; j < face.node_count; ++j) {
                CHECKED_PRINTF(" %d", face.node[j]);
            }
            CHECKED_PRINTF("\n");
        }
    }

#undef CHECKED_PRINTF

    std::fclose(f);
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
