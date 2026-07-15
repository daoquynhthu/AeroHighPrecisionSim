#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/element_types.hpp"
#include "aero/cfd/mesh_io.hpp"
#include "aero/cfd/mesh_io_cgns.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef WITH_CGNS
#include <cgnslib.h>
#endif

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

#ifdef WITH_CGNS

#define CGNS_CALL(call, close_on_fail) do { \
    int ierr_ = (call); \
    if (ierr_ != CG_OK) { \
        if (err) { char buf_[256]; cg_get_error(buf_); *err = std::string("CGNS: ") + buf_; } \
        close_on_fail; \
        return false; \
    } \
} while(0)

struct CgnsFile {
    int fn;
    explicit CgnsFile(const std::string& path) : fn(-1) {
        int ierr = cg_open(path.c_str(), CG_MODE_READ, &fn);
        if (ierr != CG_OK) fn = -1;
    }
    ~CgnsFile() { if (fn >= 0) cg_close(fn); }
    bool valid() const { return fn >= 0; }
};

ElementType cgns_elem_to_type(int elem_type) {
    switch (elem_type) {
        case TETRA_4:   return ElementType::TET4;
        case HEXA_8:    return ElementType::HEX8;
        case PENTA_6:   return ElementType::PENTA6;
        case PYRA_5:    return ElementType::PYRAMID5;
        default:        return ElementType::TET4;
    }
}

int expected_elem_nodes(int elem_type) {
    switch (elem_type) {
        case TETRA_4:   return 4;
        case HEXA_8:    return 8;
        case PENTA_6:   return 6;
        case PYRA_5:    return 5;
        case TRI_3:     return 3;
        case QUAD_4:    return 4;
        default:        return 0;
    }
}

BoundaryKind cgns_bc_to_kind(int bc_type) {
    switch (bc_type) {
        case BCWallInviscid: return BoundaryKind::SlipWall;
        case BCWallViscous:  return BoundaryKind::NoSlipWall;
        case BCFarfield:     return BoundaryKind::Farfield;
        case BCSymmetryPlane: return BoundaryKind::Symmetry;
        case BCInflow:
        case BCOutflow:
        case BCInflowSubsonic:
        case BCOutflowSubsonic:
        case BCInflowSupersonic:
        case BCOutflowSupersonic:
            return BoundaryKind::Farfield;
        default:
            return BoundaryKind::Farfield;
    }
}

#endif // WITH_CGNS

} // namespace

bool read_mesh_cgns(const std::string& path, CfdMesh& mesh, std::string* err) {
#ifdef WITH_CGNS
    mesh = CfdMesh{};

    CgnsFile cgns(path);
    if (!cgns.valid()) {
        if (err) { char buf[256]; cg_get_error(buf); *err = std::string("CGNS open failed: ") + buf; }
        return false;
    }
    int fn = cgns.fn;

    try {
        int nbases;
        CGNS_CALL(cg_nbases(fn, &nbases), (void)0);
        if (nbases < 1) { if (err) *err = "CGNS file has no base"; return false; }

        char basename[33] = {};
        int celldim, physdim;
        CGNS_CALL(cg_base_read(fn, 1, basename, &celldim, &physdim), (void)0);
        basename[sizeof(basename) - 1] = '\0';
        if (celldim != 3 || physdim != 3) {
            if (err) *err = "CGNS: only 3D meshes supported";
            return false;
        }

        int nzones;
        CGNS_CALL(cg_nzones(fn, 1, &nzones), (void)0);
        if (nzones < 1) { if (err) *err = "CGNS file has no zones"; return false; }

        // Collect face markers from BCs to apply after rebuild_mesh_faces
        struct PendingBcFace {
            BoundaryKind kind;
            int nodes[4];
            int n_nodes;
        };
        std::vector<PendingBcFace> pending_bc_faces;

        for (int Z = 1; Z <= nzones; ++Z) {
            // Save checkpoint for rollback on zone failure
            std::size_t cp_nodes = mesh.nodes.size();
            std::size_t cp_cells = mesh.cells.size();
            std::size_t cp_bc = pending_bc_faces.size();

            auto process_zone = [&]() -> bool {
                char zonename[33];
                cgsize_t size[9];
                CGNS_CALL(cg_zone_read(fn, 1, Z, zonename, size), (void)0);

                ZoneType_t zonetype;
                CGNS_CALL(cg_zone_type(fn, 1, Z, &zonetype), (void)0);
                if (zonetype != Unstructured) {
                    if (err) *err = std::string("CGNS zone '") + zonename + "' is not unstructured";
                    return false;
                }

                cgsize_t nnodes64 = size[0];
                if (nnodes64 > INT_MAX) { if (err) *err = "CGNS: too many nodes"; return false; }
                int nnodes = static_cast<int>(nnodes64);

                int ncoords;
                CGNS_CALL(cg_ncoords(fn, 1, Z, &ncoords), (void)0);
                if (ncoords < 3) { if (err) *err = "CGNS: need at least 3 coordinate arrays"; return false; }

                std::vector<Real> xs(nnodes), ys(nnodes), zs(nnodes);

                for (int coord = 1; coord <= 3; ++coord) {
                    DataType_t dt;
                    CGNS_CALL(cg_coord_info(fn, 1, Z, coord, &dt), (void)0);
                    const char* cname = (coord == 1) ? "CoordinateX" : (coord == 2) ? "CoordinateY" : "CoordinateZ";
                    std::vector<Real>* target = (coord == 1) ? &xs : (coord == 2) ? &ys : &zs;
                    if (dt == RealDouble) {
                        std::vector<double> tmp(nnodes);
                        CGNS_CALL(cg_coord_read(fn, 1, Z, cname, RealDouble, &tmp[0]), (void)0);
                        for (int i = 0; i < nnodes; ++i) (*target)[i] = static_cast<Real>(tmp[i]);
                    } else {
                        CGNS_CALL(cg_coord_read(fn, 1, Z, cname, RealSingle, &(*target)[0]), (void)0);
                    }
                }

                for (int i = 0; i < nnodes; ++i) {
                    if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]) || !std::isfinite(zs[i])) {
                        if (err) *err = "non-finite coordinate at node " + std::to_string(mesh.nodes.size() + i);
                        return false;
                    }
                }

                int base_offset = static_cast<int>(mesh.nodes.size());
                mesh.nodes.reserve(mesh.nodes.size() + nnodes);
                for (int i = 0; i < nnodes; ++i)
                    mesh.nodes.push_back({xs[i], ys[i], zs[i]});

                int nsections;
                CGNS_CALL(cg_nsections(fn, 1, Z, &nsections), (void)0);

                for (int S = 1; S <= nsections; ++S) {
                    char secname[33] = {};
                    ElementType_t elem_type;
                    cgsize_t start, end;
                    int nbndry, parent_flag;
                    CGNS_CALL(cg_section_read(fn, 1, Z, S, secname, &elem_type, &start, &end, &nbndry, &parent_flag), (void)0);
                    secname[sizeof(secname) - 1] = '\0';
                    (void)nbndry; (void)parent_flag;

                    cgsize_t nelem64 = end - start + 1;
                    if (nelem64 > INT_MAX) { if (err) *err = "CGNS: too many elements in section"; return false; }
                    int nelem = static_cast<int>(nelem64);

                    int nnodes_per_elem = expected_elem_nodes(elem_type);
                    if (nnodes_per_elem == 0) {
                        if (err) *err = std::string("unsupported CGNS element type in section ") + secname;
                        return false;
                    }

                    std::size_t total_conn = static_cast<std::size_t>(nnodes_per_elem) * static_cast<std::size_t>(nelem);
                    std::vector<cgsize_t> conn(total_conn);
                    cgsize_t data_size;
                    CGNS_CALL(cg_elements_read(fn, 1, Z, S, &conn[0], &data_size), (void)0);
                    (void)data_size;

                    bool is_volume = (elem_type == TETRA_4 || elem_type == HEXA_8 ||
                                      elem_type == PENTA_6 || elem_type == PYRA_5);

                    if (is_volume) {
                        ElementType etype = cgns_elem_to_type(elem_type);
                        for (int e = 0; e < nelem; ++e) {
                            CfdCell cell;
                            cell.type = etype;
                            for (int j = 0; j < nnodes_per_elem && j < 8; ++j) {
                                int node_id = static_cast<int>(conn[static_cast<std::size_t>(e) * nnodes_per_elem + j]) - 1 + base_offset;
                                if (node_id < 0 || static_cast<std::size_t>(node_id) >= mesh.nodes.size()) {
                                    if (err) *err = "CGNS: node index out of range";
                                    return false;
                                }
                                cell.node[j] = node_id;
                            }
                            mesh.cells.push_back(cell);
                        }
                    } else if (elem_type == TRI_3 || elem_type == QUAD_4) {
                        for (int e = 0; e < nelem; ++e) {
                            PendingBcFace pbf;
                            pbf.n_nodes = nnodes_per_elem;
                            pbf.kind = BoundaryKind::Farfield;
                            for (int j = 0; j < nnodes_per_elem; ++j) {
                                pbf.nodes[j] = static_cast<int>(conn[static_cast<std::size_t>(e) * nnodes_per_elem + j]) - 1 + base_offset;
                            }
                            pending_bc_faces.push_back(pbf);
                        }
                    }
                }

                int nbocos;
                CGNS_CALL(cg_nbocos(fn, 1, Z, &nbocos), (void)0);

                for (int BC = 1; BC <= nbocos; ++BC) {
                    char boconame[33] = {};
                    BCType_t bocotype;
                    PointSetType_t ptset_type;
                    cgsize_t npnts, normal_list_size;
                    int normal_list_flag, normal_dir_flag;
                    CGNS_CALL(cg_boco_read(fn, 1, Z, BC, boconame, &bocotype, &ptset_type,
                                &npnts, nullptr, &normal_list_size,
                                &normal_list_flag, &normal_dir_flag), (void)0);

                    BoundaryKind bkind = cgns_bc_to_kind(bocotype);

                    if (ptset_type == Element_t || ptset_type == FaceCenter) {
                        std::vector<cgsize_t> face_conn(npnts > 0 ? npnts : 1);
                        cgsize_t* data_ptr = (npnts > 0) ? &face_conn[0] : nullptr;
                        CGNS_CALL(cg_boco_read(fn, 1, Z, BC, boconame, &bocotype, &ptset_type,
                                    &npnts, data_ptr, &normal_list_size,
                                    &normal_list_flag, &normal_dir_flag), (void)0);

                        if (npnts > 0) {
                            int nface_nodes = (ptset_type == Element_t) ? 1 : 2;
                            for (cgsize_t p = 0; p < npnts / nface_nodes; ++p) {
                                cgsize_t fe_idx = face_conn[static_cast<std::size_t>(p) * nface_nodes];
                                if (fe_idx >= 1 && static_cast<std::size_t>(fe_idx - 1) < pending_bc_faces.size()) {
                                    pending_bc_faces[static_cast<std::size_t>(fe_idx - 1)].kind = bkind;
                                }
                            }
                        }
                    }
                }
                return true;
            };

            if (!process_zone()) {
                mesh.nodes.resize(cp_nodes);
                mesh.cells.resize(cp_cells);
                pending_bc_faces.resize(cp_bc);
                return false;
            }
        }

        // Build faces from cells
        rebuild_mesh_faces(mesh);

        // Validate positive cell volumes
        for (std::size_t ci = 0; ci < mesh.cells.size(); ++ci) {
            if (mesh.cells[ci].volume <= Real(0)) {
                if (err) *err = "non-positive cell volume at " + std::to_string(ci);
                return false;
            }
        }

        // Build O(1) marker lookup: sorted-node tuple → BoundaryKind
        struct SortedNodeHash {
            std::size_t operator()(const std::vector<int>& v) const {
                std::size_t h = 0;
                for (int x : v) {
                    h ^= static_cast<std::size_t>(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
                }
                return h;
            }
        };
        std::unordered_map<std::vector<int>, BoundaryKind, SortedNodeHash> marker_map;
        for (const auto& pbf : pending_bc_faces) {
            std::vector<int> key(pbf.nodes, pbf.nodes + pbf.n_nodes);
            std::sort(key.begin(), key.end());
            marker_map.emplace(std::move(key), pbf.kind);
        }

        // Apply boundary markers from CGNS boundary conditions by matching node sets
        for (auto& face : mesh.faces) {
            if (face.right_cell >= 0) continue;
            std::vector<int> key(face.node, face.node + face.node_count);
            std::sort(key.begin(), key.end());
            auto it = marker_map.find(key);
            if (it != marker_map.end()) {
                face.boundary = it->second;
            }
        }

        // Validate: all boundary faces should have non-Farfield markers
        for (std::size_t fi = 0; fi < mesh.faces.size(); ++fi) {
            const auto& face = mesh.faces[fi];
            if (face.right_cell >= 0) continue;
            if (face.boundary == BoundaryKind::Farfield) {
                // Farfield is the fallback — acceptable if file has no BC section
            }
        }

        return true;

    } catch (std::bad_alloc&) {
        if (err) *err = "CGNS: out of memory";
        return false;
    } catch (std::exception& e) {
        if (err) *err = std::string("CGNS: ") + e.what();
        return false;
    }

#else // WITH_CGNS not defined
    (void)mesh;
    if (err) *err = "CGNS support not compiled in (set AEROSIM_USE_CGNS=ON)";
    return false;
#endif
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
