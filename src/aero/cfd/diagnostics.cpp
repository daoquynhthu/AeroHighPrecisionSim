#include "aero/cfd/diagnostics.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

StateBounds compute_state_bounds(const std::vector<ConservativeState>& q, Real gamma, const std::vector<PrimitiveState>* primitive_override) {
    StateBounds bounds;
    if (q.empty()) return bounds;

    bounds.min_rho = std::numeric_limits<Real>::max();
    bounds.min_p = std::numeric_limits<Real>::max();
    bounds.min_mach = std::numeric_limits<Real>::max();
    bounds.max_rho = -std::numeric_limits<Real>::max();
    bounds.max_p = -std::numeric_limits<Real>::max();
    bounds.max_mach = -std::numeric_limits<Real>::max();
    bounds.valid = true;

    for (std::size_t i = 0; i < q.size(); ++i) {
        PrimitiveState w;
        if (primitive_override && primitive_override->size() == q.size()) {
            w = (*primitive_override)[i];
        } else if (!conservative_to_primitive(q[i], gamma, w)) {
            bounds.valid = false;
            bounds.bad_cell = i;
            return bounds;
        }

        Real a = speed_of_sound(w, gamma);
        Real vmag = std::sqrt(w.u*w.u + w.v*w.v + w.w*w.w);
        Real mach = vmag / std::max(a, Real(1e-30));
        bounds.min_rho = std::min(bounds.min_rho, w.rho);
        bounds.max_rho = std::max(bounds.max_rho, w.rho);
        bounds.min_p = std::min(bounds.min_p, w.p);
        bounds.max_p = std::max(bounds.max_p, w.p);
        bounds.min_mach = std::min(bounds.min_mach, mach);
        bounds.max_mach = std::max(bounds.max_mach, mach);
    }

    return bounds;
}

FailureSnapshot make_failure_snapshot(
    int iteration,
    int cell,
    const char* reason,
    const ConservativeState& q,
    Real gamma) {
    FailureSnapshot snapshot;
    snapshot.valid = true;
    snapshot.iteration = iteration;
    snapshot.cell = cell;
    snapshot.reason = reason ? reason : "";
    snapshot.state = q;
    conservative_to_primitive(q, gamma, snapshot.primitive);
    return snapshot;
}

static void write_binary_big_endian(std::ofstream& out, const float* data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        unsigned int bits;
        std::memcpy(&bits, &data[i], sizeof(bits));
        unsigned char bytes[4] = {
            static_cast<unsigned char>(bits >> 24),
            static_cast<unsigned char>(bits >> 16),
            static_cast<unsigned char>(bits >> 8),
            static_cast<unsigned char>(bits)
        };
        out.write(reinterpret_cast<const char*>(bytes), 4);
    }
}

static void write_binary_big_endian_int(std::ofstream& out, const int* data, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        unsigned int bits = static_cast<unsigned int>(data[i]);
        unsigned char bytes[4] = {
            static_cast<unsigned char>(bits >> 24),
            static_cast<unsigned char>(bits >> 16),
            static_cast<unsigned char>(bits >> 8),
            static_cast<unsigned char>(bits)
        };
        out.write(reinterpret_cast<const char*>(bytes), 4);
    }
}

bool write_vtk_cells(
    const char* path,
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    Real gamma,
    std::string* error) {
    if (!path || !path[0]) {
        if (error) *error = "empty path";
        return false;
    }
    if (q.size() != mesh.cells.size()) {
        if (error) *error = "state size does not match mesh cells";
        return false;
    }

    std::vector<PrimitiveState> primitive(q.size());
    std::vector<Real> mach(q.size(), 0.0f);
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (!conservative_to_primitive(q[i], gamma, primitive[i])) {
            if (error) *error = "invalid state";
            return false;
        }
        Real a = speed_of_sound(primitive[i], gamma);
        Real vmag = std::sqrt(
            primitive[i].u*primitive[i].u +
            primitive[i].v*primitive[i].v +
            primitive[i].w*primitive[i].w);
        mach[i] = vmag / std::max(a, Real(1e-30));
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) *error = "failed to open path";
        return false;
    }

    out << "# vtk DataFile Version 3.0\n";
    out << "aerosp CFD cell diagnostics\n";
    out << "BINARY\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    out << "POINTS " << mesh.nodes.size() << " float\n";
    {
        std::vector<float> pts(mesh.nodes.size() * 3);
        for (std::size_t i = 0; i < mesh.nodes.size(); ++i) {
            pts[i * 3 + 0] = static_cast<float>(mesh.nodes[i].x);
            pts[i * 3 + 1] = static_cast<float>(mesh.nodes[i].y);
            pts[i * 3 + 2] = static_cast<float>(mesh.nodes[i].z);
        }
        write_binary_big_endian(out, pts.data(), pts.size());
    }

    std::size_t total_entries = 0;
    for (const auto& cell : mesh.cells) {
        total_entries += 1 + ELEMENT_NODES[static_cast<int>(cell.type)];
    }
    out << "CELLS " << mesh.cells.size() << " " << total_entries << "\n";
    {
        std::vector<int> cells(total_entries);
        std::size_t offset = 0;
        for (const auto& cell : mesh.cells) {
            int nn = ELEMENT_NODES[static_cast<int>(cell.type)];
            cells[offset++] = nn;
            for (int i = 0; i < nn; ++i)
                cells[offset++] = cell.node[i];
        }
        write_binary_big_endian_int(out, cells.data(), cells.size());
    }

    constexpr int VTK_TETRA = 10;
    constexpr int VTK_HEX = 12;
    constexpr int VTK_WEDGE = 13;
    constexpr int VTK_PYRAMID = 14;
    out << "CELL_TYPES " << mesh.cells.size() << "\n";
    {
        std::vector<int> types(mesh.cells.size());
        for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
            switch (mesh.cells[i].type) {
                case ElementType::TET4:     types[i] = VTK_TETRA; break;
                case ElementType::HEX8:     types[i] = VTK_HEX; break;
                case ElementType::PENTA6:   types[i] = VTK_WEDGE; break;
                case ElementType::PYRAMID5: types[i] = VTK_PYRAMID; break;
            }
        }
        write_binary_big_endian_int(out, types.data(), types.size());
    }

    out << "CELL_DATA " << mesh.cells.size() << "\n";
    out << "SCALARS rho float 1\n";
    out << "LOOKUP_TABLE default\n";
    {
        std::vector<float> rho_vals(q.size());
        for (std::size_t i = 0; i < q.size(); ++i)
            rho_vals[i] = static_cast<float>(primitive[i].rho);
        write_binary_big_endian(out, rho_vals.data(), rho_vals.size());
    }
    out << "SCALARS pressure float 1\n";
    out << "LOOKUP_TABLE default\n";
    {
        std::vector<float> p_vals(q.size());
        for (std::size_t i = 0; i < q.size(); ++i)
            p_vals[i] = static_cast<float>(primitive[i].p);
        write_binary_big_endian(out, p_vals.data(), p_vals.size());
    }
    out << "SCALARS mach float 1\n";
    out << "LOOKUP_TABLE default\n";
    {
        std::vector<float> mach_vals(q.size());
        for (std::size_t i = 0; i < q.size(); ++i)
            mach_vals[i] = static_cast<float>(mach[i]);
        write_binary_big_endian(out, mach_vals.data(), mach_vals.size());
    }

    if (!out) {
        if (error) *error = "failed while writing";
        return false;
    }
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp

