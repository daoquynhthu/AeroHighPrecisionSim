#include "aero/cfd/real.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/element_types.hpp"
#include "aero/cfd/mesh_validator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

struct Vec3 {
    Real x, y, z;
};

Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator*(Vec3 a, Real s) { return {a.x * s, a.y * s, a.z * s}; }

Real dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
Real norm(Vec3 a) { return std::sqrt(dot(a, a)); }

Vec3 to_vec(const CfdNode& n) {
    if (!std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z))
        return {0, 0, 0};
    return {n.x, n.y, n.z};
}

Real signed_volume_tet(Vec3 a, Vec3 b, Vec3 c, Vec3 d) {
    return dot(b - a, cross(c - a, d - a)) / 6.0f;
}

Real element_aspect_ratio(const CfdMesh& mesh, const CfdCell& cell) {
    int nv = ELEMENT_NODES[static_cast<int>(cell.type)];
    if (nv < 2) return 1.0f;
    Real min_edge = std::numeric_limits<Real>::max();
    Real max_edge = 0.0f;
    for (int i = 0; i < nv; ++i) {
        Vec3 a = to_vec(mesh.nodes[cell.node[i]]);
        for (int j = i + 1; j < nv; ++j) {
            Vec3 b = to_vec(mesh.nodes[cell.node[j]]);
            Real len = norm(b - a);
            if (len < 1e-30f) continue;
            min_edge = std::min(min_edge, len);
            max_edge = std::max(max_edge, len);
        }
    }
    if (min_edge < 1e-30f || max_edge < 1e-30f) return 1.0f;
    return max_edge / min_edge;
}

Real tet_corner_jacobian(const Vec3 v[4]) {
    return signed_volume_tet(v[0], v[1], v[2], v[3]) * 6.0f;
}

bool tet_jacobian_sign(const Vec3 v[4], int& neg_count) {
    Real J = tet_corner_jacobian(v);
    if (J <= 0.0f) { neg_count++; return false; }
    return true;
}

Real hex_corner_jacobian(const Vec3 v[8], int corner) {
    static const int corner_to_param[8] = {0, 1, 3, 2, 4, 5, 7, 6};
    int cp = corner_to_param[corner];
    int cr = (cp & 1) ? 1 : -1;
    int cs = (cp & 2) ? 1 : -1;
    int ct = (cp & 4) ? 1 : -1;
    Real J[3][3] = {};
    for (int i = 0; i < 8; ++i) {
        int ip = corner_to_param[i];
        int nr = (ip & 1) ? 1 : -1;
        int ns = (ip & 2) ? 1 : -1;
        int nt = (ip & 4) ? 1 : -1;
        Real dNdr = nr * (1.0f + ns * cs) * (1.0f + nt * ct) / 8.0f;
        Real dNds = ns * (1.0f + nr * cr) * (1.0f + nt * ct) / 8.0f;
        Real dNdt = nt * (1.0f + nr * cr) * (1.0f + ns * cs) / 8.0f;
        J[0][0] += v[i].x * dNdr; J[0][1] += v[i].y * dNdr; J[0][2] += v[i].z * dNdr;
        J[1][0] += v[i].x * dNds; J[1][1] += v[i].y * dNds; J[1][2] += v[i].z * dNds;
        J[2][0] += v[i].x * dNdt; J[2][1] += v[i].y * dNdt; J[2][2] += v[i].z * dNdt;
    }
    Real det = J[0][0] * (J[1][1]*J[2][2] - J[1][2]*J[2][1])
             - J[0][1] * (J[1][0]*J[2][2] - J[1][2]*J[2][0])
             + J[0][2] * (J[1][0]*J[2][1] - J[1][1]*J[2][0]);
    return det;
}

bool hex_jacobian_sign(const Vec3 v[8], int& neg_count) {
    bool all_positive = true;
    for (int c = 0; c < 8; ++c) {
        Real J = hex_corner_jacobian(v, c);
        if (J <= 0.0f) { neg_count++; all_positive = false; }
    }
    return all_positive;
}

Real penta_corner_jacobian(const Vec3 v[6], int corner) {
    // Wedge parametric coordinates (xi, eta, zeta):
    //   xi, eta in [0,1]  (triangle barycentric coordinates)
    //   zeta in [-1, 1]  (extrusion direction)
    // Corner mapping:
    //   0: (1,0,-1)  3: (1,0,1)
    //   1: (0,1,-1)  4: (0,1,1)
    //   2: (0,0,-1)  5: (0,0,1)
    constexpr Real xi_c[6] = {1,0,0, 1,0,0};
    constexpr Real eta_c[6] = {0,1,0, 0,1,0};
    constexpr Real zeta_c[6] = {-1,-1,-1, 1,1,1};
    Real xi = xi_c[corner], eta = eta_c[corner], zeta = zeta_c[corner];
    Real dN[6][3] = {};
    // dN[i][0] = dNi/dxi, dN[i][1] = dNi/deta, dN[i][2] = dNi/dzeta
    dN[0][0] = -0.5f * (1.0f - zeta); dN[0][1] = -0.5f * (1.0f - zeta); dN[0][2] = -0.5f * (1.0f - xi - eta);
    dN[1][0] =  0.5f * (1.0f - zeta); dN[1][1] =  0.0f;                 dN[1][2] = -0.5f * xi;
    dN[2][0] =  0.0f;                 dN[2][1] =  0.5f * (1.0f - zeta); dN[2][2] = -0.5f * eta;
    dN[3][0] = -0.5f * (1.0f + zeta); dN[3][1] = -0.5f * (1.0f + zeta); dN[3][2] =  0.5f * (1.0f - xi - eta);
    dN[4][0] =  0.5f * (1.0f + zeta); dN[4][1] =  0.0f;                 dN[4][2] =  0.5f * xi;
    dN[5][0] =  0.0f;                 dN[5][1] =  0.5f * (1.0f + zeta); dN[5][2] =  0.5f * eta;
    Real J[3][3] = {};
    for (int i = 0; i < 6; ++i) {
        J[0][0] += v[i].x * dN[i][0]; J[0][1] += v[i].y * dN[i][0]; J[0][2] += v[i].z * dN[i][0];
        J[1][0] += v[i].x * dN[i][1]; J[1][1] += v[i].y * dN[i][1]; J[1][2] += v[i].z * dN[i][1];
        J[2][0] += v[i].x * dN[i][2]; J[2][1] += v[i].y * dN[i][2]; J[2][2] += v[i].z * dN[i][2];
    }
    Real det = J[0][0] * (J[1][1]*J[2][2] - J[1][2]*J[2][1])
             - J[0][1] * (J[1][0]*J[2][2] - J[1][2]*J[2][0])
             + J[0][2] * (J[1][0]*J[2][1] - J[1][1]*J[2][0]);
    return det;
}

bool penta_jacobian_sign(const Vec3 v[6], int& neg_count) {
    bool all_positive = true;
    for (int c = 0; c < 6; ++c) {
        Real J = penta_corner_jacobian(v, c);
        if (J <= 0.0f) { neg_count++; all_positive = false; }
    }
    return all_positive;
}

Real pyramid_corner_jacobian(const Vec3 v[5], int corner) {
    Real xi[] = {-1.0f, 1.0f, 1.0f, -1.0f, 0.0f};
    Real eta[] = {-1.0f, -1.0f, 1.0f, 1.0f, 0.0f};
    Real zeta[] = {-1.0f, -1.0f, -1.0f, -1.0f, 1.0f};
    Real r = xi[corner], s = eta[corner], t = zeta[corner];
    Real dNdr[5], dNds[5], dNdt[5];
    Real denom = 8.0f;
    for (int i = 0; i < 5; ++i) {
        Real ri = xi[i], si = eta[i], ti = zeta[i];
        if (i < 4) {
            dNdr[i] = ri * (1.0f + si*s) * (1.0f + ti*t) / denom;
            dNds[i] = si * (1.0f + ri*r) * (1.0f + ti*t) / denom;
            dNdt[i] = ti * (1.0f + ri*r) * (1.0f + si*s) / denom;
        } else {
            Real coeff = 0.5f * (1.0f - t);
            dNdr[i] = 0.5f * (1.0f - t) * r;
            dNds[i] = 0.5f * (1.0f - t) * s;
            dNdt[i] = -0.5f * (1.0f + r*ri) * (1.0f + s*si) + 0.25f * (1.0f - t*t) * ti;
        }
    }
    Real J[3][3] = {};
    for (int i = 0; i < 5; ++i) {
        J[0][0] += v[i].x * dNdr[i]; J[0][1] += v[i].y * dNdr[i]; J[0][2] += v[i].z * dNdr[i];
        J[1][0] += v[i].x * dNds[i]; J[1][1] += v[i].y * dNds[i]; J[1][2] += v[i].z * dNds[i];
        J[2][0] += v[i].x * dNdt[i]; J[2][1] += v[i].y * dNdt[i]; J[2][2] += v[i].z * dNdt[i];
    }
    return J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
         - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
         + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
}

bool pyramid_jacobian_sign(const Vec3 v[5], int& neg_count) {
    bool all_positive = true;
    for (int c = 0; c < 5; ++c) {
        Real J = pyramid_corner_jacobian(v, c);
        if (J <= 0.0f) { neg_count++; all_positive = false; }
    }
    return all_positive;
}

} // namespace

bool tet_jacobian_sign(const CfdNode* nodes, int& neg_count) {
    Vec3 v[4];
    for (int i = 0; i < 4; ++i) v[i] = to_vec(nodes[i]);
    return tet_jacobian_sign(v, neg_count);
}

bool hex_jacobian_sign(const CfdNode* nodes, int& neg_count) {
    Vec3 v[8];
    for (int i = 0; i < 8; ++i) v[i] = to_vec(nodes[i]);
    return hex_jacobian_sign(v, neg_count);
}

bool penta_jacobian_sign(const CfdNode* nodes, int& neg_count) {
    Vec3 v[6];
    for (int i = 0; i < 6; ++i) v[i] = to_vec(nodes[i]);
    return penta_jacobian_sign(v, neg_count);
}

bool pyramid_jacobian_sign(const CfdNode* nodes, int& neg_count) {
    Vec3 v[5];
    for (int i = 0; i < 5; ++i) v[i] = to_vec(nodes[i]);
    return pyramid_jacobian_sign(v, neg_count);
}

MeshQualityReport compute_mesh_quality_detail(const CfdMesh& mesh) {
    MeshQualityReport r;
    if (mesh.nodes.empty() || mesh.cells.empty() || mesh.faces.empty()) {
        r.valid = false;
        r.message = "mesh is empty";
        return r;
    }

    r.nodes = static_cast<int>(mesh.nodes.size());
    r.cells = static_cast<int>(mesh.cells.size());
    r.faces = static_cast<int>(mesh.faces.size());

    r.min_volume = std::numeric_limits<Real>::max();
    r.max_volume = 0.0f;
    r.min_h = std::numeric_limits<Real>::max();
    r.max_h = 0.0f;
    r.min_wall_distance = std::numeric_limits<Real>::max();
    r.min_aspect_ratio = std::numeric_limits<Real>::max();
    r.max_aspect_ratio = 0.0f;
    r.min_skewness = std::numeric_limits<Real>::max();
    r.max_skewness = 0.0f;
    r.min_orthogonality = std::numeric_limits<Real>::max();
    r.max_orthogonality = 0.0f;
    Real sum_aspect = 0.0f, sum_skew = 0.0f, sum_ortho = 0.0f;

    Vec3 wall_area_sum = {0.0f, 0.0f, 0.0f};

    for (int fi = 0; fi < static_cast<int>(mesh.faces.size()); ++fi) {
        const CfdFace& face = mesh.faces[fi];
        switch (face.boundary) {
            case BoundaryKind::Interior: r.interior_faces++; break;
            case BoundaryKind::Farfield: r.farfield_faces++; break;
            case BoundaryKind::SlipWall: r.slip_wall_faces++; break;
            case BoundaryKind::NoSlipWall: r.no_slip_wall_faces++; break;
            case BoundaryKind::Symmetry: r.symmetry_faces++; break;
        }
        if (face.boundary != BoundaryKind::Interior) {
            wall_area_sum.x += face.area * face.nx;
            wall_area_sum.y += face.area * face.ny;
            wall_area_sum.z += face.area * face.nz;
        }
    }
    r.closed_surface_error = norm(wall_area_sum);

    for (int ci = 0; ci < static_cast<int>(mesh.cells.size()); ++ci) {
        const CfdCell& cell = mesh.cells[ci];
        r.min_volume = std::min(r.min_volume, cell.volume);
        r.max_volume = std::max(r.max_volume, cell.volume);
        r.min_h = std::min(r.min_h, cell.h_min);
        r.max_h = std::max(r.max_h, cell.h_min);

        Real ar = element_aspect_ratio(mesh, cell);
        r.min_aspect_ratio = std::min(r.min_aspect_ratio, ar);
        r.max_aspect_ratio = std::max(r.max_aspect_ratio, ar);
        sum_aspect += ar;
        if (ar > 1000.0f) r.high_aspect_ratio_count++;

        Vec3 cc = {cell.cx, cell.cy, cell.cz};
        Real cell_skew_sum = 0.0f;
        Real cell_ortho_sum = 0.0f;
        int cell_face_count = 0;

        int nf = ELEMENT_FACES[static_cast<int>(cell.type)];
        for (int lf = 0; lf < nf; ++lf) {
            if (lf >= cell.face_count) break;
            int fi = cell.first_face + lf;
            if (fi < 0 || fi >= static_cast<int>(mesh.faces.size())) break;
            const CfdFace& face = mesh.faces[fi];
            Vec3 fc = {face.cx, face.cy, face.cz};
            Vec3 cf = fc - cc;
            Real cf_len = norm(cf);
            if (cf_len < 1e-30f) continue;
            cf.x /= cf_len; cf.y /= cf_len; cf.z /= cf_len;
            Vec3 fn = {face.nx, face.ny, face.nz};
            Real d = std::fabs(dot(cf, fn));
            Real ortho = std::acos(std::min(std::max(d, Real(-1.0)), Real(1.0))) * Real(180.0) / Real(3.14159265358979323846);
            Real skew = std::fabs(90.0f - ortho) / 90.0f;
            cell_ortho_sum += ortho;
            cell_skew_sum += skew;
            cell_face_count++;
        }

        if (cell_face_count > 0) {
            Real avg_ortho = cell_ortho_sum / cell_face_count;
            Real avg_skew = cell_skew_sum / cell_face_count;
            r.min_orthogonality = std::min(r.min_orthogonality, avg_ortho);
            r.max_orthogonality = std::max(r.max_orthogonality, avg_ortho);
            r.min_skewness = std::min(r.min_skewness, avg_skew);
            r.max_skewness = std::max(r.max_skewness, avg_skew);
            sum_ortho += avg_ortho;
            sum_skew += avg_skew;
            if (avg_skew > 0.95f) r.high_skew_count++;
        }

        Vec3 v[8];
        int nv = ELEMENT_NODES[static_cast<int>(cell.type)];
        for (int i = 0; i < nv; ++i) v[i] = to_vec(mesh.nodes[cell.node[i]]);

        switch (cell.type) {
            case ElementType::TET4: {
                Vec3 vt[4] = {v[0], v[1], v[2], v[3]};
                tet_jacobian_sign(vt, r.negative_jacobian_count);
                break;
            }
            case ElementType::HEX8: {
                hex_jacobian_sign(v, r.negative_jacobian_count);
                break;
            }
            case ElementType::PENTA6: {
                Vec3 vp[6] = {v[0], v[1], v[2], v[3], v[4], v[5]};
                penta_jacobian_sign(vp, r.negative_jacobian_count);
                break;
            }
            case ElementType::PYRAMID5: {
                Vec3 vp[5] = {v[0], v[1], v[2], v[3], v[4]};
                pyramid_jacobian_sign(vp, r.negative_jacobian_count);
                break;
            }
        }
    }

    r.avg_aspect_ratio = sum_aspect / static_cast<Real>(r.cells);
    r.avg_skewness = sum_skew / static_cast<Real>(r.cells);
    r.avg_orthogonality = sum_ortho / static_cast<Real>(r.cells);

    r.valid = (r.negative_jacobian_count == 0) && (r.min_volume > 0.0f);
    if (!r.valid) {
        if (r.negative_jacobian_count > 0)
            r.message = "negative Jacobian detected";
        else if (!(r.min_volume > 0.0f))
            r.message = "non-positive or NaN cell volume";
    }

    return r;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
