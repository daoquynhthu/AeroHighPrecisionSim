#include "aero/cfd/reconstruction.hpp"

#include <algorithm>
#include <cmath>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

PrimitiveState average_state(const PrimitiveState& a, const PrimitiveState& b) {
    PrimitiveState out;
    out.rho = 0.5f * (a.rho + b.rho);
    out.u = 0.5f * (a.u + b.u);
    out.v = 0.5f * (a.v + b.v);
    out.w = 0.5f * (a.w + b.w);
    out.p = 0.5f * (a.p + b.p);
    out.nu_tilde = 0.5f * (a.nu_tilde + b.nu_tilde);
    return out;
}

void add_face_contribution(PrimitiveGradient& g, const PrimitiveState& w, const PrimitiveState& center,
    Real nx, Real ny, Real nz, Real scale) {
    Real sx = nx * scale;
    Real sy = ny * scale;
    Real sz = nz * scale;
    g.drho_dx += (w.rho - center.rho) * sx;
    g.drho_dy += (w.rho - center.rho) * sy;
    g.drho_dz += (w.rho - center.rho) * sz;
    g.du_dx += (w.u - center.u) * sx;
    g.du_dy += (w.u - center.u) * sy;
    g.du_dz += (w.u - center.u) * sz;
    g.dv_dx += (w.v - center.v) * sx;
    g.dv_dy += (w.v - center.v) * sy;
    g.dv_dz += (w.v - center.v) * sz;
    g.dw_dx += (w.w - center.w) * sx;
    g.dw_dy += (w.w - center.w) * sy;
    g.dw_dz += (w.w - center.w) * sz;
    g.dp_dx += (w.p - center.p) * sx;
    g.dp_dy += (w.p - center.p) * sy;
    g.dp_dz += (w.p - center.p) * sz;
    g.dnu_tilde_dx += (w.nu_tilde - center.nu_tilde) * sx;
    g.dnu_tilde_dy += (w.nu_tilde - center.nu_tilde) * sy;
    g.dnu_tilde_dz += (w.nu_tilde - center.nu_tilde) * sz;
}

Real positive_theta(Real center, Real reconstructed, Real floor_value) {
    if (reconstructed >= floor_value) return 1.0f;
    if (center <= floor_value) return 0.0f;
    Real denom = center - reconstructed;
    if (denom <= 0.0f || !std::isfinite(denom)) return 0.0f;
    return std::max(Real(0.0), std::min(Real(1.0), (center - floor_value) / denom));
}

PrimitiveGradient scale_gradient(const PrimitiveGradient& g, Real theta) {
    PrimitiveGradient out = g;
    out.drho_dx *= theta;
    out.drho_dy *= theta;
    out.drho_dz *= theta;
    out.du_dx *= theta;
    out.du_dy *= theta;
    out.du_dz *= theta;
    out.dv_dx *= theta;
    out.dv_dy *= theta;
    out.dv_dz *= theta;
    out.dw_dx *= theta;
    out.dw_dy *= theta;
    out.dw_dz *= theta;
    out.dp_dx *= theta;
    out.dp_dy *= theta;
    out.dp_dz *= theta;
    out.dnu_tilde_dx *= theta;
    out.dnu_tilde_dy *= theta;
    out.dnu_tilde_dz *= theta;
    return out;
}

void accumulate_least_squares_matrix(Real a[3][3], Real dx, Real dy, Real dz) {
    Real d[3] = {dx, dy, dz};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            a[r][c] += d[r] * d[c];
        }
    }
}

void accumulate_least_squares_rhs(Real b[3], Real dx, Real dy, Real dz, Real dphi) {
    b[0] += dx * dphi;
    b[1] += dy * dphi;
    b[2] += dz * dphi;
}

void assign_gradient_component(PrimitiveGradient& g, int component, const Real x[3]) {
    switch (component) {
        case 0: g.drho_dx = x[0]; g.drho_dy = x[1]; g.drho_dz = x[2]; break;
        case 1: g.du_dx = x[0]; g.du_dy = x[1]; g.du_dz = x[2]; break;
        case 2: g.dv_dx = x[0]; g.dv_dy = x[1]; g.dv_dz = x[2]; break;
        case 3: g.dw_dx = x[0]; g.dw_dy = x[1]; g.dw_dz = x[2]; break;
        case 4: g.dp_dx = x[0]; g.dp_dy = x[1]; g.dp_dz = x[2]; break;
        default: g.dnu_tilde_dx = x[0]; g.dnu_tilde_dy = x[1]; g.dnu_tilde_dz = x[2]; break;
    }
}

Real primitive_component(const PrimitiveState& w, int component) {
    switch (component) {
        case 0: return w.rho;
        case 1: return w.u;
        case 2: return w.v;
        case 3: return w.w;
        case 4: return w.p;
        default: return w.nu_tilde;
    }
}

Real limiter_theta(Real center, Real reconstructed, Real min_value, Real max_value) {
    if (reconstructed > max_value) {
        Real denom = reconstructed - center;
        if (denom <= 0.0f) return 0.0f;
        return std::max(Real(0.0), std::min(Real(1.0), (max_value - center) / denom));
    }
    if (reconstructed < min_value) {
        Real denom = reconstructed - center;
        if (denom >= 0.0f) return 0.0f;
        return std::max(Real(0.0), std::min(Real(1.0), (min_value - center) / denom));
    }
    return 1.0f;
}

void update_minmax(PrimitiveState& min_w, PrimitiveState& max_w, const PrimitiveState& w) {
    min_w.rho = std::min(min_w.rho, w.rho);
    min_w.u = std::min(min_w.u, w.u);
    min_w.v = std::min(min_w.v, w.v);
    min_w.w = std::min(min_w.w, w.w);
    min_w.p = std::min(min_w.p, w.p);
    min_w.nu_tilde = std::min(min_w.nu_tilde, w.nu_tilde);
    max_w.rho = std::max(max_w.rho, w.rho);
    max_w.u = std::max(max_w.u, w.u);
    max_w.v = std::max(max_w.v, w.v);
    max_w.w = std::max(max_w.w, w.w);
    max_w.p = std::max(max_w.p, w.p);
    max_w.nu_tilde = std::max(max_w.nu_tilde, w.nu_tilde);
}

void update_limiter(PrimitiveLimiter& limiter, const PrimitiveState& center, const PrimitiveState& reconstructed,
    const PrimitiveState& min_w, const PrimitiveState& max_w) {
    limiter.rho = std::min(limiter.rho, limiter_theta(center.rho, reconstructed.rho, min_w.rho, max_w.rho));
    limiter.u = std::min(limiter.u, limiter_theta(center.u, reconstructed.u, min_w.u, max_w.u));
    limiter.v = std::min(limiter.v, limiter_theta(center.v, reconstructed.v, min_w.v, max_w.v));
    limiter.w = std::min(limiter.w, limiter_theta(center.w, reconstructed.w, min_w.w, max_w.w));
    limiter.p = std::min(limiter.p, limiter_theta(center.p, reconstructed.p, min_w.p, max_w.p));
    limiter.nu_tilde = std::min(limiter.nu_tilde, limiter_theta(center.nu_tilde, reconstructed.nu_tilde, min_w.nu_tilde, max_w.nu_tilde));
}

} // namespace

bool solve_3x3(Real a[3][3], Real b[3], Real x[3]) {
    Real m[3][4] = {
        {a[0][0], a[0][1], a[0][2], b[0]},
        {a[1][0], a[1][1], a[1][2], b[1]},
        {a[2][0], a[2][1], a[2][2], b[2]}
    };

    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row) {
            if (std::fabs(m[row][col]) > std::fabs(m[pivot][col])) pivot = row;
        }
        Real col_max = 0.0f;
        for (int r = 0; r < 3; ++r) col_max = (std::fabs(m[r][col]) > col_max) ? std::fabs(m[r][col]) : col_max;
        if (col_max < Real(1e-30) || std::fabs(m[pivot][col]) < col_max * Real(1e-6)) return false;
        if (pivot != col) {
            for (int j = col; j < 4; ++j) std::swap(m[col][j], m[pivot][j]);
        }
        Real inv = 1.0f / m[col][col];
        for (int j = col; j < 4; ++j) m[col][j] *= inv;
        for (int row = 0; row < 3; ++row) {
            if (row == col) continue;
            Real factor = m[row][col];
            for (int j = col; j < 4; ++j) m[row][j] -= factor * m[col][j];
        }
    }

    x[0] = m[0][3];
    x[1] = m[1][3];
    x[2] = m[2][3];
    return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
}

bool lu_factor_3x3(const Real a[3][3], Real lu[3][3], int pivot[3]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            lu[r][c] = a[r][c];
    pivot[0] = 0; pivot[1] = 1; pivot[2] = 2;

    for (int k = 0; k < 3; ++k) {
        int p = k;
        Real max_abs = std::fabs(lu[k][k]);
        for (int r = k + 1; r < 3; ++r) {
            Real abs_val = std::fabs(lu[r][k]);
            if (abs_val > max_abs) { max_abs = abs_val; p = r; }
        }
        Real col_max = 0.0f;
        for (int r = k; r < 3; ++r)
            col_max = std::max(col_max, std::fabs(lu[r][k]));
        if (col_max < Real(1e-30) || max_abs < col_max * Real(1e-6)) return false;

        if (p != k) {
            for (int j = 0; j < 3; ++j) std::swap(lu[k][j], lu[p][j]);
            std::swap(pivot[k], pivot[p]);
        }
        Real inv_diag = 1.0f / lu[k][k];
        for (int i = k + 1; i < 3; ++i) {
            Real factor = lu[i][k] * inv_diag;
            lu[i][k] = factor;
            for (int j = k + 1; j < 3; ++j)
                lu[i][j] -= factor * lu[k][j];
        }
    }
    return true;
}

void lu_solve_3x3(const Real lu[3][3], const int pivot[3], const Real b[3], Real x[3]) {
    Real y[3];
    for (int i = 0; i < 3; ++i) {
        y[i] = b[pivot[i]];
        for (int j = 0; j < i; ++j)
            y[i] -= lu[i][j] * y[j];
    }
    for (int i = 2; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < 3; ++j)
            x[i] -= lu[i][j] * x[j];
        x[i] /= lu[i][i];
    }
}

std::vector<PrimitiveGradient> compute_green_gauss_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    Real gamma,
    const std::vector<PrimitiveState>* primitive_override) {
    if (q.size() != mesh.cells.size()) return {};

    const std::vector<PrimitiveState>* local_primitive = primitive_override;
    std::vector<PrimitiveState> fallback;
    if (!local_primitive || local_primitive->size() != q.size()) {
        fallback.resize(q.size());
        for (std::size_t i = 0; i < q.size(); ++i) {
            if (!conservative_to_primitive(q[i], gamma, fallback[i])) return {};
        }
        local_primitive = &fallback;
    }

    std::vector<PrimitiveGradient> gradients(q.size());
    for (const auto& face : mesh.faces) {
        const PrimitiveState& wl = (*local_primitive)[face.left_cell];
        PrimitiveState wf = wl;
        if (face.boundary == BoundaryKind::Interior) {
            wf = average_state(wl, (*local_primitive)[face.right_cell]);
        }

        Real left_scale = face.area / mesh.cells[face.left_cell].volume;
        add_face_contribution(gradients[face.left_cell], wf, wl, face.nx, face.ny, face.nz, left_scale);

        if (face.boundary == BoundaryKind::Interior) {
            const PrimitiveState& wr = (*local_primitive)[face.right_cell];
            Real right_scale = -face.area / mesh.cells[face.right_cell].volume;
            add_face_contribution(gradients[face.right_cell], wf, wr, face.nx, face.ny, face.nz, right_scale);
        }
    }

    return gradients;
}

std::vector<PrimitiveGradient> compute_least_squares_gradients(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    Real gamma,
    const std::vector<PrimitiveState>* primitive_override) {
    if (q.size() != mesh.cells.size()) return {};

    const std::vector<PrimitiveState>* local_primitive = primitive_override;
    std::vector<PrimitiveState> fallback;
    if (!local_primitive || local_primitive->size() != q.size()) {
        fallback.resize(q.size());
        for (std::size_t i = 0; i < q.size(); ++i) {
            if (!conservative_to_primitive(q[i], gamma, fallback[i])) return {};
        }
        local_primitive = &fallback;
    }

    struct CellSystem {
        Real a[3][3] = {};
        Real b[6][3] = {};
    };

    std::vector<CellSystem> systems(q.size());
    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::Interior) continue;
        int left = face.left_cell;
        int right = face.right_cell;
        Real dx = mesh.cells[right].cx - mesh.cells[left].cx;
        Real dy = mesh.cells[right].cy - mesh.cells[left].cy;
        Real dz = mesh.cells[right].cz - mesh.cells[left].cz;
        accumulate_least_squares_matrix(systems[left].a, dx, dy, dz);
        accumulate_least_squares_matrix(systems[right].a, -dx, -dy, -dz);
        for (int component = 0; component < 6; ++component) {
            Real dphi = primitive_component((*local_primitive)[right], component) - primitive_component((*local_primitive)[left], component);
            accumulate_least_squares_rhs(systems[left].b[component], dx, dy, dz, dphi);
            accumulate_least_squares_rhs(systems[right].b[component], -dx, -dy, -dz, -dphi);
        }
    }

    std::vector<PrimitiveGradient> gradients(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) {
        Real lu[3][3];
        int pivot[3];
        if (!lu_factor_3x3(systems[i].a, lu, pivot)) continue;
        Real x[3];
        for (int component = 0; component < 6; ++component) {
            lu_solve_3x3(lu, pivot, systems[i].b[component], x);
            if (std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]))
                assign_gradient_component(gradients[i], component, x);
        }
    }

    return gradients;
}

std::vector<PrimitiveLimiter> compute_barth_jespersen_limiters(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<PrimitiveGradient>& gradients,
    Real gamma,
    const std::vector<PrimitiveState>* primitive_override) {
    if (q.size() != mesh.cells.size() || gradients.size() != mesh.cells.size()) return {};

    const std::vector<PrimitiveState>* local_primitive = primitive_override;
    std::vector<PrimitiveState> fallback;
    if (!local_primitive || local_primitive->size() != q.size()) {
        fallback.resize(q.size());
        for (std::size_t i = 0; i < q.size(); ++i) {
            if (!conservative_to_primitive(q[i], gamma, fallback[i])) return {};
        }
        local_primitive = &fallback;
    }

    std::vector<PrimitiveState> min_w = *local_primitive;
    std::vector<PrimitiveState> max_w = *local_primitive;
    for (const auto& face : mesh.faces) {
        if (face.boundary != BoundaryKind::Interior) continue;
        update_minmax(min_w[face.left_cell], max_w[face.left_cell], (*local_primitive)[face.right_cell]);
        update_minmax(min_w[face.right_cell], max_w[face.right_cell], (*local_primitive)[face.left_cell]);
    }

    std::vector<PrimitiveLimiter> limiters(q.size());
    for (const auto& face : mesh.faces) {
        const CfdCell& left = mesh.cells[face.left_cell];
        PrimitiveState left_recon = reconstruct_primitive(
            (*local_primitive)[face.left_cell],
            gradients[face.left_cell],
            face.cx - left.cx,
            face.cy - left.cy,
            face.cz - left.cz);
        update_limiter(limiters[face.left_cell], (*local_primitive)[face.left_cell], left_recon,
            min_w[face.left_cell], max_w[face.left_cell]);

        if (face.boundary == BoundaryKind::Interior) {
            const CfdCell& right = mesh.cells[face.right_cell];
            PrimitiveState right_recon = reconstruct_primitive(
                (*local_primitive)[face.right_cell],
                gradients[face.right_cell],
                face.cx - right.cx,
                face.cy - right.cy,
                face.cz - right.cz);
            update_limiter(limiters[face.right_cell], (*local_primitive)[face.right_cell], right_recon,
                min_w[face.right_cell], max_w[face.right_cell]);
        }
    }

    return limiters;
}

PrimitiveGradient apply_limiter(const PrimitiveGradient& gradient, const PrimitiveLimiter& limiter) {
    PrimitiveGradient out = gradient;
    out.drho_dx *= limiter.rho;
    out.drho_dy *= limiter.rho;
    out.drho_dz *= limiter.rho;
    out.du_dx *= limiter.u;
    out.du_dy *= limiter.u;
    out.du_dz *= limiter.u;
    out.dv_dx *= limiter.v;
    out.dv_dy *= limiter.v;
    out.dv_dz *= limiter.v;
    out.dw_dx *= limiter.w;
    out.dw_dy *= limiter.w;
    out.dw_dz *= limiter.w;
    out.dp_dx *= limiter.p;
    out.dp_dy *= limiter.p;
    out.dp_dz *= limiter.p;
    out.dnu_tilde_dx *= limiter.nu_tilde;
    out.dnu_tilde_dy *= limiter.nu_tilde;
    out.dnu_tilde_dz *= limiter.nu_tilde;
    return out;
}

PrimitiveState reconstruct_primitive(
    const PrimitiveState& center,
    const PrimitiveGradient& gradient,
    Real dx,
    Real dy,
    Real dz) {
    PrimitiveState out;
    out.rho = center.rho + gradient.drho_dx*dx + gradient.drho_dy*dy + gradient.drho_dz*dz;
    out.u = center.u + gradient.du_dx*dx + gradient.du_dy*dy + gradient.du_dz*dz;
    out.v = center.v + gradient.dv_dx*dx + gradient.dv_dy*dy + gradient.dv_dz*dz;
    out.w = center.w + gradient.dw_dx*dx + gradient.dw_dy*dy + gradient.dw_dz*dz;
    out.p = center.p + gradient.dp_dx*dx + gradient.dp_dy*dy + gradient.dp_dz*dz;
    out.nu_tilde = center.nu_tilde + gradient.dnu_tilde_dx*dx + gradient.dnu_tilde_dy*dy + gradient.dnu_tilde_dz*dz;
    return out;
}

PrimitiveState reconstruct_primitive_positive(
    const PrimitiveState& center,
    const PrimitiveGradient& gradient,
    Real dx,
    Real dy,
    Real dz,
    Real rho_floor,
    Real p_floor,
    Real* theta) {
    PrimitiveState raw = reconstruct_primitive(center, gradient, dx, dy, dz);
    Real t = std::min(
        positive_theta(center.rho, raw.rho, rho_floor),
        positive_theta(center.p, raw.p, p_floor));
    if (theta) *theta = t;
    return reconstruct_primitive(center, scale_gradient(gradient, t), dx, dy, dz);
}

} // namespace cfd
} // namespace aero
} // namespace aerosp

