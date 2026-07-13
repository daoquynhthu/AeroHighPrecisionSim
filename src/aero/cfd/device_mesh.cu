#include "aero/cfd/device_mesh.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/cuda_utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <utility>
namespace aerosp {
namespace aero {
namespace cfd {

DeviceMesh::~DeviceMesh() {
    release();
}

DeviceMesh::DeviceMesh(DeviceMesh&& other) noexcept {
    *this = std::move(other);
}

DeviceMesh& DeviceMesh::operator=(DeviceMesh&& other) noexcept {
    if (this == &other) return *this;
    release();
    cell_count_ = other.cell_count_;
    face_count_ = other.face_count_;
    d_q_ = other.d_q_;
    d_residual_ = other.d_residual_;
    d_nx_ = other.d_nx_;
    d_ny_ = other.d_ny_;
    d_nz_ = other.d_nz_;
    d_area_ = other.d_area_;
    d_left_cell_ = other.d_left_cell_;
    d_right_cell_ = other.d_right_cell_;
    d_boundary_ = other.d_boundary_;
    d_volume_ = other.d_volume_;
    d_h_min_ = other.d_h_min_;
    d_wall_distance_ = other.d_wall_distance_;
    d_cx_ = other.d_cx_;
    d_cy_ = other.d_cy_;
    d_cz_ = other.d_cz_;
    d_face_cx_ = other.d_face_cx_;
    d_face_cy_ = other.d_face_cy_;
    d_face_cz_ = other.d_face_cz_;
    d_gradients_ = other.d_gradients_;
    d_limiters_ = other.d_limiters_;
    d_minmax_ = other.d_minmax_;
    d_mu_ = other.d_mu_;
    d_lam_ = other.d_lam_;
    d_delta_ddes_ = other.d_delta_ddes_;
    d_q_k_ = other.d_q_k_;
    d_q_omega_ = other.d_q_omega_;
    d_residual_k_ = other.d_residual_k_;
    d_residual_omega_ = other.d_residual_omega_;
    d_grad_k_ = other.d_grad_k_;
    d_grad_omega_ = other.d_grad_omega_;
    d_w_ = other.d_w_;
    other.cell_count_ = 0;
    other.face_count_ = 0;
    other.d_q_ = nullptr;
    other.d_residual_ = nullptr;
    other.d_nx_ = other.d_ny_ = other.d_nz_ = nullptr;
    other.d_area_ = nullptr;
    other.d_left_cell_ = other.d_right_cell_ = nullptr;
    other.d_boundary_ = nullptr;
    other.d_volume_ = other.d_h_min_ = other.d_wall_distance_ = nullptr;
    other.d_cx_ = other.d_cy_ = other.d_cz_ = nullptr;
    other.d_face_cx_ = other.d_face_cy_ = other.d_face_cz_ = nullptr;
    other.d_gradients_ = nullptr;
    other.d_limiters_ = nullptr;
    other.d_minmax_ = nullptr;
    other.d_mu_ = nullptr;
    other.d_lam_ = nullptr;
    other.d_delta_ddes_ = nullptr;
    other.d_q_k_ = other.d_q_omega_ = nullptr;
    other.d_residual_k_ = other.d_residual_omega_ = nullptr;
    other.d_grad_k_ = other.d_grad_omega_ = nullptr;
    other.d_w_ = nullptr;
    d_halo_indices_ = other.d_halo_indices_;
    d_halo_send_buf_ = other.d_halo_send_buf_;
    d_halo_recv_buf_ = other.d_halo_recv_buf_;
    n_halo_cells_ = other.n_halo_cells_;
    other.d_halo_indices_ = nullptr;
    other.d_halo_send_buf_ = nullptr;
    other.d_halo_recv_buf_ = nullptr;
    other.n_halo_cells_ = 0;
    n_colors_ = other.n_colors_;
    d_color_offsets_ = other.d_color_offsets_;
    host_color_offsets_ = std::move(other.host_color_offsets_);
    other.n_colors_ = 0;
    other.d_color_offsets_ = nullptr;
    d_face_buf_ = other.d_face_buf_;
    d_cell_buf_ = other.d_cell_buf_;
    other.d_face_buf_ = nullptr;
    other.d_cell_buf_ = nullptr;
    return *this;
}

DeviceFaceData DeviceMesh::face_data() {
    DeviceFaceData fd;
    fd.nx = d_nx_;
    fd.ny = d_ny_;
    fd.nz = d_nz_;
    fd.area = d_area_;
    fd.left_cell = d_left_cell_;
    fd.right_cell = d_right_cell_;
    fd.boundary = d_boundary_;
    fd.cx = d_face_cx_;
    fd.cy = d_face_cy_;
    fd.cz = d_face_cz_;
    return fd;
}

ConstDeviceFaceData DeviceMesh::face_data() const {
    ConstDeviceFaceData fd;
    fd.nx = d_nx_;
    fd.ny = d_ny_;
    fd.nz = d_nz_;
    fd.area = d_area_;
    fd.left_cell = d_left_cell_;
    fd.right_cell = d_right_cell_;
    fd.boundary = d_boundary_;
    fd.cx = d_face_cx_;
    fd.cy = d_face_cy_;
    fd.cz = d_face_cz_;
    return fd;
}

DeviceCellData DeviceMesh::cell_data() {
    DeviceCellData cd;
    cd.volume = d_volume_;
    cd.h_min = d_h_min_;
    cd.wall_distance = d_wall_distance_;
    cd.cx = d_cx_;
    cd.cy = d_cy_;
    cd.cz = d_cz_;
    return cd;
}

ConstDeviceCellData DeviceMesh::cell_data() const {
    ConstDeviceCellData cd;
    cd.volume = d_volume_;
    cd.h_min = d_h_min_;
    cd.wall_distance = d_wall_distance_;
    cd.cx = d_cx_;
    cd.cy = d_cy_;
    cd.cz = d_cz_;
    return cd;
}



static int greedy_color_faces(
    const std::vector<int>& h_left_cell,
    const std::vector<int>& h_right_cell,
    std::size_t n_faces,
    int max_colors,
    std::vector<int>& color_of_face)
{
    int max_cell = 0;
    for (std::size_t f = 0; f < n_faces; ++f) {
        if (h_left_cell[f] > max_cell) max_cell = h_left_cell[f];
        if (h_right_cell[f] >= 0 && h_right_cell[f] > max_cell) max_cell = h_right_cell[f];
    }
    std::size_t n_cells = static_cast<std::size_t>(max_cell + 1);

    std::vector<std::uint64_t> cell_used(n_cells, 0);
    color_of_face.assign(n_faces, -1);
    int n_colors_used = 0;

    for (std::size_t f = 0; f < n_faces; ++f) {
        int l = h_left_cell[f];
        int r = h_right_cell[f];

        std::uint64_t used = cell_used[static_cast<std::size_t>(l)];
        if (r >= 0) used |= cell_used[static_cast<std::size_t>(r)];

        int color = 0;
        while (color < max_colors && (used & (static_cast<std::uint64_t>(1) << color))) ++color;

        if (color >= max_colors) return 0;

        color_of_face[f] = color;
        cell_used[static_cast<std::size_t>(l)] |= (static_cast<std::uint64_t>(1) << color);
        if (r >= 0) cell_used[static_cast<std::size_t>(r)] |= (static_cast<std::uint64_t>(1) << color);

        if (color + 1 > n_colors_used) n_colors_used = color + 1;
    }
    return n_colors_used;
}

void DeviceMesh::release() {
    cuda_free_safe(d_q_);
    cuda_free_safe(d_residual_);
    cuda_free_safe(d_gradients_);
    cuda_free_safe(d_limiters_);
    cuda_free_safe(d_minmax_);
    cuda_free_safe(d_mu_);
    cuda_free_safe(d_lam_);
    cuda_free_safe(d_delta_ddes_);
    cuda_free_safe(d_q_k_);
    cuda_free_safe(d_q_omega_);
    cuda_free_safe(d_residual_k_);
    cuda_free_safe(d_residual_omega_);
    cuda_free_safe(d_grad_k_);
    cuda_free_safe(d_grad_omega_);
    cuda_free_safe(d_w_);
    cuda_free_safe(d_color_offsets_);
    cuda_free_safe(d_halo_indices_);
    cuda_free_safe(d_halo_send_buf_);
    cuda_free_safe(d_halo_recv_buf_);
    // Batched face/cell buffers (PERF-G7): free the base allocations
    cuda_free_safe(d_face_buf_);
    cuda_free_safe(d_cell_buf_);
    // Null all sub-pointers (they live inside the batched buffers)
    d_nx_ = d_ny_ = d_nz_ = nullptr;
    d_area_ = nullptr;
    d_left_cell_ = d_right_cell_ = nullptr;
    d_boundary_ = nullptr;
    d_volume_ = d_h_min_ = d_wall_distance_ = nullptr;
    d_cx_ = d_cy_ = d_cz_ = nullptr;
    d_face_cx_ = d_face_cy_ = d_face_cz_ = nullptr;
    d_q_k_ = d_q_omega_ = nullptr;
    d_residual_k_ = d_residual_omega_ = nullptr;
    d_grad_k_ = d_grad_omega_ = nullptr;
    cell_count_ = 0;
    face_count_ = 0;
    n_colors_ = 0;
    n_halo_cells_ = 0;
    host_color_offsets_.clear();
}

bool DeviceMesh::upload_mesh(const CfdMesh& mesh, std::string* error, bool skip_coloring) {
    release();
    cell_count_ = mesh.cells.size();
    face_count_ = mesh.faces.size();
    if (cell_count_ == 0 || face_count_ == 0) {
        if (error) *error = "mesh is empty";
        release();
        return false;
    }

    auto alloc = [&](auto*& ptr, std::size_t bytes, const char* name) -> bool {
        if (!cuda_check(cudaMalloc(&ptr, bytes), name, error)) { release(); return false; }
        return true;
    };

    auto copy = [&](auto* dst, const auto* src, std::size_t bytes, const char* name) -> bool {
        if (!cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), name, error)) { release(); return false; }
        return true;
    };

    std::size_t nf = face_count_;
    std::size_t nc = cell_count_;

    std::size_t face_real_bytes = 7 * nf * sizeof(Real);
    std::size_t face_int_bytes = 3 * nf * sizeof(int);
    std::size_t cell_real_bytes = 6 * nc * sizeof(Real);
    if (!alloc(d_face_buf_, face_real_bytes + face_int_bytes, "cudaMalloc d_face_buf_")) return false;
    if (!alloc(d_cell_buf_, cell_real_bytes, "cudaMalloc d_cell_buf_")) return false;

    // Sub-pointers into the face buffer: Reals first, then ints
    d_nx_ = reinterpret_cast<Real*>(d_face_buf_);
    d_ny_ = d_nx_ + nf;
    d_nz_ = d_ny_ + nf;
    d_area_ = d_nz_ + nf;
    d_face_cx_ = d_area_ + nf;
    d_face_cy_ = d_face_cx_ + nf;
    d_face_cz_ = d_face_cy_ + nf;
    d_left_cell_ = reinterpret_cast<int*>(d_face_buf_ + face_real_bytes);
    d_right_cell_ = d_left_cell_ + nf;
    d_boundary_ = d_right_cell_ + nf;

    // Sub-pointers into the cell buffer
    d_volume_ = reinterpret_cast<Real*>(d_cell_buf_);
    d_h_min_ = d_volume_ + nc;
    d_wall_distance_ = d_h_min_ + nc;
    d_cx_ = d_wall_distance_ + nc;
    d_cy_ = d_cx_ + nc;
    d_cz_ = d_cy_ + nc;

    // Solution/working arrays remain separate allocations
    if (!alloc(d_q_, nc * NVAR * sizeof(Real), "cudaMalloc state")) return false;
    if (!alloc(d_residual_, nc * NVAR * sizeof(Real), "cudaMalloc residual")) return false;
    if (!alloc(d_gradients_, nc * DeviceMesh::NGRAD * sizeof(Real), "cudaMalloc gradients")) return false;
    if (!alloc(d_w_, nc * DeviceMesh::NPRIM * sizeof(Real), "cudaMalloc primitive")) return false;
    if (!alloc(d_limiters_, nc * sizeof(PrimitiveLimiter), "cudaMalloc limiters")) return false;
    if (!alloc(d_minmax_, nc * DeviceMesh::kMinmaxStride * sizeof(Real), "cudaMalloc minmax")) return false;
    if (!cuda_check(cudaMemset(d_q_, 0, nc * NVAR * sizeof(Real)), "cudaMemset state", error)) { release(); return false; }
    if (!cuda_check(cudaMemset(d_residual_, 0, nc * NVAR * sizeof(Real)), "cudaMemset residual", error)) { release(); return false; }
    if (!cuda_check(cudaMemset(d_w_, 0, nc * DeviceMesh::NPRIM * sizeof(Real)), "cudaMemset primitive", error)) { release(); return false; }
    if (!cuda_check(cudaMemset(d_gradients_, 0, nc * DeviceMesh::NGRAD * sizeof(Real)), "cudaMemset gradients", error)) { release(); return false; }
    if (!cuda_check(cudaMemset(d_limiters_, 0, nc * sizeof(PrimitiveLimiter)), "cudaMemset limiters", error)) { release(); return false; }
    if (!cuda_check(cudaMemset(d_minmax_, 0, nc * DeviceMesh::kMinmaxStride * sizeof(Real)), "cudaMemset minmax", error)) { release(); return false; }

    std::vector<Real> h_nx(nf), h_ny(nf), h_nz(nf), h_area(nf), h_face_cx(nf), h_face_cy(nf), h_face_cz(nf);
    std::vector<int> h_left_cell(nf), h_right_cell(nf), h_boundary(nf);
    for (std::size_t i = 0; i < nf; ++i) {
        const auto& f = mesh.faces[i];
        h_nx[i] = f.nx; h_ny[i] = f.ny; h_nz[i] = f.nz;
        h_area[i] = f.area;
        h_left_cell[i] = f.left_cell; h_right_cell[i] = f.right_cell;
        h_boundary[i] = static_cast<int>(f.boundary);
        h_face_cx[i] = f.cx; h_face_cy[i] = f.cy; h_face_cz[i] = f.cz;
    }
    if (!skip_coloring && nf > 0) {
        std::vector<int> color_of_face;
        int nc_used = greedy_color_faces(h_left_cell, h_right_cell, nf, kMaxColors, color_of_face);
        if (nc_used > 0) {
            n_colors_ = nc_used;
            host_color_offsets_.assign(n_colors_ + 1, 0);
            for (std::size_t f = 0; f < nf; ++f) {
                int c = color_of_face[f];
                host_color_offsets_[static_cast<std::size_t>(c) + 1]++;
            }
            for (int c = 1; c <= n_colors_; ++c)
                host_color_offsets_[c] += host_color_offsets_[c - 1];

            std::vector<int> insert_pos = host_color_offsets_;
            std::vector<int> dest(nf);
            for (std::size_t f = 0; f < nf; ++f) {
                int c = color_of_face[f];
                dest[f] = insert_pos[c]++;
            }

            std::vector<Real> reorder_tmp_real(nf);
            std::vector<int> reorder_tmp_int(nf);
            auto reorder = [&](auto& vec, auto& tmp) {
                for (std::size_t f = 0; f < nf; ++f)
                    tmp[f] = vec[f];
                for (std::size_t f = 0; f < nf; ++f)
                    vec[static_cast<std::size_t>(dest[f])] = tmp[f];
            };
            reorder(h_nx, reorder_tmp_real); reorder(h_ny, reorder_tmp_real); reorder(h_nz, reorder_tmp_real);
            reorder(h_area, reorder_tmp_real);
            reorder(h_left_cell, reorder_tmp_int); reorder(h_right_cell, reorder_tmp_int);
            reorder(h_boundary, reorder_tmp_int);
            reorder(h_face_cx, reorder_tmp_real); reorder(h_face_cy, reorder_tmp_real); reorder(h_face_cz, reorder_tmp_real);

            std::vector<int> perm(nf);
            for (std::size_t f = 0; f < nf; ++f) perm[f] = static_cast<int>(f);
            for (int c = 0; c < n_colors_; ++c) {
                int s = host_color_offsets_[c];
                int e = host_color_offsets_[c + 1];
                std::sort(perm.begin() + s, perm.begin() + e,
                    [&](int a, int b) { return h_left_cell[a] < h_left_cell[b]; });
            }
            auto apply_perm = [&](auto& vec, auto& tmp) {
                for (std::size_t f = 0; f < nf; ++f) tmp[f] = vec[f];
                for (std::size_t f = 0; f < nf; ++f) vec[f] = tmp[static_cast<std::size_t>(perm[f])];
            };
            apply_perm(h_nx, reorder_tmp_real); apply_perm(h_ny, reorder_tmp_real); apply_perm(h_nz, reorder_tmp_real);
            apply_perm(h_area, reorder_tmp_real);
            apply_perm(h_left_cell, reorder_tmp_int); apply_perm(h_right_cell, reorder_tmp_int);
            apply_perm(h_boundary, reorder_tmp_int);
            apply_perm(h_face_cx, reorder_tmp_real); apply_perm(h_face_cy, reorder_tmp_real); apply_perm(h_face_cz, reorder_tmp_real);
        }
    }

    {
        std::vector<Real> face_real_buf(7 * nf);
        Real* fr = face_real_buf.data();
        std::memcpy(fr, h_nx.data(), nf * sizeof(Real)); fr += nf;
        std::memcpy(fr, h_ny.data(), nf * sizeof(Real)); fr += nf;
        std::memcpy(fr, h_nz.data(), nf * sizeof(Real)); fr += nf;
        std::memcpy(fr, h_area.data(), nf * sizeof(Real)); fr += nf;
        std::memcpy(fr, h_face_cx.data(), nf * sizeof(Real)); fr += nf;
        std::memcpy(fr, h_face_cy.data(), nf * sizeof(Real)); fr += nf;
        std::memcpy(fr, h_face_cz.data(), nf * sizeof(Real));
        if (!copy(d_face_buf_, face_real_buf.data(), face_real_bytes, "cudaMemcpy face_real_buf")) return false;

        std::vector<int> face_int_buf(3 * nf);
        int* fi = face_int_buf.data();
        std::memcpy(fi, h_left_cell.data(), nf * sizeof(int)); fi += nf;
        std::memcpy(fi, h_right_cell.data(), nf * sizeof(int)); fi += nf;
        std::memcpy(fi, h_boundary.data(), nf * sizeof(int));
        if (!copy(d_left_cell_, face_int_buf.data(), face_int_bytes, "cudaMemcpy face_int_buf")) return false;
    }

    // Cell data: pack into contiguous host buffer → single cudaMemcpy
    std::vector<Real> h_volume(nc), h_h_min(nc), h_wall_distance(nc), h_cx(nc), h_cy(nc), h_cz(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        const auto& c = mesh.cells[i];
        h_volume[i] = c.volume; h_h_min[i] = c.h_min; h_wall_distance[i] = c.wall_distance;
        h_cx[i] = c.cx; h_cy[i] = c.cy; h_cz[i] = c.cz;
    }
    {
        std::vector<Real> cell_real_buf(6 * nc);
        Real* cr = cell_real_buf.data();
        std::memcpy(cr, h_volume.data(), nc * sizeof(Real)); cr += nc;
        std::memcpy(cr, h_h_min.data(), nc * sizeof(Real)); cr += nc;
        std::memcpy(cr, h_wall_distance.data(), nc * sizeof(Real)); cr += nc;
        std::memcpy(cr, h_cx.data(), nc * sizeof(Real)); cr += nc;
        std::memcpy(cr, h_cy.data(), nc * sizeof(Real)); cr += nc;
        std::memcpy(cr, h_cz.data(), nc * sizeof(Real));
        if (!copy(d_cell_buf_, cell_real_buf.data(), cell_real_bytes, "cudaMemcpy cell_real_buf")) return false;
    }

    if (!cuda_check(cudaMemset(d_residual_, 0, nc * NVAR * sizeof(Real)), "cudaMemset residual", error)) {
        release();
        return false;
    }

    if (n_colors_ > 0) {
        if (!alloc(d_color_offsets_, static_cast<std::size_t>(n_colors_ + 1) * sizeof(int), "cudaMalloc color_offsets")) return false;
        if (!copy(d_color_offsets_, host_color_offsets_.data(), static_cast<std::size_t>(n_colors_ + 1) * sizeof(int), "cudaMemcpy color_offsets")) return false;
    }

    return true;
}

bool DeviceMesh::upload_state(const std::vector<ConservativeState>& q, std::string* error) {
    if (cell_count_ == 0 || static_cast<int>(q.size()) != cell_count_) {
        if (error) *error = "state size mismatch";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    std::vector<Real> flat(nc * NVAR);
    for (std::size_t i = 0; i < nc; ++i) {
        flat[i * NVAR + 0] = q[i].rho;
        flat[i * NVAR + 1] = q[i].rho_u;
        flat[i * NVAR + 2] = q[i].rho_v;
        flat[i * NVAR + 3] = q[i].rho_w;
        flat[i * NVAR + 4] = q[i].rho_E;
        flat[i * NVAR + 5] = q[i].rho_nu_tilde;
    }
    return cuda_check(cudaMemcpy(d_q_, flat.data(), nc * NVAR * sizeof(Real), cudaMemcpyHostToDevice), "cudaMemcpy upload_state", error);
}

bool DeviceMesh::upload_gradients(const std::vector<PrimitiveGradient>& gradients, std::string* error) {
    if (cell_count_ == 0 || static_cast<int>(gradients.size()) != cell_count_) {
        if (error) *error = "gradient size mismatch";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    if (!d_gradients_) {
        if (!cuda_check(cudaMalloc(&d_gradients_, nc * DeviceMesh::NGRAD * sizeof(Real)), "cudaMalloc gradients", error)) return false;
    }
    return cuda_check(cudaMemcpy(d_gradients_, gradients.data(), nc * DeviceMesh::NGRAD * sizeof(Real), cudaMemcpyHostToDevice), "cudaMemcpy upload_gradients", error);
}

bool DeviceMesh::upload_limiters(const std::vector<PrimitiveLimiter>& limiters, std::string* error) {
    if (cell_count_ == 0 || static_cast<int>(limiters.size()) != cell_count_) {
        if (error) *error = "limiter size mismatch";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    if (!d_limiters_) {
        if (!cuda_check(cudaMalloc(&d_limiters_, nc * sizeof(PrimitiveLimiter)), "cudaMalloc limiters", error)) return false;
    }
    return cuda_check(cudaMemcpy(d_limiters_, limiters.data(), nc * sizeof(PrimitiveLimiter), cudaMemcpyHostToDevice), "cudaMemcpy upload_limiters", error);
}

bool DeviceMesh::clear_residual(std::string* error) {
    if (!d_residual_ || cell_count_ == 0) {
        if (error) *error = "residual buffer not allocated";
        return false;
    }
    return cuda_check(cudaMemset(d_residual_, 0, static_cast<std::size_t>(cell_count_) * NVAR * sizeof(Real)), "cudaMemset clear_residual", error);
}

bool DeviceMesh::download_state(std::vector<ConservativeState>& q, std::string* error) const {
    if (!d_q_ || cell_count_ == 0) {
        if (error) *error = "state buffer not allocated";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    std::vector<Real> flat(nc * NVAR);
    if (!cuda_check(cudaMemcpy(flat.data(), d_q_, nc * NVAR * sizeof(Real), cudaMemcpyDeviceToHost), "cudaMemcpy download_state", error)) return false;
    q.resize(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        q[i].rho = flat[i * NVAR + 0];
        q[i].rho_u = flat[i * NVAR + 1];
        q[i].rho_v = flat[i * NVAR + 2];
        q[i].rho_w = flat[i * NVAR + 3];
        q[i].rho_E = flat[i * NVAR + 4];
        q[i].rho_nu_tilde = flat[i * NVAR + 5];
    }
    return true;
}

bool DeviceMesh::download_residual(std::vector<EulerFlux>& residual, std::string* error) const {
    if (!d_residual_ || cell_count_ == 0) {
        if (error) *error = "residual buffer not allocated";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    std::vector<Real> flat(nc * NVAR);
    if (!cuda_check(cudaMemcpy(flat.data(), d_residual_, nc * NVAR * sizeof(Real), cudaMemcpyDeviceToHost), "cudaMemcpy download_residual", error)) return false;
    residual.resize(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        residual[i].mass = flat[i * NVAR + 0];
        residual[i].mom_x = flat[i * NVAR + 1];
        residual[i].mom_y = flat[i * NVAR + 2];
        residual[i].mom_z = flat[i * NVAR + 3];
        residual[i].energy = flat[i * NVAR + 4];
        residual[i].turbulence = flat[i * NVAR + 5];
    }
    return true;
}

bool DeviceMesh::download_gradients(std::vector<PrimitiveGradient>& gradients, std::string* error) const {
    if (!d_gradients_ || cell_count_ == 0) {
        if (error) *error = "gradient buffer not allocated";
        return false;
    }
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    gradients.resize(nc);
    return cuda_check(cudaMemcpy(gradients.data(), d_gradients_, nc * DeviceMesh::NGRAD * sizeof(Real), cudaMemcpyDeviceToHost), "cudaMemcpy download_gradients", error);
}

bool DeviceMesh::allocate_viscous() {
    if (cell_count_ == 0) return false;
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    if (!cuda_check(cudaMalloc(&d_mu_, nc * sizeof(Real)), "cudaMalloc d_mu_", nullptr)) return false;
    if (!cuda_check(cudaMalloc(&d_lam_, nc * sizeof(Real)), "cudaMalloc d_lam_", nullptr)) return false;
    if (!cuda_check(cudaMemset(d_mu_, 0, nc * sizeof(Real)), "cudaMemset d_mu_", nullptr)) return false;
    if (!cuda_check(cudaMemset(d_lam_, 0, nc * sizeof(Real)), "cudaMemset d_lam_", nullptr)) return false;
    return true;
}

bool DeviceMesh::allocate_ddes() {
    if (cell_count_ == 0) return false;
    if (d_delta_ddes_ != nullptr) return true;
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    if (!cuda_check(cudaMalloc(&d_delta_ddes_, nc * sizeof(Real)), "cudaMalloc d_delta_ddes_", nullptr)) return false;
    if (!cuda_check(cudaMemset(d_delta_ddes_, 0, nc * sizeof(Real)), "cudaMemset d_delta_ddes_", nullptr)) return false;
    return true;
}

bool DeviceMesh::allocate_sst() {
    if (cell_count_ == 0) return false;
    if (d_q_k_ != nullptr) return true;
    std::size_t nc = static_cast<std::size_t>(cell_count_);
    if (!cuda_check(cudaMalloc(&d_q_k_, nc * sizeof(Real)), "cudaMalloc d_q_k_", nullptr)) return false;
    if (!cuda_check(cudaMalloc(&d_q_omega_, nc * sizeof(Real)), "cudaMalloc d_q_omega_", nullptr)) return false;
    if (!cuda_check(cudaMalloc(&d_residual_k_, nc * sizeof(Real)), "cudaMalloc d_residual_k_", nullptr)) return false;
    if (!cuda_check(cudaMalloc(&d_residual_omega_, nc * sizeof(Real)), "cudaMalloc d_residual_omega_", nullptr)) return false;
    if (!cuda_check(cudaMalloc(&d_grad_k_, nc * 3 * sizeof(Real)), "cudaMalloc d_grad_k_", nullptr)) return false;
    if (!cuda_check(cudaMalloc(&d_grad_omega_, nc * 3 * sizeof(Real)), "cudaMalloc d_grad_omega_", nullptr)) return false;
    if (!cuda_check(cudaMemset(d_q_k_, 0, nc * sizeof(Real)), "cudaMemset d_q_k_", nullptr)) return false;
    if (!cuda_check(cudaMemset(d_q_omega_, 0, nc * sizeof(Real)), "cudaMemset d_q_omega_", nullptr)) return false;
    if (!cuda_check(cudaMemset(d_residual_k_, 0, nc * sizeof(Real)), "cudaMemset d_residual_k_", nullptr)) return false;
    if (!cuda_check(cudaMemset(d_residual_omega_, 0, nc * sizeof(Real)), "cudaMemset d_residual_omega_", nullptr)) return false;
    if (!cuda_check(cudaMemset(d_grad_k_, 0, nc * 3 * sizeof(Real)), "cudaMemset d_grad_k_", nullptr)) return false;
    if (!cuda_check(cudaMemset(d_grad_omega_, 0, nc * 3 * sizeof(Real)), "cudaMemset d_grad_omega_", nullptr)) return false;
    return true;
}

bool DeviceMesh::allocate_halo(int n_halo_cells) {
    cuda_free_safe(d_halo_indices_);
    cuda_free_safe(d_halo_send_buf_);
    cuda_free_safe(d_halo_recv_buf_);
    n_halo_cells_ = 0;
    if (n_halo_cells <= 0) {
        return true;
    }
    n_halo_cells_ = n_halo_cells;
    if (!cuda_check(cudaMalloc(&d_halo_indices_, n_halo_cells_ * sizeof(int)), "cudaMalloc halo_indices", nullptr)) {
        n_halo_cells_ = 0;
        return false;
    }
    if (!cuda_check(cudaMalloc(&d_halo_send_buf_, n_halo_cells_ * NVAR * sizeof(Real)), "cudaMalloc halo_send_buf", nullptr)) {
        cuda_free_safe(d_halo_indices_);
        n_halo_cells_ = 0;
        return false;
    }
    if (!cuda_check(cudaMalloc(&d_halo_recv_buf_, n_halo_cells_ * NVAR * sizeof(Real)), "cudaMalloc halo_recv_buf", nullptr)) {
        cuda_free_safe(d_halo_indices_);
        cuda_free_safe(d_halo_send_buf_);
        n_halo_cells_ = 0;
        return false;
    }
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp

