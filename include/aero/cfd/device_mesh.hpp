#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/reconstruction.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

struct GpuPartition;

struct DeviceFaceData {
    Real* nx = nullptr;
    Real* ny = nullptr;
    Real* nz = nullptr;
    Real* area = nullptr;
    int* left_cell = nullptr;
    int* right_cell = nullptr;
    int* boundary = nullptr;
    Real* cx = nullptr;
    Real* cy = nullptr;
    Real* cz = nullptr;
};

struct ConstDeviceFaceData {
    const Real* nx = nullptr;
    const Real* ny = nullptr;
    const Real* nz = nullptr;
    const Real* area = nullptr;
    const int* left_cell = nullptr;
    const int* right_cell = nullptr;
    const int* boundary = nullptr;
    const Real* cx = nullptr;
    const Real* cy = nullptr;
    const Real* cz = nullptr;
};

struct DeviceCellData {
    Real* volume = nullptr;
    Real* h_min = nullptr;
    Real* wall_distance = nullptr;
    Real* cx = nullptr;
    Real* cy = nullptr;
    Real* cz = nullptr;
};

struct ConstDeviceCellData {
    const Real* volume = nullptr;
    const Real* h_min = nullptr;
    const Real* wall_distance = nullptr;
    const Real* cx = nullptr;
    const Real* cy = nullptr;
    const Real* cz = nullptr;
};

class DeviceMesh {
public:
    static constexpr int kMaxColors = 64;

    DeviceMesh() = default;
    ~DeviceMesh();

    DeviceMesh(const DeviceMesh&) = delete;
    DeviceMesh& operator=(const DeviceMesh&) = delete;

    DeviceMesh(DeviceMesh&& other) noexcept;
    DeviceMesh& operator=(DeviceMesh&& other) noexcept;

    bool upload_mesh(const CfdMesh& mesh, std::string* error = nullptr, bool skip_coloring = false, int nvar = CFD_NVAR);

    int nvar() const { return nvar_; }
    int nprim() const { return nvar_; }
    int ngrad() const { return 3 * nvar_; }
    int minmax_stride() const { return 2 * nvar_; }
    int nvar_or_default() const { return nvar_ > 0 ? nvar_ : CFD_NVAR; }
    bool upload_state(const std::vector<ConservativeState>& q, std::string* error = nullptr);
    bool upload_gradients(const std::vector<PrimitiveGradient>& gradients, std::string* error = nullptr);
    bool upload_limiters(const std::vector<PrimitiveLimiter>& limiters, std::string* error = nullptr);
    bool clear_residual(std::string* error = nullptr);

    bool download_state(std::vector<ConservativeState>& q, std::string* error = nullptr) const;
    bool download_residual(std::vector<EulerFlux>& residual, std::string* error = nullptr) const;
    bool download_gradients(std::vector<PrimitiveGradient>& gradients, std::string* error = nullptr) const;

    void release();

    std::size_t cell_count() const { return cell_count_; }
    std::size_t face_count() const { return face_count_; }

    int color_count() const { return n_colors_; }
    const int* color_offsets_device() const { return d_color_offsets_; }
    const std::vector<int>& host_color_offsets() const { return host_color_offsets_; }

    DeviceFaceData face_data();
    ConstDeviceFaceData face_data() const;
    DeviceCellData cell_data();
    ConstDeviceCellData cell_data() const;

    Real* state_device() const { return d_q_; }
    void set_state_device(Real* d_q) { d_q_ = d_q; }
    Real* primitive_device() const { return d_w_; }
    Real* residual_device() const { return d_residual_; }
    Real* gradients_device() const { return d_gradients_; }
    Real* limiters_device() const { return d_limiters_; }
    Real* minmax_device() const { return d_minmax_; }
    // MPI halo (reserved, no-op in single-GPU mode)
    bool has_halo() const { return d_halo_indices_ != nullptr && n_halo_cells_ > 0; }
    bool allocate_halo(int n_halo_cells);
    int* halo_indices_device() const { return d_halo_indices_; }
    Real* halo_send_device() const { return d_halo_send_buf_; }
    Real* halo_recv_device() const { return d_halo_recv_buf_; }
    void set_partition(const GpuPartition* gpu_part) { gpu_part_ = gpu_part; }
    const GpuPartition* get_partition() const { return gpu_part_; }

    // Viscous buffers
    bool allocate_viscous();
    Real* mu_device() const { return d_mu_; }
    Real* lam_device() const { return d_lam_; }

    // DDES buffer
    bool allocate_ddes();
    bool has_delta_ddes() const { return d_delta_ddes_ != nullptr; }
    Real* delta_ddes_device() const { return d_delta_ddes_; }

    // SST k-omega buffers (separate from main NVAR=6 state)
    bool allocate_sst();
    bool has_sst() const { return d_q_k_ != nullptr; }
    Real* q_k_device() const { return d_q_k_; }
    Real* q_omega_device() const { return d_q_omega_; }
    Real* residual_k_device() const { return d_residual_k_; }
    Real* residual_omega_device() const { return d_residual_omega_; }
    Real* grad_k_device() const { return d_grad_k_; }
    Real* grad_omega_device() const { return d_grad_omega_; }
    Real* sst_f1_device() const { return d_sst_f1_; }

private:
    std::size_t cell_count_ = 0;
    std::size_t face_count_ = 0;

    Real* d_q_ = nullptr;
    Real* d_residual_ = nullptr;

    Real* d_nx_ = nullptr;
    Real* d_ny_ = nullptr;
    Real* d_nz_ = nullptr;
    Real* d_area_ = nullptr;
    int* d_left_cell_ = nullptr;
    int* d_right_cell_ = nullptr;
    int* d_boundary_ = nullptr;

    Real* d_volume_ = nullptr;
    Real* d_h_min_ = nullptr;
    Real* d_wall_distance_ = nullptr;
    Real* d_cx_ = nullptr;
    Real* d_cy_ = nullptr;
    Real* d_cz_ = nullptr;

    Real* d_face_cx_ = nullptr;
    Real* d_face_cy_ = nullptr;
    Real* d_face_cz_ = nullptr;

    Real* d_gradients_ = nullptr;
    Real* d_limiters_ = nullptr;
    Real* d_minmax_ = nullptr;
    Real* d_mu_ = nullptr;
    Real* d_lam_ = nullptr;
    Real* d_delta_ddes_ = nullptr;
    Real* d_q_k_ = nullptr;
    Real* d_q_omega_ = nullptr;
    Real* d_residual_k_ = nullptr;
    Real* d_residual_omega_ = nullptr;
    Real* d_grad_k_ = nullptr;
    Real* d_grad_omega_ = nullptr;
    Real* d_sst_f1_ = nullptr;
    Real* d_w_ = nullptr;

    int n_colors_ = 0;
    int* d_color_offsets_ = nullptr;
    std::vector<int> host_color_offsets_;

    // MPI halo (reserved, nullptr = single-GPU mode)
    int* d_halo_indices_ = nullptr;
    Real* d_halo_send_buf_ = nullptr;
    Real* d_halo_recv_buf_ = nullptr;
    int n_halo_cells_ = 0;

    const GpuPartition* gpu_part_ = nullptr;

    // Batched allocation base pointers (PERF-G7)
    char* d_face_buf_ = nullptr;
    char* d_cell_buf_ = nullptr;

    int nvar_ = 0;
};

} // namespace cfd
} // namespace aero
} // namespace aerosp
