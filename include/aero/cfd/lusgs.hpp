#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/device_mesh.hpp"
#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

class LusgsPreconditioner {
public:
    LusgsPreconditioner() = default;
    ~LusgsPreconditioner();

    LusgsPreconditioner(const LusgsPreconditioner&) = delete;
    LusgsPreconditioner& operator=(const LusgsPreconditioner&) = delete;

    bool allocate(DeviceMesh& mesh, std::string* error = nullptr);
    void release();

    // Rebuild cell coloring after mesh change (AMR).
    // Call after uploading new mesh to device, before compute_diagonal.
    bool rebuild_coloring(DeviceMesh& mesh, std::string* error = nullptr);

    bool compute_diagonal(DeviceMesh& mesh, const Real* d_dt_cell,
        Real gamma, bool viscous, const Real* d_mu, Real Re,
        std::string* error = nullptr);

    bool apply(DeviceMesh& mesh, const Real* d_r, Real* d_z, Real gamma,
        std::string* error = nullptr);

    bool allocated() const { return d_D_ != nullptr; }

private:
    Real* d_D_ = nullptr;
    Real* d_dz_ = nullptr;
    Real* d_inv_vol_ = nullptr;
    Real* d_spectral_radius_ = nullptr;
    int* d_cell_color_ = nullptr;
    int* d_cell_face_start_ = nullptr;
    int* d_cell_faces_ = nullptr;
    int* d_cell_face_nbr_ = nullptr;
    int n_cell_colors_ = 0;
    int n_cells_ = 0;
    int nvar_ = 0;
    int n_incidence_ = 0;
};

} // namespace cfd
} // namespace aero
} // namespace aerosp
