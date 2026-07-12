#include "aero/cfd/mms.hpp"
#include "aero/cfd/cfd_residual.hpp"
#include "aero/cfd/rans.hpp"
#include "aero/cfd/reconstruction.hpp"
#include "aero/cfd/viscous.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace aerosp {
namespace aero {
namespace cfd {
namespace {

constexpr Real kPi = 3.14159265358979323846;

Real sin_pi(Real x) { return std::sin(kPi * x); }
Real cos_pi(Real x) { return std::cos(kPi * x); }
Real sin_2pi(Real x) { return std::sin(2.0 * kPi * x); }
Real cos_2pi(Real x) { return std::cos(2.0 * kPi * x); }

} // namespace

PrimitiveState MmsSolutionEuler::eval(Real x, Real y, Real z) const {
    PrimitiveState w;
    w.rho = rho0 + rho_amp * sin_pi(freq * x) * cos_pi(freq * y) * sin_pi(freq * z);
    w.u   = u0   + u_amp   * cos_2pi(freq * x) * sin_2pi(freq * y) * cos_2pi(freq * z);
    w.v   = v0   + v_amp   * sin_2pi(freq * x) * cos_2pi(freq * y) * sin_2pi(freq * z);
    w.w   = w0   + w_amp   * cos_2pi(freq * x) * sin_2pi(freq * y) * sin_2pi(freq * z);
    w.p   = p0   + p_amp   * cos_pi(freq * x) * cos_pi(freq * y) * cos_pi(freq * z);
    return w;
}

PrimitiveGradient MmsSolutionEuler::eval_gradient(Real x, Real y, Real z) const {
    PrimitiveGradient g;
    Real fx = freq * kPi;
    g.drho_dx = rho_amp * fx * cos_pi(freq * x) * cos_pi(freq * y) * sin_pi(freq * z);
    g.drho_dy = -rho_amp * fx * sin_pi(freq * x) * sin_pi(freq * y) * sin_pi(freq * z);
    g.drho_dz = rho_amp * fx * sin_pi(freq * x) * cos_pi(freq * y) * cos_pi(freq * z);

    Real fx2 = 2.0 * kPi * freq;
    g.du_dx = -u_amp * fx2 * sin_2pi(freq * x) * sin_2pi(freq * y) * cos_2pi(freq * z);
    g.du_dy = u_amp * fx2 * cos_2pi(freq * x) * cos_2pi(freq * y) * cos_2pi(freq * z);
    g.du_dz = -u_amp * fx2 * cos_2pi(freq * x) * sin_2pi(freq * y) * sin_2pi(freq * z);

    g.dv_dx = v_amp * fx2 * cos_2pi(freq * x) * cos_2pi(freq * y) * sin_2pi(freq * z);
    g.dv_dy = -v_amp * fx2 * sin_2pi(freq * x) * sin_2pi(freq * y) * sin_2pi(freq * z);
    g.dv_dz = v_amp * fx2 * sin_2pi(freq * x) * cos_2pi(freq * y) * cos_2pi(freq * z);

    g.dw_dx = -w_amp * fx2 * sin_2pi(freq * x) * sin_2pi(freq * y) * sin_2pi(freq * z);
    g.dw_dy = w_amp * fx2 * cos_2pi(freq * x) * cos_2pi(freq * y) * sin_2pi(freq * z);
    g.dw_dz = w_amp * fx2 * cos_2pi(freq * x) * sin_2pi(freq * y) * cos_2pi(freq * z);

    g.dp_dx = -p_amp * fx * sin_pi(freq * x) * cos_pi(freq * y) * cos_pi(freq * z);
    g.dp_dy = -p_amp * fx * cos_pi(freq * x) * sin_pi(freq * y) * cos_pi(freq * z);
    g.dp_dz = -p_amp * fx * cos_pi(freq * x) * cos_pi(freq * y) * sin_pi(freq * z);
    return g;
}

PrimitiveState MmsSolutionSA::eval_sa(Real x, Real y, Real z) const {
    PrimitiveState w = eval(x, y, z);
    w.nu_tilde = nt0 + nt_amp * sin_pi(freq * x) * cos_pi(freq * y) * cos_pi(freq * z);
    return w;
}

PrimitiveGradient MmsSolutionSA::eval_gradient_sa(Real x, Real y, Real z) const {
    PrimitiveGradient g = eval_gradient(x, y, z);
    Real fx = freq * kPi;
    g.dnu_tilde_dx = nt_amp * fx * cos_pi(freq * x) * cos_pi(freq * y) * cos_pi(freq * z);
    g.dnu_tilde_dy = -nt_amp * fx * sin_pi(freq * x) * sin_pi(freq * y) * cos_pi(freq * z);
    g.dnu_tilde_dz = -nt_amp * fx * sin_pi(freq * x) * cos_pi(freq * y) * sin_pi(freq * z);
    return g;
}

void fill_mms(const CfdMesh& mesh, const MmsSolutionEuler& mms,
              std::vector<ConservativeState>& q, Real gamma) {
    q.resize(mesh.cells.size());
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        PrimitiveState w = mms.eval(
            mesh.cells[i].cx, mesh.cells[i].cy, mesh.cells[i].cz);
        q[i] = primitive_to_conservative(w, gamma);
    }
}

void fill_mms_sa(const CfdMesh& mesh, const MmsSolutionSA& mms,
                 std::vector<ConservativeState>& q, Real gamma) {
    q.resize(mesh.cells.size());
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        PrimitiveState w = mms.eval_sa(
            mesh.cells[i].cx, mesh.cells[i].cy, mesh.cells[i].cz);
        q[i] = primitive_to_conservative(w, gamma);
    }
}

bool compute_mms_source(const CfdMesh& mesh,
    const std::vector<ConservativeState>& q_exact,
    const PrimitiveState& freestream, const CfdConfig& config,
    std::vector<EulerFlux>& source)
{
    source.assign(q_exact.size(), EulerFlux{});
    std::vector<PrimitiveState> w_exact(q_exact.size());
    for (std::size_t i = 0; i < q_exact.size(); ++i)
        if (!conservative_to_primitive(q_exact[i], config.gamma, w_exact[i]))
            return false;

    if (config.reconstruction_order == 2 || config.viscous || config.turbulence) {
        auto grads = compute_green_gauss_gradients(mesh, q_exact, config.gamma, &w_exact);
        if (grads.size() != mesh.cells.size()) return false;

        if (config.reconstruction_order == 2) {
            auto limiters = compute_barth_jespersen_limiters(mesh, q_exact, grads, config.gamma, &w_exact);
            if (limiters.size() != mesh.cells.size()) return false;
            std::vector<PrimitiveGradient> limited(grads.size());
            for (std::size_t i = 0; i < grads.size(); ++i)
                limited[i] = apply_limiter(grads[i], limiters[i]);

            if (!compute_euler_residual_cpu_order2(mesh, q_exact, freestream, config.gamma, limited, source, &w_exact))
                return false;

            if (config.viscous) {
                if (!compute_viscous_flux_cpu(mesh, q_exact, limited, config.gamma,
                        config.prandtl, config.mu_ref, config.T_ref, config.sutherland_T,
                        config.Re, config.wall_temperature, config.turbulence ? 1 : 0, source, &w_exact))
                    return false;
            }
            if (config.turbulence) {
                auto rans = compute_rans_sources(mesh, q_exact, limited, config.gamma, config.Re, &w_exact);
                if (rans.size() != mesh.cells.size()) return false;
                for (std::size_t i = 0; i < q_exact.size(); ++i)
                    source[i].turbulence += rans[i].total_source * mesh.cells[i].volume;
            }
        } else {
            if (!compute_euler_residual_cpu(mesh, q_exact, freestream, config.gamma, source, &w_exact))
                return false;
            if (config.viscous) {
                if (!compute_viscous_flux_cpu(mesh, q_exact, grads, config.gamma,
                        config.prandtl, config.mu_ref, config.T_ref, config.sutherland_T,
                        config.Re, config.wall_temperature, config.turbulence ? 1 : 0, source, &w_exact))
                    return false;
            }
            if (config.turbulence) {
                auto rans = compute_rans_sources(mesh, q_exact, grads, config.gamma, config.Re, &w_exact);
                if (rans.size() != mesh.cells.size()) return false;
                for (std::size_t i = 0; i < q_exact.size(); ++i)
                    source[i].turbulence += rans[i].total_source * mesh.cells[i].volume;
            }
        }
    } else {
        if (!compute_euler_residual_cpu(mesh, q_exact, freestream, config.gamma, source, &w_exact))
            return false;
    }
    return true;
}

Real mms_l2_error(const std::vector<ConservativeState>& q,
                  const std::vector<ConservativeState>& q_ref, int nvar) {
    if (q.empty() || q.size() != q_ref.size()) return std::numeric_limits<Real>::max();
    Real num = 0.0, den = 0.0;
    Real c0 = std::numeric_limits<Real>::min();
    for (std::size_t i = 0; i < q.size(); ++i) {
        Real d0 = q[i].rho - q_ref[i].rho;
        Real d1 = q[i].rho_u - q_ref[i].rho_u;
        Real d2 = q[i].rho_v - q_ref[i].rho_v;
        Real d3 = q[i].rho_w - q_ref[i].rho_w;
        Real d4 = q[i].rho_E - q_ref[i].rho_E;
        num += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4;
        den += q_ref[i].rho * q_ref[i].rho + c0;
        if (nvar > 5) {
            Real d5 = q[i].rho_nu_tilde - q_ref[i].rho_nu_tilde;
            num += d5*d5;
        }
    }
    if (den <= 0.0) return std::numeric_limits<Real>::max();
    return std::sqrt(num / den);
}

Real mms_observed_order(Real err_coarse, int n_coarse,
                        Real err_medium, int n_medium,
                        Real err_fine, int n_fine) {
    if (err_coarse <= 0.0 || err_medium <= 0.0 || err_fine <= 0.0)
        return 0.0;
    Real h_coarse = 1.0 / static_cast<Real>(n_coarse);
    Real h_medium = 1.0 / static_cast<Real>(n_medium);
    Real h_fine = 1.0 / static_cast<Real>(n_fine);
    Real r1 = std::log(h_medium / h_coarse);
    Real r2 = std::log(h_fine / h_medium);
    Real p1 = std::log(err_medium / err_coarse) / r1;
    Real p2 = std::log(err_fine / err_medium) / r2;
    return 0.5 * (p1 + p2);
}

// --- Boundary-compatible MMS (cos modes, vanish at domain boundaries) ---

PrimitiveState MmsSolutionEulerBC::eval(Real x, Real y, Real z) const {
    PrimitiveState w;
    Real p = cos_pi(freq * x) * cos_pi(freq * y) * cos_pi(freq * z);
    w.rho = rho0 + rho_amp * p;
    w.u   = u0   + u_amp   * p;
    w.v   = v0   + v_amp   * p;
    w.w   = w0   + w_amp   * p;
    w.p   = p0   + p_amp   * p;
    return w;
}

PrimitiveGradient MmsSolutionEulerBC::eval_gradient(Real x, Real y, Real z) const {
    PrimitiveGradient g;
    Real fx = freq * kPi;
    Real sx = -sin_pi(freq * x) * cos_pi(freq * y) * cos_pi(freq * z);
    Real sy = -cos_pi(freq * x) * sin_pi(freq * y) * cos_pi(freq * z);
    Real sz = -cos_pi(freq * x) * cos_pi(freq * y) * sin_pi(freq * z);
    g.drho_dx = rho_amp * fx * sx;
    g.drho_dy = rho_amp * fx * sy;
    g.drho_dz = rho_amp * fx * sz;
    g.du_dx = u_amp * fx * sx;
    g.du_dy = u_amp * fx * sy;
    g.du_dz = u_amp * fx * sz;
    g.dv_dx = v_amp * fx * sx;
    g.dv_dy = v_amp * fx * sy;
    g.dv_dz = v_amp * fx * sz;
    g.dw_dx = w_amp * fx * sx;
    g.dw_dy = w_amp * fx * sy;
    g.dw_dz = w_amp * fx * sz;
    g.dp_dx = p_amp * fx * sx;
    g.dp_dy = p_amp * fx * sy;
    g.dp_dz = p_amp * fx * sz;
    return g;
}

PrimitiveState MmsSolutionSABC::eval_sa(Real x, Real y, Real z) const {
    PrimitiveState w = eval(x, y, z);
    w.nu_tilde = nt0 + nt_amp * cos_pi(freq * x) * cos_pi(freq * y) * cos_pi(freq * z);
    return w;
}

PrimitiveGradient MmsSolutionSABC::eval_gradient_sa(Real x, Real y, Real z) const {
    PrimitiveGradient g = eval_gradient(x, y, z);
    Real fx = freq * kPi;
    Real sx = -sin_pi(freq * x) * cos_pi(freq * y) * cos_pi(freq * z);
    Real sy = -cos_pi(freq * x) * sin_pi(freq * y) * cos_pi(freq * z);
    Real sz = -cos_pi(freq * x) * cos_pi(freq * y) * sin_pi(freq * z);
    g.dnu_tilde_dx = nt_amp * fx * sx;
    g.dnu_tilde_dy = nt_amp * fx * sy;
    g.dnu_tilde_dz = nt_amp * fx * sz;
    return g;
}

void fill_mms(const CfdMesh& mesh, const MmsSolutionEulerBC& mms,
              std::vector<ConservativeState>& q, Real gamma) {
    q.resize(mesh.cells.size());
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        PrimitiveState w = mms.eval(
            mesh.cells[i].cx, mesh.cells[i].cy, mesh.cells[i].cz);
        q[i] = primitive_to_conservative(w, gamma);
    }
}

void fill_mms_sa(const CfdMesh& mesh, const MmsSolutionSABC& mms,
                 std::vector<ConservativeState>& q, Real gamma) {
    q.resize(mesh.cells.size());
    for (std::size_t i = 0; i < mesh.cells.size(); ++i) {
        PrimitiveState w = mms.eval_sa(
            mesh.cells[i].cx, mesh.cells[i].cy, mesh.cells[i].cz);
        q[i] = primitive_to_conservative(w, gamma);
    }
}

MmsSolutionEulerBC make_default_mms_euler_bc() {
    MmsSolutionEulerBC mms{};
    mms.freq = 1.0;
    mms.rho0 = 1.0;   mms.rho_amp = 0.1;
    mms.u0   = 0.5;   mms.u_amp   = 0.05;
    mms.v0   = 0.0;   mms.v_amp   = 0.05;
    mms.w0   = 0.0;   mms.w_amp   = 0.05;
    mms.p0   = 1.0 / 1.4;  mms.p_amp = 0.05;
    return mms;
}

MmsSolutionSABC make_default_mms_sa_bc() {
    MmsSolutionSABC mms{};
    mms.freq = 1.0;
    mms.rho0 = 1.0;   mms.rho_amp = 0.1;
    mms.u0   = 0.5;   mms.u_amp   = 0.05;
    mms.v0   = 0.0;   mms.v_amp   = 0.05;
    mms.w0   = 0.0;   mms.w_amp   = 0.05;
    mms.p0   = 1.0 / 1.4;  mms.p_amp = 0.05;
    mms.nt0  = 0.1;   mms.nt_amp = 0.05;
    return mms;
}

MmsSolutionEuler make_default_mms_euler() {
    MmsSolutionEuler mms{};
    mms.freq = 1.0;
    mms.rho0 = 1.0;  mms.rho_amp = 0.1;
    mms.u0   = 0.5;  mms.u_amp   = 0.05;
    mms.v0   = 0.3;  mms.v_amp   = 0.05;
    mms.w0   = 0.2;  mms.w_amp   = 0.05;
    mms.p0   = 1.0 / 1.4;  mms.p_amp = 0.05;
    return mms;
}

MmsSolutionSA make_default_mms_sa() {
    MmsSolutionSA mms{};
    mms.freq = 1.0;
    mms.rho0 = 1.0;  mms.rho_amp = 0.1;
    mms.u0   = 0.5;  mms.u_amp   = 0.05;
    mms.v0   = 0.3;  mms.v_amp   = 0.05;
    mms.w0   = 0.2;  mms.w_amp   = 0.05;
    mms.p0   = 1.0 / 1.4;  mms.p_amp = 0.05;
    mms.nt0  = 0.1;  mms.nt_amp = 0.05;
    return mms;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
