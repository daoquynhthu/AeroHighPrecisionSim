#pragma once

#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/reconstruction.hpp"

#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

struct MmsSolutionEuler {
    Real freq;
    Real rho0, rho_amp;
    Real u0, u_amp;
    Real v0, v_amp;
    Real w0, w_amp;
    Real p0, p_amp;

    PrimitiveState eval(Real x, Real y, Real z) const;
    PrimitiveGradient eval_gradient(Real x, Real y, Real z) const;
};

struct MmsSolutionSA : MmsSolutionEuler {
    Real nt0, nt_amp;

    PrimitiveState eval_sa(Real x, Real y, Real z) const;
    PrimitiveGradient eval_gradient_sa(Real x, Real y, Real z) const;
};

void fill_mms(
    const CfdMesh& mesh,
    const MmsSolutionEuler& mms,
    std::vector<ConservativeState>& q,
    Real gamma);

void fill_mms_sa(
    const CfdMesh& mesh,
    const MmsSolutionSA& mms,
    std::vector<ConservativeState>& q,
    Real gamma);

bool compute_mms_source(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q_exact,
    const PrimitiveState& freestream,
    const CfdConfig& config,
    std::vector<EulerFlux>& source);

Real mms_l2_error(
    const std::vector<ConservativeState>& q,
    const std::vector<ConservativeState>& q_ref,
    int nvar = CFD_NVAR);

Real mms_observed_order(
    Real err_coarse, int n_coarse,
    Real err_medium, int n_medium,
    Real err_fine, int n_fine);

MmsSolutionEuler make_default_mms_euler();
MmsSolutionSA make_default_mms_sa();

} // namespace cfd
} // namespace aero
} // namespace aerosp
