#pragma once

#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/cfd_mesh.hpp"
#include "aero/cfd/cfd_result.hpp"
#include "aero/cfd/cfd_state.hpp"
#include "aero/cfd/reconstruction.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

struct CfdSolveSummary {
    CfdForceResult forces;
    CfdDiagnostics diagnostics;
    std::vector<Real> residual_history;
    std::vector<ConservativeState> final_state;
    std::vector<Real> sst_final_k;
    std::vector<Real> sst_final_omega;
    bool converged = false;
    bool failed = false;
};

void integrate_wall_forces(const CfdMesh& mesh, const std::vector<int>& wall_face_indices,
    const std::vector<ConservativeState>& q, const FreestreamCondition& condition,
    const CfdConfig& config, CfdForceResult& result,
    const std::vector<PrimitiveGradient>* grads = nullptr);

// CPU SST pipeline functions (used by CfdSolver and directly by tests)
bool compute_sst_gradients_cpu(
    const CfdMesh& mesh,
    const std::vector<Real>& k,
    const std::vector<Real>& omega,
    std::vector<Real>& grad_k,
    std::vector<Real>& grad_omega);

bool compute_sst_advection_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<Real>& k,
    const std::vector<Real>& omega,
    Real inf_k, Real inf_omega,
    Real gamma,
    std::vector<Real>& res_k,
    std::vector<Real>& res_omega);

bool compute_sst_diffusion_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<Real>& k,
    const std::vector<Real>& omega,
    const std::vector<Real>& grad_k,
    const std::vector<Real>& grad_omega,
    const std::vector<Real>& f1,
    Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    std::vector<Real>& res_k,
    std::vector<Real>& res_omega);

bool compute_sst_source_cpu(
    const CfdMesh& mesh,
    const std::vector<ConservativeState>& q,
    const std::vector<Real>& k,
    const std::vector<Real>& omega,
    const std::vector<PrimitiveGradient>& flow_grads,
    const std::vector<Real>& grad_k,
    const std::vector<Real>& grad_omega,
    Real gamma, Real Re,
    Real mu_ref, Real T_ref, Real sutherland_T,
    std::vector<Real>& res_k,
    std::vector<Real>& res_omega,
    std::vector<Real>& f1);

class CfdSolver {
public:
    bool load_mesh(const CfdMesh& mesh);

    CfdSolveSummary solve(const FreestreamCondition& condition, const CfdConfig& config);

    CfdSolveSummary solve_from_state(
        const FreestreamCondition& condition,
        const CfdConfig& config,
        const std::vector<ConservativeState>& initial_state);

    const CfdMesh& mesh() const { return mesh_; }

private:
    CfdMesh mesh_;
    std::vector<int> wall_face_indices_;

    // SST k-omega state (per-cell, separate from main NVAR=6 flow state)
    std::vector<Real> sst_k_;
    std::vector<Real> sst_omega_;
    std::vector<Real> sst_residual_k_;
    std::vector<Real> sst_residual_omega_;
    std::vector<Real> sst_grad_k_;
    std::vector<Real> sst_grad_omega_;
    std::vector<Real> sst_f1_;
};

bool assert_oracle_equivalent(
    const CfdSolveSummary& gpu,
    const CfdSolveSummary& cpu,
    Real tol_residual,
    Real tol_forces,
    std::string* error);

} // namespace cfd
} // namespace aero
} // namespace aerosp


