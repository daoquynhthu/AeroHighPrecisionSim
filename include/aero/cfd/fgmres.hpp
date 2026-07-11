#pragma once

#include "aero/cfd/real_fwd.hpp"
#include <cuda_runtime.h>
#include <functional>

namespace aerosp {
namespace aero {
namespace cfd {

class FgmresSolver {
public:
    using MatvecFunc = std::function<bool(const Real*, Real*, std::string*)>;

    FgmresSolver(int n, int restart, int max_iter, Real tol);
    ~FgmresSolver();

    FgmresSolver(const FgmresSolver&) = delete;
    FgmresSolver& operator=(const FgmresSolver&) = delete;

    bool allocate(std::string* error = nullptr);
    void release();

    bool solve(const MatvecFunc& matvec,
               const Real* d_b,
               Real* d_x,
               std::string* error = nullptr,
               cudaStream_t stream = nullptr);

    int iterations() const { return iterations_; }
    Real final_residual() const { return final_residual_; }
    bool converged() const { return converged_; }

    void set_preconditioner(const MatvecFunc& prec) { prec_ = prec; }
    void clear_preconditioner() { prec_ = MatvecFunc(); }

private:
    int n_;
    int restart_;
    int max_iter_;
    Real tol_;
    int iterations_ = 0;
    Real final_residual_ = 0;
    bool converged_ = false;
    bool allocated_ = false;

    MatvecFunc prec_;

    Real* d_v_ = nullptr;
    Real* d_z_ = nullptr;
    Real* d_w_ = nullptr;
    Real* d_hess_ = nullptr;
    Real* d_rs_ = nullptr;
    int ldv_;
    int ldz_;

    void generate_givens_rotation(Real a, Real b, Real& c, Real& s);
};

} // namespace cfd
} // namespace aero
} // namespace aerosp
