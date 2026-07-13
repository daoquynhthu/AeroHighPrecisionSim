#include "aero/cfd/cuda_utils.hpp"
#include "aero/cfd/real.hpp"
#include "aero/cfd/krylov_ops.hpp"
#include "aero/cfd/fgmres.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

constexpr int kBlockSize = 256;

// Persistent scratch buffer for two-stage dot-product reduction.
// Allocated lazily, expanded on demand.  Owned at module scope so
// the free-function API (ddot_gpu / dnrm2_gpu) needs no extra
// parameters and callers (FgmresSolver) are unaffected.
static Real* d_partial_buf = nullptr;
static int   d_partial_cap = 0;

static bool ensure_partial_buf(int min_blocks) {
    if (d_partial_cap >= min_blocks) return true;
    cuda_free_safe(d_partial_buf);
    d_partial_cap = 0;
    if (!cuda_check(cudaMalloc(&d_partial_buf, (size_t)min_blocks * sizeof(Real)),
                    "ensure_partial_buf"))
        return false;
    d_partial_cap = min_blocks;
    return true;
}

// Stage 1: each block writes its partial dot-product to d_partial[blockIdx.x].
__global__ void __launch_bounds__(256) ddot_kernel(const Real* x, const Real* y, int n, Real* d_partial) {
    __shared__ Real sdata[kBlockSize];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    Real sum = 0;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        sum += x[i] * y[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}

// Stage 2: single-block kernel sums the partials from every block.
__global__ void __launch_bounds__(256) reduce_sum_kernel(const Real* d_partial, int num_blocks, Real* result) {
    Real sum = 0;
    for (int i = 0; i < num_blocks; ++i) {
        sum += d_partial[i];
    }
    *result = sum;
}

__global__ void __launch_bounds__(256) daxpy_kernel(Real a, const Real* x, Real* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void __launch_bounds__(256) daxpy_device_kernel(Real mul, const Real* d_a, const Real* x, Real* y, int n) {
    Real alpha = mul * (*d_a);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        y[i] = alpha * x[i] + y[i];
    }
}

__global__ void __launch_bounds__(256) dscal_kernel(Real a, Real* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        x[i] = a * x[i];
    }
}

__global__ void __launch_bounds__(256) dcopy_kernel(const Real* src, Real* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void __launch_bounds__(256) daxpby_kernel(Real a, const Real* x, Real b, Real* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        y[i] = a * x[i] + b * y[i];
    }
}

int grid_size(int n) {
    int g = (n + kBlockSize - 1) / kBlockSize;
    return g < 1 ? 1 : (g > 65536 ? 65536 : g);
}

} // namespace

bool ddot_gpu(const Real* x, const Real* y, int n, Real* result, cudaStream_t stream) {
    int grid = grid_size(n);
    if (!ensure_partial_buf(grid)) return false;
    ddot_kernel<<<grid, kBlockSize, 0, stream>>>(x, y, n, d_partial_buf);
    if (!cuda_check(cudaGetLastError(), "ddot kernel")) return false;
    reduce_sum_kernel<<<1, 1, 0, stream>>>(d_partial_buf, grid, result);
    return cuda_check(cudaGetLastError(), "reduce_sum kernel");
}

bool daxpy_gpu(Real a, const Real* x, Real* y, int n, cudaStream_t stream) {
    int grid = grid_size(n);
    daxpy_kernel<<<grid, kBlockSize, 0, stream>>>(a, x, y, n);
    return cuda_check(cudaGetLastError(), "daxpy kernel");
}

bool daxpy_device_gpu(Real mul, const Real* d_a, const Real* x, Real* y, int n, cudaStream_t stream) {
    int grid = grid_size(n);
    daxpy_device_kernel<<<grid, kBlockSize, 0, stream>>>(mul, d_a, x, y, n);
    return cuda_check(cudaGetLastError(), "daxpy_device kernel");
}

bool dnrm2_gpu(const Real* x, int n, Real* result, cudaStream_t stream) {
    int grid = grid_size(n);
    if (!ensure_partial_buf(grid)) return false;
    ddot_kernel<<<grid, kBlockSize, 0, stream>>>(x, x, n, d_partial_buf);
    if (!cuda_check(cudaGetLastError(), "dnrm2 ddot kernel")) return false;
    reduce_sum_kernel<<<1, 1, 0, stream>>>(d_partial_buf, grid, result);
    return cuda_check(cudaGetLastError(), "dnrm2 reduce_sum kernel");
}

bool dscal_gpu(Real a, Real* x, int n, cudaStream_t stream) {
    int grid = grid_size(n);
    dscal_kernel<<<grid, kBlockSize, 0, stream>>>(a, x, n);
    return cuda_check(cudaGetLastError(), "dscal kernel");
}

bool dcopy_gpu(const Real* src, Real* dst, int n, cudaStream_t stream) {
    int grid = grid_size(n);
    dcopy_kernel<<<grid, kBlockSize, 0, stream>>>(src, dst, n);
    return cuda_check(cudaGetLastError(), "dcopy kernel");
}

bool daxpby_gpu(Real a, const Real* x, Real b, Real* y, int n, cudaStream_t stream) {
    int grid = grid_size(n);
    daxpby_kernel<<<grid, kBlockSize, 0, stream>>>(a, x, b, y, n);
    return cuda_check(cudaGetLastError(), "daxpby kernel");
}

// --- FgmresSolver ---

FgmresSolver::FgmresSolver(int n, int restart, int max_iter, Real tol)
    : n_(n), restart_(restart), max_iter_(max_iter), tol_(tol) {}

FgmresSolver::~FgmresSolver() { release(); }

bool FgmresSolver::allocate(std::string* error) {
    if (allocated_) return true;
    if (n_ <= 0 || restart_ <= 0) {
        if (error) *error = "FgmresSolver: invalid n or restart";
        return false;
    }
    int m = restart_;
    ldv_ = n_;
    ldz_ = n_;

    if (!cuda_check(cudaMalloc(&d_v_, (m + 1) * ldv_ * sizeof(Real)), "cudaMalloc d_v", error)) return false;
    if (!cuda_check(cudaMalloc(&d_z_, m * ldz_ * sizeof(Real)), "cudaMalloc d_z", error)) return false;
    if (!cuda_check(cudaMalloc(&d_w_, n_ * sizeof(Real)), "cudaMalloc d_w", error)) return false;
    if (!cuda_check(cudaMalloc(&d_hess_, (m + 1) * m * sizeof(Real)), "cudaMalloc d_hess", error)) return false;
    if (!cuda_check(cudaMalloc(&d_rs_, (m + 1) * sizeof(Real)), "cudaMalloc d_rs", error)) return false;

    allocated_ = true;
    return true;
}

void FgmresSolver::release() {
    cuda_free_safe(d_v_);
    cuda_free_safe(d_z_);
    cuda_free_safe(d_w_);
    cuda_free_safe(d_hess_);
    cuda_free_safe(d_rs_);
    allocated_ = false;
}

void FgmresSolver::generate_givens_rotation(Real a, Real b, Real& c, Real& s) {
    Real abs_a = real_fabs(a);
    Real abs_b = real_fabs(b);
    if (abs_a < std::numeric_limits<Real>::min() && abs_b < std::numeric_limits<Real>::min()) {
        c = 1; s = 0;
        return;
    }
    if (abs_a >= abs_b) {
        Real t = b / a;
        Real tt = real_sqrt(1 + t * t);
        c = 1 / tt;
        s = t * c;
    } else {
        Real t = a / b;
        Real tt = real_sqrt(1 + t * t);
        s = 1 / tt;
        c = t * s;
    }
}

bool FgmresSolver::solve(const MatvecFunc& matvec,
                          const Real* d_b,
                          Real* d_x,
                          std::string* error,
                          cudaStream_t stream) {
    if (!allocated_) {
        if (error) *error = "FgmresSolver not allocated";
        return false;
    }
    iterations_ = 0;
    converged_ = false;
    final_residual_ = 0;

    int m = restart_;
    int n = n_;
    Real* v = d_v_;
    Real* z = d_z_;
    Real* w = d_w_;
    Real* rs = d_rs_;

    auto krylov_ddot_device = [&](const Real* a, const Real* b, Real* d_out) -> bool {
        if (!ddot_gpu(a, b, n, d_out, stream)) { if (error) *error = "krylov ddot_device failed"; return false; }
        return true;
    };
    auto krylov_ddot_host = [&](const Real* a, const Real* b, Real& val) -> bool {
        if (!ddot_gpu(a, b, n, rs, stream)) { if (error) *error = "krylov ddot_host failed"; return false; }
        if (!cuda_check(cudaMemcpyAsync(&val, rs, sizeof(Real), cudaMemcpyDeviceToHost, stream), "ddot copy", error)) return false;
        if (!cuda_check(cudaStreamSynchronize(stream), "ddot sync", error)) return false;
        return true;
    };

    Real beta = 0;
    Real* v0 = v;

    if (!matvec(d_x, w, error)) { if (error) { std::string e = "FGMRES initial matvec failed: " + *error; *error = e; } return false; }
    if (!daxpby_gpu(1, d_b, -1, w, n, stream)) { if (error) *error = "w = b - Ax0 failed"; return false; }

    if (!krylov_ddot_host(w, w, beta)) return false;
    beta = real_sqrt(beta);
    if (beta < std::numeric_limits<Real>::min()) {
        converged_ = true;
        final_residual_ = 0;
        return true;
    }

    Real inv_beta = 1.0f / beta;
    if (!dscal_gpu(inv_beta, w, n, stream)) { if (error) *error = "v0 = r/beta failed"; return false; }
    if (!dcopy_gpu(w, v0, n, stream)) { if (error) *error = "copy v0 failed"; return false; }

    std::vector<Real> h_host((m + 1) * m, 0);
    std::vector<Real> givens_c_host(m, 0);
    std::vector<Real> givens_s_host(m, 0);
    std::vector<Real> rs_host(m + 1, 0);
    rs_host[0] = beta;

    iterations_ = 0;
    int k = 0;

    for (int outer = 0; outer < max_iter_ && !converged_; ++outer) {
        k = 0;
        for (int i = 0; i < m + 1; ++i) rs_host[i] = 0;
        rs_host[0] = beta;

        for (int j = 0; j < m; ++j) {
            Real* vj = v + j * ldv_;

            Real* zj = z + j * ldz_;
            if (prec_) {
                if (!prec_(vj, zj, error)) {
                    if (error) { std::string e = "FGMRES preconditioner failed: " + *error; *error = e; }
                    return false;
                }
            } else {
                if (!dcopy_gpu(vj, zj, n, stream)) { if (error) *error = "copy v->z failed"; return false; }
            }

            if (!matvec(zj, w, error)) {
                if (error) { std::string e = "FGMRES matvec failed: " + *error; *error = e; }
                return false;
            }

            for (int i = 0; i <= j; ++i) {
                Real* vi = v + i * ldv_;
                if (!krylov_ddot_device(w, vi, d_hess_ + i * m + j)) return false;
                if (!daxpy_device_gpu(static_cast<Real>(-1), d_hess_ + i * m + j, vi, w, n, stream)) {
                    if (error) *error = "MGS device axpy failed"; return false;
                }
            }

            if (!krylov_ddot_device(w, w, d_hess_ + (j + 1) * m + j)) return false;

            {
                size_t pitch = static_cast<size_t>(m) * sizeof(Real);
                if (!cuda_check(
                        cudaMemcpy2DAsync(h_host.data() + j, pitch,
                                          d_hess_ + j, pitch,
                                          sizeof(Real), static_cast<size_t>(j) + 2,
                                          cudaMemcpyDeviceToHost, stream),
                        "hess col copy", error))
                    return false;
                if (!cuda_check(cudaStreamSynchronize(stream), "hess col sync", error))
                    return false;
            }

            {
                Real h_j1j_sq = h_host[(j + 1) * m + j];
                Real h_j1j = real_sqrt(h_j1j_sq);
                h_host[(j + 1) * m + j] = h_j1j;

                if (h_j1j > std::numeric_limits<Real>::min()) {
                    Real inv_h = 1.0f / h_j1j;
                    Real* v_j1 = v + (j + 1) * ldv_;
                    if (!dscal_gpu(inv_h, w, n, stream)) { if (error) *error = "scale v_j+1 failed"; return false; }
                    if (!dcopy_gpu(w, v_j1, n, stream)) { if (error) *error = "copy v_j+1 failed"; return false; }
                }
            }

            for (int i = 0; i < j; ++i) {
                Real temp = givens_c_host[i] * h_host[i * m + j] + givens_s_host[i] * h_host[(i + 1) * m + j];
                h_host[(i + 1) * m + j] = -givens_s_host[i] * h_host[i * m + j] + givens_c_host[i] * h_host[(i + 1) * m + j];
                h_host[i * m + j] = temp;
            }
            {
                Real a = h_host[j * m + j];
                Real b = h_host[(j + 1) * m + j];
                Real c = 1, s = 0;
                generate_givens_rotation(a, b, c, s);
                givens_c_host[j] = c;
                givens_s_host[j] = s;
                h_host[j * m + j] = c * a + s * b;
                h_host[(j + 1) * m + j] = 0;

                Real rs_j = rs_host[j];
                rs_host[j] = c * rs_j;
                rs_host[j + 1] = -s * rs_j;
            }

            Real res = real_fabs(rs_host[j + 1]);
            iterations_++;

            if (res <= tol_) {
                converged_ = true;
                k = j;
                break;
            }
            k = j;
        }

        int k_eff = k;
        {
            std::vector<Real> R(k_eff + 1, 0);
            for (int i = 0; i <= k_eff; ++i) R[i] = rs_host[i];

            for (int i = k_eff; i >= 0; --i) {
                Real sum = R[i];
                for (int j = i + 1; j <= k_eff; ++j) {
                    sum -= h_host[i * m + j] * R[j];
                }
                if (real_fabs(h_host[i * m + i]) > std::numeric_limits<Real>::epsilon()) {
                    R[i] = sum / h_host[i * m + i];
                } else {
                    R[i] = 0;
                }
            }

            if (!dscal_gpu(0, w, n, stream)) { if (error) *error = "zero w failed"; return false; }
            if (!daxpy_gpu(R[0], z, w, n, stream)) { if (error) *error = "w = R[0]*z0 failed"; return false; }
            for (int j = 1; j <= k_eff; ++j) {
                Real* zj = z + j * ldz_;
                if (!daxpy_gpu(R[j], zj, w, n, stream)) { if (error) *error = "w += R[j]*zj failed"; return false; }
            }

            if (!daxpy_gpu(1, w, d_x, n, stream)) { if (error) *error = "x += w failed"; return false; }
        }

        if (!converged_) {
            if (!matvec(d_x, w, error)) { if (error) { std::string e = "FGMRES restart matvec failed: " + *error; *error = e; } return false; }
            if (!daxpby_gpu(1, d_b, -1, w, n, stream)) { if (error) *error = "w = b - Ax failed"; return false; }

            if (!krylov_ddot_host(w, w, beta)) return false;
            beta = real_sqrt(beta);
            if (beta < std::numeric_limits<Real>::min()) {
                converged_ = true;
                final_residual_ = 0;
                break;
            }

            Real inv_beta2 = 1.0f / beta;
            if (!dscal_gpu(inv_beta2, w, n, stream)) { if (error) *error = "v0 = r/beta failed"; return false; }
            if (!dcopy_gpu(w, v0, n, stream)) { if (error) *error = "copy v0 failed"; return false; }
        }
    }

    final_residual_ = real_fabs(rs_host[k + 1]);
    return true;
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
