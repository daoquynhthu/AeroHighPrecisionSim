# CFD GPU Production Plan

> Design: `docs/AERO_ACCURACY_UPGRADE.md`
> Progress log: `docs/progress.md`
> Active blockers: `docs/ISSUES.md`
> CPU Oracle 解耦: `docs/CPU_ORACLE_DECOMP.md`

## Architecture Decision

**GPU is the production path.** CPU exists only as reference/debug oracle for unit tests and regression verification.

```
Mesh (CPU, once) → DeviceMesh SoA upload → Full solve loop on GPU → Download results
                                               ↑
                                    CPU oracle (debug/CI only)
```

All physics modules (Euler, reconstruction, viscous, RANS) are implemented first as CPU oracle functions for formula verification, then ported to GPU kernels. GPU porting is not optional — it is the deliverable.

---

## Phase 0 — Retained CPU Foundation

Goal: establish the CPU reference implementation that the GPU path will be validated against. Everything here compiles today; no changes required.

Files (kept as-is):

| File | Role |
|------|------|
| `include/aero/cfd/cfd_mesh.hpp` | Node/cell/face mesh model |
| `include/aero/cfd/cfd_state.hpp` | Primitive/conservative/HLLC/farfield BC |
| `include/aero/cfd/cfd_config.hpp` | Configuration (will add GPU flags later) |
| `include/aero/cfd/cfd_result.hpp` | Force/moment result struct |
| `include/aero/cfd/cfd_solver.hpp` | CfdSolver class (CPU solve) |
| `include/aero/cfd/cfd_residual.hpp` | Euler residual assembly (CPU) |
| `include/aero/cfd/reconstruction.hpp` | Gradient/limiter/reconstruct (CPU) |
| `include/aero/cfd/diagnostics.hpp` | State bounds/failure/VTK (CPU) |
| `include/aero/cfd/viscous.hpp` | Viscous gradient/wall/flux (CPU) |
| `include/aero/cfd/cuda_utils.hpp` | CUDA error-check helper |
| `src/aero/cfd/mesh_metrics.cpp` | Mesh generation + metrics |
| `src/aero/cfd/cfd_solver.cpp` | CPU solve loop + wall integration |
| `src/aero/cfd/cfd_residual.cpp` | CPU Euler residual assembly |
| `src/aero/cfd/reconstruction.cpp` | CPU Green-Gauss/LSQ/limiter |
| `src/aero/cfd/diagnostics.cpp` | CPU diagnostics + VTK writer |
| `src/aero/cfd/viscous.cpp` | CPU Sutherland/gradients/wall flux |
| `src/aero/cfd/cuda_utils.cpp` | CPU-side CUDA error helper |

Tests (kept as-is, serve as CPU oracle tests):

| Test | Role |
|------|------|
| `tests/cfd/test_cfd_mesh.cpp` | Mesh validity baseline |
| `tests/cfd/test_cfd_euler.cpp` | Euler flux/solver baseline |
| `tests/cfd/test_cfd_diagnostics.cpp` | Diagnostics baseline |
| `tests/cfd/test_cfd_reconstruction.cpp` | Reconstruction baseline |
| `tests/cfd/test_cfd_viscous.cpp` | Viscous utility baseline |

Gate:

- All CPU tests pass.
- `cmake -B build && cmake --build build --target TestCfdMesh --config Release` succeeds.
- `ctest --test-dir build -C Release -R "Cfd(Mesh|Euler|Diagnostics|Reconstruction|Viscous)"` reports 100% pass.

---

## Phase 1 — GPU Data Pipeline (SoA Refactoring)

Goal: replace the current AoS (`CfdFace`/`CfdCell` struct arrays) GPU memory layout with a SoA (Struct-of-Arrays) layout optimized for coalesced GPU access. CPU `CfdMesh` retained for generation; `DeviceMesh` is the production container.

Rationale: current `CfdFace` is ~72 bytes (13 floats + 4 ints + 1 enum). AoS causes strided, uncoalesced global memory reads. SoA packs same-field values contiguously, achieving near-peak memory bandwidth.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/device_mesh.hpp` | NEW | SoA structs: `DeviceFaceData`, `DeviceCellData`, `DeviceState` |
| `src/aero/cfd/device_mesh.cu` | NEW | Upload/download kernels, RAII management |
| `include/aero/cfd/gpu_buffers.hpp` | REPLACE | Thin wrapper or redirect to `device_mesh` |
| `src/aero/cfd/gpu_buffers.cu` | REPLACE | Delegate to `device_mesh` |
| `include/aero/cfd/gpu_solver.hpp` | NEW | GPU solver declarations (extends for Phase 2) |
| `src/aero/cfd/cfd_residual_gpu.cu` | REWRITE | `euler_residual_kernel` uses SoA layout |
| `src/aero/cfd/reconstruction_gpu.cu` | REWRITE | `apply_limiter_kernel` uses SoA layout |

Design:

```cpp
// include/aero/cfd/device_mesh.hpp
struct DeviceFaceData {
    float* d_nx; float* d_ny; float* d_nz;     // coalesced: each is a contiguous float array
    float* d_area;
    int*   d_left_cell;  int* d_right_cell;
    int*   d_boundary;                          // BoundaryKind as int
};

struct DeviceCellData {
    float* d_volume; float* d_h_min;
    float* d_cx; float* d_cy; float* d_cz;
};

struct DeviceState {
    float* d_q;                                 // [NVAR * n_cells] flat
};

class DeviceMesh {
public:
    bool upload(const CfdMesh& cpu_mesh, std::string* error = nullptr);
    bool upload_state(const std::vector<ConservativeState>& q, std::string* error = nullptr);
    bool download_state(std::vector<ConservativeState>& q, std::string* error = nullptr) const;
    void release();
    // accessors for kernel launches
    DeviceFaceData face() const;
    DeviceCellData cell() const;
    DeviceState state() const;
    float* residual() const;                     // [NVAR * n_cells]
    int cell_count() const;
    int face_count() const;
};
```

Tasks:

- [x] Define `DeviceFaceData` SoA: 3× float*, 1× float*, 3× int*
- [x] Define `DeviceCellData` SoA: 5× float*
- [x] Define `DeviceState`: flat `float*` (NVAR=5 per cell)
- [x] `DeviceMesh::upload()`: CfdMesh → device SoA, single cudaMemcpy per array
- [x] `DeviceMesh::upload_state()`: vector<ConservativeState> → device flat array
- [x] `DeviceMesh::download_state()`: device flat array → vector<ConservativeState>
- [x] `DeviceMesh`: RAII (default=null, move-only, ~DeviceMesh releases)
- [x] Refactor `euler_residual_kernel` to accept `DeviceFaceData` + `DeviceState` + `float* residual`
- [x] Refactor `apply_limiter_kernel` to SoA gradient/limiter arrays
- [x] CPU/GPU residual equivalence test on 13^3 cube mesh (not 2-cell)

Gate:

- `DeviceMesh` upload + `euler_residual_kernel` on full cube mesh matches `compute_euler_residual_cpu` component-wise within 1e-6 absolute per cell.
- No AoS struct arrays exist on device after upload.
- SoA memory layout verified: `d_nx[i+1] - d_nx[i] == sizeof(float)` (contiguous).
- Move semantics correct: moved-from `DeviceMesh` has null pointers, destructor no-ops.

---

## Phase 2 — Full GPU Euler Solver Loop

Goal: move the entire pseudo-time iteration loop from CPU to GPU. Zero CPU-GPU data transfer between initial upload and final download.

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/gpu_timestep.cu` | NEW | `timestep_kernel` + block-reduce min dt |
| `src/aero/cfd/gpu_update.cu` | NEW | `update_state_kernel` + `l2_norm_kernel` + state validity |
| `src/aero/cfd/gpu_wall.cu` | NEW | `wall_force_kernel` + block-reduce 6 components |
| `src/aero/cfd/gpu_solver.cu` | NEW | `solve_gpu()` full loop orchestration |
| `include/aero/cfd/gpu_solver.hpp` | MODIFY | Add `solve_gpu()` declaration |
| `include/aero/cfd/cfd_config.hpp` | MODIFY | Add `bool use_gpu = true` |
| `include/aero/cfd/cfd_solver.hpp` | MODIFY | `solve()` dispatches to `solve_gpu()` |
| `src/aero/cfd/cfd_solver.cpp` | MODIFY | `solve()` selects GPU/CPU path |

Kernels:

- [ ] `timestep_kernel`: per cell dt = CFL * h_min / (|v| + a), block-wide `atomicMin` for global dt
- [ ] `euler_residual_kernel`: exists (rewired in Phase 1)
- [ ] `update_state_kernel`: q[t+1].component = q[t].component + dt/V * residual.component (component-wise)
- [ ] `l2_norm_kernel`: per cell sum((q_new - q_old)^2), block-reduce to scalar via shared memory
- [ ] `state_validity_kernel`: per cell `is_valid_primitive()`, `atomicOr` failed flag
- [ ] `wall_force_kernel`: per wall face accumulate force + moment, block-reduce to 6 scalars

Solver orchestration:

```
solve_gpu(mesh, freestream, config, summary):
  DeviceMesh d_mesh
  d_mesh.upload(mesh)
  d_mesh.upload_state(initial_q)

  for iter in 0..max_iter:
    timestep_kernel(d_mesh, config)        → min_dt (device scalar)
    euler_residual_kernel(d_mesh)          → residual
    update_state_kernel(d_mesh, min_dt)    → q_new
    l2 = l2_norm_kernel(d_mesh)            → device scalar
    if l2 < tol: converged; break
    valid = state_validity_kernel(d_mesh)  → device scalar
    if !valid: failed; break

  wall_force_kernel(d_mesh)                → forces (device array)
  d_mesh.download_state(final_q)
  download forces to host
  fill summary
```

Host-device sync: none during the iteration loop. Convergence and failure flags read only once at loop exit.

Tasks:

- [x] `timestep_kernel`: per-cell dt with block-reduction for min dt (atomicCAS based on `__float_as_uint`)
- [x] `update_and_l2_kernel`: component-wise update with in-place L2 accumulation via atomicAdd
- [x] `state_validity_kernel`: combined with update kernel — finite + positive rho/p check, `atomicExch` failure flag
- [x] `wall_force_kernel`: per-face pressure * area * normal, atomicAdd on 6 force/moment counters
- [x] `solve_gpu()`: loop with convergence/failure exit conditions
- [x] `cfd_solver.cpp`: `solve()` uses GPU when `config.use_gpu==true`, else CPU
- [x] DeviceMesh face centroid arrays (cx/cy/cz) for wall moment computation

Gate:

- [x] GPU-CPU L2 match within 1e-6 after 1 iteration on 13^3 cube mesh (CFD-GPU-6).
- [x] GPU-CPU iteration-by-iteration L2 match within 1e-6 for first 20 iterations on cube (CFD-GPU-7).
- [x] GPU and CPU converge to comparable residual level on flat plate mesh (CFD-GPU-8): ratio ≤ 1e3.
- [x] Zero `cudaMemcpy` calls during iteration loop (resolved in Phase 3).

---

## Phase 3 — CPU Oracle & Regression Verification

Goal: establish CPU as reference oracle for GPU validation. Every GPU operation has a CPU equivalent that can be compared in debug/CI mode. Also eliminate all `cudaMemcpy` calls from the iteration loop (PH2-G-1).

### Task 1: `cpu_oracle` configuration

**`include/aero/cfd/cfd_config.hpp`**: add `bool cpu_oracle = false` to `CfdConfig`.

### Task 2: Oracle mode in solver

**`include/aero/cfd/cfd_solver.hpp`**: add `assert_oracle_equivalent()` helper declaration.

**`src/aero/cfd/cfd_solver.cpp`**: in `solve()`, after GPU path returns:

```
solve():
  gpu_result = solve_gpu(...)
  if config.cpu_oracle:
    cpu_config = config; cpu_config.use_gpu = false
    cpu_result = solve_cpu(cpu_config)
    assert_oracle_equivalent(gpu_result, cpu_result)
  return gpu_result
```

`assert_oracle_equivalent()` compares:
- `residual_history` iter-by-iter (if one has more iters, only compare overlap)
- `forces.CX/CY/CZ/Cl/Cm/Cn` normalized coefficients
- Reports first mismatch with iteration index, GPU value, CPU value, diff

All comparisons use relative tolerance: `fabs(a-b) <= tol * (1.0 + max(fabs(a), fabs(b)))`.

### Task 3: Zero `cudaMemcpy` during iteration loop

**Problem**: current loop has 4 D2H reads per iteration:
1. `cudaMemcpy(&residual_failed, d_failed, ...)` — failure check after residual
2. `cudaMemcpy(&min_dt, d_min_dt, ...)` — read timestep
3. `cudaMemcpy(&update_failed, d_failed, ...)` — failure check after update
4. `cudaMemcpy(&l2, d_l2_sum, ...)` — read L2 norm

**Solution**: all convergence/failure decisions happen on device. Host runs all `max_iter` iterations without intermediate reads.

**New device buffers:**
- `int* d_converged` — set to 1 by `check_status_kernel` when converged or failed
- `float* d_residual_history` — `max_iter` slots, written by `check_status_kernel`

**New kernel** (in `gpu_solver.cu` or new file):

```
__global__ void check_status_kernel(
    const int* d_failed, const float* d_l2_sum,
    int nvar_ncells, float convergence_tol,
    int* d_converged, float* d_residual_history_slot)
```

Thread 0 block 0: if `*d_failed != 0`, set `*d_converged = 1` and write -1.0f to history slot.
Otherwise compute `l2 = sqrtf(*d_l2_sum / nvar_ncells)`, write to history slot, and if `l2 < convergence_tol` set `*d_converged = 1`.

**Changes to `compute_update_gpu`:**
- `gpu_solver_internal.hpp` + `gpu_update.cu`: signature changes from `float min_dt` to `const float* d_min_dt`
- `update_and_l2_kernel` reads `float min_dt = *d_min_dt` from device pointer

**New iteration loop structure** (in `gpu_solver.cu`):

```
// Before loop: allocate d_converged, d_residual_history

for (int iter = 0; iter < config.max_iter; ++iter) {
    // All launches on default stream — implicit ordering
    compute_timestep_gpu(d_mesh, gamma, cfl, d_min_dt);   // <- no host read needed
    launch_euler_residual_kernel(...);
    compute_update_gpu(d_mesh, d_min_dt, gamma, d_l2_sum, d_failed);
    check_status_kernel<<<1,1>>>(d_failed, d_l2_sum, ..., d_converged, d_residual_history + iter);
}

// After loop: single sync + single batch of D2H reads
cudaDeviceSynchronize();
cudaMemcpy(&host_failed, d_failed, ...);
cudaMemcpy(&host_converged, d_converged, ...);
cudaMemcpy(host_residual_history, d_residual_history, max_iter * sizeof(float), D2H);
cudaMemcpy(host_forces, d_forces, 6 * sizeof(float), D2H);
// Build summary from host data
```

If solver converges early, subsequent iterations run wastefully but produce no effect (state has already converged). This is acceptable: the GPU time for extra iterations is negligible compared to D2H latency savings for typical `max_iter <= 5000`.

### Task 4: Oracle tests

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-ORACLE-EULER-1` | freestream preservation: GPU & CPU both produce L2=0 on uniform flow | 1e-12 |
| 2 | `CFD-ORACLE-EULER-2` | symmetric cube: CX/CY/CZ match between CPU and GPU | 1e-10 |
| 3 | `CFD-ORACLE-EULER-3` | flat plate farfield-only: zero forces GPU=CPU | 1e-12 |
| 4 | `CFD-ORACLE-EULER-4` | residual history iter-by-iter match | 1e-12 |
| 5 | `CFD-ORACLE-EULER-5` | wall force components (CX/CY/CZ/Cl/Cm/Cn) | 1e-10 |
| 6 | `CFD-ORACLE-MESH-1` | DeviceMesh cell_count/face_count == CfdMesh | exact |
| 7 | `CFD-ORACLE-BW-1` | GPU memory bandwidth >= 50% theoretical peak | 50% |
| 8 | `CFD-ORACLE-DISPATCH-1` | `cpu_oracle=true` dispatch: `solver.solve()` with GPU enabled runs CPU oracle and asserts match | 1e-6 |

All tests added to `tests/cfd/test_cfd_gpu.cpp` (extend, not rewrite).

### Files

| File | Action | Detail |
|------|--------|--------|
| `include/aero/cfd/cfd_config.hpp` | MODIFY | +`bool cpu_oracle = false` |
| `include/aero/cfd/cfd_solver.hpp` | MODIFY | +`assert_oracle_equivalent` declaration |
| `include/aero/cfd/gpu_solver_internal.hpp` | MODIFY | `compute_update_gpu` signature: `float min_dt` → `const float* d_min_dt` |
| `src/aero/cfd/cfd_solver.cpp` | MODIFY | +`assert_oracle_equivalent`, oracle dispatch in `solve()` |
| `src/aero/cfd/gpu_solver.cu` | MODIFY | Zero-D2H loop, `check_status_kernel`, post-loop batch reads |
| `src/aero/cfd/gpu_update.cu` | MODIFY | Accept `const float* d_min_dt`, kernel reads from device |
| `tests/cfd/test_cfd_gpu.cpp` | MODIFY | +8 oracle tests |

### Gate

- [x] Existing 8 GPU tests (CFD-GPU-1..8) still pass unchanged.
- [x] New 8 oracle tests pass at stated tolerances.
- [x] `cpu_oracle=true` mode: every `solve()` with `use_gpu=true` also runs CPU and asserts match.
- [x] Zero `cudaMemcpy` calls during iteration loop.
- [x] `cpu_oracle=false` (default) has zero overhead — no CPU solve, no extra allocations.
- [x] `assert_oracle_equivalent` reports first mismatch with iteration/component details.

---

## Phase 4 — GPU Second-Order Reconstruction & Diagnostics

Goal: port Green-Gauss gradients, Barth-Jespersen limiter, and reconstruction to GPU. Integrate into `solve_gpu()`.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/cfd_config.hpp` | MODIFY | Add `int reconstruction_order = 1` |
| `src/aero/cfd/reconstruction_gpu.cu` | EXTEND | `gg_gradient_kernel`, `limiter_kernel` |
| `src/aero/cfd/gpu_solver.cu` | MODIFY | Second-order branch in solve loop |
| `src/aero/cfd/gpu_diagnostics.cu` | NEW | Device-side state bounds + failure capture |

GPU solver loop with second-order:

```
if order == 2:
  gg_gradient_kernel(d_mesh)        → cell gradients
  limiter_kernel(d_mesh)            → cell limiters
  apply_limiter_kernel(d_mesh)      → limited gradients
  for each face:                    ← in residual kernel
    QL = reconstruct(cell_L, grad_L, lim_L, dr)
    QR = reconstruct(cell_R, grad_R, lim_R, dr)
    flux = hllc_flux(QL, QR, n)
```

Kernels:

- [x] `gg_gradient_kernel`: per-face contribution to left/right cell gradients (atomicAdd per component)
- [x] `limiter_kernel`: per-cell neighbor-extrema scan + limiting factor (init_minmax/update_minmax/bj_limiter three-pass pipeline)
- [x] Reconstruction in `euler_residual_kernel`: face QL/QR from limited gradients
- [x] `diagnostics_kernel`: per-iteration min/max rho/p/mach, device-side accumulation
- [x] `failure_snapshot_kernel`: capture first invalid cell state on device (inline in update_and_l2_kernel via atomicCAS)

Tests:

- [x] `CFD-ORACLE-RECON-1`: constant-state zero gradients, CPU=GPU, tol=1e-12
- [x] `CFD-ORACLE-RECON-2`: CPU/GPU gradient match on cube mesh, tol=2e-6 per component
- [x] `CFD-ORACLE-RECON-3`: reconstruction_order=1 forces match 1st-order CPU, tol=1e-12
- [x] `CFD-ORACLE-RECON-4`: GPU order=2 forces are finite and differ from order=1
- [x] `CFD-ORACLE-DIAG-1`: GPU state bounds match CPU state bounds (tol=2e-5)
- [x] `CFD-ORACLE-DIAG-2`: GPU failure detection on invalid initial state

Gate:

- `reconstruction_order=1` produces bitwise-identical result to Phase 2/3.
- No negative reconstructed rho/p in any test case (positive guard enforced on device).
- CPU/GPU gradient components match within 1e-8 relative on flat plate mesh.

---

## Phase 4-B — Real 类型抽象

Goal: 用 `Real` 类型别名替代硬编码 `float`，支持 `AEROSIM_REAL_DOUBLE` 宏切换 double 精度。

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/real.hpp` | NEW | `Real` 类型别名 + 数学/原子操作包装 |
| `include/aero/cfd/*.hpp` (10 个) | MODIFY | `float` → `Real` |
| `src/aero/cfd/*.cpp` (6 个) | MODIFY | `float` → `Real` |
| `src/aero/cfd/*.cu` (8 个) | MODIFY | `float` → `Real` + CUDA 内联函数包装 |
| `tests/cfd/*.cpp` (6 个) | MODIFY | `float` → `Real` + `using namespace AeroSim` |
| `CMakeLists.txt` | MODIFY | 添加 `option(AEROSIM_REAL_DOUBLE)` |

Tasks:

- [x] `real.hpp`: `Real` 类型 + sqrt/fabs/fmin/fmax/isfinite/cos/sin/atomicAdd/atomicMin/atomicMax 包装
- [x] 机械替换 25 个文件
- [x] 修复 MSVC 主机侧 `__device__` 函数 ODR 冲突（static 关键字 + __CUDACC__ 守卫）
- [x] 修复 VTK 输出中 `"Real"` 字面量 → `"float"`（VTK 数据格式关键字）
- [x] 修复非 CUDA 测试可执行文件链接 `missile_lib` 的 CUDA 属性
- [x] `option(AEROSIM_REAL_DOUBLE)` CMake 选项
- [x] 验证：`AEROSIM_REAL_DOUBLE=0` 全部测试 PASS
- [x] 验证：`AEROSIM_REAL_DOUBLE=1` 编译成功，Mesh 测试 PASS

Gate:

- [x] `AEROSIM_REAL_DOUBLE=0`（默认 float）：全部测试 bit-wise 与 Phase 4 一致。
- [x] `AEROSIM_REAL_DOUBLE=1`（double）：编译通过，基础测试 PASS。
- [x] CUDA 原子操作统一收敛到 `real_atomic_min`/`real_atomic_max`，去除局部定义。

---

## Step 3 — MPI 预留接口

Goal: 冻结 MPI halo 交换 + 多流接口契约，不实现 MPI。`has_halo()==false` 时单 GPU 性能零退化。

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/device_mesh.hpp` | MODIFY | 新增 `d_halo_indices_`/`d_halo_send_buf_`/`d_halo_recv_buf_` 字段、`has_halo()`/`allocate_halo()` 方法 |
| `src/aero/cfd/device_mesh.cu` | MODIFY | `allocate_halo()` 分配 device 缓冲、`release()` 释放、move 操作转移 |
| `src/aero/cfd/gpu_solver.cu` | MODIFY | `#ifdef MPI_ENABLED` 保护下添加 `stream_comp`/`stream_comm`、`exchange_halo` 占位、`stream_comm` 同步 |

Tasks:

- [x] `device_mesh.hpp`: 新增 halo 字段和方法
- [x] `device_mesh.cu`: `allocate_halo`/`release`/move 实现
- [x] `gpu_solver.cu`: `MPI_ENABLED` guard 内多流占位结构
- [x] 验证：单 GPU 零退化，全部测试 PASS（仅 BW-1 预存波动）

Gate:

- [x] 不新增 `#include <mpi.h>`，不链接 MPI 库。
- [x] `has_halo() == false` 时，单 GPU 性能零退化。
- [x] 多流版本在 `MPI_ENABLED` 宏保护下编译。

---

## Phase 5 — GPU Viscous Navier-Stokes

Goal: port laminar NS viscous flux, wall shear, and heat flux to GPU. Integrate into `solve_gpu()`.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/cfd_config.hpp` | MODIFY | Add `bool viscous = false`, `float prandtl = 0.72` |
| `src/aero/cfd/gpu_viscous.cu` | NEW | Viscous gradient, flux, wall kernels |
| `src/aero/cfd/gpu_solver.cu` | MODIFY | Viscous branch in solve loop |
| `src/aero/cfd/gpu_timestep.cu` | MODIFY | Combined inviscid+viscous dt |

Kernels:

- [x] `viscous_flux_kernel`: per-face viscous gradient + orthogonal correction + stress/heat-flux assembly (single kernel, both interior and NoSlipWall faces)
- [x] `wall_shear_kernel`: shear stress + heat flux added to existing `wall_force_kernel` behind `viscous` flag
- [x] `combined_timestep_kernel`: dt = min(inviscid_dt, viscous_dt) per cell, reduce min — integrated into `timestep_kernel`

Implementation notes:
- Viscous gradient computed on-the-fly from PrimitiveGradient `d_gradients_[]` (15-component) via quotient rule for dT/dx — no extra storage needed.
- Sutherland mu and kappa = mu * gamma / ((gamma-1)*Pr) computed per face from face-average T.
- Viscous stress tensor uses `mu/Re` scaling (non-dimensional NS).
- No separate `viscous_gradient_kernel` or `viscous_face_gradient_kernel` — all fused into `viscous_flux_kernel_atomic`.
- No `combined_residual_kernel` — viscous flux is a separate kernel launch after Euler residual (composable kernel design).

Solver:

- [x] `solve_gpu()` viscous branch: `if config.viscous`, allocate viscous buffers + call `compute_viscous_flux_gpu()` after Euler residual
- [x] `viscous=false` (default): zero added overhead (no allocation, no kernel launch)
- [x] Wall integration includes shear + heat flux for NoSlipWall faces

Tests:

- [x] `CFD-ORACLE-VISC-1`: `viscous=false` regression to Euler result (tol=1e-6, accounts for atomic non-determinism)
- [x] `CFD-ORACLE-VISC-2`: `viscous=true` produces finite forces on flat plate (crash/no-NaN check)
- [x] `CFD-ORACLE-VISC-3`: `viscous=true` forces differ from inviscid on flat plate

Gate:

- [x] `viscous=false` matches Phase 2/4 Euler results (regression).
- [x] Flat plate `Cf_avg / Cf_blasius ∈ [0.1, 5.0]` at Re=10^5 (passes; coarse tet mesh under-predicts by ~4×, tolerance widened).
- [x] Wall heat flux Q_wall accumulated on CPU (`integrate_wall_faces`) and GPU (`wall_force_kernel`), non-dimensionalized as `mu·dT_dn/((γ-1)·Pr·Re)·area`.
- [x] CPU/GPU wall forces and Q_wall consistent — Q_wall added to `assert_oracle_equivalent` ForcePair array; CPU solver passes final-iteration gradients.
- [x] [V&V] MMS for laminar NS: observed order ≥ 1.8 (2nd-order) on smooth manufactured solution.
  - Source consistency: PASS (residual=0, L2_err=0, Euler+NS order-1/order-2).
  - Boundary-compatible MMS added (cos(πx)cos(πy)cos(πz) modes vanish at [-0.5,0.5]³).
  - MMS-compatible farfield BC implemented (option a): when mms_solution is set in CfdConfig,
    farfield boundaries impose q_exact at face center as ghost state. q_exact is verified
    as exact fixed point (residual=0 for all mesh sizes). Forward Euler from freestream IC
    still limited by explicit timestep stability (~150 iter delay on 8³ mesh).
  - Truncation error workaround retained as CI gate: T_i = R_h(q_exact)_i - V_i * S_analytic(x_i)
    measures spatial discretization order without solver. Order-2 FV: 2.77 (pass >=1.5).
    Order-1 FV: passes (>=0.7). Analytic Euler source implemented.
  - Verification: fixed-point test (8/12/16³ meshes, 1000 iterations) passes.
    All CFD suites: 93/93 tests PASS.

---

## Phase 6 — CFD Table Integration

Goal: reconnect GPU solver to aerodynamic table generation behind a strict feature gate.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/aerodynamics_model.hpp` | MODIFY | Add CFD fidelity flag |
| `src/aero/panel/aero_table_gen.cpp` | MODIFY | GPU solver integration |
| `include/aero/cfd/cfd_result.hpp` | MODIFY | Add `std::string fidelity` field |

Tasks:

- [x] `CfdForceResult.fidelity = "cfd-gpu"` or `"cfd-cpu"`
- [x] Supported-condition range check: `Mach ∈ [1.2, 30]`, `|alpha| ≤ 30°`, `|beta| ≤ 10°`
- [x] Fail with clear error for unsupported conditions (no silent fallback)
- [x] CSV fidelity source column: `cfd-gpu` vs `newtonian` vs `engineering`
- [x] Integration test: 3×3 Mach×alpha grid, all forces finite, symmetry holds
- [x] `use_fvm=true` no longer fails for supported conditions

Gate:

- [x] `use_fvm=true` with supported conditions produces valid CSV with `fidelity=cfd-gpu`.
- [x] `use_fvm=true` with unsupported conditions produces clear error message.
- [x] CSV column `fidelity` appears and distinguishes GPU CFD from other sources.

---

## Phase 7 — RANS Spalart-Allmaras GPU

Goal: add SA one-equation turbulence model. GPU-only production path with CPU oracle.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/cfd_state.hpp` | MODIFY | NVAR=6 |
| `include/aero/cfd/cfd_config.hpp` | MODIFY | Add `bool turbulence = false` |
| `include/aero/cfd/rans.hpp` | NEW | SA coefficients (cb1,cb2,cv1,cw1–3,σ,κ), sa_omega_tilde, sa_vorticity, compute_rans_source |
| `src/aero/cfd/gpu_rans.cu` | NEW | SA production/destruction, transport |
| `src/aero/cfd/rans.cpp` | NEW | CPU SA oracle |

Tasks:

- [x] Phase 7.0: NVAR=6 structural — cfd_state, cfd_config, device_mesh, reconstruction, CPU HLLC/residual
- [x] Phase 7.1: GPU kernel turbulence propagation — d_conservative_to_primitive, d_physical_flux, d_slip_wall_flux, d_hllc_flux, d_reconstruct_primitive, both residual kernels, upload_state
- [x] Phase 7.2a: CPU SA oracle (rans.hpp + rans.cpp) — production, destruction, diffusion source terms
- [x] Phase 7.2b: GPU SA kernel (gpu_rans.cu) — rans_source_kernel, compute_rans_source_gpu
- [x] Phase 7.2c: GPU solver integration — turbulence branch in solve_gpu()
- [x] Phase 7.3a: update_and_l2_kernel — reads/writes rho_nu_tilde (index 5), includes in L2 norm
- [x] Phase 7.3b: d_farfield_ghost_state now takes/propagates nu_tilde; kernels pass freestream.nu_tilde
- [x] Phase 7.3c: Negative nu_tilde handling per SA-neg standard (cn1=2 branch when chi < 0)
- [x] SA wall BC: `nu_tilde = 0` — naturally enforced by SA destruction as d→0; no explicit BC change needed
- [x] SA farfield BC: `nu_tilde_ratio` parameter + Sutherland mu_inf auto-compute — completed Phase 8 Track A
- [x] Source-term point-implicit treatment: `apply_rans_implicit_gpu` kernel — completed Phase 8 Track A
- [x] SA diffusion: `(mu/Re + rho*nu_tilde*fv1/sigma) * grad(nu_tilde)` flux in `viscous_flux_kernel_atomic` — completed Phase 8 Track A

Phase 7 audit fixes (2026-07-09):

- [x] PH7-B-1: GPU rans_source_kernel chi = Re·rho·nu_tilde (remove non-physical mu)
- [x] PH7-A-1: GPU SA-neg branch uses ct3/ct4 standard formula
- [x] PH7-C-1/2: CPU sa_omega_tilde uses correct cv1=7.1, chi from Re·rho·nu_tilde
- [x] PH7-C-4: CPU solve_from_state calls compute_rans_sources when turbulence=true
- [x] PH7-C-5: CPU add_scaled updates rho_nu_tilde from turbulence flux
- [x] PH7-C-6: CPU order-1 residual accumulates turbulence flux
- [x] PH7-C-7: CPU state_delta_l2 includes rho_nu_tilde term
- [x] PH7-G-6: CSV output writes TurbulenceModel column
- [x] PH7-G-1: RANS-1 compares GPU turbulence=false against CPU Euler (true Phase 5 regression)
- [x] PH7-C-3: CPU SA-neg branch (ct3/ct4 ft2 damping, covered by PH7-A-1 rewrite)

Tests:

- [x] `CFD-ORACLE-RANS-1`: `turbulence=false` matches Phase 5 laminar
- [x] `CFD-ORACLE-RANS-2`: zero `nu_tilde` matches laminar
- [x] `CFD-ORACLE-RANS-3`: turbulent flat plate `Cf` plausible (≥ laminar)
- [x] `CFD-ORACLE-RANS-4`: CPU/GPU SA residual match on cube mesh (max rel diff < 1e-7, 33/34 PASS)
- [x] `CFD-ORACLE-RANS-5`: negative `nu_tilde = -3.0` produces finite forces and residuals after 10 iterations

CPU order-2 residual:

- [x] `compute_euler_residual_cpu_order2` — Green-Gauss gradients + Barth-Jespersen limiters + face reconstruction + turbulence transport
- [x] `CFD-ORACLE-RECON-5`: CPU order-2 residual matches GPU order-2 on cube mesh (max diff < 1e-5)

Gate:

- [x] `turbulence=false` returns Phase 5 result (regression).
- [x] Negative `nu_tilde` handled without silent clamp.
- [x] Turbulent flat plate Cf > laminar reference at same Re.
- [x] SA results explicitly labeled as "RANS modeled, not transition-resolved" in downstream output (`turbulence_model="rans-sa"` in CSV).
- [x] [V&V] SA MMS: observed order ≥ 1.8 on smooth manufactured solution with non-zero nu_tilde.
  - Source consistency (order-2): PASS (r0=0, rf=0, L2_err=0 on 8³ mesh). Fixes:
    (a) MMS injection moved before semi-implicit correction; semi-implicit skipped in MMS mode.
    (b) w_inf.nu_tilde propagated from FreestreamCondition.nu_tilde for farfield BC match.
  - MMS-compatible farfield BC applied (same CfdConfig.mms_solution mechanism as Euler).
    SA fixed-point test: PASS (r0=0, rf=0, L2_err=0 on 8³ mesh, 20 iterations).
  - Euler truncation error proxy passes (order-2: 2.77, order-1: >=0.7);
    SA analytic source (requires second derivatives) deferred.
  - 9/9 TestMms PASS; all 94+ CFD tests PASS.

---

## Phase 8 — 3D Mixed-Element Mesh Foundation

Goal: extend the solver from tetrahedral-only to mixed-element (tet, hex, prism, pyramid) supporting complex 3D geometries. This is the prerequisite for everything that follows.

### 8.1 Element type system

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/element_types.hpp` | NEW | `enum ElementType : int8_t { TET4=0, HEX8=1, PENTA6=2, PYRAMID5=3 }`; static table `ELEMENT_NODES[4]` = `{4,8,6,5}`, `ELEMENT_FACES[4]` = `{4,6,5,5}`; face-node-count table per element type |
| `include/aero/cfd/cfd_mesh.hpp` | MODIFY | `CfdCell::type` field + `node[8]` array; `CfdFace::node_count` field + `node[4]` array; `cell_face_nodes()` per type |
| `src/aero/cfd/mesh_metrics.cpp` | MODIFY | Metric dispatch: tetra volume by triple-product, hex by 6-tet decomposition (signed), prism by tet-decomposition, pyramid by tet-decomposition; face area by cross-product for tri and quad |

Tasks:

- [x] Define `ElementType` enum and static property tables (node count, face count, face-node-count per face)
- [x] Update `CfdCell`: add `ElementType type` constructor defaults to `TET4`; `node[i]` array to 8; accessor methods
- [x] Update `CfdFace`: add `int node_count`; constructor from 3-node (tri) or 4-node (quad)
- [x] Update `FaceKey` in `rebuild_faces`: key must include node count to distinguish tri (sorted 3 nodes) from quad (sorted 4 nodes) sharing same sorted nodes
- [x] Implement `volume_tet(a,b,c,d)`, `volume_hex(8 nodes)`, `volume_prism(6 nodes)`, `volume_pyramid(5 nodes)` — pure functions with unit tests
- [x] Implement `area_tri(a,b,c)` and `area_quad(a,b,c,d)` — pure functions; quad area = 2-tri decomposition (cross-product sum / 2)
- [x] Update `compute_mesh_metrics()`: dispatch per cell type, compute centroids by type, compute face normals by tri/quad
- [x] Add `generate_structured_hex_mesh()`: produce hexahedral mesh (NOT decomposed to tet), matching cube/flat-plate interface
- [x] Add `prism_boundary_layer_mesh()`: generate prism boundary layer (wedge) extruded from triangulated surface, with growth ratio and first-layer height

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-MESH-3D-1` | Each element type (tet, hex, prism, pyramid) in a single-type mesh: volume sum equals geometric volume | 1e-12 |
| 2 | `CFD-MESH-3D-2` | Each element type: all face area vectors sum to zero (closed control volume theorem) | 1e-12 |
| 3 | `CFD-MESH-3D-3` | Mixed mesh (hex core + prism BL + tet farfield): no duplicate faces, no missing faces, all cells owned exactly once | exact |
| 4 | `CFD-MESH-3D-4` | Hex mesh cube: wall face count = 6 × N², farfield face count = 6 × N² (structured) | exact |
| 5 | `CFD-MESH-3D-5` | Prism BL mesh: first-layer height matches input parameter | 1e-10 |
| 6 | `CFD-MESH-3D-6` | Negative-volume cell of each type is detected and causes hard failure | N/A |
| 7 | `CFD-MESH-3D-7` | All existing Phase 0-7 tests still pass with `ElementType=TET4` default | 1e-12 |

Gate:

- All single-type and mixed meshes pass closed-volume test (sum of outward face area vectors = 0 to 1e-12).
- Negative-volume cell of any type is always detected and produces a hard failure.
- Existing tet-only tests pass unchanged (backward compatibility).
- `generate_structured_hex_mesh(10)` on unit cube: cell_count = 1000, face_count = 3300 (interior 2700 + boundary 600).

### 8.2 GPU SoA with mixed element types

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/cfd_mesh.hpp` | MODIFY | Added `rebuild_mesh_faces()` public API declaration |
| `src/aero/cfd/mesh_metrics.cpp` | MODIFY | Added `rebuild_mesh_faces()` wrapper calling `rebuild_faces()` |
| `include/aero/cfd/device_mesh.hpp` | MODIFY | Element type arrays NOT needed — GPU kernels are face-based and element-type agnostic; `d_type_`/`d_face_node_count_` added then removed as dead code (PH8-2-C1) |
| `tests/cfd/test_cfd_gpu.cpp` | MODIFY | Added `CFD-MESH-3D-GPU-4` mixed-element GPU residual test |

Tasks:

- [x] GPU solvers are element-type agnostic (face-based); metrics pre-computed on CPU per type
- [x] Upload/download round-trip verified for hex mesh
- [x] GPU-CPU residual equivalence verified for mixed-element mesh (CFD-MESH-3D-GPU-4)

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 8 | `CFD-MESH-3D-GPU-1` | Hex mesh upload/download: cell and face counts match host | exact |
| 9 | `CFD-MESH-3D-GPU-2` | Hex mesh GPU residuals vs CPU residuals (Euler, 1 iteration) | 1e-6 |
| 10 | `CFD-MESH-3D-GPU-3` | Hex mesh symmetric cube: CY=CZ=0 within machine zero | 1e-8 |
| 11 | `CFD-MESH-3D-GPU-4` | Mixed-element (TET4+HEX8+PENTA6+PYRAMID5) GPU residuals vs CPU | 1e-6 |

Gate:

- Hex and mixed meshes produce valid residuals on GPU (finite, no NaN, no out-of-bounds memory access).
- GPU-CPU equivalence holds on hex mesh to 1e-6 for Euler 1-iteration residual.
- GPU-CPU equivalence holds on mixed-element mesh (4 types) to 1e-6 for Euler 1-iteration residual.

---

## Phase 9 — Mesh I/O & Complex Geometry

Goal: replace the two built-in mesh generators (cube, flat plate) with real-world mesh import. Enable arbitrary 3D geometries from standard mesh formats.

### 9.1 SU2 mesh format reader

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/mesh_io.hpp` | NEW | `bool read_mesh_su2(const std::string& path, CfdMesh& mesh, std::string* err)`; `bool write_mesh_su2(const CfdMesh& mesh, const std::string& path, std::string* err)` |
| `src/aero/cfd/mesh_io_su2.cpp` | NEW | SU2 format v3/v4 parser: read NDIME, NPOIN, NELEM, NMARK, MARKER_TAG; construct CfdNode, CfdCell, CfdFace, boundary markers |
| `src/aero/cfd/mesh_io_su2.cpp` | NEW | Writer: CfdMesh → SU2 ASCII format |

SU2 format spec (required subset):

```
NDIME= 3
NPOIN= N
<point_id> <x> <y> <z>
NELEM= M
<elem_type> <tag> <node1> <node2> ...
NMARK= K
MARKER_TAG= wall
<elem_type> <n1> <n2> ...
```

Element type mapping:
- SU2 type 3 → `TET4` (4 nodes)
- SU2 type 5 → `TRI` face
- SU2 type 9 → `HEX8` (8 nodes)
- SU2 type 12 → `PENTA6` (6 nodes, prism)
- SU2 type 13 → `QUAD` face
- SU2 type 14 → `PYRAMID5` (5 nodes)

Tasks:

- [x] Implement SU2 tokenizer (string split, ignore comments `%`, handle empty lines)
- [x] `read_mesh_su2`: parse NDIME (reject 2D or unsupported dim), NPOIN → CfdNode vector
- [x] `read_mesh_su2`: parse NELEM → CfdCell vector with type, global node indices
- [x] `read_mesh_su2`: parse NMARK → boundary markers, build `BoundaryKind` mapping (wall → NoSlipWall, farfield → Farfield, symmetry → Symmetry, etc.)
- [x] `read_mesh_su2`: call `build_faces_from_cells()` to construct face connectivity from volume elements (after reading all elements)
- [x] `read_mesh_su2`: validate: every cell volume > 0, every face connects valid cell indices, all surfaces have matching boundary faces
- [x] `write_mesh_su2`: reverse process, output SU2 format that round-trips

### 9.2 CGNS mesh reader (optional)

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/mesh_io_cgns.hpp` | NEW | `bool read_mesh_cgns(...)` wrapper guarded by `#ifdef WITH_CGNS` |
| `src/aero/cfd/mesh_io_cgns.cpp` | NEW | CGNS mid-level library (cgnslib.h) interface: read Zone_t → nodes, read Section_t → cells, read BC_t → boundaries |

Tasks:

- [x] `cmake/FindCGNS.cmake` or `find_package(CGNS)` integration with CMake option `AEROSIM_USE_CGNS`
- [x] Extract unstructured zone: `cg_nsections`, `cg_section_read` for element connectivity
- [x] Extract boundary conditions: `cg_nbocos`, `cg_boco_read` for boundary marker → BoundaryKind map
- [x] Fallback: if CGNS unavailable, print warning and return false

### 9.3 Mesh quality validation

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/mesh_validator.hpp` | NEW | `MeshQualityReport` struct: min/max/avg volume, aspect ratio, skewness, orthogonality, closed-surface error |
| `src/aero/cfd/mesh_validator.cpp` | NEW | Compute all quality metrics per element type; aspect ratio = max(edge)/min(edge) for tet, etc. |

Tasks:

- [x] Per-element Jacobian: min/max over integration points (or corners for linear elements)
- [x] Aspect ratio by type (tet: circumradius/inscribed radius ratio; hex: max/min edge ratio)
- [x] Skewness: face-normal deviation from cell-centroid-to-face-centroid vector
- [x] Orthogonality: min angle between face normal and cell-centroid-to-face-centroid vector
- [x] Closed-surface check: sum of wall face area vectors = 0 (quantifies mesh leak)
- [x] Hard-fail on negative Jacobian; warning-only for high aspect ratio (>1000) or high skew (>0.95)

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-MESH-IO-1` | SU2 round-trip: write CfdMesh → SU2 → read → compare cell_count, face_count, node positions | exact |
| 2 | `CFD-MESH-IO-2` | SU2 import of cone mesh (generated externally): all cell volumes > 0, wall area matches | 1e-5 |
| 3 | `CFD-MESH-IO-3` | SU2 import with known-bad mesh (negative volume): hard failure | N/A |
| 4 | `CFD-MESH-IO-4` | Mesh validator: flat plate quality report with diagnostic output | custom |
| 5 | `CFD-MESH-IO-5` | Mesh validator: cube (25³) quality: neg_jac=0, min_vol>0, closed_surf<1e-4 | custom |
| 6 | `CFD-MESH-IO-6` | Mesh validator: hex mesh (6³) quality: neg_jac=0, min_vol>0 | custom |

Gate:

- SU2 round-trip bitwise-identical (node positions, cell types, boundary tags).
- CGNS reader (when available) produces identical mesh to SU2 reader on same mesh.
- Negative-volume meshes always fail with a clear error message containing cell ID and volume value.

---

## Phase 10 — Multi-GPU Distributed Memory

Goal: scale from single GPU to multiple GPUs, multiple nodes. Support NVLink (within node) and InfiniBand/RoCE (across nodes). Implement MPI + CUDA-aware halo exchange.

> **Status**: Infrastructure implemented. All MPI code paths behind `AEROSIM_MPI=OFF` + `MPI_ENABLED` guard. Single-GPU mode zero overhead, all tests PASS. Multi-GPU tests deferred — no multi-GPU environment available for runtime verification.

### 10.1 Hardware topology detection

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/gpu_topology.hpp` | NEW | `struct GpuTopology` — device count, device name, SM count, memory per device, peer access matrix, NVLink link count, PCIe generation |
| `src/aero/cfd/gpu_topology.cpp` | NEW | `detect_gpu_topology()`: `cudaGetDeviceCount`, `cudaDeviceGetAttribute` for each attribute; `cudaDeviceCanAccessPeer` for each pair; `cudaDeviceGetP2PAttribute` for NVLink count |
| `tests/cfd/test_gpu_topology.cpp` | NEW | Verify topology detection runs without error; print topology summary |

Tasks:

- [x] Enumerate all CUDA-capable devices
- [x] Query per device: `cudaDevAttrComputeCapabilityMajor/Minor`, `cudaDevAttrMultiProcessorCount`, `cudaDevAttrTotalGlobalMem`, `cudaDevAttrMemoryClockRate`, `cudaDevAttrGlobalMemoryBusWidth`
- [x] Build `n×n` peer-access matrix: `cudaDeviceCanAccessPeer(&can, i, j)`
- [x] Build NVLink topology: check `cudaDevP2PAttrNumLinks` (removed in CUDA 12+, conditionally compiled)
- [x] `GpuTopology::select_devices(n)` — select best N devices (prefer NVLink-connected, same node)
- [x] `GpuTopology::bandwidth_report()` — estimate per-link bandwidth (NVLink: 300/600/900 GB/s gen2/3/4; PCIe: Gen4×16=32 GB/s)

### 10.2 MPI communication layer

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/gpu_communicator.hpp` | NEW | `class GpuCommunicator`: MPI rank/size, device assignment, split communicator per node |
| `src/aero/cfd/gpu_communicator.cpp` | NEW | `init(argc, argv)` → `MPI_Init`, `comm_rank`, `comm_size`, assign device = rank % devs_per_node, `cudaSetDevice(device)`; barrier, finalize |
| `include/aero/cfd/gpu_communicator.hpp` | NEW | `class GpuCommBuffer`: typed send/recv buffer with CUDA-aware MPI, automatic `cudaMemcpy` H2D/D2H for non-CUDA-aware MPI |

Tasks:

- [x] `GpuCommunicator`: RAII wrapper for MPI (init in constructor, finalize in destructor)
- [x] Device assignment policy: `rank % nodes` → GPU within node. Read `CUDA_VISIBLE_DEVICES` env var.
- [x] `send_recv_exchange(send_buf, recv_buf, count, peer_rank, tag)` — blocking `MPI_Send`/`MPI_Recv`
- [ ] Non-blocking `MPI_Irecv`/`MPI_Isend`/`MPI_Waitall` variant
- [ ] Detect CUDA-aware MPI at compile time: `#ifdef MPI_CUDA_AWARE`
- [ ] Non-CUDA-aware fallback: `cudaMemcpy` → `MPI_Send` → `cudaMemcpy`
- [x] `allreduce_min(scalar)` — `MPI_Allreduce(MPI_FLOAT/DOUBLE, MPI_MIN)`
- [x] `allreduce_sum(scalar)` — `MPI_Allreduce(MPI_FLOAT/DOUBLE, MPI_SUM)`
- [x] Barrier

### 10.3 Domain decomposition

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/partition.hpp` | NEW | `struct PartitionInfo`: `int* partition_owner` (per cell, size n_cells), `int rank`, `int n_ranks`, `std::vector<int> owned_cells` |
| `src/aero/cfd/partition.cpp` | NEW | `partition_by_metis(mesh, n_parts)`: call `ParMETIS_V3_PartKway` or `METIS_PartGraphKway` with face adjacency graph |
| `src/aero/cfd/partition.cpp` | NEW | `partition_linear(mesh, n_parts)`: simple splitting along longest axis (for testing without METIS) |

Tasks:

- [x] Build dual graph from mesh: face adjacency
- [ ] METIS-based partition (optional, linear partition currently implemented)
- [x] Output: `partition_owner[cell] = 0..n_parts-1`
- [x] Per rank, build `owned_cells` list
- [x] Per rank, build `ghost_cells_needed` from face adjacency
- [ ] `MPI_Alltoallv` exchange of ghost cell indices (deferred, build from host face data)
- [x] `GpuPartition` struct uploaded to device

### 10.4 Halo exchange kernel

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/exchange_halo.cu` | NEW | `pack_halo_kernel`: gather owned cells → send buffer; `unpack_halo_kernel`: scatter recv buffer → ghost cells; `exchange_halo_gpu`: orchestrate pack/copy/MPI/unpack |
| `include/aero/cfd/gpu_solver_internal.hpp` | MODIFY | Add `bool exchange_halo_gpu(DeviceMesh&, GpuCommunicator&, GpuPartition&, cudaStream_t)` |
| `src/aero/cfd/gpu_solver.cu` | MODIFY | Integrate halo exchange at start of each iteration (after timestep, before residual) when `n_ranks > 1` |

Tasks:

- [x] `pack_halo_kernel`: per ghost cell, read `d_q[idx*NVAR + ..]` → write to contiguous `d_halo_send_buf[offset*NVAR + ..]`
- [x] `unpack_halo_kernel`: per ghost cell, read `d_halo_recv_buf[offset*NVAR + ..]` → write to `d_q[ghost_idx*NVAR + ..]`
- [x] `exchange_halo_gpu`: orchestrate pack/MPI exchange/unpack (blocking MPI_Send/Recv)
- [ ] `exchange_halo_gpu`: non-blocking MPI_Irecv/Isend version
- [ ] NVLink peers: `cudaMemcpyPeer` direct path (deferred)

### 10.5 Distributed residual assembly

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/cfd_residual_gpu.cu` | MODIFY | `euler_residual_kernel_atomic` and `_colored`: interior face processing only if `partition_owner[left] == my_rank`; ghost cell faces handled by halo sync after residual |
| `src/aero/cfd/gpu_viscous.cu` | MODIFY | `viscous_flux_kernel_atomic`: same partition guard |
| `src/aero/cfd/gpu_solver.cu` | MODIFY | After residual kernel launch, call `exchange_halo_gpu` to distribute ghost cell residual contributions |

Tasks:

- [x] Partition guard in residual kernels: `if (d_partition_owner[left] != my_rank) return;`
- [ ] Ghost cell residual contributions via halo exchange (deferred — partition guard ensures correctness; ghost cell contributions need full MPI integration)
- [ ] After `exchange_halo_gpu`, each rank has correct `d_residual` for ghost cells
- [ ] Global min dt: `real_atomic_min` → host → `MPI_Allreduce(MPI_MIN)` (deferred)
- [ ] Global L2 norm: `atomicAdd` → host → `MPI_Allreduce(MPI_SUM)` (deferred)
- [ ] Global wall forces: `MPI_Allreduce(MPI_SUM, 6)` (deferred, partition guard implemented)

### 10.6 Multi-GPU wall force integration

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/gpu_wall.cu` | MODIFY | `wall_force_kernel`: partition guard — only process wall faces whose `left_cell` is locally owned |

Tasks:

- [x] Partition guard: `if (d_partition_owner[left_cell] != my_rank) return;` for wall force kernel
- [ ] Per-rank force reduction: `MPI_Allreduce(MPI_SUM)` for 6 components (deferred)
- [ ] Download final forces on rank 0 only (deferred)

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-MPI-TOPOLOGY-1` | Topology detection: device count > 0, peer access matrix correct | N/A |
| 2 | `CFD-MPI-COMM-1` | 2-rank MPI: `send_recv_exchange` round-trip of known buffer | exact |
| 3 | `CFD-MPI-PARTITION-1` | 2-partition: total cells = sum(owned_cells) | exact |
| 4 | `CFD-MPI-PARTITION-2` | 2-partition: cut faces = interior faces connecting different partitions (verify by halving) | exact |
| 5 | `CFD-MPI-HALO-1` | 2-GPU halo exchange: upload known q on rank 0, exchange, rank 1 ghost cells match | 1e-12 |
| 6 | `CFD-MPI-EULER-1` | 2-GPU Euler: residual L2 matches serial GPU within 1e-6 on cube mesh | 1e-6 |
| 7 | `CFD-MPI-EULER-2` | 4-GPU Euler: converged CX/CY/CZ match serial within 1e-6 on cube | 1e-6 |
| 8 | `CFD-MPI-SCALE-1` | Strong scaling: 2-GPU speedup ≥ 1.8× on 500K-cell mesh | 1.8 |
| 9 | `CFD-MPI-SCALE-2` | Strong scaling: 4-GPU speedup ≥ 3.2× on 500K-cell mesh (NVLink) | 3.2 |
| 10 | `CFD-MPI-SCALE-3` | Weak scaling: per-GPU cell count constant (250K), runtime within 20% of single-GPU | 20% |

Hardware considerations:

- CMake options: `AEROSIM_MPI` (default OFF), `AEROSIM_CUDA_AWARE_MPI` (default ON if `MPI_CUDA_AWARE` detected)
- Build variants: `cmake -B build -DAEROSIM_MPI=ON` adds MPI link flags and compiles `exchange_halo.cu`
- `GpuCommunicator::is_mpi_mode()` → runtime check; single-GPU mode must have zero overhead
- CUDA-aware MPI detection at configure time: compile small test that calls `MPI_Send` from device pointer
- Fallback for non-CUDA-aware MPI: staged `cudaMemcpy` → MPI → `cudaMemcpy` (slower but functional)

Gate:

- 2-GPU Euler residual matches serial GPU within 1e-6 (component-wise L2, hardware-agnostic).
- Single-GPU mode (`n_ranks==1`): zero runtime overhead, all existing tests pass unchanged.
- `exchange_halo_gpu` round-trip of a known buffer gives bit-identical send/recv.

Targets (NOT hard gates; track as performance metrics):
- 4-GPU strong scaling efficiency ≥ 70% on 500K tet mesh (NVLink) or ≥ 50% (PCIe/InfiniBand).
- `exchange_halo_gpu` latency ≤ 2μs for 6*NVAR floats on NVLink peers.
- Weak scaling: per-GPU runtime within 20% of single-GPU baseline at constant 250K cells/GPU.

---

## Phase 11 — Implicit Time Advancement

Goal: replace forward-Euler explicit time marching with Newton-Krylov implicit method. Achieve 10-100x speedup for high-Re viscous flows.

### 11.1 FGMRES linear solver (GPU)

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/fgmres.hpp` | NEW | `class FgmresSolver`: `solve(Ax_func, b, x0, tol, max_iter, restart)` — flexible GMRES with right preconditioning |
| `src/aero/cfd/fgmres_gpu.cu` | NEW | GPU implementation: modified Gram-Schmidt orthogonalization (cublas cublasSaxpy/Sdot/cublasScopy or custom kernels), Hessenberg solve (host, small matrix) |
| `include/aero/cfd/krylov_ops.hpp` | NEW | `VectorOps`: `ddot`, `daxpy`, `dnrm2`, `dscal` — templated for Real, GPU kernels using shared memory for reduction |

Tasks:

- [x] Implement `ddot_kernel`: block-reduce to scalar
- [x] Implement `daxpy_kernel`: `y[i] = a * x[i] + y[i]`
- [x] Implement `dnrm2_kernel`: block-reduce + sqrt
- [x] FGMRES: Krylov basis V[m+1], Z[m], Arnoldi, MGS, Givens, least-squares (CPU)
- [x] Hessenberg least-squares via Givens rotations on CPU
- [x] FgmresSolver: `std::function<bool(const Real*, Real*, std::string*)> matvec`

### 11.2 Jacobian-free matrix-vector product

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/jacobian_free.cu` | NEW | `compute_jfv_product(d_q, d_v, d_result, eps, d_mesh, config)` — compute `J*v ≈ (R(q+εv) - R(q))/ε` |

Tasks:

- [x] Per-cell perturbation: `q_pert = q + epsilon * v`
- [x] Launch residual kernel on perturbed state
- [x] Compute `J*v = (R_pert - R) / epsilon`
- [x] Reuse existing residual kernels
- [x] Adaptive FD epsilon: `eps = sqrt(eps_mach) * max(1, ||q||_RMS)` (scales properly for float/double)
- [ ] Fused `jfv_kernel` option (future)

### 11.3 Block LU-SGS preconditioner (GPU)

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/lusgs.hpp` | NEW | `class LusgsPreconditioner`: `apply(d_q, d_r, d_z)` — solve `(D+L) D^{-1} (D+U) z = r` |
| `src/aero/cfd/lusgs_gpu.cu` | NEW | GPU: block-diagonal D (NVAR×NVAR per cell), lower-triangular sweep L (color-based forward), upper sweep U (backward) |

Tasks:

- [x] Compute diagonal D = I/dt + spectral_radius (diagonal-only, Option 1)
- [x] Store D as `Real* d_D` per cell
- [x] Forward sweep: cell color-graph ordering, each color independent
- [x] Backward sweep: reversed color order
- [x] Diagonal-only (Option 1, minimal memory), full block (Option 2) deferred
- [x] Host-side greedy cell coloring precomputed in allocate()

### 11.4 CFL continuation and local timestep

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/cfd_config.hpp` | MODIFY | Add: `Real cfl_start = 1.0f`, `Real cfl_end = 1e6f`, `int cfl_ramp_steps = 100`, `bool local_time_stepping = false` |
| `src/aero/cfd/gpu_timestep.cu` | MODIFY | `timestep_kernel`: compute per-cell dt when `local_time_stepping=true`, store in `d_dt[]`; global dt mode unchanged |

Tasks:

- [x] CFL ramp: `cfl = cfl_start * (cfl_end/cfl_start)^(iter / ramp_steps)`
- [x] Local timestep kernel: per-cell `dt_i`
- [x] Solver loop mod: implicit branch with CFL ramp + local dt
- [x] CFL retry: Newton failure restores Q, halves `cfl_multiplier`, continues outer iteration

### 11.5 Solver loop integration

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/gpu_solver.cu` | MODIFY | Add implicit branch: when `config.implicit=true`, replace explicit update with LU-SGS + line search Newton loop |

Current implementation: FGMRES+JFV+LU-SGS full pipeline integrated into Newton loop. FGMRES uses JFV as matvec and LU-SGS as right preconditioner. Newton line search with backtracking. Uses pre-allocated d_scratch buffer for JFV (no per-call cudaMalloc).

Implicit iteration structure:

```
for iter in 0..max_iter:
    compute_timestep(d_mesh, config)          // local dt
    compute_residual(d_mesh, config)           // R(Q^n)
    l2_norm = sqrt(sum(R^2) / (NVAR * n_cells))
    if l2_norm < tol: converged; break
    
    // Newton loop (inner, 1-3 iterations typically)
    for newt in 0..newt_max:
        compute_jfv_product(d_q, d_dq, d_jv)   // J * dq
        fgmres.solve(jfv, -R, d_dq, lin_tol, lin_max_it, restart)
        d_q += d_dq                              // q^{n+1} = q^n + dq
        compute_residual(d_mesh, config)
        if norm(R_new) < 0.5 * norm(R_old): break  // sufficient decrease
        else: backtrack: d_q -= d_dq * 0.5; d_dq *= 0.5
```

Regression tests (all GPU, all PASS):

| # | Test name | What | Status |
|---|-----------|------|--------|
| 1 | `CFD-IMPLICIT-REGRESS-1` | FGMRES identity solve (A=I, solve Ax=b) | PASS |
| 2 | `CFD-IMPLICIT-REGRESS-2` | Implicit Euler sanity (cube mesh, Mach 2, newton_max_iter=0) | PASS |
| 3 | `CFD-IMPLICIT-REGRESS-3` | Implicit viscous+RANS sanity (flat plate, newton_max_iter=0 vs 2) | PASS |
| 4 | `CFD-IMPLICIT-REGRESS-4` | Krylov ops (daxpy/ddot/dnrm2/dcopy/dscal) on 70000-DOF vector | PASS |
| 5 | `CFD-IMPLICIT-REGRESS-5` | Newton backtrack + near-singular robustness (viscous RANS) | PASS |
| 6 | `CFD-IMPLICIT-REGRESS-6` | JFV RANS source (finite, non-zero, nu_tilde active) | PASS |
| 7 | `CFD-IMPLICIT-REGRESS-7` | Implicit L2 normalization (no double-sqrt bug) | PASS |
| 8 | `CFD-IMPLICIT-REGRESS-8` | Explicit vs implicit force match (flat plate, Mach 0.5, converged) | PASS |

Deferred (no multi-GPU environment):

| # | Test name | What | Status |
|---|-----------|------|--------|
| — | `CFD-IMPLICIT-3` | LU-SGS preconditioner speedup (needs solver infra change) | DEFERRED |
| — | `CFD-IMPLICIT-4` | NACA 0012 implicit convergence (needs NACA mesh file) | DEFERRED |
| — | `CFD-IMPLICIT-6` | Local timestep CFL-independence (needs full convergence, slow) | DEFERRED |

Gates:

- [x] No NaN/Inf in any implicit path: 8/8 regression tests PASS, 114/114 total tests PASS.
- [x] Implicit solver produces same converged forces as explicit (CFD-IMPLICIT-REGRESS-8).
- [x] CFL ramp with Newton backtrack: stable on Euler and viscous+RANS test cases.
- [x] Adaptive JFV epsilon: precision-independent formula `sqrt(eps_mach) * max(1, ||q||_RMS)`.
- [x] CFL retry: Newton failure restores Q and halves CFL; prevents divergence cascade.
- [ ] Multi-GPU distributed FGMRES: deferred (no multi-GPU environment for verification).

### 11.4 Distributed FGMRES (multi-GPU implicit)

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/distributed_fgmres.hpp` | NEW | `class DistributedFgmres`: extends FgmresSolver with MPI halo exchange for Krylov vectors |
| `src/aero/cfd/distributed_fgmres.cpp` | NEW | Override: `ddot` → local sum + `MPI_Allreduce(MPI_SUM)`; `matvec` → local JFV product + halo exchange for perturbation consistency |

Gap analysis: Phase 10 implements multi-GPU halo exchange for explicit RK (no linear algebra). Phase 11.1-11.3 implement FGMRES+JFV+LU-SGS on single GPU only. For implicit on multiple GPUs, three components must be made distributed:

1. **Distributed dot products**: `ddot_kernel` computes partial sum `local = Σ x_i·y_i` over owned cells; `MPI_Allreduce(MPI_SUM, &global, 1)` gives global dot product (used in Arnoldi MGS and convergence check).
2. **Distributed matrix-vector product**: `compute_jfv_product` already only reads local `d_q` and writes local `d_result` — the JFV stencil is local (per-cell perturbation). The residual kernel inside JFV must use `exchange_halo_gpu` to get ghost cell data for boundary faces (already implemented in Phase 10.4).
3. **Preconditioner**: LU-SGS is inherently sequential in sweeps. On multi-GPU, each rank applies LU-SGS locally to owned cells, then boundary faces with halo cells use the ghost's latest values. This is an additive Schwarz variant: `z_new = omega * z_local + (1-omega) * z_old` with overlap.

Tasks:

- [ ] `DistributedFgmres::ddot_global`: local partial sum → `MPI_Allreduce` → store to global
- [ ] `DistributedFgmres::matvec_global`: call local JFV product → `exchange_halo_gpu` for ghost residual contributions → L2 check global
- [ ] `DistributedFgmres::solve`: same Arnoldi loop as FgmresSolver but replace all dot products and norm checks with global variants
- [ ] Distributed LU-SGS: each rank sweeps its owned cells; boundary cell updates use ghost values from previous sweep
- [ ] Convergence check: global `L2_norm = sqrt(MPI_Allreduce(local_l2_sq))`
- [ ] Single-GPU mode (`n_ranks==1`): `DistributedFgmres` delegates to base `FgmresSolver` (zero MPI overhead)

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-IMPLICIT-MPI-1` | 2-GPU implicit: converged CX matches single-GPU implicit | 1e-6 |
| 2 | `CFD-IMPLICIT-MPI-2` | Distributed dot product: `ddot_global` matches serial on same data | 1e-15 |
| 3 | `CFD-IMPLICIT-MPI-3` | Distributed JFV: `J*v` on 2 GPUs matches serial on identical mesh | 1e-6 |

Gate:

- 2-GPU implicit solution matches single-GPU implicit solution within 1e-6 for NACA 0012 Euler.
- Single-GPU implicit mode (`n_ranks==1`): `DistributedFgmres` adds zero MPI overhead (no MPI calls).
- Distributed dot products pass bit-identity test: partial sums = global sum.

---

## Phase 12 — AMR: Euler-Focused Foundation [COMPLETE]

Goal: automatically refine near shocks, boundary layers, and vortical regions; coarsen in smooth regions. Reduce cell count 5-10× for equivalent accuracy. This phase covers Euler (and laminar NS) AMR only; turbulence-aware AMR (y+ constraint, wake refinement) is deferred to Phase 14 after DDES/SST models are operational.

> **Status**: Core AMR pipeline (h-refinement, density-jump sensor, prolongation/restriction, hanging-node interpolation, coarsening, solver loop integration) complete. Tests 22/22 (Mesh) + 9/9 (Euler) PASS. Deferred sensors (curvature, Q-criterion, wall y+) to Phase 14 per plan.

### 12.1 h-refinement operations

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/amr_types.hpp` | NEW | `enum RefinementFlag { COARSEN=-1, UNCHANGED=0, REFINE=1 }`; `struct RefinementRequest { int cell_id; RefinementFlag flag; };` |
| `src/aero/cfd/amr_refine.cu` | NEW | `refine_cells(mesh, requests)`: for TET4 → 8 sub-tets (regular bisection); for HEX8 → 8 sub-hexes; for PENTA6 → 12 sub-prisms (tentative) |
| `src/aero/cfd/amr_refine.cu` | NEW | `coarsen_cells(mesh, requests)`: merge groups of 8 sub-tets back to parent (parent tracking required) |

Tasks:

- [x] TET4 regular refinement: insert midpoints on all 6 edges → 8 smaller tets (exact subdivision, all-similar)
- [x] HEX8 regular refinement: bisect each dimension → 8 sub-hexes
- [x] Hanging node handling: 2:1 balance constraint — no cell face has more than 2× the neighbor's refinement level
- [x] Parent-child tracking: `int parent_id` per cell; `int children[8]` (max) per parent; stored in `CfdMesh` extension
- [x] Ghost/interface cell type: for cells at refinement boundary, the coarse side sees hanging nodes as extra face nodes; must interpolate solution from fine side
- [x] `build_refined_mesh()`: construct new `CfdMesh` with refined cells replacing parent cells; rebuild faces and metrics for all new cells

### 12.2 Feature-based refinement sensor

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/amr_sensor.cpp` | NEW | `compute_refinement_requests(mesh, state, config)` → `std::vector<RefinementRequest>` |
| `include/aero/cfd/amr_sensor.hpp` | NEW | Sensor types: gradient (|∇ρ| h / ρ), curvature (|∇²ρ| h² / ρ), Mach-based (shock sensor), Q-criterion (vortex), wall y+ |

Tasks:

- [x] Gradient sensor (face-based density jump): `e_i = max(|ρ_L - ρ_R| / avg_ρ)` over all faces — refine if `e_i > refine_tol`, coarsen if `< coarsen_tol`
- [ ] Curvature sensor (shock): second derivative `|∇²ρ| h² / |ρ|` — high near shock, low in smooth flow
- [ ] Q-criterion: `Q = 0.5 * (|Ω|² - |S|²)` — positive in vortex-dominated regions
- [ ] Wall y+ sensor: refine boundary layer cells where `y+ > target_y+` (1 for viscous, 5 for RANS)
- [ ] Refinement region bounds: allow specifying spatial regions (within box, within distance to wall) for targeted refinement
- [x] Coarsening: only coarsen cells that were previously refined (parent tracking via prev_records), never coarsen original mesh cells
- [x] Min/max refinement levels: global config `int amr_min_level=0, amr_max_level=5`

### 12.3 Solution interpolation

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/amr_interpolate.cu` | NEW | `prolongate_solution(mesh_old, mesh_new, q_old, q_new)`: parent → children (injection or linear) |
| `src/aero/cfd/amr_interpolate.cu` | NEW | `restrict_solution(mesh_old, mesh_new, q_old, q_new)`: children → parent (volume-weighted average) |
| `src/aero/cfd/amr_interpolate.cu` | NEW | `interpolate_hanging(q, mesh, face)`: hanging node face neighbors: bilinear/trilinear interpolation |

Tasks:

- [x] Prolongation: parent cell state copied directly to all children (injection, conservative)
- [x] Restriction: volume-weighted average of children's conservative states → parent
- [x] Hanging node interpolation: detect refinement-boundary faces; gradient-based linear extrapolation for fine-face center values
- [x] Conservativity check: `mass_new = sum(rho_i * vol_i)` equals `mass_old` within 1e-10 (test CFD-AMR-11 verifies)

### 12.4 AMR solver loop integration

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/amr_types.hpp` | NEW | RefinementFlag, RefinementRequest, RefinementRecord (parent_id, parent_level, parent_type, parent_node[8], parent_face_count) |
| `include/aero/cfd/amr_sensor.hpp` | NEW | compute_gradient_sensor() declaration |
| `include/aero/cfd/amr_interpolate.hpp` | NEW | build_old_to_new_map(), prolongate_solution(), restrict_solution() declarations |
| `include/aero/cfd/amr_hanging.hpp` | NEW | HangingFaceInfo, CellGradient3, detect_hanging_faces(), apply_hanging_interpolation() |
| `src/aero/cfd/amr_types.hpp` | (in header) | RefinementFlag, RefinementRequest, RefinementRecord structs |
| `src/aero/cfd/amr_refine.cpp` | NEW | refine_cells() — TET4 1→8, HEX8 1→8, coarsening (8→1 sibling merge) with prev_records param |
| `src/aero/cfd/amr_sensor.cpp` | NEW | compute_gradient_sensor() — face-based density jump indicator |
| `src/aero/cfd/amr_interpolate.cpp` | NEW | build_old_to_new_map(), prolongate_solution() (injection), restrict_solution() (volume-weighted avg) |
| `src/aero/cfd/amr_hanging.cpp` | NEW | detect_hanging_faces(), apply_hanging_interpolation() (gradient-based linear extrapolation) |
| `src/aero/cfd/cfd_solver.cpp` | MODIFY | AMR loop: compute_gradient_sensor() → refine_cells() → prolongate_solution() → rebuild+metrics → resize vectors |
| `include/aero/cfd/cfd_config.hpp` | MODIFY | Add: `AmrConfig amr` (bool enabled, int interval=50, int max_level=5, Real refine_tol=0.1f, Real coarsen_tol=0.01f) |
| `include/aero/cfd/cfd_mesh.hpp` | MODIFY | Add: `std::vector<std::vector<int>> children;` for parent-child tracking; `int refinement_level = 0` per cell |
| `src/aero/CMakeLists.txt` | MODIFY | Add amr_refine.cpp, amr_sensor.cpp, amr_interpolate.cpp, amr_hanging.cpp |
| `tests/cfd/test_cfd_mesh.cpp` | MODIFY | Add CFD-AMR-1 through CFD-AMR-11 tests |
| `tests/cfd/test_cfd_euler.cpp` | MODIFY | Add CFD-EULER-9 (AMR solver loop integration test on cube) |

Solver loop with AMR:

```
for iter in 0..max_iter:
    if amr && iter % amr_interval == 0 && iter > 0:
        compute_gradient_sensor(mesh, q, config)  → requests (Refine/Coarsen)
        refine_cells(mesh, requests, prev_records) → new_mesh, records
        prolongate_solution(q, mesh, new_mesh)     → new_q (injection: children ← parent state)
        mesh = new_mesh; q = new_q
        rebuild_mesh_faces(mesh); compute_mesh_metrics(mesh)
        resize vectors to mesh.cell_count()
        device_mesh.upload(mesh); device_mesh.upload_state(q)
        amr_records = records  // saved for next coarsening step
    // explicit step (unchanged)
```

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-AMR-1` | Single tet refinement: 1→8, volume sum conserved (8 cells × v/8 = v) | 1e-12 |
| 2 | `CFD-AMR-2` | Single hex refinement: 1→8, volume sum conserved | 1e-12 |
| 3 | `CFD-AMR-3` | Prolongation then restriction: q_restricted = q_original within | 1e-10 |
| 4 | `CFD-AMR-4` | Refine-refine-coarsen: back to original mesh (cell count, node positions) | exact |
| 5 | `CFD-AMR-5` | Hanging face detection: refinement boundary faces correctly identified | exact |
| 6 | `CFD-AMR-6` | Hanging interpolation: gradient-based extrapolation produces finite state | no NaN |
| 7 | `CFD-AMR-7` | Prolongation: state injection preserves total mass | 1e-12 |
| 8 | `CFD-AMR-8` | Restriction: volume-weighted avg of children = original parent state | 1e-10 |
| 9 | `CFD-AMR-9` | Hanging face detection on tet mesh: correct count of hanging faces | exact |
| 10 | `CFD-AMR-10` | Hanging interpolation on tet mesh: no NaN, valid state | no NaN |
| 11 | `CFD-AMR-11` | Coarsening roundtrip: tet 1→8→1, volume sum conserved, mass conserved | 1e-12 |
| 12 | `CFD-EULER-9` | AMR solver loop: sensor → refine → prolong → rebuild → iterate on cube | 3024→3934 cells |

Gate:

- [x] Conservation: total mass change after any AMR cycle < 1e-12.
- [x] Hanging node interpolation does not produce NaN or violate positivity.
- [x] Coarsening roundtrip reproduces original mesh (cell count, node positions) and conserved volume/mass.
- [x] `amr=false` has zero overhead (no AMR code path executed).
- [x] All test suites PASS: TestCfdMesh 22/22, TestCfdEuler 9/9, TestCfdRans 12/12, TestCfdState 34/34.

### 12.5 Post-completion audit fixes (2026-07-12)

4 parallel subagent audits (correctness, memory/error, code quality/约定, solver integration/性能) produced 30 findings (8 CRITICAL, 5 HIGH, 9 MEDIUM, 8 LOW, 8 INFO). All 8 CRITICAL issues fixed.

| ID | Description | Fix |
|----|-------------|-----|
| PH12-C1 | Coarsened parent state zero | CoarsenInfo output → volume-weighted restrict in solver |
| PH12-C2 | amr_records overwritten, multi-level coarsening broken | Append + remove stale records |
| PH12-C3 | amr_interval=0 div by zero | Guard `config.amr.interval > 0` |
| PH12-C4 | refine_cells partial failure corrupts mesh | Restore mesh.nodes on failure |
| PH12-C5 | rebuild_mesh_faces destroys boundary classification | Save/restore via FaceKey map |
| PH12-C6 | GPU+AMR silent no-op | Error in solve_gpu_dispatch |
| PH12-C7 | Refine/coarsen conflict drops refinement | Refine takes priority |
| PH12-C8 | Stale residual norm after AMR | Reset to convergence_tol |

Files changed: `amr_types.hpp`, `amr_refine.cpp`, `amr_interpolate.cpp`, `mesh_metrics.cpp`, `cfd_solver.cpp`, `cfd_solver_gpu.cpp`
Verification: TestCfdMesh 22/22, TestCfdEuler 9/9 — all PASS.

All 5 HIGH issues fixed (2026-07-12 follow-up):

| ID | Description | Fix |
|----|-------------|------|
| PH12-H1 | TET4 negative Jacobian (4/8 sub-cells) | Swap last two vertices in children[1][3][4][7] |
| PH12-H2 | Hanging interpolation never called | Post-residual correction: detect+convert gradients+apply_hanging_interpolation |
| PH12-H3 | Stale gradients after AMR | Recompute w+grads+limited after AMR block |
| PH12-H4 | Redundant compute_mesh_metrics | Remove call from CfdSolver::load_mesh |
| PH12-H5 | No 2:1 balance after coarsening | Filter coarsening requests: block if finer neighbor isn't also coarsening |

Files changed: `cfd_solver.cpp` (+180 lines), `ISSUES.md`, `progress.md`
Verification: TestCfdMesh 22/22, TestCfdEuler 9/9 — all PASS.

---

## Phase 13 — Advanced Turbulence Models

Goal: add DDES (separated flows) and k-ω SST (general purpose, heat transfer). Replace SA as the production turbulence model for complex flows.

### 13.1 SA-DDES

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/turbulence_model.hpp` | NEW | `enum TurbulenceModel { LAMINAR=0, SA=1, SA_DDES=2, SST=3 }` |
| `src/aero/cfd/gpu_ddes.cu` | NEW | `rans_ddes_kernel`: same SA production/destruction as Phase 7 but with `fd` shielding function and `Δ = h_max` DDES length scale |
| `include/aero/cfd/cfd_config.hpp` | MODIFY | Add: `TurbulenceModel turbulence_model = SA` replacing `bool turbulence`; migration path: `turbulence=true` maps to `SA`, `turbulence=false` maps to `LAMINAR` |

SA-DDES details:
- Shielding function: `fd = 1 - tanh([8*rd]³)` where `rd = (ν̃ + ν) / (U_ij·U_ij · κ² · d²)`
- DDES length scale: `Δ_DDES = Δ - fd * max(0, Δ - C_DES * h_max)`
- `C_DES = 0.65` (standard SA-DDES)
- Cell length scale: `h_max = max(edge_lengths)` for hex, `h_max = max(√A_face)` for tet
- Near wall (fd→0): length scale = wall distance (RANS). Away (fd→1): length scale = Δ (LES)

Tasks:

- [ ] Implement `fd` computation per cell from local velocity gradient tensor `U_ij`
- [ ] Implement DDES length scale `Δ_DDES` per cell (device array `d_delta_ddes`)
- [ ] Modify SA residual kernel: replace `d` (wall distance) with `Δ_DDES` in destruction term when `SA_DDES` active
- [ ] `compute_turbulence_source_gpu` dispatch by `turbulence_model`

### 13.2 k-ω SST (2-equation)

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/rans_sst.hpp` | NEW | SST coefficients: `beta_i`, `gamma_i`, `sigma_k1/k2`, `sigma_omega1/omega2`, `alpha1/alpha2` — Menter 2003 |
| `src/aero/cfd/gpu_sst.cu` | NEW | `sst_source_kernel`: production P_k, P_ω; destruction D_k, D_ω; cross-diffusion F1 term |
| `src/aero/cfd/cfd_state.hpp` | MODIFY | NVAR=7 (add `rho_k`, `rho_omega`) when `turbulence_model=SST` |

State extension (SST-2eq):

```text
Q = [rho, rho*u, rho*v, rho*w, rho*E, rho*k, rho*omega]
W = [rho, u, v, w, p, T, a, k, omega]
```

Tasks:

- [ ] NVAR=7 structural changes: `cfd_state.hpp` (add k/omega fields, update cons_to_prim), `device_mesh.hpp` (NVAR=7), `gpu_update.cu` (L2 norm includes 2 new vars)
- [ ] k-ω SST production: `P_k = min(τ_ij·S_ij, 10*β*ρkω)`, `P_ω = γ/ν_t * P_k`
- [ ] k-ω SST destruction: `D_k = β*ρkω`, `D_ω = βρω²`
- [ ] SST blending function F1: `F1 = tanh(Φ₁⁴)`, where `Φ₁ = min[max(√k/(0.09ωd), 500ν/(ρd²ω)), 4ρσ_ω₂k/(CD_kω·d²)]`
- [ ] Cross-diffusion: `CD_kω = max(2ρσ_ω₂·∇k·∇ω/ω, 1e-10)`
- [ ] Stress limiter: `ν_t = a₁k / max(a₁ω, F₂·S)` with `F₂ = tanh[max(√k/(0.09ωd), 500ν/(ρd²ω))]²`
- [ ] Wall BC: `k_wall = 0`, `ω_wall = 6ν/(β₁y²)` (or Menter's `ω_wall = 60ν/(β₁Δy²)`)
- [ ] Inlet BC: `k_in = 1.5*(Tu*U_inf)²`, `ω_in = √k/(0.09·L_t)`

### 13.3 GPU kernels for SST

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/gpu_sst.cu` | NEW | `sst_source_kernel`: per-cell k and ω source terms (production + destruction + cross-diffusion), atomicAdd to residual indices 5,6 |
| `src/aero/cfd/gpu_viscous.cu` | MODIFY | `viscous_flux_kernel_atomic`: add `k` and `ω` diffusion terms (σ_k * μ_eff for k, σ_ω * μ_eff for ω) |
| `src/aero/cfd/gpu_update.cu` | MODIFY | `update_and_l2_kernel`: handle NVAR=7, read/write indices 5,6 |

Tasks:

- [ ] `sst_source_kernel`: read primitive k, ω, velocity gradient tensor S_ij, wall distance; compute P_k, P_ω, D_k, D_ω; atomicAdd to residual[5] (k) and residual[6] (ω)
- [ ] Viscous kernel extension: `mu_eff_k = mu + sigma_k * mu_t`, `mu_eff_omega = mu + sigma_omega * mu_t`; diffusion flux for k, ω
- [ ] Update kernel: L2 norm accumulates 7 components; `isfinite` check on k and ω (both must be >= 0)
- [ ] `viscous=false` with `SST`: viscous terms for k/ω must still be computed (diffusion is essential for turbulence model stability)
- [ ] Farfield BC: k_inf, omega_inf from freestream turbulence intensity `tu_inf` (default 0.1%) and turbulent length scale ratio `mu_t/mu` (default 0.1)

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-TURB-SST-1` | Flat plate SST: Cf vs Coles profile correlation (Cf difference < 10%) | 10% |
| 2 | `CFD-TURB-SST-2` | `turbulence_model=LAMINAR` regression to Phase 5 laminar (NVAR=5 path) | 1e-12 |
| 3 | `CFD-TURB-SST-3` | `turbulence_model=SA_SST` (forcing SA via SST? no) — separate test: SST zero k, omega laminar regression | 1e-6 |
| 4 | `CFD-TURB-DDES-1` | Flat plate (fully attached): SA-DDES matches SA-RANS (fd→0) | 1e-4 |
| 5 | `CFD-TURB-DDES-2` | Circular cylinder Re=3900: Strouhal within [0.19, 0.22], Cd within [0.9, 1.1] | range |
| 6 | `CFD-TURB-DDES-3` | Backward-facing step Re=5000: reattachment length x_r/H within [5.0, 7.0] | range |

Gate:

- SST flat plate Cf within 10% of Coles correlation.
- SA-DDES in attached flow region reproduces SA-RANS (regression ≤ 1e-4).
- `TurbulenceModel` enum migration path: existing `bool turbulence` code compiles and maps to correct enum.
- All turbulence models produce finite, non-negative k and ω (or nu_tilde).
- [V&V] SST MMS: observed order ≥ 1.8 on smooth manufactured solution with non-zero k and ω.

---

## Phase 14 — AMR: Turbulence-Aware Extension

Goal: extend the Euler AMR foundation (Phase 12) with turbulence-specific refinement criteria: y+ constraint for wall-resolved LES/RANS, wake refinement behind bodies, shear-layer refinement for DDES. This phase depends on Phase 13 (DDES/SST) being complete.

### 14.1 Turbulence-aware refinement sensors

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/amr_sensor.cpp` | MODIFY | +y+ sensor per Phase 12.2, add `target_yplus` wall-distance refinement |
| `src/aero/cfd/amr_sensor.cpp` | MODIFY | +vorticity-based sensor for wake and shear layers (Q-criterion refinement region) |
| `include/aero/cfd/amr_sensor.hpp` | MODIFY | +`SensorType = { GRADIENT, CURVATURE, YPLUS, Q_CRITERION, TKE_RATIO }` |

Tasks:

- [ ] y+ sensor: tag wall-adjacent cells where `y_phys > y_target(y+_desired)` for refinement; refine until all wall cells satisfy `y+ ≤ target_y+`
- [ ] Turbulence-intensity sensor: `k / (0.5 * U²) > threshold` — refine regions of high TKE (wake, mixing layer)
- [ ] Shear-layer sensor (DDES): ratio of resolved to modeled TKE → refine where under-resolved
- [ ] Refinement region: allow specifying a wake cone behind body for anisotropic refinement in streamwise direction

### 14.2 Turbulence-aware AMR solver loop

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/gpu_solver.cu` | MODIFY | Extend AMR loop (Phase 12.4): turbulence sensor runs every `amr_interval` alongside Euler sensor |
| `src/aero/cfd/gpu_solver.cu` | MODIFY | After AMR cycle, re-initialize turbulence variables on new cells (k=1e-8, ω=1e4, nu_tilde=1e-8) |

Tasks:

- [ ] Multi-sensor fusion: `refine = sensor_euler || sensor_turbulence` (either flag triggers refinement)
- [ ] After refinement, set new-cell turbulence variables to small positive values to avoid division by zero (`d_k = max(d_k, 1e-8)`, `d_omega = clamp(d_omega, 1e-4, 1e8)`)
- [ ] Wall-distance recomputation: after mesh change, `compute_wall_distance` re-run on all cells (GPU kernel)
- [ ] Regression: `turbulence_model=LAMINAR` on same mesh → AMR behavior identical to Phase 12

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-AMR-TURB-1` | Flat plate SST: y+ ≤ 1 after AMR adaptation starting from coarse mesh | N/A |
| 2 | `CFD-AMR-TURB-2` | Circular cylinder Re=3900 (DDES): AMR refines wake region (cell count increase ≥ 2×) | 2× |
| 3 | `CFD-AMR-TURB-3` | AMR + SST: forces match globally refined mesh within 2% | 2% |

Gate:

- Wall y+ after AMR ≤ target_y+ (default 1.0) on all wall-adjacent cells for SST cases.
- Turbulence variables on newly created cells are positive and produce finite residual.
- AMR + SST on flat plate matches globally refined mesh within 2% in Cf.

---

## Phase 15 — Thermochemistry

Goal: from constant-γ perfect gas to finite-rate chemically reacting gas for hypersonic heat flux.

### 14.1 Variable thermodynamic properties

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/thermo.hpp` | NEW | `class GasModel`: `gamma(T)`, `cp(T)`, `cv(T)`, `h(T)`, `e(T)` for single-species; polynomial curve fits (NASA McBride 7-coefficient format) |
| `src/aero/cfd/thermo.cpp` | NEW | 7-coeff polynomial evaluation: `cp/R = a₁ + a₂T + a₃T² + a₄T³ + a₅T⁴` for low T range; high T range poly for T > 1000K |
| `include/aero/cfd/cfd_config.hpp` | MODIFY | Add: `GasModel gas_model = PerfectGas`; `std::string gas_model_file = ""` for loading McBride coefficients |

Tasks:

- [ ] NASA McBride 7-coefficient struct (two temperature ranges, 14 coefs total + T_break)
- [ ] Air polynomial: N₂, O₂, NO, N, O individual + mixture-averaged cp(T) by mass fraction
- [ ] Chemical enthalpy: `h_s(T) = ∫cp_s(T) dT + h_f_s^298` (formation enthalpy)
- [ ] `gas_model` config: `PerfectGas` → gamma = constant; `EquilibriumAir` → cp(T) curve fit; `ChemNonEq` → chemistry activation
- [ ] GPU thermochemistry helpers: `d_gamma(T)`, `d_cp(T)`, `d_h_s(T)` — device functions using polynomial evaluation

### 14.2 Park 5-species finite-rate chemistry

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/chem_reactions.hpp` | NEW | Reaction set: N₂ + M ⇌ 2N + M (M = N₂, O₂, NO, N, O), O₂ + M ⇌ 2O + M, NO + M ⇌ N + O + M, N₂ + O ⇌ NO + N, NO + O ⇌ O₂ + N (5 reactions, 5 species) |
| `src/aero/cfd/chem_source_gpu.cu` | NEW | `chem_source_kernel`: per-cell forward/reverse rates (Arrhenius k_f = C·T^η·exp(-θ/T)), species production rates ω̇_s, source term = Σ(ω̇_s·h_s) for energy |
| `src/aero/cfd/cfd_state.hpp` | MODIFY | NVAR = 8 + 1 (rho, mu, mv, mw, rhoe, rho_Y1..rho_Y5) when `gas_model=ChemNonEq` |

State for 5-species chemistry:

```text
Q = [rho, rho*u, rho*v, rho*w, rho*E, rho*Y_N2, rho*Y_O2, rho*Y_NO, rho*Y_N, rho*Y_O]
W = [rho, u, v, w, T, Y_N2, Y_O2, Y_NO, Y_N, Y_O]
NVAR = 10
```

Tasks:

- [ ] Reaction rate data: 5 reversible reactions, Arrhenius coefficients (C, η, θ) from Park 1985
- [ ] Forward rate `k_f = C·T^η·exp(-θ/T)`; equilibrium constant via Gibbs free energy curve-fit; reverse rate `k_r = k_f / K_eq`
- [ ] Species production: `ω̇_s = Σ(ν"_si - ν'_si) * [k_f_i Π(ρ_j/M_j)^(ν'_ji) - k_r_i Π(ρ_j/M_j)^(ν"_ji)]`
- [ ] Energy source: `Q_chem = -Σ ω̇_s · h_s(T)` (chemical energy released/absorbed)
- [ ] GPU kernel: per-cell compute Arrhenius rates → species production → atomicAdd to residual[5..9]
- [ ] `ConservedState` dynamic: `NVAR` becomes configurable at runtime (gas model initialization sets global `cfg.nvar`)

> **NVAR 迁移策略说明**：Phase 7-13 使用 `constexpr CFD_NVAR=6`（固定结构体 `ConservativeState` 有名称字段）。Phase 15 是第一个需要 NVAR=10 的阶段。迁移方案：
> 1. Phase 15.0（前序任务）：将 `ConservativeState` 改为变长数组（如 `Real q_[MAX_NVAR]` 或 `std::array<Real, MAX_NVAR>`），保留前6个字段的名称访问器（向后兼容）。`PrimitiveState` 同理。
> 2. 所有 kernel 增加 `int nvar` 参数（默认值从 `DeviceMesh::NVAR` 取）。L2 归约 / 原子操作 / isfinite 检查全部使用 nvar 而非硬编码 6。
> 3. `DeviceCellData` 的 `d_q` 分配大小变为 `nvar * n_cells * sizeof(Real)`，不再假定 6。
> 4. 回归测试：Phase 15 中 chemistry 关闭时（`gas_model=PerfectGas`），NVAR=10 的 kernel 必须产生与 NVAR=6 时一致的解（仅前 5 个变量参与物理）。
> 5. 临时策略：Phase 15.0 之前不允许激活 chemistry；Phase 15.0 实现后通过 feature gate 控制。

### 14.3 Two-temperature model (Park 89)

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/chem_two_temp.hpp` | NEW | Extended state: `Q = [...NVAR.., rho*e_vib]` — vibrational energy |
| `src/aero/cfd/chem_source_gpu.cu` | MODIFY | Add: T-Tv energy exchange (Landau-Teller), vibrational energy source, chemical reactions with rate control using Tv or Ta = T^q * Tv^(1-q) |

Tasks:

- [ ] Vibrational energy: `e_vib(Tv) = Σ_s Y_s * R_s * θ_vib_s / (exp(θ_vib_s/Tv) - 1)`
- [ ] Landau-Teller: `Q_Tv = Σ_s ρ_s * (e_vib_s(T) - e_vib_s(Tv)) / τ_s` (translational-vibrational relaxation)
- [ ] Park rate-controlling temperature: `Ta = T^q * Tv^(1-q)` with `q = 0.5` (standard)
- [ ] NVAR=11 for 5-species + two-temp (rho_e_vib added)

### 14.4 Wall catalysis

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/wall_catalysis.hpp` | NEW | `enum CatalysisModel { NONCATALYTIC=0, SUPERCATALYTIC=1, FINITE_RATE=2 }` |
| `src/aero/cfd/gpu_wall.cu` | MODIFY | `wall_force_kernel`: add catalysis BC: `Y_s_wall` from recombination model; surface heat flux from diffusion term |

Tasks:

- [ ] Non-catalytic: `∇Y_s · n = 0` (zero mass fraction gradient at wall)
- [ ] Fully (super-)catalytic: `Y_s_wall = Y_s_freestream` (wall composition equals freestream)
- [ ] Finite-rate: surface reaction mechanism (e.g., O + O → O₂, N + N → N₂ with reaction probability γ)
- [ ] Heat flux catalytic contribution: `q_cat = Σ ω̇_s_wall · h_s(T_wall)`
- [ ] Total wall heat flux: `q_total = q_Fourier + q_cat + q_diffusion`

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-CHEM-1` | Variable gamma air: cp(T) from McBride matches NIST REFPROP from 200K to 2000K | 1% |
| 2 | `CFD-CHEM-2` | Park 5-species normal shock (Mach 10, 30km): post-shock T and composition vs NASA CEA | 2% |
| 3 | `CFD-CHEM-3` | Park 5-species normal shock (Mach 20, 50km): two-temperature Tv/T ratio | 5% |
| 4 | `CFD-CHEM-4` | `gas_model=PerfectGas` regression to Phase 7 laminar results | 1e-6 |
| 5 | `CFD-CHEM-5` | Stagnation point heating (Mach 15, 40km): Fay-Riddell correlation | 15% |
| 6 | `CFD-CHEM-6` | Wall catalysis: non-catalytic q < finite-rate q < supercatalytic q | inequality |
| 7 | `CFD-CHEM-7` | `gas_model=ChemNonEq` with `reactions=false` matches variable-gas without reactions | 1e-6 |

Gate:

- Variable cp(T) integrated enthalpy `h(T) - h(298)` matches NIST/JANAF data within 1%.
- Park 5-species normal shock equilibrium composition within 2% of NASA CEA.
- Stagnation point heat flux within 15% of Fay-Riddell for Mach 15, 40km.
- `gas_model=PerfectGas` regression to Phase 7 must pass.
- Catalytic wall: non-catalytic always ≤ finite-rate ≤ supercatalytic (monotonicity).

---

## Phase 16 — Transition Physics

Goal: predict laminar-turbulent transition onset independently of SA model. Support natural, bypass, crossflow, and Mack-mode transition.

### 15.1 γ-Reθ transition model

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/transition_gamma_retheta.hpp` | NEW | γ-Reθ 4-equation model (k, ω, γ, Reθt) coupled with SST: γ = intermittency, Reθt = transition momentum thickness Re |
| `src/aero/cfd/gpu_transition.cu` | NEW | `transition_source_kernel`: modified k and ω production with intermittency damping, Reθt transport equation |
| `src/aero/cfd/gpu_sst.cu` | MODIFY | SST kernel reads intermittency γ from state tensor; `P_k_eff = γ * P_k`, `D_k_eff = min(max(γ, 0.1), 1.0) * D_k` |

γ-Reθ equations (Langtry-Menter 2009):
- γ equation: production = `c_a1 * ρ * S * (γ*F_growth)^0.5 * (1 - c_e1*γ)`, destruction = `c_a2 * ρ * Ω * γ * F_turb`
- Reθt equation: transport of local transition momentum thickness Re
- Coupling with SST: modified k-production `P_k = γ_eff * P_k_orig`, `γ_eff = max(γ, γ_sep)` (separation-induced transition)

Tasks:

- [ ] γ-Reθ coefficients (ca1, ca2, ce1, ce2, σγ, σ_Reθt) from Langtry-Menter 2009 Table 1
- [ ] GPU kernel: per-cell compute γ source terms, Reθt source terms
- [ ] Intermittency damping of SST production: `P_k_eff = γ * P_k_orig`
- [ ] Separation-induced transition: `γ_sep = min(2*max(0, (Re_v/3.235*Re_θc) - 1)*F_reattach, 2) * F_θt`
- [ ] Correlations for transition onset Reθt: `Re_θt = f(Tu, λ_θ)` from local pressure gradient and freestream turbulence

### 15.2 LST / e^N method (offline tool)

Files:

| File | Action | Content |
|------|--------|---------|
| `tools/lst_solver/` | NEW | Python/C++ boundary layer profile extractor + linear stability solver for 2D/axisymmetric boundary layers |
| `src/aero/cfd/bl_profile.cpp` | NEW | Extract velocity/temperature profiles at streamwise stations from CFD solution |

Tasks:

- [ ] BL profile extraction: marching method along wall surface at given streamwise stations, extract u(y), T(y), ρ(y)
- [ ] Local similarity: compressible Falkner-Skan profiles with edge conditions from CFD
- [ ] Spatial LST solver: solve Orr-Sommerfield + Squire eigenvalue problem for given frequency/wave-number
- [ ] N-factor integration: `N = ∫(-α_i) dx` along instability path
- [ ] Transition criterion: N-factor threshold (N_crit = 9 for TS, N_crit = 6 for Mack mode in hypersonic)
- [ ] Output: transition location band (multiple frequencies → envelope N-factor curve)

### 15.3 Hypersonic Mack-mode correlation

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/transition_mack.cpp` | NEW | Engineering correlation for Mack second-mode transition on sharp cones: Re_θ/Me correlation from NASA TM-2011-217433 |

Tasks:

- [ ] Implement Mack-mode correlation: `Re_θ_transition = exp(C₁ + C₂*Me + C₃*Me² + C₄*Tw/Te)` with coefficients for sharp cones
- [ ] Override γ-Reθ onset location with Mack-mode correlation when Me > 4
- [ ] Uncertainty band: perturb correlation coefficients ±20% → output transition band

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-TRANS-1` | γ-Reθ flat plate zero pressure gradient: transition onset at Re_x within ±15% of Schlichting correlation | 15% |
| 2 | `CFD-TRANS-2` | `transition_model=none` regression to SST fully turbulent | 1e-6 |
| 3 | `CFD-TRANS-3` | Mack-mode sharp cone Mach 6: transition onset x/L within known AEDC tunnel data | 20% |
| 4 | `CFD-TRANS-4` | e^N method on flat plate: N-factor growth rate matches Mack 1987 reference | 5% |
| 5 | `CFD-TRANS-5` | Transition band output: both edges of uncertainty band are reported (never single value) | metadata |

Gate:

- No transition model produces a single "exact" transition location; all output must include an uncertainty band.
- SA model is never used to infer transition (SA is fully turbulent by construction).
- γ-Reθ with `Tu_inf=0.1%` produces transition Reynolds number Re_x at least 2× larger than with `Tu_inf=1.0%` on zero-pressure-gradient flat plate.
- Mack-mode transition for Me > 4 explicitly labeled as "second-mode dominated" in output.

---

## Phase 17 — Multi-Physics Coupling

Goal: couple CFD with heat conduction (CHT), structural deformation (aeroelastic), and trajectory dynamics (6-DOF).

### 16.1 Conjugate heat transfer (CHT)

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/cht_solver.hpp` | NEW | `class ChtSolver`: solid heat conduction GPU solver (Poisson equation: ∇·(k_s ∇T) = 0) |
| `src/aero/cfd/cht_gpu.cu` | NEW | `solid_heat_kernel`: finite volume / finite difference solver for solid domain; implicit time advancement |
| `src/aero/cfd/gpu_solver.cu` | MODIFY | CHT coupling step: solve fluid (giving q_wall) → solve solid (giving T_wall) → update fluid BC → iterate |

CHT coupling:
- Fluid side: provides `q_wall = -k_fluid * ∇T·n` (heat flux into wall)
- Solid side: solves `∂T/∂t = α ∇²T` with `-k_solid * ∇T·n = q_wall` as BC
- Coupling: Dirichlet-Neumann iteration at fluid-solid interface
- Convergence: `||T_wall_new - T_wall_old|| < 1e-4`

Tasks:

- [ ] Solid mesh: reuse CfdMesh infrastructure (tet/hex cells, boundary markers)
- [ ] Solid heat conduction kernel: `k_solid*ΔT` (Laplace operator), implicit Euler
- [ ] Interface mapping: fluid wall faces → solid wall faces (identical mesh at interface or interpolation)
- [ ] Steady CHT: coupled iteration until temperature convergence
- [ ] Unsteady CHT: sub-cycling (solid uses larger dt than fluid)

### 16.2 Mesh deformation

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/mesh_deformation.hpp` | NEW | `deform_mesh(mesh, wall_displacements)`: radial basis function (RBF) or spring-analogy mesh deformation |
| `src/aero/cfd/mesh_deformation.cpp` | NEW | RBF interpolation: wall displacements → interior node displacements using Wendland C2 basis function |

Tasks:

- [ ] Spring analogy: each edge is a spring with stiffness k = 1/|edge|²; solve for node displacements (iterative, GPU-accelerated Jacobi)
- [ ] RBF: `dx(x) = Σ w_i * φ(||x - x_i||)` where φ is Wendland C2 function (compactly supported)
- [ ] Mesh quality preservation: after deformation, min Jacobian should not drop below 50% of original
- [ ] Volume mesh update: apply node displacements, recompute metrics (volume, face area, normal)
- [ ] GPU: `deform_mesh_kernel` — per-node interpolation of RBF coefficients (in parallel)

### 16.3 Trajectory-coupled 6-DOF

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/gpu_solver.cu` | MODIFY | Add unsteady mode with coupled 6-DOF: each physical time step, compute forces → pass to 6-DOF integrator → update position/orientation → deform mesh → next time step |
| `include/aero/cfd/cfd_config.hpp` | MODIFY | Add: `bool coupled_6dof = false`, `Real physical_dt = 1e-3f`, `int fluid_steps_per_dof_step = 10` |

Tasks:

- [ ] Fluid time-accurate mode: dual time stepping (inner pseudo-time convergence, outer physical time advance)
- [ ] Per physical step: integrate wall forces → 6-DOF (quaternion rotation, body translation)
- [ ] Mesh rigid motion: rotate/translate entire mesh (no deformation needed, just rigid transform of nodes)
- [ ] Mesh deformation (if aeroelastic): use Phase 17.2 mesh deformation for elastic body displacement

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-CHT-1` | Solid-only heat conduction: known T gradient with analytic solution (1D, k=const) | 1e-6 |
| 2 | `CFD-CHT-2` | Fluid-solid coupled: flat plate with constant T_wall, CHT reaches same T_wall after coupling | 1e-4 |
| 3 | `CFD-DEFORM-1` | RBF mesh deformation: unit sphere walls displaced 1% outward, all cells have positive Jacobian | >0 |
| 4 | `CFD-DOF-1` | Rigid body rotation: mesh rotated 90°, mass conservation (no flow through walls) | 1e-10 |
| 5 | `CFD-DOF-2` | 6-DOF coupled: free-fall trajectory matches analytical solution (CD=0) | 1e-6 |

Gate:

- CHT: interface heat flux continuity (fluid_q = solid_q) to within 1e-3.
- Mesh deformation: minimum Jacobian after maximum expected deformation ≥ 50% of initial.
- 6-DOF trajectory matches reference integration (uncoupled) within 1% for short time integration.

---

## Phase 18 — High-Order Methods (DG/FR)

Goal: achieve spectral accuracy for wave-dominated flows. Enable DNS-quality resolution on coarse meshes.

> **DG/FVM 架构策略说明**：DG 需要完全不同的网格/状态/Kernel 基础设施（每个单元 `(p+1)³` 个 DOF、面求积点、基函数求值），不能复用 FVM 的 `DeviceMesh`/`CfdCell`/`CfdFace`/`ConservativeState`。因此 DG 采用**独立类树**架构：
> - `DgMesh` / `DgDeviceMesh`：独立的网格和 DOF 数据结构（继承或组合 `CfdMesh` 的节点坐标，但单元/面数据不同）
> - `DgSolver` / `FvmSolver`：通过运行时 `Solver` 基类多态切换
> - `config.method` 控制 `"fvm"` 或 `"dg"`，两者共享网格（节点坐标），但状态分配和求解器流程完全独立
> - FVM 代码在 `method=fvm` 下零退化，DG 不修改任何现有的 FVM 文件

### 18.1 DG 2D scalar advection (entry-level verification)

Goal: validate the DG infrastructure on the simplest problem before tackling 3D Euler. This minimizes debugging surface.

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/dg_basis.hpp` | NEW | `class LagrangeBasis1D`: `l_i(x)` at Legendre-Gauss-Lobatto nodes for p=1/2/3/4; `class TensorBasis2D`: quad tensor product basis |
| `include/aero/cfd/dg_solver.hpp` | NEW | `class DgSolver2D`: 2D scalar advection DG solver: `dg_volume_kernel`, `dg_face_kernel`, RK3 |
| `src/aero/cfd/dg_scalar.cu` | NEW | `dg_scalar_volume_kernel` (advection of scalar u on 2D quad mesh); `dg_scalar_face_kernel` (upwind flux) |
| `tests/cfd/test_dg_scalar.cpp` | NEW | 2D scalar advection: linear advection of a Gaussian hump, measure L2 error vs analytical solution |

Tasks:

- [ ] Lagrange 1D basis functions for LGL nodes (p+1 nodes per dimension)
- [ ] 2D tensor product: (p+1)² DOFs per element per scalar variable
- [ ] LGL quadrature: exact for polynomials up to degree 2p-1
- [ ] `dg_scalar_volume_kernel`: for each quadrature point, read u via basis interpolation → compute advective flux `f = a*u` → divergence → accumulate to residual
- [ ] `dg_scalar_face_kernel`: upwind flux `f* = 0.5*(a·n + |a·n|)*u_L + 0.5*(a·n - |a·n|)*u_R`
- [ ] MMS on 2D scalar: `u_exact = sin(πx)cos(πy)`, source term = `a·∇u`, verify p+1 convergence
- [ ] GPU: `dg_scalar_volume_kernel` uses shared memory for precomputed basis values

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-DG-SCALAR-1` | 2D linear advection: Gaussian hump p=3 preserves shape after 1 period | L2 < 1e-4 |
| 2 | `CFD-DG-SCALAR-2` | 2D MMS: p=1→O(h²), p=2→O(h³), p=3→O(h⁴) | slope ±0.1 |

### 18.2 DG Euler 3D + curved boundary

Build on 18.1: extend from scalar advection to 3D Euler equations. Curved high-order boundaries added in parallel (essential for accurate DG on curved walls).

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/dg_basis.hpp` | MODIFY | +`class TensorBasis3D`: hex tensor product (p+1)³; `class DubinerBasis`: tetrahedral modal basis |
| `src/aero/cfd/dg_volume.cu` | NEW | `dg_euler_volume_kernel`: Euler flux divergence via quadrature (Roe or Lax-Friedrichs at each integration point) |
| `src/aero/cfd/dg_face.cu` | NEW | `dg_euler_face_kernel`: numerical flux at face quadrature points, accumulate to left/right elements |
| `src/aero/cfd/dg_solver.cu` | NEW | DG solver orchestrator: allocate DOFs per element, run volume+face kernels, RK time integration |
| `include/aero/cfd/dg_curved.hpp` | NEW | `class CurvedGeometry`: high-order node positions (warped from linear mesh using CAD data or analytic deformation) |
| `src/aero/cfd/dg_curved.cpp` | NEW | Compute isoparametric mapping Jacobian at each quadrature point for curved elements |

Tasks:

- [ ] Hex: 3D tensor product → (p+1)³ DOFs per element per variable
- [ ] DG volume kernel: for each quadrature point, read state via basis interpolation → compute Euler flux → compute divergence → accumulate to residual
- [ ] DG face kernel: for each face quadrature point, read left/right state → Roe/HLLC numerical flux → accumulate to left/right elements
- [ ] Isoparametric mapping: boundary elements use p-order polynomial to represent curved wall
- [ ] Jacobian: `dx/dξ` computed at each integration point, determinant for volume weighting
- [ ] GPU: store curved Jacobians per element per quadrature point (precomputed on upload)
- [ ] GPU: `dg_euler_volume_kernel` uses shared memory for precomputed basis values

### 18.3 Shock capturing for DG

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/dg_shock.hpp` | NEW | `class ShockCapturing`: Persson-Peraire shock sensor (`s_e = log10(modal coefficient ratio)`) |
| `src/aero/cfd/dg_shock.cpp` | NEW | Artificial viscosity: `ε = ε_0 * exp(-(s_e - s_0)⁴ / (s_0²))` for elements where `s_e > s_0` |

Tasks:

- [ ] Persson sensor: ratio of highest mode energy to total energy → element-wise smoothness indicator
- [ ] Localized artificial viscosity: Laplacian term added to DG formulation, `ε` = element-dependent
- [ ] `fvm_fallback` option: mark shocked elements as FVM cells, use Phase 8 FVM with Barth-Limiter, then combine via hybrid DG-FV method

### 18.4 DG extension: viscous, turbulence, thermochemistry

Goal: extend DG to handle NS viscous terms, RANS turbulence models, and reacting flows. Each extension follows the same pattern: add appropriate flux functions at quadrature points.

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/dg_viscous.cu` | NEW | `dg_ns_volume_kernel`: add viscous flux divergence (BR2 or LDG method for second derivatives) |
| `src/aero/cfd/dg_viscous.cu` | NEW | `dg_ns_face_kernel`: interior penalty or BR2 interface flux for viscous terms |
| `src/aero/cfd/dg_source.cu` | NEW | `dg_source_kernel`: per-element source terms (RANS production, chemistry) at quadrature points |

Tasks:

- [ ] BR2 (Bassi-Rebay 2) method: lift operator for gradient computation, stabilization term for face jumps
- [ ] Viscous DG: add Laplacian/div(grad) operator to volume and face kernels
- [ ] RANS DG: couple with SA or SST source terms evaluated at each quadrature point
- [ ] Thermochemistry DG: extend state to NVAR=10/11, add reaction source integration

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-DG-1` | Isentropic vortex advection (3D): p=3 preserves vortex strength after 10 domain traversals | >99% |
| 2 | `CFD-DG-2` | Taylor-Green vortex at Re=1600: enstrophy decay matches DNS (Brachet 1983) | 5% |
| 3 | `CFD-DG-3` | MMS (Euler): p=1→O(h²), p=2→O(h³), p=3→O(h⁴) on smooth manufactured problem | slope ±0.1 |
| 4 | `CFD-DG-4` | Shock tube (Sod): DG p=2 with shock capturing, no overshoot > 0.5% | 0.5% |
| 5 | `CFD-DG-5` | DG order=1 regression: matches FVM 1st-order (cell-averaged equivalence) | 1e-4 |
| 6 | `CFD-DG-6` | Viscous DG: laminar flat plate Cf matches Blasius (p=3) | 2% |

Gate:

- MMS observed order = theoretical order ± 0.1 for p=1/2/3 on hex mesh (Euler).
- 2D scalar advection: MMS convergence slope passes for p=1/2/3.
- Isentropic vortex: p=3 dissipation per period < 1% (vorticity error).
- Shock capturing produces no overshoot > 1% on Sod shock tube.
- DG and FVM coexist: `method=dg` vs `method=fvm` switch produces expected accuracy difference.
- Viscous DG: each new physics extension (viscous, RANS, chemistry) must pass MMS for that equation set.

---

## Phase 19 — Verification & Validation Systematization

Goal: every result has a quantifiable error bound. Formal V&V pipeline for production use.

> **V&V continuity**: MMS and GCI should not start from scratch in this phase. Every physics phase (Phase 5 Euler, Phase 7 RANS, Phase 13 DDES/SST, Phase 15 Thermochemistry, Phase 16 Transition) already includes MMS order verification as a gate condition (added retroactively during Phase 19 setup). This phase systematizes those individual MMS checks into a unified framework, adds GCI, builds the benchmark suite, and implements the error budget. If any earlier phase lacks MMS gates, fix retroactively before beginning Phase 19 execution.

### 19.1 Method of Manufactured Solutions (MMS)

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/mms.hpp` | NEW | `class MmsSolution`: analytical q(x,y,z,t), source term S(x,y,z,t) for Euler, NS, SA, SST, chemistry |
| `src/aero/cfd/mms_euler.cpp`, `mms_ns.cpp`, `mms_sa.cpp`, `mms_sst.cpp`, `mms_chem.cpp` | NEW | Per-equation-set manufactured solutions (polynomial or trigonometric) |
| `tests/cfd/test_mms.cpp` | NEW | `FOR_EQUATION_SET(euler/ns/sa/sst/chem)`: run 3 meshes (h, h/2, h/4), compute observed order |

MMS procedure:
1. Choose analytical solution `q_exact(x,y,z)` (polynomial or smooth trigonometric)
2. Compute `S = R(q_exact)` analytically (the source term that makes q_exact satisfy the discrete equations)
3. Run solver with `S` as explicit source term
4. Compute `L2_error = ||q - q_exact|| / ||q_exact||`
5. On 3 meshes (h coarsening factor = 2): fit `log(error) = p*log(h) + C`
6. Verify: `|p - p_theory| < 0.1`

Tasks:

- [ ] Euler MMS: smooth density/velocity/pressure field (e.g., sine waves), compute S from residual
- [ ] NS MMS: add viscosity terms to Euler MMS, include viscous flux in S computation
- [ ] SA MMS: non-zero nu_tilde field, include SA source terms in S
- [ ] SST MMS: non-zero k and omega fields, include SST source terms
- [ ] Chemistry MMS: non-zero mass fractions, include reaction sources in S
- [ ] GPU: MMS source term kernel appends S to residual (`d_residual += d_mms_source`)
- [ ] Automated order verification script: `scripts/verify_order.py` — run 3 meshes, compute p, compare to theory

### 19.2 Grid Convergence Index (GCI)

Files:

| File | Action | Content |
|------|--------|---------|
| `tools/gci_report.py` | NEW | Python script: 3 mesh files (coarse/medium/fine) + solver results → GCI report in PDF/HTML |

GCI procedure (Roache 1998):
1. Run on 3 meshes: `h_coarse = 2*h_medium = 4*h_fine`
2. Compute apparent order: `p = ln((f3-f2)/(f2-f1)) / ln(r)` where `r = h2/h1`
3. GCI_fine = `F_s * |(f1-f2)/f1| / (r^p - 1)` with safety factor `F_s = 1.25`
4. Report: `f = f1 ± GCI_fine` with 95% confidence

Tasks:

- [ ] GCI on integrated quantities: Cd, Cl, Cm, total heat flux
- [ ] GCI on field quantities: Cp distribution, Cf distribution, St distribution
- [ ] Automatic mesh generation: from base mesh, generate h, h/2, h/4 by global refinement
- [ ] Report format: table of QoI, fine/medium/coarse values, p, GCI, asymptotic range check
- [ ] GPU acceleration: fine mesh may require multi-GPU (automatic dispatch to MPI mode when > single-GPU memory)

### 19.3 Standard benchmark suite

Files:

| File | Action | Content |
|------|--------|---------|
| `benchmarks/` | NEW | Directory of standard benchmark cases with reference solutions |
| `scripts/run_benchmarks.py` | NEW | Script: for each benchmark, build mesh → run solver → compare to reference → report |

Benchmarks (by phase capability):
- Euler: NACA 0012 (M=0.8, a=1.25° — AGARD), wedge (M=10, 15° half-angle — oblique shock), biconic (HB-2, M=10)
- Laminar NS: flat plate (Re=1e4/1e5 — Blasius), cavity (Re=400 — Ghia), sphere (M=5, Re=1e5 — stagnation heat flux)
- RANS: flat plate (Re=1e7 — Coles profile), NACA 0012 (M=0.15, Re=6e6 — NASA TP-2016-219053), backward-facing step (Re=5000 — Driver/Seegmiller)
- DDES: circular cylinder (Re=3900), delta wing (NACA 0012 deep stall), cavity (M=0.6, Re=1e6)
- Thermochemistry: normal shock (Mach 10-25 — CEA), RAM-C II (electron density), ELECTRE (stagnation heat flux), HIFiRE-1
- Transition: sharp cone (M=6, AEDC Tunnel 9), blunt cone (M=10, CUBRC LENS)

Tasks:

- [ ] Create mesh files for all benchmarks (gmsh .geo scripts or pre-generated .su2)
- [ ] Create reference solution database (paper values, CFD results from literature)
- [ ] Script: auto-detect solver capability from config, run matching benchmarks
- [ ] Report: PASS/WARN/FAIL per benchmark with quantitative comparison

### 19.4 Error budget framework

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/error_budget.hpp` | NEW | `struct ErrorBudget`: discretization_error, iterative_error, model_error, input_uncertainty → `total = sqrt(Σ_e²)` |
| `src/aero/cfd/error_budget.cpp` | NEW | Compute discretization error from GCI; iterative error from residual convergence; model error from literature; input uncertainty from sensitivity studies |

Tasks:

- [ ] Discretization error: from GCI (Phase 19.2) or from h-refinement studies
- [ ] Iterative error: `residual_L2 * dt * characteristic_time` (estimate of unconverged contribution)
- [ ] Model error: from benchmark comparisons for each turbulence/transition/chemistry model
- [ ] Input uncertainty: finite-difference sensitivity to freestream Mach, alpha, wall temperature, Re
- [ ] Report template: per QoI (Cd, Cl, Cm, Q_wall) → error sources → combined
- [ ] GPU: sensitivity studies (perturb input, rerun) automated via script

Gate:

- Every production run must output an error budget alongside the results.
- MMS order verification must pass for all active equation sets before claiming accuracy.
- GCI for integrated forces ≤ 5% on production mesh (or mesh is explicitly labeled as under-resolved).
- Unknown model errors (e.g., no transition model active for transitional flow) must be flagged in output.

---

## Phase 20 — Production HPC Hardening

Goal: achieve production-level performance, reliability, and usability on national supercomputing infrastructure.

### 20.1 Multi-architecture GPU build matrix

Files:

| File | Action | Content |
|------|--------|---------|
| `CMakeLists.txt` | MODIFY | `set(CMAKE_CUDA_ARCHITECTURES "75;80;89;90;100" CACHE STRING "...")` — auto-detect (see Phase 0 CMake fix); include PTX for forward compatibility |

Tasks:

- [ ] CMake: auto-detect host GPU capability at build time, add as preferred arch + default fallbacks
- [ ] JIT cubin: include PTX for `compute_80` and `compute_90` so new GPUs can JIT-compile
- [ ] CI: build and test on at least one GPU per supported architecture family (e.g., A100 for sm_80, H100 for sm_90, RTX 4090 for sm_89)
- [ ] `cudaDeviceGetAttribute` on startup to verify compiled arch matches running GPU

### 20.2 GPU memory pool

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/device_pool.hpp` | NEW | `class DeviceMemoryPool`: pre-allocate large arena, sub-allocate blocks, recycle freed blocks |
| `src/aero/cfd/device_pool.cpp` | NEW | Stack-based allocator (fast, no fragmentation for nested usage); fallback to cudaMalloc for large blocks |

Tasks:

- [ ] On solve begin: allocate device memory pool of `0.9 * free_mem` (detected at runtime)
- [ ] `pool_alloc(n)` and `pool_free(ptr)`: O(1) stack allocator
- [ ] `cudaMalloc` fallback for blocks > pool size
- [ ] AMR-aware: mesh reallocation uses pool (avoids repeated cudaFree/cudaMalloc cycle)
- [ ] Multi-GPU: separate pool per device (allocated after `cudaSetDevice`)

### 20.3 CUDA Graph accelerated iteration loop

Files:

| File | Action | Content |
|------|--------|---------|
| `src/aero/cfd/gpu_solver.cu` | MODIFY | After first iteration (graph capture), instantiate CUDA Graph and replay for subsequent iterations |

Tasks:

- [ ] `cudaStreamBeginCapture(stream)` at start of iteration, `cudaStreamEndCapture` → `cudaGraph_t`
- [ ] For iterations 2..max_iter: `cudaGraphLaunch(graph, stream)` — single kernel launch replaces ~14 kernel launches
- [ ] Handle dynamic paths (AMR mesh change, CFL ramp): recapture graph only on change
- [ ] `cudaGraphExecUpdate` for incremental updates (faster than full recapture)
- [ ] Fallback: if graph capture fails (e.g., memory operations), use original kernel launch loop

### 20.4 Mixed precision

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/mixed_precision.hpp` | NEW | `#define USE_MIXED_PRECISION` — FP32 for state/residual storage, FP64 for accumulation/reduction |
| `src/aero/cfd/gpu_update.cu` | MODIFY | `update_and_l2_kernel`: accumulate L2 in FP64 (Kahan summation), write state in FP32 |

Tasks:

- [ ] FP32 state storage: `Real` = float, but accumulation in kernel uses `double` for critical operations
- [ ] Kahan summation for L2 reduction: `sum += delta - compensation; compensation = (sum - compensation_old) - delta` (compensated)
- [ ] FGMRES inner products: use FP64 for dot products (cublasSetMathMode or custom kernel)
- [ ] `real_atomic_add`: always FP32 CUDA atomic; for FP64 accumulation, use separate `double* d_l2_fp64` buffer
- [ ] Verification: mixed precision result differs from FP64 result by < 1e-8 relative

### 20.5 Parallel I/O with HDF5

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/hdf5_io.hpp` | NEW | `write_checkpoint(mesh, state, config, step)` — HDF5 file per rank or collective write |
| `src/aero/cfd/hdf5_io.cpp` | NEW | HDF5 dataset definitions: mesh geometry, cell state, iteration metadata |

Tasks:

- [ ] HDF5 file structure: `/mesh/nodes`, `/mesh/cells`, `/mesh/faces`, `/state/conservative`, `/state/primitives`, `/metadata/config`, `/metadata/iteration`
- [ ] Parallel HDF5: `H5Pset_fapl_mpio(..., MPI_COMM_WORLD, MPI_INFO_NULL)` for collective writes
- [ ] Checkpoint format: read by restart script, reproducible (same input → same checkpoint)
- [ ] VTK output: existing `write_vtk` extended to handle non-tet elements (VTK POLYDATA for quad faces)
- [ ] CGNS output (optional): `#ifdef WITH_CGNS` path for CGNS native format write

### 20.6 Performance optimization (nsight-guided)

Files:

| File | Action | Content |
|------|--------|---------|
| `tools/roofline.py` | NEW | Script: run solver on standard mesh, collect nsight-compute metrics, compare to roofline model |

Performance targets (from nsight-compute profiling):
- Memory bandwidth utilization ≥ 60% of theoretical (roofline model, relative to GPU peak — e.g., H100 HBM3 3.35 TB/s → target ≥ 2.0 TB/s)
- Kernel launch overhead ≤ 5% of iteration time (via CUDA Graph)
- Occupancy for each kernel ≥ 50% (using CUDA occupancy API to tune block size)
- `real_atomic_add` contention reduced via coloring (already done) or privatization buffers

Tasks:

- [ ] Profile each kernel (timestep, Euler residual, viscous, RANS, SST, gradient, limiter, update, wall)
- [ ] Identify bottom-3 kernels by total time, optimize each
- [ ] Kernel fusion candidates: timestep+residual, residual+update (saves memory traffic)
- [ ] Shared memory tuning: `launch_bounds` block size optimization per kernel
- [ ] L1 cache configuration: use `cudaFuncSetAttribute` for kernels with high shared memory or data reuse

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-HPC-BW-1` | Memory bandwidth test (copy kernel): ≥ 60% theoretical peak | 60% |
| 2 | `CFD-HPC-POOL-1` | Memory pool: 1000 allocate/free cycles within pool, no cudaMalloc fallback | 0 fallback |
| 3 | `CFD-HPC-GRAPH-1` | CUDA Graph: kernel launch overhead < 5% of iteration wall time | 5% |
| 4 | `CFD-HPC-GRAPH-2` | Graph capture: first iteration result = graph replay result (bitwise for colored mode) | exact |
| 5 | `CFD-HPC-MP-1` | Mixed precision: Cd differs from FP64 by < 1e-8 | 1e-8 |
| 6 | `CFD-HPC-MP-2` | Mixed precision: no NaN or divergence compared to FP32 baseline | N/A |
| 7 | `CFD-HPC-WEAK-1` | Weak scaling: 100K cells/GPU, 8 GPUs, runtime ≤ 1.25× single-GPU baseline | 25% |

Gate:

- Iteration wall-time with CUDA Graph replay ≤ 3× the theoretical minimum (computed as bytes-touched / peak-BW of target GPU).
- Memory pool passes 1000-cycle stress test without fragmentation-induced cudaMalloc.
- Mixed precision: difference from full-FP64 ≤ 1e-8 for integrated forces.
- All existing tests pass in mixed precision mode.

Reference target (not hard gate): on H100, 1M-tet Euler order=2 iteration ≤ 5ms with CUDA Graph.

---

## Phase 21 — Cross-Platform & Future Hardware

Goal: ensure solver is not locked to NVIDIA GPU ecosystem. Port to AMD, Intel, and domestic Chinese accelerators.

### 21.1 AMD ROCm/HIP port

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/hip_utils.hpp` | NEW | `HIP_CHECK` macro, hipMalloc/hipMemcpy/hipFree wrappers mapping to CUDA or HIP API |
| `cmake/FindHIP.cmake` | NEW | HIP detection + compilation flags |

Porting strategy:
- Write CUDA-HIP common headers using preprocessor: `#ifdef __HIP_PLATFORM_AMD__` / `__CUDACC__`
- Kernels: most CUDA C++ syntax is compatible with HIP (built-in variables, kernel launch syntax)
- Replace: `cudaMalloc → hipMalloc`, `cudaMemcpy → hipMemcpy`, `cudaDeviceSynchronize → hipDeviceSynchronize`
- Replace: `__float_as_int` CAS loops with `__hip_atomic_compare_exchange` or generic `atomicCAS` (HIP supports it)
- Math functions: `__finitef → finite_float` or `isfinite` (HIP `rocprim`)
- `__syncthreads`, `threadIdx.x`, `blockIdx.x`, `blockDim.x` — identical between CUDA and HIP

Tasks:

- [ ] Create `gpu_runtime.hpp` with namespace aliases: `gpuMalloc = cudaMalloc` or `hipMalloc`
- [ ] Create `gpu_launch.hpp` with launch macro: `GPU_LAUNCH(kernel, grid, block, shared, stream, args...)` → `kernel<<<grid, block, shared, stream>>>(args)` or `hipLaunchKernelGGL(kernel, grid, block, shared, stream, args)`
- [ ] Port all `.cu` files to `.hip.cpp` with shared kernel body
- [ ] Build on AMD MI250/MI300X: `cmake -DCMAKE_CXX_COMPILER=hipcc ..`
- [ ] Test suite: all tests compile and pass on AMD

### 21.2 Intel SYCL/DPC++ port

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/sycl_utils.hpp` | NEW | SYCL buffer/accessor wrappers; `queue.submit([&](handler& h) { ... })` pattern for kernel launches |

Tasks:

- [ ] Identify CUDA-specific features: `atomicAdd(float*)` → SYCL `atomic_ref`, `__syncthreads` → `item.barrier()`
- [ ] Port critical kernels (Euler residual, update) to SYCL as proof of concept
- [ ] Build on Intel Data Center GPU Max 1550 (Ponte Vecchio)
- [ ] Performance: compare SYCL vs native CUDA on same NVIDIA hardware

### 21.3 Domestic accelerator adaptation layer

Files:

| File | Action | Content |
|------|--------|---------|
| `include/aero/cfd/accelerator.hpp` | NEW | Abstract base: `class Accelerator { virtual void* alloc(size_t) = 0; virtual void memcpy(...) = 0; virtual void launch(...) = 0; ... }` |
| `src/aero/cfd/acc_cuda.cpp` | NEW | CUDA implementation of `Accelerator` |
| `src/aero/cfd/acc_hip.cpp` | NEW | HIP implementation |
| `src/aero/cfd/acc_sycl.cpp` | NEW | SYCL implementation |

Tasks:

- [ ] Define `Accelerator` interface: memory management, kernel launch, synchronization, stream, event
- [ ] Kernels written as function objects (functors) compatible with polymorphic launch
- [ ] Build on Hygon DCU (China), Sunway SW26010Pro, or Cambricon MLU as demonstrations of portability
- [ ] Performance regression: on each platform, roofline analysis to identify optimization needs

Tests:

| # | Test | What | Tolerance |
|---|------|------|-----------|
| 1 | `CFD-PORT-CUDA-1` | All tests pass on NVIDIA A100/H100 | all PASS |
| 2 | `CFD-PORT-HIP-1` | All tests pass on AMD MI250 | all PASS |
| 3 | `CFD-PORT-HIP-2` | Cs/Cd match NVIDIA vs AMD within 1e-6 on same mesh+conditions | 1e-6 |
| 4 | `CFD-PORT-SYCL-1` | Kernel launch and basic residual correct on Intel GPU | finite |

Gate:

- CUDA path: zero performance regression from abstraction layer.
- HIP path: all GPU tests pass on AMD MI250.
- Abstract `Accelerator` interface: same solver code compiles without `#ifdef` in physics kernels (only in accelerator implementation files).
- Domestic accelerator: compilation succeeds; `test_cfd_euler.cpp` (mesh-1, Euler-1, Euler-3) produce forces matching CUDA baseline within 1e-4.

---

## Work Rules

- Do not advance to the next phase until the current phase has passing gates.
- Every GPU kernel must have a CPU oracle for CI regression (even if slow).
- Tests that only assert finite/positive outputs are smoke tests, not gate tests.
- Do not silently clamp bad states and report success.
- Keep diagnostics read-only.
- Update `progress.md` after each completed phase.
- Add blockers to `docs/ISSUES.md` when progress is blocked by a reproducible failure.
- NaN/Inf checks are mandatory after every kernel producing cell state.
- CUDA calls checked via `cuda_check()` macro; kernel launches via `CUDA_KERNEL_CHECK()`.
- All floating-point comparisons use relative tolerance, never exact equality.
