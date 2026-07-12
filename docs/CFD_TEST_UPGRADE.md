# CFD Test Infrastructure Upgrade

> Scope: CFD module only. Non-CFD modules (propulsion, guidance, autopilot, launch, aero table gen) are excluded — kept at current coverage level.
> 
> Approved operating modes:
> - **Write docs** — explicitly requested by user
> - **Plan and build** — active development mode
> - **Write code** — all changes in this document are implementations, not research

## 0. Current State Assessment

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| Code coverage measurement | None | No `--coverage`, no gcov/llvm-cov in any CMakeLists.txt |
| Sanitizer builds | None | No `-fsanitize` in any CMakeLists.txt |
| CI/CD | None | No GitHub Actions, GitLab CI, Jenkins. Only local `scripts/ci.ps1` |
| Fuzz / property-based | None | No fuzzing infrastructure. Zero `fuzz` references in codebase |
| Struct/function unit tests | Weak | `is_valid_primitive()` and `speed_of_sound()` never directly tested. `PrimitiveGradient`/`PrimitiveLimiter` arithmetic never validated in isolation |
| Edge-case robustness tests | Partial | NaN and negative-value tests exist; missing: Inf injection, division-by-zero, out-of-bounds, null-ptr, integer overflow |
| GPU numerical safety | Good | NaN guards, positivity checks, failure snapshot, oracle cross-validation |
| GPU deterministic reduction | Good | Face coloring eliminates atomic non-associativity, byte-level reproducibility verified (CFD-COLOR-4) |
| Test volume | Moderate | ~113 `TEST()` macros across 7 CFD test files |
| Pre-existing broken targets | Known | 8 CPU targets (TestPropulsion etc.) have Eigen include/link errors — excluded from ci.ps1 |

### 0.1 Gap Analysis: What a NaN/Inf/UB test would catch today

| Scenario | Current guard | Test coverage | Would sanitizer catch? |
|----------|---------------|---------------|----------------------|
| Negative rho in solver | `!(rho > 0.0f)` in cfd_solver.cpp:48 | CFD-DIAG-3 injects rho=-1, checks failure | ASAN: no. UBSAN: no |
| NaN in wall distance | `d_wall_distance[i] <= 0.0f → 1e30f` | Not tested | UBSAN: no |
| HLLC denom=0 at symmetric state | `fabs(denom) < 1e-30f → clamp` | CFD-HLLC-NAN tests NaN resilience | UBSAN: no (intentional guard) |
| Out-of-bounds face index | `fi < 0` guard in mesh_validator.cpp | Not tested | ASAN: yes (crash) |
| Integer overflow in cell count | `int cell_count_` (limited to <2B) | Not tested | ASAN: no. UBSAN: yes (with `__builtin_add_overflow`) |
| Device memory out-of-bounds | None in kernels (performance) | Not tested | ASAN: no (GPU). cuda-memcheck: yes |

---

## 1. Phase A — Build & Measurement Infrastructure

Goal: instrument the build so coverage and UB/memory safety can be measured, and CI can enforce regressions.

### 1.1 Coverage build variant

**`CMakeLists.txt` (root)** — add option:

```
option(AEROSIM_COVERAGE "Enable code coverage flags (gcov)" OFF)
if(AEROSIM_COVERAGE)
    add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
    add_link_options(--coverage)
endif()
```

Requires GCC or Clang (not MSVC). Document as Linux/CI-only.

**`scripts/ci.ps1`** — add `coverage` subcommand:

```
"^coverage$" {
    cmake -B build -DAEROSIM_COVERAGE=ON
    cmake --build build --config Debug
    ctest --test-dir build -C Debug -R "Cfd"
    # generate HTML report
    gcovr -r . --html --html-details -o build/coverage/index.html
}
```

Output: `build/coverage/index.html` — per-file line/branch coverage.

### 1.2 Sanitizer build variant

**`CMakeLists.txt` (root)** — add option and per-compiler detection:

```
option(AEROSIM_SANITIZE "Enable address+undefined behavior sanitizers" OFF)
if(AEROSIM_SANITIZE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
        # UBSAN trap on all undefined behavior, not just crash-level
        add_compile_options(-fno-sanitize-recover=all)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # MSVC: ASAN available in VS 16.9+, UBSAN not available
        # Use /RTC1 for basic checks instead
        add_compile_options(/RTC1)
    endif()
endif()
```

**Script** — `scripts/ci.ps1` add `sanitize` subcommand:

```
"^sanitize$" {
    cmake -B build_san -DAEROSIM_SANITIZE=ON
    cmake --build build_san --config Debug
    ctest --test-dir build_san -C Debug -R "Cfd"
    # Non-zero exit on any sanitizer error
}
```

**Known limitation**: CUDA GPU code cannot be sanitized by host ASAN/UBSAN. Only CPU oracle code (`missile_cpu` + CPU test executables) is covered. GPU memory errors must be caught by:
- `cuda-memcheck` (manual, slow)
- Device-side `d_failed` propagation (already implemented)
- Deterministic coloring (already implemented, eliminates atomic race UB)

### 1.3 CI pipeline stub

**`.github/workflows/cfd.yml`** — GitHub Actions workflow:

```yaml
name: CFD
on: [push, pull_request]
jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: .github/actions/setup-cuda  # install CUDA toolkit
      - run: cmake -B build -DAEROSIM_COVERAGE=ON -DAEROSIM_USE_CGNS=OFF
      - run: cmake --build build --config Debug -j$(nproc)
      - run: ctest --test-dir build -C Debug -R "Cfd"
      - run: gcovr -r . --html --html-details -o build/coverage/index.html
      - uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: build/coverage/
  sanitize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cmake -B build_san -DAEROSIM_SANITIZE=ON -DAEROSIM_USE_CGNS=OFF
      - run: cmake --build build_san --config Debug -j$(nproc)
      - run: ctest --test-dir build_san -C Debug -R "Cfd"
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - run: cmake -B build
      - run: cmake --build build --config Release
      - run: ctest --test-dir build -C Release -R "Cfd"
```

### 1.4 Pre-commit hook fix

**`.git/hooks/pre-commit`** — the `#!/usr/bin/env pwsh` shebang fails on Windows (Git Bash cannot resolve it). Fix: detect shell and fall back:

```
#!/bin/sh
# pre-commit: delegate to pwsh if available, else skip checks
if command -v pwsh &> /dev/null; then
    exec pwsh -NoProfile -File "$(dirname "$0")/pre-commit.ps1"
fi
# No pwsh available — skip hook checks
exit 0
```

Rename existing hook content to `.git/hooks/pre-commit.ps1`. The `install-hooks` subcommand in `ci.ps1` generates both files.

### 1.5 Tasks

- [x] `CMakeLists.txt`: add `AEROSIM_COVERAGE` option + compile/link flags
- [x] `CMakeLists.txt`: add `AEROSIM_SANITIZE` option + per-compiler flags
- [x] `scripts/ci.ps1`: add `coverage` and `sanitize` subcommands
- [x] `.github/workflows/cfd.yml`: create CI pipeline stub (3 jobs)
- [x] `.git/hooks/pre-commit` → `.git/hooks/pre-commit` (sh wrapper) + `.git/hooks/pre-commit.ps1` (actual checks)
- [x] `scripts/ci.ps1`: update `install-hooks` to generate both files

### 1.6 Files Changed

| File | Action | Content |
|------|--------|---------|
| `CMakeLists.txt` | MODIFY | Add `AEROSIM_COVERAGE` and `AEROSIM_SANITIZE` options |
| `scripts/ci.ps1` | MODIFY | Add `coverage`, `sanitize` subcommands; fix `install-hooks` for dual-file hook |
| `.github/workflows/cfd.yml` | NEW | CI pipeline (coverage + sanitize + build) |
| `.git/hooks/pre-commit` | MODIFY | POSIX shell wrapper delegating to `pre-commit.ps1` |
| `.git/hooks/pre-commit.ps1` | NEW | Actual hook logic (extracted from current `pre-commit`) |

### 1.7 Gate

- `cmake -B build -DAEROSIM_COVERAGE=ON` configures without error.
- `cmake -B build_san -DAEROSIM_SANITIZE=ON` configures without error.
- `ctest -R Cfd` under sanitizer build: 0 runtime errors reported by ASAN/UBSAN.
- `scripts/ci.ps1 coverage` produces `build/coverage/index.html` with per-file line coverage.
- `scripts/ci.ps1 sanitize` runs all CFD tests under ASAN+UBSAN, exit code 0.

---

## 2. Phase B — Unit & Regression Test Expansion

Goal: bring untested core functions under direct test, and add missing edge-case coverage.

### 2.1 New test file: `tests/cfd/test_cfd_state.cpp`

Target functions in `include/aero/cfd/cfd_state.hpp`:

| Function | Tests | Count |
|----------|-------|-------|
| `is_valid_primitive` | Valid state (rho>0,p>0,finite); rho=0 edge; rho<0; p=0; p<0; NaN rho; NaN u; Inf rho; mixed NaN/valid | 9 |
| `speed_of_sound` | gamma=1.4; gamma=1.0 (isothermal); gamma_extreme=1.66 (monatomic); rho=0 (division guard); p=0; negative p; NaN p; NaN rho | 8 |
| `primitive_to_conservative` | Existing coverage sufficient (36 refs across test files). No new tests needed. | 0 |
| `conservative_to_primitive` | Compare against `primitive_to_conservative` round-trip; inject NaN rho_E; inject negative rho_E | 3 |
| `hllc_flux` | Symmetric states (PH2-RA-H1 guard); vacuum left/right; sonic point (PH2-RA-H2 guard); strong shock (M=10); subsonic; supersonic | 6 |

Total: 26 new test cases in `test_cfd_state.cpp`.

### 2.2 Robustness injection tests

New test section in `tests/cfd/test_cfd_gpu.cpp`:

| Test | Category | What | Expected |
|------|----------|------|----------|
| `CFD-ROBUST-INF-RHO` | Inf injection | Set `d_q[i*NVAR+0] = INFINITY` on one cell, run 1 iter | Solver fails with `d_failed=1` |
| `CFD-ROBUST-INF-P` | Inf injection | Set `d_q[i*NVAR+4] = INFINITY` | Solver fails |
| `CFD-ROBUST-NAN-VEL` | NaN injection | Set `d_q[i*NVAR+1] = NAN` | Solver fails |
| `CFD-ROBUST-ZERO-VOLUME` | Degenerate mesh | Face area = 0 for one face | Residual is finite (no division by zero) |
| `CFD-ROBUST-NEGATIVE-WALL-DIST` | Negative metric | Inject -1 wall distance | SA source finite (uses 1e30 fallback) |
| `CFD-ROBUST-OOB-INDEX` | Out-of-bounds | Solver with `cell_count < mesh.n_cells` | Not applicable (cannot test without corrupting device memory). Document as "covered by cuda-memcheck" |

### 2.3 CUDA memory safety

GPU out-of-bounds writes cannot be tested by host code — they crash the CUDA context. Testing strategy:

- **CI gate**: `cuda-memcheck build/bin/Release/TestCfdGpu.exe` must report 0 errors.
- **Manual check**: `cuda-memcheck --leak-check full` before each Phase release.
- **Documented as**: "GPU memory safety verified by cuda-memcheck, not by TEST macro."

Add a `scripts/run_cuda_memcheck.ps1` helper:

```powershell
param([string]$Test = "TestCfdGpu")
cuda-memcheck "build/bin/Release/$Test.exe"
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] cuda-memcheck detected errors"; exit 1 }
```

### 2.4 Fuzz integration (GPU kernel parameter space)

Add `tests/cfd/test_cfd_fuzz_gpu.cpp` with randomized parameter sweeps:

| Test | Parameter | Range | Iterations | Check |
|------|-----------|-------|------------|-------|
| FUZZ-MACH-1 | CFL | 0.01 to 10.0 (log scale) | 20 | finite L2, no failure |
| FUZZ-MACH-2 | `Re` | 1e3 to 1e8 (log scale) | 20 | finite forces |
| FUZZ-MACH-3 | `gamma` | 1.05 to 1.67 | 20 | finite L2 |
| FUZZ-MACH-4 | `alpha` | -30 to +30 deg | 20 | CY ≈ 0 symmetry |
| FUZZ-RANS-1 | `nu_tilde_ratio` | 0.001 to 10.0 | 10 | finite L2 |
| FUZZ-RANS-2 | `wall_temperature` | 0.5 to 5.0 (ratio to T_inf) | 10 | finite heat flux |

Each fuzz test reports:
- Parameter value
- L2 norm
- Force coefficients
- Convergence status
- Failure: "FAIL at param=X: L2=NaN" or "FAIL at param=X: force=NaN"

No hard pass/fail on convergence — only NaN/finite check. Documented as "parameter space exploration, not convergence guarantees."

### 2.5 Tasks

- [x] Create `tests/cfd/test_cfd_state.cpp` with 26 test cases for `is_valid_primitive`, `speed_of_sound`, `conservative_to_primitive` round-trip, `hllc_flux` edge cases
- [x] Register `test_cfd_state` target in `tests/CMakeLists.txt` (links `missile_cpu`, no CUDA needed)
- [x] Add 5 robustness injection tests (`CFD-ROBUST-*`) to `test_cfd_gpu.cpp`
- [x] Create `tests/cfd/test_cfd_fuzz_gpu.cpp` with randomized parameter sweeps
- [x] Register `test_cfd_fuzz_gpu` target in `tests/CMakeLists.txt`
- [x] Create `scripts/run_cuda_memcheck.ps1`
- [x] Run full suite: all new tests PASS at stated tolerances

### 2.6 Files Changed

| File | Action | Content |
|------|--------|---------|
| `tests/cfd/test_cfd_state.cpp` | NEW | 26 unit tests for `cfd_state.hpp` functions |
| `tests/cfd/test_cfd_gpu.cpp` | MODIFY | Add 5 `CFD-ROBUST-*` injection tests |
| `tests/cfd/test_cfd_fuzz_gpu.cpp` | NEW | 6 randomized parameter sweep tests |
| `tests/CMakeLists.txt` | MODIFY | Register `test_cfd_state` + `test_cfd_fuzz_gpu` |
| `scripts/run_cuda_memcheck.ps1` | NEW | CUDA memory safety runner |
| `src/aero/CMakeLists.txt` | MODIFY | Ensure `missile_cpu` exports symbols needed by `test_cfd_state` |

### 2.7 Gate

- `tests/cfd/test_cfd_state.cpp`: 26/26 tests PASS.
- `CFD-ROBUST-INF-RHO`, `CFD-ROBUST-INF-P`, `CFD-ROBUST-NAN-VEL`: solver correctly detects failure (`d_failed=1`).
- `CFD-ROBUST-ZERO-VOLUME`, `CFD-ROBUST-NEGATIVE-WALL-DIST`: solver produces finite residual (no NaN propagation).
- `FUZZ-*` tests: all parameter combinations produce finite L2 and forces. Zero NaN/Inf in output.
- `cuda-memcheck build/bin/Release/TestCfdGpu.exe`: 0 errors.
- No regression in existing 57 tests.

---

## 3. Phase C — Coverage Hardening & CI Enforcement

Goal: use coverage data to identify untested paths, then close gaps. Enforce minimum coverage in CI.

### 3.1 Coverage-guided gap analysis

After Phase A produces `build/coverage/index.html`:

1. Identify functions/blocks with < 80% line coverage.
2. For each uncovered block, decide: write test, or mark as "defensive guard, not reachable via public API."
3. Track in `docs/COVERAGE_GAPS.md`:

```markdown
### Uncovered: src/aero/cfd/viscous.cpp:42-48
Function: `sutherland_mu(T)` — cold-wall branch (T < 50K)
Verdict: Test. Add `CFD-COVERAGE-SUTHERLAND-COLD` with T=10K.
```

### 3.2 Coverage regression gate

**`.github/workflows/cfd.yml`** — add coverage threshold job:

```yaml
coverage-threshold:
  runs-on: ubuntu-latest
  steps:
    - ... (build + ctest as above)
    - run: |
        gcovr -r . --fail-under-line=70 --fail-under-branch=60
```

If line coverage drops below 70% or branch below 60%, CI fails.

Initial thresholds are set low (accounting for GPU code that gcov cannot measure). Revised upward each quarter.

### 3.3 Sanitizer regression gate

**`.github/workflows/cfd.yml`** — add sanitizer job with `-fno-sanitize-recover=all`:

```yaml
sanitize:
  runs-on: ubuntu-latest
  steps:
    - ...
    - run: ctest --test-dir build_san -C Debug -R "Cfd" --output-on-failure
```

Zero tolerance: any ASAN/UBSAN error is a CI failure.

### 3.4 Tasks

- [x] Run coverage build on all CFD tests, generate `docs/COVERAGE_GAPS.md` with initial uncovered block list
- [x] Write tests for uncovered blocks (priority: core arithmetic, boundary conditions, reconstruction)
- [x] Add coverage threshold gate to CI pipeline
- [x] Add sanitizer gate to CI pipeline
- [ ] Set up cron job (weekly) to regenerate coverage report and detect regressions

### 3.5 Files Changed

| File | Action | Content |
|------|--------|---------|
| `docs/COVERAGE_GAPS.md` | NEW | Coverage gap tracking (initially generated, manually curated) |
| `.github/workflows/cfd.yml` | MODIFY | Add coverage-threshold and sanitizer gate jobs |
| `tests/cfd/test_cfd_gpu.cpp` | MODIFY | Add tests for uncovered blocks (specifics determined by gap analysis) |

### 3.6 Gate

- CI pipeline has 4 jobs: `build` (Windows), `coverage` (Linux, generates HTML), `coverage-threshold` (Linux, line≥70%, branch≥60%), `sanitize` (Linux, zero ASAN/UBSAN errors).
- Coverage report visible as CI artifact.
- All 57 existing tests + Phase B new tests still PASS.

---

## 4. Timeline & Dependencies

```
Phase A (infrastructure)     → enables Phase C (coverage measurement)
  ↓
Phase B (unit/robustness)    → independent, can run in parallel with A
  ↓
Phase C (coverage hardening) → depends on A + B
```

| Phase | Estimated effort | Parallelizable |
|-------|-----------------|----------------|
| A | 2 sessions | No (build system changes) |
| B | 3 sessions | Yes with A (test content, no build deps) |
| C | 2 sessions | No (depends on coverage data from A) |

---

## 5. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| `--coverage` changes compiler behavior (inlining, optimizations) | Coverage data may not reflect Release behavior | Coverage measured in Debug build; final validation always in Release |
| ASAN adds 2-3x runtime overhead | CI pipeline slow | Limit sanitizer job to CFD CPU tests only; exclude GPU (cannot be sanitized) |
| MSVC lacks `--coverage` and full ASAN | Windows CI cannot enforce coverage/sanitizer parity | Windows CI only validates build + test PASS. Coverage/sanitizer gates run on Linux |
| GPU kernel out-of-bounds cannot be tested by host | False sense of safety | `cuda-memcheck` script + documented limitation. "GPU memory safety: verified by cuda-memcheck" |
| Coverage tooling breaks with CUDA `.cu` files | gcov cannot instrument device code | Coverage measured only on `.cpp` files (CPU oracle path). GPU code coverage is 0% by design — document as expected |
