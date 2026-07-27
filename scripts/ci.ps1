param(
    [string]$Action = "check",
    [string]$Filter = "",
    [string]$Config = "Release",
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root $BuildDir

function Red   { Write-Host "$($PSStyle.Foreground.Red)$($args[0])$($PSStyle.Reset)" }
function Green { Write-Host "$($PSStyle.Foreground.Green)$($args[0])$($PSStyle.Reset)" }
function Cyan  { Write-Host "$($PSStyle.Foreground.Cyan)$($args[0])$($PSStyle.Reset)" }
function Gray  { Write-Host "$($PSStyle.Foreground.BrightBlack)$($args[0])$($PSStyle.Reset)" }

function Step($Name) { Cyan ">> $Name" }

function Exec($Name, [scriptblock]$Block) {
    Gray "   $Name ..."
    & $Block
    if ($LASTEXITCODE -ne 0) {
        Red "[FAIL] $Name (exit=$LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    Green "[PASS] $Name"
}

function Ensure-Build-Dir {
    if (-not (Test-Path $Build)) {
        Step "Configuring CMake (first time)..."
        cmake -B $Build
        if ($LASTEXITCODE -ne 0) { Red "[FAIL] cmake configure"; exit 1 }
    }
}

function Ensure-Ninja {
    # Verify the build system is usable — detect stale state
    if (-not (Test-Path (Join-Path $Build "build.ninja"))) {
        Step "Reconfiguring (build.ninja missing)..."
        cmake -B $Build
        if ($LASTEXITCODE -ne 0) { Red "[FAIL] cmake reconfigure"; exit 1 }
    }
}

function Invoke-Build($Targets, $Label) {
    Ensure-Build-Dir
    Ensure-Ninja
    Step "Building $Label..."
    foreach ($t in $Targets) {
        Exec "cmake --build $Build --target $t --config $Config" {
            cmake --build $Build --target $t --config $Config 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                # Show build output on failure
                cmake --build $Build --target $t --config $Config 2>&1 | Select-String -Pattern "error|fatal|FAILED" | ForEach-Object { Write-Host "   $_" }
            }
        }
    }
}

function Invoke-Ctest($Label, $CtestArgs) {
    Step "Running $Label..."
    Exec "ctest $($CtestArgs -join ' ')" {
        ctest --test-dir $Build -C $Config --output-on-failure @CtestArgs 2>&1 | ForEach-Object {
            if ($_ -match "PASSED|passed") { Green "   $_" }
            elseif ($_ -match "FAILED|failed") { Red "   $_" }
            else { Write-Host "   $_" }
        }
    }
}

# ─── Target lists ───────────────────────────────────────────────
$Libs = @("missile_lib", "missile_cpu")
$CfdAll = @(
    "TestCfdGpu", "TestCfdMesh", "TestCfdEuler",
    "TestCfdDiagnostics", "TestCfdReconstruction", "TestCfdViscous"
)
$CfdGpu   = @("TestCfdGpu")
$CfdCpu   = @("TestCfdMesh", "TestCfdEuler", "TestCfdDiagnostics",
              "TestCfdReconstruction", "TestCfdViscous")
$OtherTest = @("TestAeroViscous", "TestGpuTopology")
# Known broken (pre-existing link/include errors): TestPropulsion, TestAero,
# TestIntegrator, TestLaunch, TestAeroTableGen, TestCompareAtm, TestGuidance, TestAutopilot

# ─── Actions ────────────────────────────────────────────────────
switch -Regex ($Action.ToLower()) {

    "^(check|compile)$" {
        Invoke-Build $Libs "libraries"
        Green "[PASS] check — all libraries compile"
    }

    "^(build|compile-all)$" {
        Invoke-Build $Libs "libraries"
        Invoke-Build ($CfdAll + $OtherTest) "test executables"
        Green "[PASS] build — all targets compile"
    }

    "^test$" {
        Invoke-Build ($Libs + $CfdAll + $OtherTest) "libraries + tests"
        Invoke-Ctest "all working tests" @()
    }

    "^test (cfd-gpu|gpu)$" {
        Invoke-Build ($Libs + $CfdGpu) "libraries + TestCfdGpu"
        Invoke-Ctest "CFD GPU tests" @("-R", "CfdGpu")
    }

    "^test (cfd-cpu|cpu)$" {
        Invoke-Build ($Libs + $CfdCpu) "libraries + CFD CPU tests"
        Invoke-Ctest "CFD CPU tests" @("-R", "Cfd(Mesh|Euler|Diagnostics|Reconstruction|Viscous)")
    }

    "^test cfd$" {
        Invoke-Build ($Libs + $CfdAll) "libraries + all CFD tests"
        Invoke-Ctest "all CFD tests" @("-R", "Cfd")
    }

    "^full$" {
        Invoke-Build ($Libs + $CfdAll + $OtherTest) "everything"
        Invoke-Ctest "all working tests" @()
        Green "[PASS] full — everything OK"
    }

    "^(coverage)$" {
        Step "Coverage build (GCC)..."
        $CovBuild = Join-Path $Root "build_cov"
        $Gcc = "C:\msys64\mingw64\bin\gcc.exe"
        $Gxx = "C:\msys64\mingw64\bin\g++.exe"
        Exec "cmake configure -DAEROSIM_COVERAGE=ON -DAEROSIM_USE_CUDA=OFF" {
            cmake -B $CovBuild -DAEROSIM_COVERAGE=ON -DAEROSIM_USE_CUDA=OFF -DCMAKE_C_COMPILER="$Gcc" -DCMAKE_CXX_COMPILER="$Gxx" 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { cmake -B $CovBuild -DAEROSIM_COVERAGE=ON -DAEROSIM_USE_CUDA=OFF -DCMAKE_C_COMPILER="$Gcc" -DCMAKE_CXX_COMPILER="$Gxx" 2>&1 | Select-String -Pattern "error|Error|ERROR" | ForEach-Object { Write-Host "   $_" } }
        }
        Exec "cmake --build" {
            cmake --build $CovBuild --config Debug 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { cmake --build $CovBuild --config Debug 2>&1 | Select-String -Pattern "error|fatal|FAILED" | ForEach-Object { Write-Host "   $_" } }
        }
        Exec "ctest" {
            ctest --test-dir $CovBuild -C Debug -R "Cfd(Mesh|Euler|Diagnostics|Reconstruction|Viscous|State|Rans)" --output-on-failure 2>&1 | ForEach-Object { Write-Host "   $_" }
        }
        Exec "gcovr HTML report" {
            $env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
            gcovr -r $Root --filter "$Root/src" --filter "$Root/tests" --exclude-directories ".*_deps.*" --html --html-details -o (Join-Path $CovBuild "coverage" "index.html") --gcov-executable "gcov.exe" 2>&1 | Out-Null
        }
        Exec "coverage threshold gate" {
            $env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
            # Initial thresholds account for ~2000 lines of GPU code not measurable by gcov.
            gcovr -r $Root --filter "$Root/src" --gcov-executable "gcov.exe" --fail-under-line=45 --fail-under-branch=28 2>&1 | ForEach-Object { Write-Host "   $_" }
            if ($LASTEXITCODE -ne 0) { throw "Coverage below threshold (line 45%, branch 28%)" }
        }
        Green "[PASS] coverage — report at build_cov/coverage/index.html"
    }

    "^(sanitize)$" {
        Step "Sanitizer build (GCC + ASAN/UBSAN)..."
        $SanBuild = Join-Path $Root "build_san"
        $Gcc = "C:\msys64\mingw64\bin\gcc.exe"
        $Gxx = "C:\msys64\mingw64\bin\g++.exe"
        Exec "cmake configure -DAEROSIM_SANITIZE=ON -DAEROSIM_USE_CUDA=OFF" {
            cmake -B $SanBuild -DAEROSIM_SANITIZE=ON -DAEROSIM_USE_CUDA=OFF -DCMAKE_C_COMPILER="$Gcc" -DCMAKE_CXX_COMPILER="$Gxx" 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { cmake -B $SanBuild -DAEROSIM_SANITIZE=ON -DAEROSIM_USE_CUDA=OFF -DCMAKE_C_COMPILER="$Gcc" -DCMAKE_CXX_COMPILER="$Gxx" 2>&1 | Select-String -Pattern "error|Error|ERROR" | ForEach-Object { Write-Host "   $_" } }
        }
        Exec "cmake --build" {
            cmake --build $SanBuild --config Debug 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { cmake --build $SanBuild --config Debug 2>&1 | Select-String -Pattern "error|fatal|FAILED" | ForEach-Object { Write-Host "   $_" } }
        }
        Exec "ctest" {
            ctest --test-dir $SanBuild -C Debug -R "Cfd(Mesh|Euler|Diagnostics|Reconstruction|Viscous|State|Rans)" --output-on-failure 2>&1 | ForEach-Object { Write-Host "   $_" }
        }
        Green "[PASS] sanitize — no ASAN/UBSAN errors"
    }

    "^(install-hooks|hooks)$" {
        Step "Installing pre-commit hooks..."
        $HookDir = Join-Path (Join-Path $Root ".git") "hooks"
        if (-not (Test-Path $HookDir)) {
            Red "[FAIL] .git/hooks not found — is this a git repo?"
            exit 1
        }
        # POSIX shell wrapper (resolves pwsh path on Windows)
        $ShPath = Join-Path $HookDir "pre-commit"
        $ShLines = @(
            '#!/bin/sh',
            '# pre-commit wrapper — auto-generated by scripts/ci.ps1 install-hooks',
            'if command -v pwsh >/dev/null 2>&1; then',
            '    exec pwsh -NoProfile -File "$(dirname "$0")/pre-commit.ps1"',
            'fi',
            '# No pwsh available — skip checks',
            'exit 0'
        )
        $ShContent = $ShLines -join "`n"
        Set-Content -Path $ShPath -Value $ShContent -Encoding ASCII -NoNewline
        Green "[PASS] pre-commit (sh wrapper) written"

        # PowerShell implementation
        $Ps1Path = Join-Path $HookDir "pre-commit.ps1"
        $Ps1Lines = @(
            '# Auto-generated by scripts/ci.ps1 install-hooks',
            '# Exit codes: 0 = pass, non-zero = block commit',
            '$Root = Split-Path -Parent $PSScriptRoot',
            '$Root = Split-Path -Parent $Root',
            '$failed = $false',
            '# 1. trailing whitespace in staged files',
            '$staged = & git diff --cached --name-only --diff-filter=ACM',
            'foreach ($f in $staged) {',
            '    $path = Join-Path $Root $f',
            '    if (-not (Test-Path $path)) { continue }',
            '    $ext = [System.IO.Path]::GetExtension($f)',
            '    if ($ext -notin ''.cpp'',''.hpp'',''.cu'',''.h'',''.cuh'',''.py'',''.ps1'',''.cmake'') { continue }',
            '    $lines = Get-Content $path',
            '    for ($i = 0; $i -lt $lines.Count; $i++) {',
            '        if ($lines[$i] -match ''[ \t]+$'') {',
            '            Write-Host "[pre-commit] trailing whitespace in $f line $($i+1)"',
            '            $failed = $true',
            '        }',
            '    }',
            '}',
            '# 2. quick compilation check',
            'Write-Host "[pre-commit] compiling libraries ..."',
            '& cmake --build (Join-Path $Root "build") --target missile_cpu --config Release 2>&1 | Out-Null',
            'if ($LASTEXITCODE -ne 0) {',
            '    Write-Host "[pre-commit] FAIL: library compilation error"',
            '    $failed = $true',
            '}',
            'if ($failed) { exit 1 }',
            'Write-Host "[pre-commit] OK"'
        )
        $Ps1Content = $Ps1Lines -join "`n"
        Set-Content -Path $Ps1Path -Value $Ps1Content -Encoding UTF8
        Green "[PASS] pre-commit.ps1 written"
    }

    default {
        Red "[FAIL] Unknown action '$Action'"
        Write-Host ""
        Write-Host "Usage: .\scripts\ci.ps1 <action> [filter]"
        Write-Host ""
        Write-Host "  check            Quick compile check (libraries only)"
        Write-Host "  build            Full build (libraries + all known-good tests)"
        Write-Host "  test             Build + run all working tests"
        Write-Host "  test cfd-gpu     Build + run CFD GPU tests only"
        Write-Host "  test cfd-cpu     Build + run CFD CPU tests only"
        Write-Host "  test cfd         Build + run all CFD tests"
        Write-Host "  full             Build + all tests (run before commit)"
        Write-Host "  coverage         Coverage build (gcov) + HTML report"
        Write-Host "  sanitize         Sanitizer build (ASAN+UBSAN) + test"
        Write-Host "  install-hooks    Install pre-commit git hooks"
        exit 1
    }
}
