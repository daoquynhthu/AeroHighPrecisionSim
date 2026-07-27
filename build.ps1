param(
    [ValidateSet('Debug','Release','RelWithDebInfo','MinSizeRel')]
    [string]$Config = 'Debug',
    [string]$Target = '',
    [switch]$Clean,
    [switch]$Rebuild,
    [switch]$SkipTests,
    [switch]$Quiet,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Usage: .\build.ps1 [[-Config] <Debug|Release|...>] [-Target <name>] [-Clean] [-SkipTests] [-Quiet]

  -Config       Build configuration (default: Debug)
  -Target       Build only this target (e.g. TestThermoDb, missile_lib)
  -Clean        Delete build/ and reconfigure from scratch
  -Rebuild      Same as -Clean
  -SkipTests    Do not build test targets
  -Quiet        Suppress CUDA nvcc warnings (adds -w to CMAKE_CUDA_FLAGS)
  -Help         Show this help

Examples:
  .\build.ps1                          # Debug, full build
  .\build.ps1 -Config Release          # Release, full build
  .\build.ps1 -Target TestThermoDb     # Debug, single test target
  .\build.ps1 -Target missile_lib      # Debug, library only
  .\build.ps1 -SkipTests -Quiet        # Debug, no tests, quiet CUDA
"@
    exit 0
}

$ErrorActionPreference = 'Continue'
$rootDir = $PSScriptRoot
$buildDir = Join-Path $rootDir 'build'
$logFile = Join-Path $rootDir "build_$($Config.ToLower()).log"

function Write-Banner { param([string]$Msg); Write-Host "`n=== $Msg ===" -ForegroundColor Cyan }
function Write-Info   { param([string]$Msg); Write-Host "  $Msg" -ForegroundColor Gray }
function Write-OK     { param([string]$Msg); Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Fail   { param([string]$Msg); Write-Host "  [FAIL] $Msg" -ForegroundColor Red; exit 1 }

# ─── Find VS 2022 ──────────────────────────────────────────
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
)
$vsPath = $null
foreach ($cand in $vsPaths) {
    $bat = Join-Path $cand 'VC\Auxiliary\Build\vcvars64.bat'
    if (Test-Path -LiteralPath $bat) { $vsPath = $cand; break }
}
if (-not $vsPath) { Write-Fail "Visual Studio 2022 not found" }
Write-OK "Visual Studio 2022: $vsPath"

# ─── Init MSVC x64 environment ─────────────────────────────
$vcvarsBat = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
Write-Info "Initializing MSVC x64 environment..."
& cmd /c "`"$vcvarsBat`" > nul 2>&1 && set" | ForEach-Object {
    $idx = $_.IndexOf('=')
    if ($idx -gt 0) { Set-Item -Path "env:$($_.Substring(0,$idx))" -Value $_.Substring($idx+1) -ErrorAction SilentlyContinue }
}
Write-OK "MSVC x64 environment initialized"

# ─── Detect Ninja ──────────────────────────────────────────
$ninjaPath = (Get-Command ninja -ErrorAction SilentlyContinue).Source
$generator = if ($ninjaPath) { 'Ninja' } else { 'Visual Studio 17 2022' }
$genLabel = if ($generator -eq 'Ninja') { "Ninja ($ninjaPath)" } else { $generator }
Write-OK "Generator: $genLabel"

# ─── Clean / Rebuild ────────────────────────────────────────
if ($Clean -or $Rebuild) {
    Write-Banner "Cleaning build directory"
    if (Test-Path -LiteralPath $buildDir) { Remove-Item -LiteralPath $buildDir -Recurse -Force; Write-OK "Deleted $buildDir" }
}

if (-not (Test-Path -LiteralPath $buildDir)) { New-Item -ItemType Directory -Path $buildDir -Force | Out-Null }

# ─── Decide whether to re-configure ────────────────────────
$cmakeCache = Join-Path $buildDir 'CMakeCache.txt'
$cmakeLists = Join-Path $rootDir 'CMakeLists.txt'
$stampFile = Join-Path $buildDir '.build_stamp.txt'
$stamp = @{ SkipTests = $SkipTests; Config = $Config; Generator = $generator; Quiet = $Quiet }
$stampJson = $stamp | ConvertTo-Json -Compress

$needConfigure = $true
if ((Test-Path -LiteralPath $cmakeCache) -and -not ($Clean -or $Rebuild)) {
    $cacheTime = (Get-Item $cmakeCache).LastWriteTime
    $srcTime = (Get-Item $cmakeLists).LastWriteTime
    $stampChanged = $true
    if (Test-Path -LiteralPath $stampFile) {
        $prevStamp = (Get-Content -LiteralPath $stampFile -Raw).Trim()
        if ($prevStamp -eq $stampJson) { $stampChanged = $false }
    }
    if ($srcTime -le $cacheTime -and -not $stampChanged) { $needConfigure = $false; Write-OK "CMake cache is current" }
}

# ─── Configure ──────────────────────────────────────────────
if ($needConfigure) {
    Write-Banner "Configuring ($generator, $Config)"
    $cmakeArgs = "-B", $buildDir, "-G", $generator, "-DCMAKE_BUILD_TYPE=$Config"
    if ($generator -eq 'Visual Studio 17 2022') { $cmakeArgs += '-A', 'x64' }
    if ($SkipTests) { $cmakeArgs += '-DBUILD_TESTING=OFF' } else { $cmakeArgs += '-DBUILD_TESTING=ON' }

    if ($Quiet) {
        $cmakeArgs += '-DCMAKE_CUDA_FLAGS=--expt-relaxed-constexpr;-w'
    }

    & cmake @cmakeArgs 2>&1 | Tee-Object -FilePath $logFile
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { Write-Fail "Configure failed (see $logFile)" }

    $stampJson | Out-File -LiteralPath $stampFile -Encoding utf8 -Force
    Write-OK "Configured successfully"
} else {
    Write-OK "Skipping configure"
}

# ─── Build ──────────────────────────────────────────────────
$cpuCores = [Math]::Max(1, [int]$env:NUMBER_OF_PROCESSORS)
$targetLabel = if ($Target) { " target=$Target" } else { "" }
Write-Banner "Building ($generator, $Config, $cpuCores cores$targetLabel)"

$buildArgs = @("--build", $buildDir, "--config", $Config, "--parallel", $cpuCores)
if ($Target) { $buildArgs += "--target", $Target }
& cmake @buildArgs 2>&1 | Tee-Object -FilePath $logFile -Append
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) { Write-Fail "Build failed (see $logFile)" }

Write-OK "Build succeeded ($generator, $Config$targetLabel)"
Write-Info "Log: $logFile"
