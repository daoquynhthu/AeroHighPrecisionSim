param(
    [string]$Test = "TestCfdGpu",
    [string]$BuildDir = "build"
)

$TestPath = Join-Path $BuildDir "bin" "Release" "$Test.exe"
if (-not (Test-Path $TestPath)) {
    Write-Host "[FAIL] Test executable not found: $TestPath"
    exit 1
}

Write-Host "[INFO] Running cuda-memcheck on $Test ..."
& cuda-memcheck $TestPath 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] cuda-memcheck detected errors (exit code: $LASTEXITCODE)"
    exit 1
}

Write-Host "[PASS] cuda-memcheck: 0 errors"
exit 0
