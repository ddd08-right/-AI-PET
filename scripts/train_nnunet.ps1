param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [Parameter(Mandatory = $true)][string]$Configuration,
    [Parameter(Mandatory = $true)][string]$Fold,
    [Parameter(Mandatory = $true)][string]$Trainer,
    [string]$Plans,
    [string]$Device = "cuda",
    [string]$NnUNetTrainExe = "nnUNetv2_train",
    [string]$RunManifestOut,
    [int]$Seed
)

$ErrorActionPreference = "Stop"

foreach ($name in @("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
        throw "Required environment variable $name is not set."
    }
}

$env:nnUNet_n_proc_DA = if ($env:nnUNet_n_proc_DA) { $env:nnUNet_n_proc_DA } else { "1" }
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "1" }
$env:MKL_NUM_THREADS = if ($env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS } else { "1" }
$env:OPENBLAS_NUM_THREADS = if ($env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS } else { "1" }

$cmd = @($Dataset, $Configuration, $Fold, "-tr", $Trainer, "-device", $Device)
if (-not [string]::IsNullOrWhiteSpace($Plans)) {
    $cmd += @("-p", $Plans)
}

Write-Host "nnUNet_n_proc_DA=$env:nnUNet_n_proc_DA"
Write-Host "OMP_NUM_THREADS=$env:OMP_NUM_THREADS"
Write-Host "MKL_NUM_THREADS=$env:MKL_NUM_THREADS"
Write-Host "OPENBLAS_NUM_THREADS=$env:OPENBLAS_NUM_THREADS"
Write-Host "COMMAND: $NnUNetTrainExe $($cmd -join ' ')"

& $NnUNetTrainExe @cmd
$exitStatus = $LASTEXITCODE

if ($RunManifestOut) {
    $manifestCmd = @("scripts/create_run_manifest.py", "--output", $RunManifestOut, "--exit-status", "$exitStatus", "--notes", "nnUNet training wrapper")
    if ($PSBoundParameters.ContainsKey("Seed")) {
        $manifestCmd += @("--seed", "$Seed")
    }
    $manifestCmd += @("--", $NnUNetTrainExe) + $cmd
    Write-Host "RUN_MANIFEST_COMMAND: python $($manifestCmd -join ' ')"
    python @manifestCmd
}

exit $exitStatus
