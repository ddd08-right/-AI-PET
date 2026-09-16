param(
    [Parameter(Mandatory = $true)][string]$InputDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$Dataset,
    [Parameter(Mandatory = $true)][string]$Configuration,
    [Parameter(Mandatory = $true)][string]$Fold,
    [string]$Trainer,
    [string]$Plans,
    [string]$Checkpoint,
    [string]$Device = "cuda",
    [string]$NnUNetPredictExe = "nnUNetv2_predict",
    [string]$RunManifestOut
)

$ErrorActionPreference = "Stop"

foreach ($name in @("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
        throw "Required environment variable $name is not set."
    }
}

if (-not (Test-Path -LiteralPath $InputDir)) {
    throw "InputDir does not exist: $InputDir"
}

$cmd = @("-i", $InputDir, "-o", $OutputDir, "-d", $Dataset, "-c", $Configuration, "-f", $Fold, "-device", $Device)
if ($Trainer) { $cmd += @("-tr", $Trainer) }
if ($Plans) { $cmd += @("-p", $Plans) }
if ($Checkpoint) { $cmd += @("-chk", $Checkpoint) }

Write-Host "COMMAND: $NnUNetPredictExe $($cmd -join ' ')"
& $NnUNetPredictExe @cmd
$exitStatus = $LASTEXITCODE

if ($RunManifestOut) {
    $manifestCmd = @("scripts/create_run_manifest.py", "--output", $RunManifestOut, "--exit-status", "$exitStatus", "--notes", "nnUNet inference wrapper", "--", $NnUNetPredictExe) + $cmd
    Write-Host "RUN_MANIFEST_COMMAND: python $($manifestCmd -join ' ')"
    python @manifestCmd
}

exit $exitStatus
