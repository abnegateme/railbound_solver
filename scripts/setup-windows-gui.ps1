[CmdletBinding()]
param(
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $VenvPath = Join-Path $projectRoot ".venv-gui"
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $pythonLauncher) {
        throw "Python Launcher 'py' was not found. Install Python 3.14 from python.org."
    }
    $pythonCommand = $pythonLauncher.Source
    & $pythonCommand "-3.14" "-m" "venv" $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment at $VenvPath."
    }
}

$package = "${projectRoot}[gui]"
& $venvPython -m pip install --upgrade --force-reinstall $package
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the Railbound GUI environment."
}

Write-Host "Railbound GUI environment is ready: $venvPython"
Write-Host "Run: & '$venvPython' -m railbound calibrate --interactive --image data\\Railbound-1-13.PNG"
