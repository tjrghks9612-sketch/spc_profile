$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

Write-Host "Creating virtual environment in .venv"
$created = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -eq 0) {
        $created = $true
    } else {
        & py -3 -m venv .venv
        if ($LASTEXITCODE -eq 0) {
            $created = $true
        }
    }
} else {
    Invoke-Checked python -m venv .venv
    $created = $true
}

if (-not $created) {
    throw "Could not create a Python virtual environment. Install 64-bit Python 3.12 or newer and retry."
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "Upgrading packaging tools"
Invoke-Checked $python -m pip install --upgrade pip setuptools wheel

Write-Host "Removing stale Qt bindings"
& $python -m pip uninstall -y PySide6 PySide6_Addons PySide6_Essentials shiboken6

Write-Host "Installing pinned dependencies"
Invoke-Checked $python -m pip install --no-cache-dir --force-reinstall -r requirements.txt

Write-Host "Running Qt diagnostic"
Invoke-Checked $python tools\diagnose_qt.py

Write-Host ""
Write-Host "Install complete. Start the app with:"
Write-Host "  .\.venv\Scripts\python.exe main.py"
