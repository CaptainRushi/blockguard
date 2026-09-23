# BlockGuard — Windows Install Script
# Run in PowerShell: powershell -ExecutionPolicy Bypass -File install.ps1

Write-Host "BlockGuard — Installing..." -ForegroundColor Cyan

# Check Python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Install Python 3.10+ first." -ForegroundColor Red
    exit 1
}
Write-Host "Python: $pythonVersion" -ForegroundColor Green

# Check uv
$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    Write-Host "uv found — syncing..." -ForegroundColor Green
    uv sync
} else {
    Write-Host "uv not found — using pip..." -ForegroundColor Yellow
    pip install -e .
}

# Download spaCy model
Write-Host "Downloading spaCy model..." -ForegroundColor Yellow
python -m spacy download en_core_web_sm

# Create seed data
Write-Host "Creating seed training data..." -ForegroundColor Yellow
uv run python -m blockguard.scripts.train_scorer --seed-only

# Verify
Write-Host ""
Write-Host "Verifying..." -ForegroundColor Yellow
uv run python -m blockguard.cli version
uv run python -m pytest tests/ -v 2>&1 | Select-String "passed|failed"

Write-Host ""
Write-Host "BlockGuard installed! Run: blockguard start" -ForegroundColor Green
