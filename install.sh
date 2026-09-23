#!/bin/bash
# BlockGuard — Linux/Mac Install Script
# Usage: bash install.sh

echo "╔══════════════════════════════════════════════╗"
echo "║         BlockGuard — Installer                 ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3.10+ not found. Install it first."
    exit 1
fi
echo "Python: $(python3 --version)"

# Check uv
if command -v uv &> /dev/null; then
    echo "uv found — using uv sync"
    uv sync
    if [ $? -ne 0 ]; then
        echo "ERROR: uv sync failed. Run 'uv sync' manually."
        exit 1
    fi
else
    echo "uv not found — using pip install"
    pip install -e .
    if [ $? -ne 0 ]; then
        echo "ERROR: pip install failed. Run 'pip install -e .' manually."
        exit 1
    fi
fi

# Download spaCy model
echo ""
echo "Downloading spaCy model..."
python3 -m spacy download en_core_web_sm

# Create seed data
echo "Creating seed training data..."
uv run python -m blockguard.scripts.train_scorer --seed-only

# Verify
echo ""
echo "Verifying install..."
uv run python -m blockguard.cli version
uv run python -m pytest tests/ -v 2>&1 | tail -3

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  BlockGuard installed successfully!            ║"
echo "║                                             ║"
echo "║  Start server: blockguard start              ║"
echo "║  Status:     blockguard status               ║"
echo "║  Tests:      blockguard test                 ║"
echo "╚══════════════════════════════════════════════╝"
