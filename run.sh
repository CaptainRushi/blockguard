#!/bin/bash
# BlockGuard — Start the proxy server
# Usage: bash run.sh [port] [provider]
#   provider: openai (default), anthropic, vllm
#
# Required env vars for LLM:
#   OPENAI_API_KEY       (for openai provider)
#   ANTHROPIC_API_KEY    (for anthropic provider)
#   BLOCKGUARD_VLLM_URL  (for vllm provider, default: http://localhost:8000/v1/completions)
#
# Optional env vars:
#   BLOCKGUARD_PORT      (default: 8000)
#   BLOCKGUARD_HOST      (default: 0.0.0.0)
#   BLOCKGUARD_TIER      (fast | strict)
#   BLOCKGUARD_RISK_BUDGET  (default: 0.95)
#   BLOCKGUARD_SAMPLES   (default: 5, for strict mode)
#   BLOCKGUARD_DOMAIN    (general | medical | legal | financial)

echo "╔══════════════════════════════════════════════╗"
echo "║         BlockGuard — Hallucination Filter    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

PORT=${1:-8000}
PROVIDER=${2:-openai}

export BLOCKGUARD_PORT="$PORT"
export BLOCKGUARD_LLM_PROVIDER="$PROVIDER"

echo "Provider: $PROVIDER"
echo "Port: $PORT"
echo "Tier: ${BLOCKGUARD_TIER:-fast}"
echo "Domain: ${BLOCKGUARD_DOMAIN:-general}"
echo "Risk budget: ${BLOCKGUARD_RISK_BUDGET:-0.95}"
echo ""
echo "Starting server..."
echo "Health: http://localhost:$PORT/health"
echo "Guard:  http://localhost:$PORT/guard"
echo ""

uv run python -m blockguard.proxy
