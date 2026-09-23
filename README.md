# BlockGuard
Hidden hallucination-filtering funnel for AI model outputs.

Pipeline: **S0 Anchor → S1 Segment → S2 Verify → S3 Sort → S4 Decide → S5 Splice**

## Quick Start

```bash
# Install dependencies
uv sync

# Download spaCy model
uv run python -m spacy download en_core_web_sm

# Create seed data and train scorer
uv run python -m blockguard.scripts.train_scorer --seed-only
uv run python -m blockguard.scripts.train_scorer --data blockguard/data/seed_blocks.csv

# Run tests
uv run python -m pytest tests/ -v

# Start proxy server
bash run.sh 8000 openai
# or: BLOCKGUARD_LLM_PROVIDER=anthropic bash run.sh 8000 anthropic
```

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status |
| `/guard` | POST | Filter a prompt through BlockGuard |
| `/guard/stream` | POST | Streaming variant |

## Architecture

Each stage is a pluggable component. The full pipeline runs in ~600–800ms (FAST mode) and filters hallucinated claim-blocks before they reach the user.

See `blockguard/` for the 6-stage implementation.
