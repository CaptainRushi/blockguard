# BlockGuard

**Hidden hallucination-filtering funnel for AI model outputs.**

BlockGuard sits between an LLM and the user as invisible middleware. It splits every response into atomic claim-blocks, scores each block's truthfulness against multiple independent signals (SelfCheckGPT, NLI entailment, semantic entropy, LightGBM), fuses those signals into a hallucination-risk score, and keeps/trims/removes/regenerates/flags each block — repairing text seams where blocks are cut.

## One-Command Install

```bash
git clone https://github.com/CaptainRushi/blockguard.git
cd blockguard
pip install -e .
blockguard start
```

Or with uv:
```bash
uv sync
uv run blockguard start
```

Or use the quick install script:
```bash
bash install.sh        # Linux/Mac
powershell install.ps1 # Windows
```

## Quick Start

```bash
# Start the proxy server (default: port 8000, FAST mode)
blockguard start --provider openai --model gpt-4o

# Strict mode for medical/legal/financial
blockguard start --tier strict --domain medical

# With Anthropic
blockguard start --provider anthropic --model claude-sonnet-4

# With vLLM (open-weight, self-hosted)
blockguard start --provider vllm --model llama-3.1-70b

# Check status
blockguard status

# Run tests
blockguard test
```

## How It Works

```
User prompt → S0 Anchor → LLM → S1 Segment → S2 Verify → S3 Sort → S4 Decide → S5 Splice → Cleaned output
```

| Stage | What It Does |
|-------|-------------|
| **S0** Anchor Extractor | Parses user prompt for entities, constraints, negations |
| **S1** Segmenter | Splits LLM output into atomic claim-blocks |
| **S2** Verifier | 4 independent signals: SelfCheckGPT, NLI entailment, semantic entropy, LightGBM |
| **S3** Optical Sorter | Detector → edge detection → rolling risk budget ejector |
| **S4** Decision Layer | KEEP / TRIM / REGENERATE / FLAG per block |
| **S5** Splicer | Junction repair for removed blocks (prevents incoherence) |

## Endpoints (when running)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server status |
| `/guard` | POST | Filter a prompt through BlockGuard |
| `/guard/stream` | POST | Streaming variant |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BLOCKGUARD_TIER` | `fast` | `fast` or `strict` |
| `BLOCKGUARD_DOMAIN` | `general` | `general`, `medical`, `legal`, `financial` |
| `BLOCKGUARD_RISK_BUDGET` | `0.95` | Cumulative risk threshold |
| `BLOCKGUARD_SAMPLES` | `5` | N variants for SelfCheckGPT |
| `BLOCKGUARD_PORT` | `8000` | Server port |
| `BLOCKGUARD_LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `vllm` |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| `BLOCKGUARD_VLLM_URL` | `http://localhost:8000/v1/completions` | vLLM endpoint |

## CLI Commands

| Command | Description |
|---|---|
| `blockguard start` | Start the proxy server |
| `blockguard status` | Show pipeline configuration |
| `blockguard test` | Run test suite (17 tests) |
| `blockguard train` | Train LightGBM on custom data |
| `blockguard version` | Show version |

## Project Structure

```
blockguard/
├── cli.py              # CLI entry point (blockguard command)
├── config.py           # Runtime configuration
├── models/pipeline.py  # Pydantic models + BlockCode enum
├── pipeline.py         # S0→S5 orchestrator
├── proxy.py            # FastAPI server + OpenAI/Anthropic/vLLM
├── stages/             # 6 stage modules
│   ├── s0_anchor.py
│   ├── s1_segment.py
│   ├── s2_verify.py
│   ├── s3_sort.py
│   ├── s4_decide.py
│   └── s5_splice.py
├── scripts/train_scorer.py  # LightGBM training
└── tests/              # 17 unit tests
```

## Plugin for AI Agents

BlockGuard ships as a skill for:
- **Hermes Agent** — `~/.hermes/skills/blockguard/SKILL.md`
- **Claude Code** — `~/.hermes/skills/claude-blockguard/SKILL.md`
- **Codex** — `~/.hermes/skills/codex-blockguard/SKILL.md`

Install via `blockguard install` or copy the skill file to your agent's skills directory.

## Honest Limits

- **Zero hallucination is not achievable** — FLAG over silent delete, always
- **Binary true/false is wrong** — continuous confidence score + AX uncertain state
- **LightGBM is a meta-fuser** — never a standalone truth oracle
- **STRICT mode costs 5–10× more tokens** — use only where errors are expensive
- **The filter is an attack surface** — prompt injection possible
- **Real accuracy needs 2K+ human-labeled blocks** — synthetic seed data is illustrative only

## License

MIT
