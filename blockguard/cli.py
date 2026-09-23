# BlockGuard CLI
# Usage: blockguard start, blockguard train, blockguard test, blockguard version

import sys
import os
import argparse
import asyncio
import uvicorn

def main():
    parser = argparse.ArgumentParser(
        prog="blockguard",
        description="BlockGuard — Hidden hallucination-filtering funnel for AI model outputs",
    )
    sub = parser.add_subparsers(dest="command")

    # Start the proxy server
    p_start = sub.add_parser("start", help="Start the BlockGuard proxy server")
    p_start.add_argument("--port", type=int, default=int(os.getenv("BLOCKGUARD_PORT", "8000")))
    p_start.add_argument("--host", default=os.getenv("BLOCKGUARD_HOST", "0.0.0.0"))
    p_start.add_argument("--tier", default=os.getenv("BLOCKGUARD_TIER", "fast"))
    p_start.add_argument("--domain", default=os.getenv("BLOCKGUARD_DOMAIN", "general"))
    p_start.add_argument("--provider", default=os.getenv("BLOCKGUARD_LLM_PROVIDER", "openai"))
    p_start.add_argument("--model", default="gpt-4o")

    # Train the scorer
    p_train = sub.add_parser("train", help="Train the LightGBM meta-scorer")
    p_train.add_argument("--data", default="blockguard/data/seed_blocks.csv")
    p_train.add_argument("--output", default="blockguard/data/lightgbm_model.pkl")

    # Test the pipeline
    p_test = sub.add_parser("test", help="Run the test suite")

    # Show version and status
    sub.add_parser("version", help="Show version")
    sub.add_parser("status", help="Show pipeline configuration")

    args = parser.parse_args()

    if args.command == "start":
        start_server(args)
    elif args.command == "train":
        train_scorer(args)
    elif args.command == "test":
        run_tests()
    elif args.command == "version":
        print("BlockGuard v0.1.0")
        print("Hidden hallucination-filtering funnel for AI model outputs")
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()

def start_server(args):
    """Start the FastAPI proxy server."""
    os.environ["BLOCKGUARD_TIER"] = args.tier
    os.environ["BLOCKGUARD_DOMAIN"] = args.domain
    os.environ["BLOCKGUARD_LLM_PROVIDER"] = args.provider
    os.environ["BLOCKGUARD_PORT"] = str(args.port)

    from blockguard.proxy import app

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║         BlockGuard v0.1.0                    ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  Tier: {args.tier:<28}║")
    print(f"║  Domain: {args.domain:<26}║")
    print(f"║  Provider: {args.provider:<25}║")
    print(f"║  Model: {args.model:<28}║")
    print(f"║  Server: http://{args.host}:{args.port}")
    print(f"║  Health: http://{args.host}:{args.port}/health")
    print(f"║  Guard:  http://{args.host}:{args.port}/guard")
    print(f"╚══════════════════════════════════════════════╝")
    print()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

def train_scorer(args):
    """Train the LightGBM meta-scorer."""
    from blockguard.scripts.train_scorer import train
    model = train(args.data, args.output)
    if model:
        print(f"Model trained and saved to {args.output}")

def run_tests():
    """Run the test suite."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    sys.exit(result.returncode)

def show_status():
    """Show current pipeline configuration."""
    from blockguard.config import load_config
    cfg = load_config()
    print(f"BlockGuard v0.1.0")
    print(f"  Tier: {cfg.domain.value}")
    print(f"  Risk budget: {cfg.risk_budget}")
    print(f"  Sampling count: {cfg.sampling_count}")
    print(f"  NLI model: {cfg.nli_model_name}")
    print(f"  Embedding model: {cfg.embedding_model_name}")
    print(f"  SpaCy model: {cfg.spacy_model}")
    print(f"  P95 latency target: {cfg.p95_latency_ms}ms")
    print(f"  Removal precision: {cfg.removal_precision}")

if __name__ == "__main__":
    main()
