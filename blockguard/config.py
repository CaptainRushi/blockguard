# BlockGuard — Config
import os
from enum import Enum
from typing import Optional

class DomainTier(str, Enum):
    FAST = "fast"
    STRICT = "strict"

class BlockAction(str, Enum):
    KEEP = "keep"
    TRIM = "trim"
    REGENERATE = "regenerate"
    FLAG = "flag"

class BlockCode(str, Enum):
    A1 = "A1"  # Factual, verified
    AA1 = "AA1"  # Double-verified
    B1 = "B1"  # Reasoning, consistent
    C1 = "C1"  # Context/filler
    D1 = "D1"  # Code/formula
    A0 = "A0"  # Failed verification
    B0 = "B0"  # Failed reasoning
    AX = "AX"  # Uncertain — flag, never delete

class BlockGuardConfig:
    """Runtime configuration for BlockGuard pipeline."""
    domain: DomainTier = DomainTier.FAST
    sampling_count: int = 5  # N variants for SelfCheckGPT (STRICT mode)
    risk_budget: float = 0.95  # Cumulative risk threshold for S3 ejector
    anchor_weight_entities: float = 1.0
    anchor_weight_negation: float = 0.9
    anchor_weight_constraint: float = 0.85
    anchor_weight_verb: float = 0.7
    lightgbm_model_path: Optional[str] = None
    nli_model_name: str = "MoritzLaF/deberta-v3-large-mnli"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    p95_latency_ms: int = 400  # FAST tier target
    removal_precision: float = 0.90  # G2 threshold
    # Streaming buffer: hold ~1 block of lag
    stream_buffer_size: int = 1
    # Redis for rolling risk budget state (set to None for in-memory)
    redis_url: Optional[str] = None

# Load from env overrides
def load_config() -> BlockGuardConfig:
    cfg = BlockGuardConfig()
    if tier := os.getenv("BLOCKGUARD_TIER"):
        cfg.domain = DomainTier(tier.lower())
    if budget := os.getenv("BLOCKGUARD_RISK_BUDGET"):
        cfg.risk_budget = float(budget)
    if samples := os.getenv("BLOCKGUARD_SAMPLES"):
        cfg.sampling_count = int(samples)
    return cfg

cfg = load_config()
