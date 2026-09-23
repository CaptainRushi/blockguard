# BlockGuard — Pydantic models
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class BlockType(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    CODE = "code"
    FILLER = "filler"

class BlockCode(str, Enum):
    A1 = "A1"  # Factual claim, verified
    AA1 = "AA1"  # Double-verified
    B1 = "B1"  # Reasoning step, internally consistent
    C1 = "C1"  # Context/filler, harmless
    D1 = "D1"  # Code/formula, passes parse or test
    A0 = "A0"  # Failed verification
    B0 = "B0"  # Failed reasoning
    AX = "AX"  # Uncertain — flag, never silently delete

class Anchor(BaseModel):
    phrase: str
    weight: float
    embedding: List[float] = Field(default_factory=list)
    kind: str = "entity"  # entity | negation | constraint | verb

class ClaimBlock(BaseModel):
    text: str
    position: int  # Start index in raw response
    end_position: int  # End index
    block_type: BlockType = BlockType.FACTUAL
    code: str = "AX"  # BlockCode — defaults to uncertain
    confidence: float = 0.5  # 0.0-1.0 continuous score
    signals: Dict[str, float] = Field(default_factory=dict)  # Raw signal scores
    anchor_alignment: float = 0.0
    is_important: bool = True  # Load-bearing for the answer?
    context_before: str = ""
    context_after: str = ""

class AnchorSet(BaseModel):
    anchors: List[Anchor] = Field(default_factory=list)

class PipelineInput(BaseModel):
    """What the user sends in — prompt + optional context."""
    prompt: str
    context: Optional[str] = None  # RAG docs, user-supplied facts
    domain: str = "general"  # general | medical | legal | financial
    tier: str = "fast"

class BlockDecision(BaseModel):
    block: ClaimBlock
    action: str = "keep"  # keep | trim | regenerate | flag
    reason: str = ""

class PipelineResult(BaseModel):
    cleaned_text: str
    blocks: List[ClaimBlock] = Field(default_factory=list)
    decisions: List[BlockDecision] = Field(default_factory=list)
    flagged_blocks: List[ClaimBlock] = Field(default_factory=list)
    removed_count: int = 0
    total_latency_ms: float = 0.0
    risk_score: float = 0.0
