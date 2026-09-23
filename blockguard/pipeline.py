# BlockGuard — Main Pipeline Orchestrator
# Ties all stages S0→S1→S2→S3→S4→S5 into a single execution flow.
# The filter is invisible — the calling app sees only the cleaned response.

import time
from typing import List, Optional
from fastapi import HTTPException
from .models.pipeline import (
    PipelineInput, PipelineResult, ClaimBlock, 
    BlockDecision, BlockCode, AnchorSet
)
from .config import BlockGuardConfig, load_config
from .stages.s0_anchor import extract_anchors
from .stages.s1_segment import segment
from .stages.s2_verify import SignalScorer
from .stages.s3_sort import OpticalSorter
from .stages.s4_decide import DecisionLayer
from .stages.s5_splice import Splicer

class BlockGuardPipeline:
    """
    Executes the full BlockGuard pipeline on an LLM response.
    
    Flow:
    S0 Anchor Extractor → S1 Segmenter → S2 Verifier → 
    S3 Optical Sorter → S4 Decision Layer → S5 Splicer → cleaned output
    """
    
    def __init__(self, config: Optional[BlockGuardConfig] = None):
        self.config = config or load_config()
        self.sorter = OpticalSorter()
        self.decision_layer = DecisionLayer()
        self.splicer = Splicer()
        self.scorer = SignalScorer()
        self._trained = False
    
    def train_scorer(self, X, y):
        """Train the LightGBM meta-scorer on labeled data."""
        self.decision_layer.train(X, y)
        self._trained = True
    
    async def run(self, input: PipelineInput, generate_fn) -> PipelineResult:
        """
        Run the full pipeline.
        
        Args:
            input: PipelineInput with prompt, context, domain, tier
            generate_fn: callable(prompt) -> str — generates the raw LLM response
        
        Returns:
            PipelineResult with cleaned text and all decisions.
        """
        start_time = time.time()
        
        # Reset state
        self.sorter.reset()
        self.splicer.reset()
        
        # --- S0: Anchor Extractor ---
        anchor_set = extract_anchors(input.prompt)
        
        # --- Generate the raw LLM response (untouched) ---
        raw_response = await generate_fn(input.prompt)
        
        # --- S1: Segmenter ---
        blocks = segment(raw_response)
        
        # Inject context before/after for each block
        for i, block in enumerate(blocks):
            if i > 0:
                blocks[i].context_before = blocks[i-1].text
            if i < len(blocks) - 1:
                blocks[i].context_after = blocks[i+1].text
        
        # --- S2: Verifier (score each block) ---
        for block in blocks:
            self.scorer.anchor_set = anchor_set
            self.scorer.context_docs = [input.context] if input.context else []
            self.scorer.score_block(block, generate_fn=generate_fn, anchor_set=anchor_set)
        
        # --- S3: Optical Sorter ---
        detector_flags = set(self.sorter.detect_suspicious(blocks))
        edge_flags = set(self.sorter.detect_edges(blocks))
        ejections = self.sorter.eject(blocks, detector_flags, edge_flags)
        
        # Mark ejected blocks
        ejected_positions = {blocks[i].position for i, _ in ejections}
        for block in blocks:
            if block.position in ejected_positions:
                block.code = BlockCode.A0.value  # Mark as failed
        
        # --- S4: Decision Layer ---
        decisions = []
        for block in blocks:
            # Get LightGBM probability if trained, else heuristic
            features = self.scorer.compute_static_features(block, anchor_set)
            lgbm_prob = self.decision_layer.predict(features)
            
            decision = self.decision_layer.decide(
                block, lgbm_prob, block.anchor_alignment
            )
            decisions.append(decision)
        
        # --- S5: Splicer ---
        # Build decision lookup by block position
        decisions_by_pos = {}
        for d in decisions:
            decisions_by_pos[d.block.position] = d
        
        # Determine which blocks to remove
        remove_positions = set()
        for d in decisions:
            if d.action in ("trim",):
                remove_positions.add(d.block.position)
            elif d.action == "regenerate":
                # For regenerate, we keep the block but mark it
                pass
        
        cleaned_text = self.splicer.splice(blocks, decisions, raw_response)
        
        # --- Compile results ---
        elapsed = (time.time() - start_time) * 1000
        
        # Collect flagged blocks
        flagged = [d.block for d in decisions if d.action == "flag"]
        
        # Compute overall risk
        total_risk = sum(
            self.decision_layer.predict(
                self.scorer.compute_static_features(b, anchor_set)
            ) for b in blocks
        ) if blocks else 0.0
        
        return PipelineResult(
            cleaned_text=cleaned_text,
            blocks=blocks,
            decisions=decisions,
            flagged_blocks=flagged,
            removed_count=len(remove_positions),
            total_latency_ms=round(elapsed, 2),
            risk_score=round(total_risk, 4),
        )

# --- Pipeline as a service ---
pipeline = BlockGuardPipeline()

async def guard_generate(prompt: str, context: Optional[str] = None, 
                          domain: str = "general", tier: str = "fast") -> PipelineResult:
    """
    Convenience entry point: runs the full pipeline on an LLM response.
    
    Usage:
        result = await guard_generate("What is quantum computing?", domain="medical")
        print(result.cleaned_text)
    """
    input = PipelineInput(prompt=prompt, context=context, domain=domain, tier=tier)
    return await pipeline.run(input, generate_fn=lambda p: _fake_generate(p))

def _fake_generate(prompt: str) -> str:
    """Placeholder — in production, this calls the actual LLM API."""
    return ("Quantum computing is a type of computation that harnesses the phenomena of "
            "quantum mechanics, such as superposition and entanglement. It is expected to "
            "solve certain problems much faster than classical computers. The first quantum "
            "computer was built in 2019 by Google. It uses qubits instead of bits. "
            "Quantum computers can break RSA encryption. The market for quantum computing "
            "is expected to reach $50 billion by 2030. However, this technology is not yet "
            "ready for widespread commercial use and remains experimental in nature.")
