# S3 — Optical Sorter
# Three-stage cascade: Detector → Edge Detection → Ejector with rolling risk budget.
# Maps the "optical sorter" concept to a real detector → edge-detection → ejector pipeline.

import numpy as np
from typing import List, Tuple
from ..models.pipeline import ClaimBlock, BlockCode
from ..config import cfg

class OpticalSorter:
    """
    Three-stage cascade for filtering blocks before the decision layer.
    Stage 1 (Detector): cheap screen to flag suspicious blocks.
    Stage 2 (Edge Detection): find where the model drifted off-task.
    Stage 3 (Ejector): rolling risk budget — cumulative risk threshold.
    """
    
    def __init__(self):
        self.cumulative_risk = 0.0
        self.ejected_indices = set()
    
    def reset(self):
        """Reset the rolling risk budget for a new response."""
        self.cumulative_risk = 0.0
        self.ejected_indices = set()
    
    # --- Stage 1: Detector (cheap, fast triage) ---
    def detect_suspicious(self, blocks: List[ClaimBlock]) -> List[int]:
        """
        Fast pre-screen: flag blocks that are likely problematic.
        Returns indices of suspicious blocks.
        """
        suspicious = []
        for i, block in enumerate(blocks):
            # Flag if: low confidence from S2, high hedge density, or unknown entity
            if block.confidence < 0.5:
                suspicious.append(i)
            elif "?" in block.text and block.confidence < 0.7:
                suspicious.append(i)
        return suspicious
    
    # --- Stage 2: Edge Detection ---
    def detect_edges(self, blocks: List[ClaimBlock]) -> List[int]:
        """
        Compute anchor-alignment curve across blocks.
        Local minima = drift points where the model went off-topic.
        
        Returns: indices where embedding discontinuities occur.
        """
        if len(blocks) < 2:
            return []
        
        # Compute alignment scores for all blocks
        alignments = [b.anchor_alignment for b in blocks]
        
        if not alignments:
            return []
        
        # Simple gradient-based edge detection
        edges = []
        for i in range(1, len(alignments)):
            delta = alignments[i] - alignments[i-1]
            # Large drop in alignment = drift edge
            if delta < -0.2:  # Threshold for "sudden drop"
                edges.append(i)
        
        # Also flag blocks with very low alignment
        for i, align in enumerate(alignments):
            if align < 0.1 and i not in edges:
                edges.append(i)
        
        return edges
    
    # --- Stage 3: Ejector with rolling risk budget ---
    def eject(self, blocks: List[ClaimBlock], detector_flags: set, edge_flags: set) -> List[Tuple[int, float]]:
        """
        Track cumulative risk as blocks stream.
        If cumulative risk crosses the threshold, eject the block.
        
        Returns: list of (index, risk_contribution) for ejected blocks.
        """
        ejections = []
        
        for i, block in enumerate(blocks):
            # Compute risk contribution for this block
            risk = self._compute_risk(block)
            
            # Detector and edge flags increase risk
            if i in detector_flags:
                risk *= 1.5  # Suspicious blocks weighted higher
            if i in edge_flags:
                risk *= 2.0  # Edge blocks are doubled
            
            self.cumulative_risk += risk
            
            if self.cumulative_risk >= cfg.risk_budget:
                # Eject this block
                ejections.append((i, risk))
                self.ejected_indices.add(i)
                # Reset budget after ejection (new "clean" section)
                self.cumulative_risk = 0.0
        
        return ejections
    
    def _compute_risk(self, block: ClaimBlock) -> float:
        """
        Compute risk contribution for a single block.
        Higher confidence = lower risk.
        """
        if block.code == BlockCode.AX:
            return 0.3  # Uncertain blocks have moderate risk
        elif block.code in (BlockCode.A0, BlockCode.B0):
            return 0.8  # Failed verification = high risk
        elif block.confidence < 0.4:
            return 0.6
        else:
            return 0.1  # Verified blocks have low risk
        
    def get_risk_status(self) -> dict:
        """Current state of the rolling budget."""
        return {
            "cumulative_risk": round(self.cumulative_risk, 4),
            "budget_remaining": round(cfg.risk_budget - self.cumulative_risk, 4),
            "ejected_count": len(self.ejected_indices),
            "threshold": cfg.risk_budget,
        }
