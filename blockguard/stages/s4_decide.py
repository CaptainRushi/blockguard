# S4 — Decision Layer (the "judge")
# Fuses LightGBM probability + anchor alignment + block importance → verdict.
# Four verdicts: KEEP / TRIM / REGENERATE / FLAG
# Default rule for high-stakes domains: FLAG over silent delete, always.

import numpy as np
from typing import List, Optional
from lightgbm import LGBMClassifier
from ..models.pipeline import ClaimBlock, BlockDecision, BlockCode
from ..config import BlockAction
from ..config import cfg

class DecisionLayer:
    """
    The judge that fuses all signals into a final verdict per block.
    
    final_score = f(LightGBM_probability, anchor_alignment_score, block_importance_score)
    
    Uses LightGBM as a meta-fuser over engineered features — never as a standalone truth oracle.
    """
    
    def __init__(self):
        self.model: Optional[LGBMClassifier] = None
        self._trained = False
    
    def train(self, X, y):
        """
        Train the LightGBM meta-scorer on labeled data.
        X: feature matrix (n_samples, n_features) from compute_static_features
        y: labels (0=bad, 1=good) — human-labeled or SelfCheckGPT auto-labeled
        """
        self.model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="binary",
            metric="binary_logloss",
            verbose=-1,
        )
        self.model.fit(X, y)
        self._trained = True
    
    def predict(self, features: dict) -> float:
        """
        Predict hallucination probability for a single block's features.
        Returns: probability (0.0-1.0) that this block is hallucinated.
        """
        if not self._trained or self.model is None:
            # Untrained fallback: heuristic score
            return self._heuristic_predict(features)
        
        # Convert features dict to array in consistent order
        feature_order = ["entity_overlap", "hedge_density", "block_length", 
                        "has_numbers", "has_question", "position_ratio"]
        X = np.array([[features.get(f, 0.0) for f in feature_order]])
        return float(self.model.predict_proba(X)[0][1])  # Probability of "hallucinated"
    
    def _heuristic_predict(self, features: dict) -> float:
        """Fallback heuristic when LightGBM isn't trained yet."""
        score = 0.0
        score += (1.0 - features.get("entity_overlap", 0.0)) * 0.4
        score += features.get("hedge_density", 0.0) * 0.3
        score += features.get("has_question", 0.0) * 0.15
        score += min(features.get("block_length", 20) / 50.0, 1.0) * 0.15
        return float(np.clip(score, 0.0, 1.0))
    
    def decide(self, block: ClaimBlock, lgbm_prob: float, anchor_alignment: float) -> BlockDecision:
        """
        Fuse signals into a final verdict.
        
        Args:
            block: The claim block to decide on
            lgbm_prob: LightGBM hallucination probability
            anchor_alignment: Alignment score vs. anchor set
        
        Returns:
            BlockDecision with action and reason.
        """
        # Block importance heuristic
        importance = self._compute_importance(block)
        
        # Fuse: final hallucination risk score
        # Higher lgbm_prob + lower anchor_alignment = higher risk
        final_score = (
            0.50 * lgbm_prob +
            0.30 * (1.0 - anchor_alignment) +
            0.20 * (1.0 - block.confidence)
        )
        
        # Decision logic
        is_high_stakes = cfg.domain.value in ("medical", "legal", "financial")
        
        if final_score < 0.25:
            # High confidence, well-anchored → Keep
            return BlockDecision(
                block=block,
                action=BlockAction.KEEP.value,
                reason=f"Low risk ({final_score:.3f}), well-anchored",
            )
        elif final_score < 0.5:
            # Moderate risk → Keep but note
            return BlockDecision(
                block=block,
                action=BlockAction.KEEP.value,
                reason=f"Moderate risk ({final_score:.3f}), kept with watch",
            )
        elif final_score < 0.7:
            # Uncertain territory → FLAG (never silently delete in high-stakes)
            if is_high_stakes or block.code == BlockCode.AX:
                return BlockDecision(
                    block=block,
                    action=BlockAction.FLAG.value,
                    reason=f"Uncertain ({final_score:.3f}), flagged for user",
                )
            else:
                return BlockDecision(
                    block=block,
                    action=BlockAction.TRIM.value,
                    reason=f"Low importance, trimmed ({final_score:.3f})",
                )
        elif final_score < 0.85:
            # High risk but potentially salvageable → REGENERATE
            if importance > 0.5:
                return BlockDecision(
                    block=block,
                    action=BlockAction.REGENERATE.value,
                    reason=f"High-risk load-bearing block ({final_score:.3f}), regenerate",
                )
            else:
                return BlockDecision(
                    block=block,
                    action=BlockAction.TRIM.value,
                    reason=f"Unimportant high-risk block ({final_score:.3f}), trim",
                )
        else:
            # Very high risk → FLAG in high-stakes, else remove
            if is_high_stakes:
                return BlockDecision(
                    block=block,
                    action=BlockAction.FLAG.value,
                    reason=f"Critical risk ({final_score:.3f}), flagged for user",
                )
            else:
                return BlockDecision(
                    block=block,
                    action=BlockAction.REGENERATE.value,
                    reason=f"Critical risk ({final_score:.3f}), regenerate",
                )
    
    def _compute_importance(self, block: ClaimBlock) -> float:
        """
        Score how load-bearing a block is for answering the question.
        Heuristic: longer blocks, blocks with numbers, first blocks = higher importance.
        """
        importance = 0.3  # Base
        importance += min(len(block.text.split()) / 30.0, 0.3)  # Length up to 0.3
        importance += float(block.code in (BlockCode.A1, BlockCode.AA1)) * 0.2
        importance += float(any(w in block.text.lower() for w in ["key", "main", "critical", "important", "result"])) * 0.2
        return float(np.clip(importance, 0.0, 1.0))
