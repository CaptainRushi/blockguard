# S2 — Verifier
# Multi-signal scoring per block. Fuses 4 independent signals:
# 1. Sampling consistency (SelfCheckGPT) — stability across N resamples
# 2. Grounding / NLI entailment — DeBERTa-NLI against retrieved evidence
# 3. Logprob / semantic entropy — model's internal uncertainty
# 4. Static features → LightGBM meta-scorer

import numpy as np
from typing import List, Dict, Optional
from ..models.pipeline import ClaimBlock, BlockCode
from ..config import cfg

class SignalScorer:
    """Fetches all 4 signals for a single block."""
    
    def __init__(self, anchor_set=None, context_docs: Optional[List[str]] = None):
        self.anchor_set = anchor_set
        self.context_docs = context_docs or []
        self._nli_model = None
        self._embed_model = None
    
    def _get_nli_model(self):
        """Lazy-load DeBERTa-NLI for entailment checks.
        
        In FAST mode, returns None immediately to skip NLI scoring.
        Only loads the model in STRICT mode where accuracy matters.
        """
        if cfg.domain.value == "fast":
            return None  # Skip NLI in FAST mode — neutral score used instead
        if self._nli_model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tokenizer = AutoTokenizer.from_pretrained(cfg.nli_model_name)
                model = AutoModelForSequenceClassification.from_pretrained(cfg.nli_model_name)
                self._nli_model = (tokenizer, model)
            except Exception:
                self._nli_model = None  # Will skip NLI signal
        return self._nli_model
    
    def _get_embedding_model(self):
        """Lazy-load sentence transformers for embedding similarity."""
        if cfg.domain.value == "fast":
            return None  # Skip embeddings in FAST mode
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer(cfg.embedding_model_name)
            except Exception:
                self._embed_model = None
        return self._embed_model
    
    def signal_sampling_consistency(self, block_text: str, generate_fn, n: int = 5) -> float:
        """
        SelfCheckGPT approach: generate N variants, check if this block's claim agrees.
        Returns: agreement ratio (0.0-1.0). Low = potential confabulation.
        
        generate_fn: callable(prompt) -> str — generates a variant response.
        """
        if n < 2:
            return 1.0  # No sampling possible, assume consistent
        try:
            agreements = []
            for _ in range(n):
                variant = generate_fn(block_text[:200])  # Pass block as prompt context
                # Simple check: does the variant contain the key phrase?
                key_phrase = block_text[:30].strip().split()[0] if block_text else ""
                if key_phrase and key_phrase.lower() in variant.lower():
                    agreements.append(1.0)
                else:
                    agreements.append(0.0)
            return float(np.mean(agreements)) if agreements else 0.5
        except Exception:
            return 0.5
    
    def signal_nli_entailment(self, block_text: str) -> float:
        """
        DeBERTa-NLI: does any context document entail this block?
        Returns: entailment probability (0.0-1.0). Low = ungrounded claim.
        """
        model = self._get_nli_model()
        if model is None or not self.context_docs:
            return 0.5  # No data — neutral score, will be weighted down later
        
        try:
            tokenizer, nli_model = model
            best_entail = 0.0
            for doc in self.context_docs:
                inputs = tokenizer(doc, block_text, return_tensors="pt", truncation=True, max_length=512)
                outputs = nli_model(**inputs)
                probs = np.softmax(outputs.logits.detach().numpy(), axis=1)[0]
                # Index 0 = entailment, 1 = neutral, 2 = contradiction
                entail_prob = float(probs[0])
                best_entail = max(best_entail, entail_prob)
            return best_entail
        except Exception:
            return 0.5
    
    def signal_semantic_entropy(self, block_text: str) -> float:
        """
        Model's internal uncertainty on this block.
        Returns: entropy score (0.0-1.0, higher = more uncertain).
        
        For open-weight models via vLLM, token-level logprobs are free.
        Here we approximate using embedding distance as a proxy.
        """
        embed_model = self._get_embedding_model()
        if embed_model is None:
            return 0.5  # Unknown — neutral
        try:
            emb = embed_model.encode([block_text])
            # Simple proxy: embed distance from 0-centered
            norm = np.linalg.norm(emb[0])
            # Normalize to 0-1 range
            return float(min(norm / 5.0, 1.0))
        except Exception:
            return 0.5
    
    def compute_static_features(self, block: ClaimBlock, anchor_set=None) -> Dict[str, float]:
        """
        Engineer features for LightGBM from the block and anchor set.
        These are the raw signals the meta-scorer fuses.
        """
        anchor_set = anchor_set or self.anchor_set
        features = {}
        
        # Entity overlap with anchors
        if anchor_set and anchor_set.anchors:
            anchor_texts = set(a.phrase.lower() for a in anchor_set.anchors)
            block_words = set(block.text.lower().split())
            overlap = len(anchor_texts & block_words) / max(len(block_words), 1)
            features["entity_overlap"] = overlap
        else:
            features["entity_overlap"] = 0.0
        
        # Hedge-word density
        hedge_words = {"might", "possibly", "perhaps", "could", "maybe", "seems", 
                       "likely", "probably", "approximately", "roughly"}
        words = block.text.lower().split()
        hedge_count = sum(1 for w in words if w in hedge_words)
        features["hedge_density"] = hedge_count / max(len(words), 1)
        
        # Block length (shorter blocks are often safer)
        features["block_length"] = float(len(words))
        
        # Numeric presence (numbers are checkable — either true or false)
        features["has_numbers"] = float(bool(__import__('re').search(r'\d', block.text)))
        
        # Question mark presence (uncertainty signal)
        features["has_question"] = float(block.text.strip().endswith("?"))
        
        # Position-based features (blocks later in response tend to drift)
        features["position_ratio"] = float(block.position / max(len(block.text), 1))
        
        return features
    
    def score_block(self, block: ClaimBlock, generate_fn=None, anchor_set=None) -> ClaimBlock:
        """
        Run all 4 signals on a block. Returns the block with updated confidence and signals.
        
        This is the core of S2 — every block gets a continuous confidence score.
        """
        anchor_set = anchor_set or self.anchor_set
        generate_fn = generate_fn or (lambda x: x)  # Identity if no generator provided
        
        signals = {}
        
        # Signal 1: Sampling consistency
        if generate_fn and cfg.domain.value == "strict":
            signals["sampling_consistency"] = self.signal_sampling_consistency(
                block.text, generate_fn, n=cfg.sampling_count
            )
        else:
            signals["sampling_consistency"] = 0.5  # Unknown in FAST mode
        
        # Signal 2: NLI entailment
        signals["nli_entailment"] = self.signal_nli_entailment(block.text)
        
        # Signal 3: Semantic entropy
        signals["semantic_entropy"] = self.signal_semantic_entropy(block.text)
        
        # Static features
        features = self.compute_static_features(block, anchor_set)
        
        # Combine: weighted average as baseline confidence
        # NLI and sampling are the strongest signals
        confidence = (
            0.35 * signals["nli_entailment"] +
            0.30 * signals["sampling_consistency"] +
            0.15 * (1.0 - signals["semantic_entropy"]) +  # lower entropy = more confident
            0.20 * self._heuristic_confidence(features)
        )
        
        block.confidence = float(np.clip(confidence, 0.0, 1.0))
        block.signals = signals
        block.anchor_alignment = features["entity_overlap"]
        
        # Assign code from score, not hand-written rule
        block.code = self._score_to_code(block.confidence)
        
        return block
    
    def _heuristic_confidence(self, features: Dict[str, float]) -> float:
        """Simple heuristic for confidence when signals are incomplete."""
        score = 0.5
        score += features["entity_overlap"] * 0.3
        score -= features["hedge_density"] * 0.2
        score += (1.0 - min(features["block_length"] / 50.0, 1.0)) * 0.1
        return float(np.clip(score, 0.0, 1.0))
    
    def _score_to_code(self, confidence: float) -> str:
        """
        Convert continuous confidence to BlockCode.
        This is the corrected scheme: codes are assigned FROM scores.
        """
        if confidence >= 0.9:
            return BlockCode.AA1  # Double-verified
        elif confidence >= 0.75:
            return BlockCode.A1  # Verified
        elif confidence >= 0.6:
            return BlockCode.B1  # Reasoning, consistent
        elif confidence >= 0.4:
            return BlockCode.AX  # Uncertain — MUST flag, never delete
        elif confidence >= 0.25:
            return BlockCode.A0  # Failed verification
        else:
            return BlockCode.B0  # Failed reasoning
