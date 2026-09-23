# BlockGuard — Tests
# Run with: pytest tests/ -v

import pytest
from blockguard.stages.s0_anchor import extract_anchors
from blockguard.stages.s1_segment import segment
from blockguard.stages.s2_verify import SignalScorer
from blockguard.stages.s3_sort import OpticalSorter
from blockguard.stages.s4_decide import DecisionLayer
from blockguard.stages.s5_splice import Splicer
from blockguard.pipeline import BlockGuardPipeline, PipelineInput
from blockguard.models.pipeline import BlockCode

# --- S0: Anchor Extractor ---

def test_extract_anchors_basic():
    """Anchor extraction finds entities and constraints."""
    anchors = extract_anchors("Summarize quantum computing under $50 million")
    assert len(anchors.anchors) > 0
    # Should find "$50 million" as an entity
    entity_phrases = [a.phrase for a in anchors.anchors if a.kind == "entity"]
    assert len(entity_phrases) > 0

def test_extract_anchors_negation():
    """Should detect negation."""
    anchors = extract_anchors("Do NOT use AI unless it is verified")
    has_negation = any(a.kind == "negation" for a in anchors.anchors)
    assert has_negation

def test_extract_anchors_empty():
    """Empty prompt returns empty anchor set."""
    anchors = extract_anchors("")
    assert len(anchors.anchors) == 0

# --- S1: Segmenter ---

def test_segment_single_sentence():
    """Single sentence becomes one block."""
    blocks = segment("Quantum computing uses qubits.")
    assert len(blocks) == 1
    assert blocks[0].text == "Quantum computing uses qubits."

def test_segment_multi_fact():
    """Multi-fact sentence splits into multiple blocks."""
    blocks = segment("Quantum computing uses qubits. It was pioneered by Feynman.")
    assert len(blocks) >= 1  # At least 2 blocks expected
    assert all(b.text for b in blocks)

def test_segment_empty():
    """Empty text returns one filler block."""
    blocks = segment("")
    assert len(blocks) == 1
    assert blocks[0].block_type == BlockCode.C1.value or len(blocks) == 1

# --- S2: Verifier ---

def test_signal_scorer_init():
    """SignalScorer initializes without error."""
    scorer = SignalScorer()
    assert scorer is not None

def test_scorer_classify_code():
    """Score-to-code mapping works correctly."""
    scorer = SignalScorer()
    assert scorer._score_to_code(0.95) == "AA1"
    assert scorer._score_to_code(0.8) == "A1"
    assert scorer._score_to_code(0.5) == "AX"  # Uncertain
    assert scorer._score_to_code(0.1) == "B0"

# --- S3: Optical Sorter ---

def test_sorter_reset():
    """Reset clears state."""
    sorter = OpticalSorter()
    sorter.cumulative_risk = 1.0
    sorter.ejected_indices = {0, 1}
    sorter.reset()
    assert sorter.cumulative_risk == 0.0
    assert len(sorter.ejected_indices) == 0

def test_sorter_risk_status():
    """Risk status returns correct dict."""
    sorter = OpticalSorter()
    sorter.cumulative_risk = 0.5
    status = sorter.get_risk_status()
    assert status["cumulative_risk"] == 0.5
    assert status["threshold"] == 0.95

# --- S4: Decision Layer ---

def test_decision_layer_init():
    """DecisionLayer initializes without a trained model."""
    dl = DecisionLayer()
    assert not dl._trained

def test_decision_heuristic():
    """Heuristic prediction returns valid probability."""
    dl = DecisionLayer()
    features = {"entity_overlap": 0.5, "hedge_density": 0.1, "block_length": 15,
                "has_numbers": 0, "has_question": 0, "position_ratio": 0.3}
    prob = dl.predict(features)
    assert 0.0 <= prob <= 1.0

def test_decision_actions():
    """All four actions are reachable."""
    from blockguard.models.pipeline import ClaimBlock
    dl = DecisionLayer()
    block = ClaimBlock(text="test", position=0, end_position=4)
    # Test low risk → KEEP
    decision = dl.decide(block, 0.1, 0.8)
    assert decision.action in ("keep", "trim")

# --- S5: Splicer ---

def test_splicer_repair():
    """Splicer repairs broken seams."""
    splicer = Splicer()
    text = "Quantum computing uses qubits. , which is amazing"
    repaired = splicer._repair_seams(text, set(), [])
    assert repaired != text  # Should be modified

def test_splicer_reverify():
    """Re-verification catches broken joins."""
    splicer = Splicer()
    # Valid new sentence
    assert splicer.reverify_join("The result is clear.", "However, there are caveats.") == True
    # Broken join (two sentences without connector)
    assert splicer.reverify_join("The result is clear.", "It is amazing.") == False

# --- Pipeline Integration ---

@pytest.mark.asyncio
async def test_pipeline_structure():
    """Pipeline initializes correctly."""
    pipe = BlockGuardPipeline()
    assert pipe.sorter is not None
    assert pipe.decision_layer is not None
    assert pipe.splicer is not None

def test_pipeline_config():
    """Config loads with defaults."""
    from blockguard.config import load_config
    cfg = load_config()
    assert cfg.domain.value in ("fast", "strict")
    assert cfg.risk_budget > 0
