# S0 — Anchor Extractor
# Parses user prompt to build weighted anchor set — things the answer can't drift from.
# Runs once per request, sub-100ms target.

import spacy
from typing import List, Tuple
from ..config import cfg, BlockGuardConfig
from ..models.pipeline import Anchor, AnchorSet, BlockCode

# Lazy-loaded spaCy model
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(cfg.spacy_model)
        except OSError:
            # Model not downloaded — fall back to lightweight regex-only
            _nlp = None
    return _nlp

def _extract_entities(doc) -> List[Anchor]:
    """Extract named entities with weights."""
    anchors = []
    for ent in doc.ents:
        weight = cfg.anchor_weight_entities
        # Numbers and dates get highest weight — these are checkable facts
        if ent.label_ in ("DATE", "NUMBER", "MONEY", "PERCENT", "QUANTITY"):
            weight *= 1.2
        anchors.append(Anchor(
            phrase=ent.text,
            weight=weight,
            kind="entity",
        ))
    return anchors

def _extract_negations(doc) -> List[Anchor]:
    """Find negation tokens and their scope."""
    anchors = []
    for token in doc:
        if token.dep_ in ("neg", "negation") or token.lower_ in ("not", "never", "no", "none", "neither", "nor"):
            # Get the head token's subtree as the negation scope
            scope = " ".join(t.text for t in token.head.subtree)
            anchors.append(Anchor(
                phrase=f"NOT {scope}",
                weight=cfg.anchor_weight_negation,
                kind="negation",
            ))
    return anchors

def _extract_constraints(doc) -> List[Anchor]:
    """Find constraint words: 'only', 'must', 'excluding', 'under $X', etc."""
    constraint_words = {"only", "must", "mustn't", "excluding", "except", 
                       "under", "over", "above", "below", "at least", "at most",
                       "between", "within", "without"}
    anchors = []
    for token in doc:
        if token.lower_ in constraint_words or token.lemma_.lower() in constraint_words:
            scope = " ".join(t.text for t in token.head.subtree)
            anchors.append(Anchor(
                phrase=scope,
                weight=cfg.anchor_weight_constraint,
                kind="constraint",
            ))
    return anchors

def _extract_verb(doc) -> List[Anchor]:
    """Find the core task verb."""
    anchors = []
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            anchors.append(Anchor(
                phrase=token.lemma_,
                weight=cfg.anchor_weight_verb,
                kind="verb",
            ))
            break
    return anchors

def extract_anchors(prompt: str) -> AnchorSet:
    """
    Build the anchor set from a user prompt.
    
    Returns AnchorSet with weighted (phrase, weight, kind) tuples.
    This is carried through the pipeline; every block gets an alignment score.
    """
    doc = _get_nlp()(prompt)
    if doc is None:
        # Regex fallback — no spaCy model available
        return _regex_fallback(prompt)
    
    anchors = []
    anchors.extend(_extract_entities(doc))
    anchors.extend(_extract_negations(doc))
    anchors.extend(_extract_constraints(doc))
    anchors.extend(_extract_verb(doc))
    
    # Deduplicate by phrase
    seen = set()
    unique = []
    for a in anchors:
        if a.phrase not in seen:
            seen.add(a.phrase)
            unique.append(a)
    
    return AnchorSet(anchors=unique)

def _regex_fallback(prompt: str) -> AnchorSet:
    """Lightweight fallback when spaCy isn't available."""
    import re
    anchors = []
    # Find capitalized named entities (rough)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', prompt)
    for e in entities[:5]:
        anchors.append(Anchor(phrase=e, weight=cfg.anchor_weight_entities, kind="entity"))
    # Find numbers
    numbers = re.findall(r'\b\d+\.?\d*\s*(?:million|billion|%|dollars|$\d+)\b', prompt)
    for n in numbers[:3]:
        anchors.append(Anchor(phrase=n, weight=cfg.anchor_weight_entities * 1.2, kind="entity"))
    return AnchorSet(anchors=anchors)
