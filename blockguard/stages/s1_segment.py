# S1 — Segmenter
# Splits LLM output into atomic claim-blocks — smallest units that can be independently true/false.
# Dynamic block size: a sentence with 3 facts = 3 blocks, one sentence with 1 fact = 1 block.

import spacy
import re
from typing import List, Optional
from ..models.pipeline import ClaimBlock, BlockType, BlockCode
from ..config import cfg

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(cfg.spacy_model)
        except OSError:
            _nlp = None
    return _nlp

def _classify_block_type(text: str) -> BlockType:
    """Classify a block as factual, reasoning, code, or filler."""
    code_patterns = [r'\bdef\s+\w+', r'\bclass\s+\w+', r'```', r'=\s*\d+', r'import\s+\w+']
    if any(re.search(p, text) for p in code_patterns):
        return BlockType.CODE
    # Reasoning: contains logical connectors, steps
    reasoning_words = {"because", "therefore", "thus", "consequently", "since", 
                       "first", "second", "finally", "step"}
    if any(w in text.lower() for w in reasoning_words):
        return BlockType.REASONING
    # Filler: short, hedged, non-specific
    filler_patterns = [r'\b(might|possibly|perhaps|could be|seems like|I think)\b']
    if any(re.search(p, text, re.IGNORECASE) for p in filler_patterns):
        return BlockType.FILLER
    return BlockType.FACTUAL

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using spaCy or naive split."""
    doc = _get_nlp()(text)
    if doc and doc.is_parsed:
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # Naive fallback: split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def _split_clauses(sentence: str) -> List[str]:
    """Split a sentence into independent clauses using conjunctions."""
    # Split on coordinating conjunctions preceded by comma or semicolon
    clauses = re.split(r',?\s+(?:and|but|or|yet|so|for|nor|although|while|because|since|however|therefore|thus)\s+', sentence)
    return [c.strip() for c in clauses if len(c.strip()) > 5]

def segment(text: str) -> List[ClaimBlock]:
    """
    Break LLM output into atomic claim-blocks.
    
    Dynamic sizing: multi-fact sentences become multiple blocks.
    Each block tagged with type, position, and surrounding context.
    """
    sentences = _split_sentences(text)
    blocks = []
    pos = 0
    
    for sent in sentences:
        # Check if this sentence has multiple independent facts
        clauses = _split_clauses(sent)
        if len(clauses) > 1 and len(clauses) <= 4:
            # Multi-fact sentence → split into blocks
            for i, clause in enumerate(clauses):
                clause_pos = text.find(clause, pos)
                blocks.append(ClaimBlock(
                    text=clause,
                    position=clause_pos if clause_pos >= 0 else pos,
                    end_position=clause_pos + len(clause) if clause_pos >= 0 else pos + len(clause),
                    block_type=_classify_block_type(clause),
                    code=BlockCode.AX,  # Default to uncertain — will be scored in S2
                    context_before=text[max(0, clause_pos-50):clause_pos] if clause_pos >= 0 else "",
                    context_after=text[clause_pos+len(clause):clause_pos+len(clause)+50] if clause_pos >= 0 else "",
                ))
        else:
            sent_pos = text.find(sent, pos)
            blocks.append(ClaimBlock(
                text=sent,
                position=sent_pos if sent_pos >= 0 else pos,
                end_position=sent_pos + len(sent) if sent_pos >= 0 else pos + len(sent),
                block_type=_classify_block_type(sent),
                code=BlockCode.AX,
                context_before=text[max(0, (sent_pos if sent_pos >= 0 else pos)-50):sent_pos if sent_pos >= 0 else pos],
                context_after=text[(sent_pos + len(sent)) if sent_pos >= 0 else pos + len(sent):(sent_pos + len(sent) + 50) if sent_pos >= 0 else pos + len(sent) + 50],
            ))
        
        if sent_pos >= 0:
            pos = sent_pos + len(sent)
    
    # If no blocks detected (empty text or parse failure), return the whole text as one block
    if not blocks:
        blocks.append(ClaimBlock(
            text=text,
            position=0,
            end_position=len(text),
            block_type=BlockType.FILLER,
            code=BlockCode.AX,
        ))
    
    return blocks
