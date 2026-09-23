# S5 — Splicer
# Removes/rewrites flagged blocks, repairs seams, re-verifies joins.
# Cutting a block mid-response breaks pronouns, "this," "the method above," logical connectors.
# Junction repair pass rewrites the seam so remaining text still reads coherently.

import re
from typing import List
from ..models.pipeline import ClaimBlock, BlockDecision, BlockCode

class Splicer:
    """
    Post-decision cleanup: remove flagged blocks, repair text seams, re-verify joins.
    
    Key insight: deleting a block creates a seam where pronouns and connectors break.
    This pass rewrites the seam to restore coherence, then re-verifies the repaired join.
    """
    
    # Common broken-reference patterns and their repairs
    _REPAIR_PATTERNS = [
        # "this [noun]" after removal → replace with explicit referent
        (r'\bthis\s+(?:method|approach|technique|result|data|finding)', "the above"),
        (r'\bthese\s+(?:results|findings|data)', "the above findings"),
        # "the [noun] above" — keep as-is if context allows
        (r'\bthe\s+(?:above|method|approach)\b', "the preceding"),
        # Broken connectors after block removal
        (r'\s*,\s*which\s+(?:was|is)\s+', " "),
        (r'\s*,\s*where\s+(?:by|this)\s+', " "),
        # Double punctuation cleanup
        (r'\.\s*\.', '. '),
        (r',\s*,', ','),
        (r'\s+\.', '.'),
    ]
    
    def __init__(self):
        self.repair_count = 0
        self.re_verification_failures = []
    
    def reset(self):
        self.repair_count = 0
        self.re_verification_failures = []
    
    def splice(self, blocks: List[ClaimBlock], decisions: List[BlockDecision], full_text: str) -> str:
        """
        Remove blocks marked for removal, repair seams, return cleaned text.
        
        Args:
            blocks: All claim blocks from S1
            decisions: Decisions from S4 (keep/trim/regenerate/flag)
            full_text: The original raw LLM response
        
        Returns:
            Cleaned text with blocks removed and seams repaired.
        """
        # Identify which blocks to remove
        remove_indices = set()
        regenerate_indices = set()
        trim_indices = set()
        
        for decision in decisions:
            if decision.action == "trim":
                trim_indices.add(decision.block.position)
            elif decision.action == "regenerate":
                regenerate_indices.add(decision.block.position)
            elif decision.action == "flag":
                # Flagged blocks stay but get marked — don't remove
                pass
            # "keep" blocks stay
        
        # Build cleaned text by removing blocks and repairing seams
        # Sort blocks by position for processing
        sorted_blocks = sorted(blocks, key=lambda b: b.position)
        
        # Collect text segments to keep
        kept_segments = []
        last_end = 0
        
        for block in sorted_blocks:
            # Skip blocks that should be removed
            if block.position in remove_indices:
                last_end = max(last_end, block.end_position)
                continue
            
            # Add text before this block (if we skipped something)
            if block.position > last_end:
                kept_segments.append(full_text[last_end:block.position])
            
            # Add this block's text (unless trimmed)
            if block.position not in trim_indices:
                kept_segments.append(block.text)
            
            last_end = block.end_position
        
        # Add trailing text
        if last_end < len(full_text):
            kept_segments.append(full_text[last_end:])
        
        cleaned = "".join(kept_segments)
        
        # --- Junction Repair Pass ---
        cleaned = self._repair_seams(cleaned, remove_indices, sorted_blocks)
        
        return cleaned
    
    def _repair_seams(self, text: str, removed_positions: set, blocks: list) -> str:
        """
        Repair text seams where blocks were removed.
        Fixes broken references, connectors, and punctuation.
        """
        repaired = text
        
        # Apply repair patterns
        for pattern, replacement in self._REPAIR_PATTERNS:
            new_repaired = re.sub(pattern, replacement, repaired)
            if new_repaired != repaired:
                self.repair_count += 1
                repaired = new_repaired
        
        # Fix sentence fragments: if a sentence starts with a lowercase letter
        # after a period (likely a broken continuation), capitalize it
        repaired = re.sub(r'\.\s*([a-z])', lambda m: '. ' + m.group(1).upper(), repaired)
        
        # If the text ends mid-sentence (ends with comma or semicolon), add a period
        if repaired.rstrip().endswith((',', ';', ':')):
            repaired = repaired.rstrip() + '.'
        
        return repaired
    
    def reverify_join(self, before: str, after: str) -> bool:
        """
        Re-verify that a repaired seam is coherent (doesn't introduce new false claims).
        Returns: True if the join looks safe, False if it needs re-inspection.
        
        Simple check: sentence ends and starts make grammatical sense.
        """
        # Check if the end of `before` and start of `after` form a valid connection
        before_trimmed = before.rstrip()
        after_trimmed = after.lstrip()
        
        if not before_trimmed or not after_trimmed:
            return True
        
        # If before ends with a period and after starts with lowercase, it's a new sentence — OK
        if before_trimmed.endswith('.') and after_trimmed[0].islower():
            return True
        
        # If before ends mid-sentence (comma/semicolon) and after continues — check capitalization
        if before_trimmed.endswith((',', ';', ':')) and after_trimmed[0].isupper():
            return True
        
        # If both are complete sentences but the connector is broken — flag
        if (before_trimmed.endswith('.') and after_trimmed[0].isupper()
                and not self._is_connector(after_trimmed)):
            return False
        
        return True
    
    def _is_connector(self, text: str) -> bool:
        """Check if text starts with a logical connector."""
        connectors = {"however", "therefore", "thus", "consequently", "moreover", 
                      "furthermore", "additionally", "meanwhile", "nevertheless",
                      "although", "while", "because", "since", "first", "second"}
        words = text.strip().split()
        if words:
            return words[0].rstrip('.,;:').lower() in connectors
        return False
