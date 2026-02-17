"""
Igbo phoneme symbol set for VITS.

We use pre-computed integer phoneme indices in filelists,
so this module defines the symbol list for model vocab_size
and for any runtime text→sequence conversion.

The symbol order matches igbo_tts/phonemes.py exactly.
Symbol 0 is the blank/pad token used by VITS for MAS alignment.
"""

import sys
from pathlib import Path

# Import our canonical Igbo phoneme inventory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from igbo_tts.phonemes import (
    SPECIAL_TOKENS, VOWELS, CONSONANTS, TONES, PUNCTUATION,
    PHONEME_TO_ID, VOCAB_SIZE,
)

# Build ordered symbol list matching igbo_tts.phonemes indices
# Index 0 = <pad> (also serves as VITS blank token for MAS)
_all_tokens = SPECIAL_TOKENS + VOWELS + CONSONANTS + TONES + PUNCTUATION
symbols = _all_tokens

# Verify consistency
assert len(symbols) == VOCAB_SIZE, f"Symbol count {len(symbols)} != VOCAB_SIZE {VOCAB_SIZE}"
assert symbols[0] == "<pad>", f"Symbol 0 must be <pad>, got {symbols[0]}"
