"""
Igbo text processing for VITS.

Two modes:
1. Pre-computed: filelists contain space-separated integer indices.
   Used during training (fastest, no G2P at runtime).
2. Runtime: Igbo text → G2P → integer sequence.
   Used during inference.
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from igbo_tts.g2p import IgboG2P
from igbo_tts.phonemes import PHONEME_TO_ID, VOCAB_SIZE

# Singleton G2P instance
_g2p = None


def get_g2p() -> IgboG2P:
    global _g2p
    if _g2p is None:
        _g2p = IgboG2P()
    return _g2p


def text_to_sequence(text: str) -> List[int]:
    """Convert Igbo text to phoneme index sequence at runtime."""
    g2p = get_g2p()
    return g2p.text_to_indices(text)


def cleaned_text_to_sequence(text: str) -> List[int]:
    """Parse pre-computed integer sequence from filelist.

    Expected format: space-separated integers, e.g. "3 4 12 7 2 15"
    """
    return [int(x) for x in text.strip().split()]


def sequence_to_text(sequence: List[int]) -> str:
    """Convert integer sequence back to phoneme string (for debugging)."""
    from igbo_tts.phonemes import ID_TO_PHONEME
    return " ".join(ID_TO_PHONEME.get(idx, "?") for idx in sequence)
