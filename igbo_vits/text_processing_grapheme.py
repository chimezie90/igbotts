"""
Igbo grapheme-level text processing for multi-speaker VITS.

Unlike text_processing.py (phoneme-based), this uses character-level
input for training with large multi-speaker datasets where implicit
grapheme-to-phoneme learning is sufficient.
"""
from typing import List

from igbo_vits.graphemes import (
    text_to_sequence as _text_to_seq,
    normalize_text,
    VOCAB_SIZE,
    symbols,
)


def text_to_sequence(text: str, cleaner_names=None) -> List[int]:
    """Convert Igbo text to grapheme index sequence."""
    return _text_to_seq(text)


def cleaned_text_to_sequence(text: str) -> List[int]:
    """Convert pre-normalized text to grapheme index sequence.

    For multi-speaker filelists, text is already normalized by
    generate_filelists.py, so we just tokenize without re-normalizing.
    """
    return _text_to_seq(text)
