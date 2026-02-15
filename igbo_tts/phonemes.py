#!/usr/bin/env python3
"""
Igbo Phoneme Inventory and Vocabulary Mapping

Defines the complete Igbo phoneme set (IPA), tone markers,
and the integer vocabulary mapping used by the TTS model.

Igbo phonology:
- 8 vowels (with ATR harmony: +ATR {i, e, o, u} vs -ATR {ɪ, ɔ, ʊ} + open {a})
- 28 consonants (including digraph-mapped labio-velars, palatals, labials)
- 2 level tones: HIGH (˥) and LOW (˩), plus downstep (ꜜ)
"""

from typing import Dict, List


# ── Igbo IPA Phoneme Sets ──────────────────────────────────────────

VOWELS: List[str] = [
    "a",   # open central
    "e",   # close-mid front (+ATR)
    "i",   # close front (+ATR)
    "ɪ",   # near-close front (-ATR)  — orthographic ị
    "o",   # close-mid back (+ATR)
    "ɔ",   # open-mid back (-ATR)     — orthographic ọ
    "u",   # close back (+ATR)
    "ʊ",   # near-close back (-ATR)   — orthographic ụ
]

CONSONANTS: List[str] = [
    "b",    # voiced bilabial plosive
    "d",    # voiced alveolar plosive
    "f",    # voiceless labiodental fricative
    "ɡ",    # voiced velar plosive
    "h",    # voiceless glottal fricative
    "dʒ",   # voiced postalveolar affricate   — orthographic j
    "k",    # voiceless velar plosive
    "l",    # alveolar lateral approximant
    "m",    # bilabial nasal
    "n",    # alveolar nasal
    "ŋ",    # velar nasal                     — orthographic ṅ
    "p",    # voiceless bilabial plosive
    "r",    # alveolar trill/tap
    "s",    # voiceless alveolar fricative
    "t",    # voiceless alveolar plosive
    "v",    # voiced labiodental fricative
    "w",    # labio-velar approximant
    "j",    # palatal approximant             — orthographic y
    "z",    # voiced alveolar fricative
    "tʃ",   # voiceless postalveolar affricate — orthographic ch
    "ʃ",    # voiceless postalveolar fricative — orthographic sh
    "ɡ͡b",   # voiced labial-velar plosive     — orthographic gb
    "k͡p",   # voiceless labial-velar plosive  — orthographic kp
    "ɣ",    # voiced velar fricative           — orthographic gh
    "ɡʷ",   # labialized voiced velar plosive  — orthographic gw
    "kʷ",   # labialized voiceless velar plos. — orthographic kw
    "ɲ",    # palatal nasal                    — orthographic ny
    "ŋʷ",   # labialized velar nasal           — orthographic nw
]

# ── Tone Markers ────────────────────────────────────────────────────

TONE_HIGH = "˥"       # high tone (acute accent: á, é, í, ó, ú, etc.)
TONE_LOW = "˩"        # low tone  (grave accent: à, è, ì, ò, ù, etc.)
TONE_DOWNSTEP = "ꜜ"   # downstep  (phonological — inserted between words)

TONES: List[str] = [TONE_HIGH, TONE_LOW, TONE_DOWNSTEP]

# ── Special Tokens ──────────────────────────────────────────────────

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPACE_TOKEN = " "
SIL_TOKEN = "<sil>"  # sentence boundary silence

SPECIAL_TOKENS: List[str] = [PAD_TOKEN, UNK_TOKEN, SPACE_TOKEN, SIL_TOKEN]

# ── Punctuation ─────────────────────────────────────────────────────

PUNCTUATION: List[str] = [".", ",", "!", "?", ";", ":", "-"]


def build_vocab() -> Dict[str, int]:
    """
    Build the complete phoneme-to-integer vocabulary.

    Order: special tokens → vowels → consonants → tones → punctuation.
    Total vocab size is ~50 tokens.

    Returns:
        Dict mapping phoneme string to integer ID.
    """
    vocab: Dict[str, int] = {}

    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    for ph in VOWELS:
        vocab[ph] = len(vocab)

    for ph in CONSONANTS:
        vocab[ph] = len(vocab)

    for tone in TONES:
        vocab[tone] = len(vocab)

    for p in PUNCTUATION:
        vocab[p] = len(vocab)

    return vocab


# Pre-built vocabulary (module-level constant)
PHONEME_TO_ID: Dict[str, int] = build_vocab()
ID_TO_PHONEME: Dict[int, str] = {v: k for k, v in PHONEME_TO_ID.items()}
VOCAB_SIZE: int = len(PHONEME_TO_ID)


if __name__ == "__main__":
    print(f"Igbo phoneme vocabulary size: {VOCAB_SIZE}")
    print(f"  Vowels:     {len(VOWELS)}")
    print(f"  Consonants: {len(CONSONANTS)}")
    print(f"  Tones:      {len(TONES)}")
    print(f"  Special:    {len(SPECIAL_TOKENS)}")
    print(f"  Punctuation:{len(PUNCTUATION)}")
    print()
    for ph, idx in PHONEME_TO_ID.items():
        print(f"  {idx:3d} → {ph!r}")
