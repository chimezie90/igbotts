"""
Igbo grapheme (character-level) symbol set for multi-speaker VITS.

For multi-speaker training with 119k+ samples, we use character-level input
instead of phoneme-level. Igbo orthography is fairly phonemic, so the model
can learn grapheme-to-sound mapping implicitly from the data.

The symbol set covers:
- Standard Igbo alphabet (36 letters including dotted variants)
- Toned vowels (acute = high tone, grave = low tone)
- Common punctuation
- Space and special tokens
"""

import re
import unicodedata
from typing import Dict, List, Tuple

# ── Special Tokens ──────────────────────────────────────────────
PAD = "<pad>"      # index 0, also VITS blank token
UNK = "<unk>"      # unknown character
SPACE = " "

SPECIAL_TOKENS = [PAD, UNK, SPACE]

# ── Igbo Base Letters (lowercase) ───────────────────────────────
# Standard Igbo alphabet: 28 consonants + 8 vowels
BASE_VOWELS = list("aeiouọịụ")
BASE_CONSONANTS = list("bcdfghjklmnṅprstwvyz")

# ── Toned Vowels ────────────────────────────────────────────────
# Acute accent (high tone) and grave accent (low tone)
TONED_VOWELS = list("àáèéìíòóùú")

# ── Punctuation ─────────────────────────────────────────────────
PUNCTUATION = list(".,!?;:-'")

# ── Complete Symbol List ────────────────────────────────────────
symbols = (
    SPECIAL_TOKENS
    + BASE_VOWELS
    + BASE_CONSONANTS
    + TONED_VOWELS
    + PUNCTUATION
)

# Build lookup
SYMBOL_TO_ID: Dict[str, int] = {s: i for i, s in enumerate(symbols)}
ID_TO_SYMBOL: Dict[int, str] = {i: s for i, s in enumerate(symbols)}
VOCAB_SIZE = len(symbols)


def normalize_text(text: str) -> str:
    """Normalize Igbo text for model input.

    - Lowercases
    - Normalizes Unicode (NFC)
    - Strips disfluency markers [um], [?], [uh]
    - Maps uncommon characters to closest equivalents
    - Removes characters not in our symbol set
    """
    text = text.lower().strip()
    text = unicodedata.normalize("NFC", text)

    # Remove disfluency markers
    text = re.sub(r'\[(?:um|uh|\?)\]', '', text)

    # Normalize quotes and dashes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # smart quotes
    text = text.replace('\u201c', '').replace('\u201d', '')    # double quotes
    text = text.replace('\u2014', '-').replace('\u2013', '-')  # em/en dash

    # Map dotted variants (ȧ→a, ė→e, etc.) — these are annotation artifacts
    dot_above_map = {'ȧ': 'a', 'ė': 'e', 'ȯ': 'o'}
    for src, dst in dot_above_map.items():
        text = text.replace(src, dst)

    # Map macron variants (ā→a, etc.) — length markers not standard in Igbo
    macron_map = {'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u'}
    for src, dst in macron_map.items():
        text = text.replace(src, dst)

    # Map ǹ → n (grave n)
    text = text.replace('ǹ', 'n')

    # Strip combining diacritics we don't handle
    # Keep combining dot below (0323) and combining acute/grave (0301, 0300)
    # but remove others
    cleaned = []
    for ch in text:
        if ch in SYMBOL_TO_ID:
            cleaned.append(ch)
        elif unicodedata.category(ch).startswith('M'):
            # Skip combining marks we don't recognize
            continue
        else:
            # Skip unrecognized characters entirely
            continue
    text = ''.join(cleaned)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_to_sequence(text: str) -> List[int]:
    """Convert normalized text to integer sequence."""
    text = normalize_text(text)
    sequence = []
    for ch in text:
        if ch in SYMBOL_TO_ID:
            sequence.append(SYMBOL_TO_ID[ch])
        else:
            sequence.append(SYMBOL_TO_ID[UNK])
    return sequence


def sequence_to_text(sequence: List[int]) -> str:
    """Convert integer sequence back to text."""
    return ''.join(ID_TO_SYMBOL.get(i, '?') for i in sequence)


if __name__ == '__main__':
    print(f"Igbo grapheme vocabulary: {VOCAB_SIZE} symbols")
    print(f"  Special: {len(SPECIAL_TOKENS)}")
    print(f"  Base vowels: {len(BASE_VOWELS)}")
    print(f"  Base consonants: {len(BASE_CONSONANTS)}")
    print(f"  Toned vowels: {len(TONED_VOWELS)}")
    print(f"  Punctuation: {len(PUNCTUATION)}")
    print()
    for sym, idx in SYMBOL_TO_ID.items():
        print(f"  {idx:3d} → {sym!r}")
    print()

    # Test
    test = "Nnọọ, kedụ ka ị mere?"
    norm = normalize_text(test)
    seq = text_to_sequence(test)
    print(f"Input:      {test}")
    print(f"Normalized: {norm}")
    print(f"Sequence:   {seq}")
    print(f"Roundtrip:  {sequence_to_text(seq)}")
