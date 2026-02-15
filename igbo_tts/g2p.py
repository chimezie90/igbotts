#!/usr/bin/env python3
"""
Igbo Grapheme-to-Phoneme (G2P) Converter

Rule-based G2P for Standard Igbo. Igbo orthography is highly phonemic
(nearly 1:1 letter-to-sound), so a rule-based approach works well.

Pipeline:
  1. Unicode NFC normalization
  2. Tokenize into words (split on whitespace/punctuation)
  3. For each word: match digraphs first (longest match), then single graphemes
  4. Extract tone diacritics from vowels (acute → HIGH, grave → LOW)
  5. Output: list of phoneme tokens with interleaved tone markers

The converter also provides `text_to_indices()` for direct model input.
"""

import unicodedata
import re
from typing import Dict, List, Tuple

from .phonemes import (
    PHONEME_TO_ID,
    ID_TO_PHONEME,
    VOCAB_SIZE,
    TONE_HIGH,
    TONE_LOW,
    PAD_TOKEN,
    UNK_TOKEN,
    SPACE_TOKEN,
    SIL_TOKEN,
    VOWELS,
)

import logging

logger = logging.getLogger(__name__)


# ── Orthography → IPA Mapping ──────────────────────────────────────
# Digraphs MUST be checked before single characters (longest-match-first).

_DIGRAPH_MAP: Dict[str, str] = {
    "ch": "tʃ",
    "sh": "ʃ",
    "gb": "ɡ͡b",
    "kp": "k͡p",
    "gh": "ɣ",
    "gw": "ɡʷ",
    "kw": "kʷ",
    "nw": "ŋʷ",
    "ny": "ɲ",
}

_SINGLE_MAP: Dict[str, str] = {
    "a": "a",
    "b": "b",
    "d": "d",
    "e": "e",
    "f": "f",
    "g": "ɡ",
    "h": "h",
    "i": "i",
    "ị": "ɪ",
    "j": "dʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "ṅ": "ŋ",
    "o": "o",
    "ọ": "ɔ",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "ụ": "ʊ",
    "v": "v",
    "w": "w",
    "y": "j",
    "z": "z",
}

# Vowel graphemes (used to detect tone-bearing characters)
_VOWEL_GRAPHEMES = set("aeioịọụuɪɔʊ")

# Punctuation that we keep as tokens
_PUNCT_SET = set(".,!?;:-")

# Regex to split text into word/punctuation tokens
_TOKEN_RE = re.compile(r"[a-zA-ZạẠịỊọỌụỤṅṄàáèéìíòóùúÀÁÈÉÌÍÒÓÙÚ]+|[.,!?;:\-]")


def _normalize(text: str) -> str:
    """Unicode NFC normalization + lowercase."""
    return unicodedata.normalize("NFC", text).lower()


def _strip_tone(char: str) -> Tuple[str, str]:
    """
    Decompose a (possibly accented) vowel into base + tone.

    Returns (base_char, tone_marker) where tone_marker is one of
    TONE_HIGH, TONE_LOW, or "" (unmarked).

    Igbo convention:
      - acute accent (á) → high tone
      - grave accent (à) → low tone
      - no accent → contextual default (we emit no tone marker)
    """
    # NFD decomposes e.g. 'á' → 'a' + '\u0301'
    decomposed = unicodedata.normalize("NFD", char)
    base = ""
    tone = ""

    for c in decomposed:
        cat = unicodedata.category(c)
        if cat.startswith("M"):
            # Combining mark — check for tone accents
            if c == "\u0301":      # combining acute accent
                tone = TONE_HIGH
            elif c == "\u0300":    # combining grave accent
                tone = TONE_LOW
            elif c == "\u0323":    # combining dot below (ị, ọ, ụ)
                base += c          # keep the dot-below as part of base
            else:
                base += c          # keep other combining marks
        else:
            base += c

    # Re-compose the base (so ị stays as ị, not i + dot)
    base = unicodedata.normalize("NFC", base)
    return base, tone


def _graphemes_to_phonemes(word: str) -> List[str]:
    """
    Convert a single Igbo word (lowercase, NFC) to a list of IPA phoneme
    tokens, including interleaved tone markers.

    Uses longest-match-first: digraphs before single characters.
    """
    phonemes: List[str] = []
    i = 0

    while i < len(word):
        # Try digraph match first (2 characters)
        if i + 1 < len(word):
            # Strip tones from both chars to check the digraph
            base1, tone1 = _strip_tone(word[i])
            base2, _ = _strip_tone(word[i + 1])
            digraph = base1 + base2

            if digraph in _DIGRAPH_MAP:
                phonemes.append(_DIGRAPH_MAP[digraph])
                # Tone on the first character of a digraph applies to
                # the whole consonant (unusual but preserved for completeness)
                if tone1:
                    phonemes.append(tone1)
                i += 2
                continue

        # Single character match
        base, tone = _strip_tone(word[i])

        if base in _SINGLE_MAP:
            phonemes.append(_SINGLE_MAP[base])
            # Append tone marker right after the vowel phoneme
            if tone:
                phonemes.append(tone)
        else:
            # Unknown character — skip with warning
            logger.debug(f"Unknown grapheme: {word[i]!r} (base={base!r})")

        i += 1

    return phonemes


class IgboG2P:
    """
    Igbo grapheme-to-phoneme converter.

    Usage:
        g2p = IgboG2P()
        phonemes = g2p.text_to_phonemes("Nnọọ, kedụ ka ị mere?")
        indices  = g2p.text_to_indices("Nnọọ, kedụ ka ị mere?")
    """

    def __init__(self):
        self.phoneme_to_id = PHONEME_TO_ID
        self.id_to_phoneme = ID_TO_PHONEME

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert Igbo text to a flat list of IPA phoneme tokens.

        Words are separated by space tokens. Punctuation is preserved.
        A <sil> token is prepended and appended for sentence boundaries.
        """
        if not text or not text.strip():
            return []

        text = _normalize(text)
        tokens = _TOKEN_RE.findall(text)

        phonemes: List[str] = [SIL_TOKEN]

        for i, token in enumerate(tokens):
            if token in _PUNCT_SET:
                phonemes.append(token)
            else:
                word_phonemes = _graphemes_to_phonemes(token)
                phonemes.extend(word_phonemes)

            # Add space between tokens (but not after last)
            if i < len(tokens) - 1:
                phonemes.append(SPACE_TOKEN)

        phonemes.append(SIL_TOKEN)
        return phonemes

    def text_to_indices(self, text: str) -> List[int]:
        """Convert Igbo text to a list of phoneme integer indices."""
        phonemes = self.text_to_phonemes(text)
        indices = []
        for ph in phonemes:
            if ph in self.phoneme_to_id:
                indices.append(self.phoneme_to_id[ph])
            else:
                logger.warning(f"Unknown phoneme: {ph!r}, using <unk>")
                indices.append(self.phoneme_to_id[UNK_TOKEN])
        return indices

    def indices_to_phonemes(self, indices: List[int]) -> List[str]:
        """Convert phoneme indices back to phoneme strings."""
        return [self.id_to_phoneme.get(idx, UNK_TOKEN) for idx in indices]

    def process_text(self, text: str) -> List[List[str]]:
        """Compatibility method: returns [[phonemes]]."""
        phonemes = self.text_to_phonemes(text)
        return [phonemes] if phonemes else [[]]

    def to_dict(self) -> dict:
        """Serialize processor state."""
        return {
            "phoneme_to_id": self.phoneme_to_id,
            "id_to_phoneme": self.id_to_phoneme,
            "processor_type": "igbo_rule_g2p",
        }

    @classmethod
    def from_dict(cls, state_dict: dict) -> "IgboG2P":
        """Restore processor from serialized state."""
        g2p = cls()
        if "phoneme_to_id" in state_dict:
            g2p.phoneme_to_id = state_dict["phoneme_to_id"]
            g2p.id_to_phoneme = {v: k for k, v in g2p.phoneme_to_id.items()}
        return g2p


# ── MFA Dictionary Generation ──────────────────────────────────────

def generate_mfa_dictionary(word_list: List[str], output_path: str):
    """
    Generate an MFA-compatible pronunciation dictionary from a word list.

    Each line: WORD\tPHONEME1 PHONEME2 ...
    (tab-separated, phonemes space-separated, no tone markers)

    Args:
        word_list: list of Igbo words
        output_path: path to write the dictionary file
    """
    g2p = IgboG2P()
    seen = set()

    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted(set(word_list)):
            word_lower = _normalize(word)
            if word_lower in seen or not word_lower.strip():
                continue
            seen.add(word_lower)

            phonemes = _graphemes_to_phonemes(word_lower)
            # Remove tone markers for MFA (it uses acoustic info for prosody)
            phonemes_no_tone = [
                p for p in phonemes if p not in {TONE_HIGH, TONE_LOW}
            ]
            if phonemes_no_tone:
                f.write(f"{word_lower}\t{' '.join(phonemes_no_tone)}\n")


if __name__ == "__main__":
    g2p = IgboG2P()

    test_sentences = [
        "Nnọọ, kedụ ka ị mere?",
        "Igbo bụ asụsụ dị mma",
        "Ànyị nà-àgụ akwụkwọ",
        "Chineke dị mma",
        "Ọ nà-àchọ nri",
    ]

    print("=" * 70)
    print("Igbo G2P Test")
    print("=" * 70)

    for sentence in test_sentences:
        phonemes = g2p.text_to_phonemes(sentence)
        indices = g2p.text_to_indices(sentence)
        print(f"\nText:     {sentence}")
        print(f"Phonemes: {' '.join(phonemes)}")
        print(f"Indices:  {indices[:30]}{'...' if len(indices) > 30 else ''}")

    print(f"\nVocabulary size: {g2p.get_vocab_size()}")
