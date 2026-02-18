"""
Multi-speaker Igbo VITS training script.

Wraps vits_repo/train_ms.py with Igbo-specific setup:
- Monkey-patches text processing to use Igbo grapheme symbols
- Patches load_wav_to_torch to support FLAC files
- Uses igbo_vits/configs/igbo_multispeaker.json config

Usage:
    python train_ms_igbo.py -c igbo_vits/configs/igbo_multispeaker.json -m output_vits_igbo_ms
"""
import os
import sys
import types

# ── Setup paths ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VITS_DIR = os.path.join(PROJECT_ROOT, "vits_repo")
sys.path.insert(0, VITS_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ── Monkey-patch monotonic_align (skip Cython build) ──
ma_module = types.ModuleType("monotonic_align")
sys.modules["monotonic_align"] = ma_module

# We need a working maximum_path for training (MAS alignment)
import numpy as np
import torch


def _maximum_path_numpy(neg_cent, mask):
    """Monotonic alignment search (numpy fallback for training).

    This is the core MAS algorithm used by VITS to find the optimal
    monotonic alignment between encoder and decoder sequences.
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    mask = mask.data.cpu().numpy()

    b, t_t, t_s = neg_cent.shape
    path = np.zeros_like(neg_cent, dtype=np.int32)

    for batch in range(b):
        # Build Q table
        Q = np.full((t_t, t_s), fill_value=-np.inf, dtype=np.float32)
        for t in range(t_t):
            for s in range(t_s):
                if mask[batch, t, s] == 0:
                    continue
                if t == 0 and s == 0:
                    Q[t, s] = neg_cent[batch, t, s]
                elif t == 0:
                    continue  # can't be at s>0 when t=0
                elif s == 0:
                    Q[t, s] = Q[t - 1, s] + neg_cent[batch, t, s]
                else:
                    Q[t, s] = max(Q[t - 1, s], Q[t - 1, s - 1]) + neg_cent[batch, t, s]

        # Backtrace
        s = t_s - 1
        for t in range(t_t - 1, -1, -1):
            path[batch, t, s] = 1
            if t > 0 and s > 0:
                if Q[t - 1, s - 1] > Q[t - 1, s]:
                    s -= 1

    return torch.from_numpy(path).to(device=device, dtype=dtype)


ma_module.maximum_path = _maximum_path_numpy

# ── Monkey-patch VITS text processing to use Igbo graphemes ──
import igbo_vits.text_processing_grapheme as igbo_text
import text as vits_text_module

vits_text_module.cleaned_text_to_sequence = igbo_text.cleaned_text_to_sequence
vits_text_module.text_to_sequence = igbo_text.text_to_sequence

# Replace symbols in text.symbols with our grapheme set
from igbo_vits.graphemes import symbols as igbo_symbols
from text import symbols as _orig_symbols

# Patch the symbols list that VITS uses for vocab_size
import text.symbols
text.symbols.symbols = igbo_symbols

# ── Monkey-patch load_wav_to_torch to support FLAC ──
import soundfile as sf
import utils as vits_utils


def load_wav_to_torch(full_path):
    """Load audio file (WAV or FLAC) and return as torch tensor."""
    data, sampling_rate = sf.read(full_path, dtype='float32')
    # soundfile returns float32 in [-1, 1], VITS expects int16-scale
    data = data * 32768.0
    return torch.FloatTensor(data), sampling_rate


vits_utils.load_wav_to_torch = load_wav_to_torch

# Also patch it in data_utils which imports it directly
import data_utils
data_utils.load_wav_to_torch = load_wav_to_torch

# ── Now import and run the multi-speaker training ──
print(f"Igbo multi-speaker VITS training")
print(f"  Grapheme vocab size: {len(igbo_symbols)}")
print(f"  Symbols: {igbo_symbols[:10]}...")
print()

# Import train_ms which will use our patched modules
import train_ms
train_ms.main()
