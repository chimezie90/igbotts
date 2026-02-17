#!/usr/bin/env python3
"""
Igbo TTS Inference using VITS.

End-to-end: Igbo text → G2P → VITS → WAV audio.
No separate vocoder needed (HiFi-GAN is built into VITS).

Usage:
    python -m igbo_vits.inference \
        --model output_vits_igbo \
        --text "Nnọọ, kedụ ka ị mere?"

    python -m igbo_vits.inference \
        --model output_vits_igbo \
        --interactive
"""

import argparse
import json
import sys
import os
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Add vits_repo to path
VITS_DIR = str(Path(__file__).resolve().parent.parent / "vits_repo")
sys.path.insert(0, VITS_DIR)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import commons
import utils
from models import SynthesizerTrn

from igbo_tts.g2p import IgboG2P
from igbo_vits.symbols import symbols

# Patch VITS text module
import igbo_vits.text_processing as igbo_text
import text as vits_text_module
vits_text_module.cleaned_text_to_sequence = igbo_text.cleaned_text_to_sequence
vits_text_module.text_to_sequence = lambda t, c: igbo_text.text_to_sequence(t)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IgboVITS:
    """Igbo text-to-speech using VITS."""

    def __init__(self, model_dir: str, device: str = None):
        self.model_dir = Path(model_dir)

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.g2p = IgboG2P()
        self._load_model()

    def _load_model(self):
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        self.hps = utils.HParams(**config)
        self.sample_rate = self.hps.data.sampling_rate

        self.model = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)

        # Find latest checkpoint
        ckpt = self._find_checkpoint()
        logger.info(f"Loading checkpoint: {ckpt}")

        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        # Handle DDP state dicts (keys prefixed with "module.")
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {total_params:,} params on {self.device}")

    def _find_checkpoint(self) -> str:
        ckpts = sorted(self.model_dir.glob("G_*.pth"))
        if not ckpts:
            raise FileNotFoundError(f"No G_*.pth checkpoints in {self.model_dir}")
        return str(ckpts[-1])

    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Convert Igbo text to speech.

        Args:
            text: Igbo text
            output_path: optional WAV save path
            noise_scale: VAE sampling temperature (lower = more stable)
            noise_scale_w: duration predictor noise (lower = more stable)
            length_scale: speech speed (1.0 = normal, >1 = slower)

        Returns:
            Audio waveform as numpy array
        """
        if not text.strip():
            return np.array([], dtype=np.float32)

        # G2P
        phonemes = self.g2p.text_to_phonemes(text)
        indices = self.g2p.text_to_indices(text)
        logger.info(f"Text: {text}")
        logger.info(f"Phonemes ({len(phonemes)}): {' '.join(phonemes)}")

        # Intersperse with blank tokens (as VITS expects when add_blank=True)
        if getattr(self.hps.data, "add_blank", True):
            indices = commons.intersperse(indices, 0)

        x = torch.LongTensor(indices).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(indices)]).to(self.device)

        with torch.no_grad():
            audio_tensor = self.model.infer(
                x, x_lengths,
                noise_scale=noise_scale,
                length_scale=length_scale,
                noise_scale_w=noise_scale_w,
            )[0][0, 0]

        audio = audio_tensor.cpu().numpy()
        logger.info(f"Generated {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")

        if output_path:
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)
            logger.info(f"Saved: {output_path}")

        return audio


def main():
    parser = argparse.ArgumentParser(description="Igbo VITS Inference")
    parser.add_argument("--model", "-m", required=True, help="Model directory")
    parser.add_argument("--text", "-t", help="Igbo text")
    parser.add_argument("--output", "-o", default="output_vits.wav")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--noise-scale-w", type=float, default=0.8)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    tts = IgboVITS(model_dir=args.model, device=args.device)

    if args.interactive:
        print("Interactive Igbo VITS — type 'quit' to exit")
        while True:
            text = input("\nIgbo text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                out = f"vits_{abs(hash(text)) % 10000}.wav"
                tts.synthesize(text, out,
                               noise_scale=args.noise_scale,
                               noise_scale_w=args.noise_scale_w,
                               length_scale=args.length_scale)
    elif args.text:
        tts.synthesize(args.text, args.output,
                       noise_scale=args.noise_scale,
                       noise_scale_w=args.noise_scale_w,
                       length_scale=args.length_scale)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
