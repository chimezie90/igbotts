#!/usr/bin/env python3
"""
Igbo TTS Inference — convert Igbo text to speech audio.

Loads a trained Kokoro model checkpoint + Igbo G2P, runs inference,
and synthesises audio via HiFi-GAN vocoder.
"""

import argparse
import json
import pickle
import sys
import logging
from pathlib import Path
from typing import List, Optional

import torch

# Kokoro model & audio utilities — imported from the cloned training repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kokoro_training"))
from kokoro.model import KokoroModel
from audio.vocoder_manager import VocoderManager
from audio.audio_utils import AudioUtils

from .g2p import IgboG2P

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IgboTTS:
    """Igbo text-to-speech inference wrapper."""

    def __init__(
        self,
        model_dir: str,
        device: str = None,
        vocoder_type: str = "hifigan",
        vocoder_path: str = None,
    ):
        self.model_dir = Path(model_dir)

        # Device
        self.device = AudioUtils.validate_device(device)
        logger.info(f"Using device: {self.device}")

        # Load config
        self._load_config()
        self.audio_utils = AudioUtils(self.sample_rate)

        # Igbo G2P
        self.phoneme_processor = self._load_phoneme_processor()

        # Model
        self.model = self._load_model()

        # Vocoder
        self.vocoder_manager = VocoderManager(vocoder_type, vocoder_path, self.device)

    # ── Config ──────────────────────────────────────────────────────

    def _load_config(self):
        config_path = self.model_dir / "model_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                cfg = json.load(f)
            self.sample_rate = cfg.get("sample_rate", 22050)
            self.hop_length = cfg.get("hop_length", 256)
            self.n_mels = cfg.get("n_mels", 80)
            self.hidden_dim = cfg.get("hidden_dim", 512)
            self.n_encoder_layers = cfg.get("n_encoder_layers", 6)
            self.n_decoder_layers = cfg.get("n_decoder_layers", 6)
            self.n_heads = cfg.get("n_heads", 8)
            self.encoder_ff_dim = cfg.get("encoder_ff_dim", 2048)
            self.decoder_ff_dim = cfg.get("decoder_ff_dim", 2048)
            self.encoder_dropout = cfg.get("encoder_dropout", 0.1)
            self.max_decoder_seq_len = cfg.get("max_decoder_seq_len", 4000)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning("model_config.json not found, using defaults")
            self.sample_rate = 22050
            self.hop_length = 256
            self.n_mels = 80
            self.hidden_dim = 512
            self.n_encoder_layers = 6
            self.n_decoder_layers = 6
            self.n_heads = 8
            self.encoder_ff_dim = 2048
            self.decoder_ff_dim = 2048
            self.encoder_dropout = 0.1
            self.max_decoder_seq_len = 4000

    # ── Phoneme processor ───────────────────────────────────────────

    def _load_phoneme_processor(self) -> IgboG2P:
        pkl_path = self.model_dir / "phoneme_processor.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            processor = IgboG2P.from_dict(data)
            logger.info(f"Loaded Igbo G2P from {pkl_path}")
        else:
            logger.warning("phoneme_processor.pkl not found, creating fresh G2P")
            processor = IgboG2P()
        return processor

    # ── Model ───────────────────────────────────────────────────────

    def _load_model(self) -> KokoroModel:
        final = self.model_dir / "kokoro_igbo_final.pth"
        checkpoints = sorted(
            self.model_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        best = self.model_dir / "checkpoint_best.pth"

        if final.exists():
            model_path = final
        elif best.exists():
            model_path = best
        elif checkpoints:
            model_path = checkpoints[-1]
        else:
            raise FileNotFoundError(f"No model found in {self.model_dir}")

        logger.info(f"Loading model from {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("model", checkpoint)
        )

        # Detect max PE length from checkpoint (may have been extended during training)
        max_seq = self.max_decoder_seq_len
        for key in state_dict:
            if "positional_encoding.pe" in key:
                pe_len = state_dict[key].shape[1]
                if pe_len > max_seq:
                    max_seq = pe_len

        vocab_size = len(self.phoneme_processor.phoneme_to_id)
        model = KokoroModel(
            vocab_size=vocab_size,
            mel_dim=self.n_mels,
            hidden_dim=self.hidden_dim,
            n_encoder_layers=self.n_encoder_layers,
            n_heads=self.n_heads,
            encoder_ff_dim=self.encoder_ff_dim,
            encoder_dropout=self.encoder_dropout,
            n_decoder_layers=self.n_decoder_layers,
            decoder_ff_dim=self.decoder_ff_dim,
            max_decoder_seq_len=max_seq,
        )

        model.load_state_dict(state_dict, strict=False)

        # Fix NaN batch norm running stats (bfloat16 training on MPS accumulates NaN)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    if torch.isnan(module.running_mean).any():
                        logger.warning(f"Resetting NaN batch norm stats in {name}")
                        module.running_mean.zero_()
                        module.running_var.fill_(1.0)

        model.to(self.device)
        model.eval()
        logger.info(f"Model loaded (vocab_size={vocab_size})")
        return model

    # ── Synthesis ───────────────────────────────────────────────────

    def text_to_speech(
        self,
        text: str,
        output_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Convert Igbo text to speech.

        Args:
            text: Igbo text string
            output_path: optional WAV output path

        Returns:
            1-D audio tensor (float32, at self.sample_rate)
        """
        if not text:
            return torch.empty(0)

        logger.info(f"Input: {text}")

        # G2P
        phonemes = self.phoneme_processor.text_to_phonemes(text)
        indices = self.phoneme_processor.text_to_indices(text)
        logger.info(f"Phonemes ({len(phonemes)}): {' '.join(phonemes)}")

        phoneme_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)

        # Generate mel
        with torch.no_grad():
            mel_spec = self.model.forward_inference(
                phoneme_indices=phoneme_tensor,
                max_len=800,
                stop_threshold=0.01,
            )

        mel_spec = mel_spec.squeeze(0).float().cpu()
        # Replace NaN/Inf with zeros (can happen with bfloat16 inference)
        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            nan_count = torch.isnan(mel_spec).sum().item()
            logger.warning(f"Mel has {nan_count} NaN values — replacing with zeros")
            mel_spec = torch.nan_to_num(mel_spec, nan=0.0, posinf=0.0, neginf=-11.5)
        # Clamp to training range
        mel_spec = torch.clamp(mel_spec, min=-11.5, max=0.0)
        # Transpose (frames, mel_dim) → (mel_dim, frames) for vocoder
        mel_spec = mel_spec.transpose(0, 1)
        logger.info(f"Mel shape: {mel_spec.shape}")

        # Vocoder
        audio = self.vocoder_manager.mel_to_audio(mel_spec)
        logger.info(f"Audio: {len(audio) / self.sample_rate:.2f}s")

        if output_path:
            self.audio_utils.save_audio(audio, output_path)
            logger.info(f"Saved to {output_path}")

        return audio

    def batch_text_to_speech(self, texts: List[str], output_dir: str):
        """Convert multiple texts, saving each to output_dir/."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(texts):
            path = out / f"igbo_{i:03d}.wav"
            self.text_to_speech(text, str(path))


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Igbo TTS Inference")
    parser.add_argument("--model", "-m", required=True, help="Model directory")
    parser.add_argument("--text", "-t", help="Igbo text to synthesize")
    parser.add_argument("--text-file", "-f", help="File with texts (one per line)")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV path")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--vocoder", choices=["hifigan", "griffin_lim"], default="hifigan")
    parser.add_argument("--vocoder-path", help="Path to custom vocoder model")
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    tts = IgboTTS(
        model_dir=args.model,
        device=args.device,
        vocoder_type=args.vocoder,
        vocoder_path=args.vocoder_path,
    )

    if args.interactive:
        print("Interactive Igbo TTS — type 'quit' to exit")
        while True:
            text = input("\nIgbo text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                out = f"interactive_{abs(hash(text)) % 10000}.wav"
                tts.text_to_speech(text, out)
                print(f"Saved: {out}")

    elif args.text:
        tts.text_to_speech(args.text, args.output)

    elif args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [l.strip() for l in f if l.strip()]
        out_dir = Path(args.output).parent if Path(args.output).suffix else args.output
        tts.batch_text_to_speech(texts, str(out_dir))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
