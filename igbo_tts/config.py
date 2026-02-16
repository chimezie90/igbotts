#!/usr/bin/env python3
"""
Training Configuration for Igbo TTS (WAXAL dataset)

Adapted from kokoro_training/training/config_english.py for Igbo.
Key differences:
  - vocab_size ~50 (Igbo IPA + tones) vs ~100 (English ARPA + stress)
  - Data directory points to WAXAL Igbo dataset
  - Same model architecture — single-speaker, ~180 hrs
"""

import torch
from dataclasses import dataclass
from typing import Optional

from .phonemes import VOCAB_SIZE


@dataclass
class IgboTrainingConfig:
    """Training configuration for Igbo TTS with WAXAL dataset."""

    # Dataset paths
    data_dir: str = "data/waxal_igbo"
    output_dir: str = "output_models_igbo"

    # Training parameters
    num_epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Validation
    validation_split: float = 0.05
    validate_every: int = 1
    save_best_only: bool = False

    # Learning rate schedule
    warmup_epochs: int = 10
    lr_eta_min: float = 5e-6

    # Optimizer (AdamW)
    weight_decay: float = 0.01
    adam_eps: float = 1e-8
    adam_betas: tuple = (0.9, 0.999)

    # Model architecture (same as kokoro default)
    n_mels: int = 80
    hidden_dim: int = 512
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    n_heads: int = 8
    encoder_ff_dim: int = 2048
    decoder_ff_dim: int = 2048
    encoder_dropout: float = 0.1
    max_decoder_seq_len: int = 4253

    # Loss weights
    duration_loss_weight: float = 0.02
    stop_token_loss_weight: float = 0.1
    mel_coarse_loss_weight: float = 0.5
    mel_refined_loss_weight: float = 1.0

    # Audio processing
    max_seq_length: int = 2500
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    f_min: float = 0.0
    f_max: float = 8000.0

    # Data loading
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 3
    persistent_workers: bool = True

    # Checkpointing
    save_every: int = 1
    resume_checkpoint: str = "auto"
    keep_last_n_checkpoints: int = 2

    # Gradient checkpointing
    gradient_checkpointing: bool = True
    checkpoint_segments: int = 2
    auto_optimize_checkpointing: bool = True

    # Mixed precision
    use_mixed_precision: bool = True
    mixed_precision_dtype = torch.bfloat16
    amp_init_scale: float = 2**12
    amp_growth_factor: float = 2.0
    amp_backoff_factor: float = 0.5
    amp_growth_interval: int = 1000

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Scheduled sampling — gradually replace teacher-forced decoder input
    # with model's own predictions to prevent train/inference mismatch
    enable_scheduled_sampling: bool = True
    scheduled_sampling_warmup_batches: int = 5000
    scheduled_sampling_max_prob: float = 0.5
    scheduled_sampling_zero_input_ratio: float = 0.1

    # Ground truth durations — use MFA durations for first N epochs
    # to stabilize training before letting the duration predictor take over
    use_gt_durations_until_epoch: int = 50

    # Profiling
    enable_profiling: bool = False
    profile_epoch_start: int = 1
    profile_wait_steps: int = 1
    profile_warmup_steps: int = 1
    profile_steps: int = 5
    run_standalone_profiling: bool = False

    # Interbatch profiling
    enable_interbatch_profiling: bool = False
    interbatch_report_interval: int = 100

    # Adaptive memory
    enable_adaptive_memory: bool = True
    memory_report_interval: int = 500

    # Logging
    log_dir: str = "runs"
    log_interval: int = 50

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "igbo-tts"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list = None
    wandb_notes: Optional[str] = None

    def __post_init__(self):
        import os

        quiet = os.environ.get("TESTING")

        if self.checkpoint_segments < 1:
            self.checkpoint_segments = 1

        if self.device == "mps":
            self.pin_memory = False

        if self.auto_optimize_checkpointing and self.gradient_checkpointing:
            self._optimize_checkpointing()

        if not quiet:
            self._log_config()

    def _optimize_checkpointing(self):
        import os
        quiet = os.environ.get("TESTING")

        if torch.cuda.is_available():
            try:
                total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                if total_mb > 20000:
                    self.checkpoint_segments = 2
                elif total_mb > 10000:
                    self.checkpoint_segments = 3
                elif total_mb > 6000:
                    self.checkpoint_segments = 4
                else:
                    self.checkpoint_segments = 6
            except Exception:
                pass
        elif torch.backends.mps.is_available():
            self.checkpoint_segments = 4

    def _log_config(self):
        import os
        if os.environ.get("TESTING"):
            return

        print("\n" + "=" * 60)
        print("Igbo TTS Training Configuration")
        print("=" * 60)
        print(f"Dataset: {self.data_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Vocab Size: {VOCAB_SIZE}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        if self.gradient_checkpointing:
            print(f"Gradient Checkpointing: {self.checkpoint_segments} segments")
        print(f"\nAudio: {self.sample_rate} Hz, {self.n_mels} mels, hop={self.hop_length}")
        print(f"Model: {self.hidden_dim}d, {self.n_encoder_layers}L enc, {self.n_decoder_layers}L dec, {self.n_heads}H")
        print("=" * 60 + "\n")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> "IgboTrainingConfig":
        return cls(**d)


def get_default_config() -> IgboTrainingConfig:
    return IgboTrainingConfig()


def get_small_config() -> IgboTrainingConfig:
    """Small model for testing."""
    return IgboTrainingConfig(
        batch_size=8,
        n_encoder_layers=4,
        n_decoder_layers=4,
        hidden_dim=256,
        encoder_ff_dim=1024,
        decoder_ff_dim=1024,
        num_epochs=10,
    )


def get_medium_config() -> IgboTrainingConfig:
    """
    Medium model (~25-30M params) — optimal for single-speaker data.
    Faster training, reliable convergence.
    """
    return IgboTrainingConfig(
        batch_size=32,
        n_encoder_layers=4,
        n_decoder_layers=4,
        hidden_dim=384,
        encoder_ff_dim=1536,
        decoder_ff_dim=1536,
        n_heads=8,
        gradient_checkpointing=False,
    )


def get_cloud_config() -> IgboTrainingConfig:
    """
    Cloud GPU config (RTX 4090 / A100) — max speed + good quality.
    Medium model (25-30M params), large batches, optimized data loading.
    """
    return IgboTrainingConfig(
        batch_size=24,
        learning_rate=5e-4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        hidden_dim=384,
        encoder_ff_dim=1536,
        decoder_ff_dim=1536,
        n_heads=8,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        gradient_checkpointing=True,
        max_seq_length=1300,
        save_every=10,
        keep_last_n_checkpoints=5,
        log_interval=20,
    )


def get_large_config() -> IgboTrainingConfig:
    """Large model for high-end GPUs."""
    return IgboTrainingConfig(
        batch_size=32,
        n_encoder_layers=8,
        n_decoder_layers=8,
        hidden_dim=768,
        encoder_ff_dim=3072,
        decoder_ff_dim=3072,
        n_heads=12,
    )
