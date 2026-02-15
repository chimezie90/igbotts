#!/usr/bin/env python3
"""
Igbo TTS Training Script

Main entry point for training the Kokoro model on the WAXAL Igbo dataset.
Wraps the kokoro_training codebase with Igbo-specific G2P, dataset, and config.

Usage:
    python train.py                          # default config
    python train.py --model-size medium      # medium model (~25M params)
    python train.py --test-mode              # quick smoke test
    python train.py --wandb                  # enable W&B logging
    python train.py --resume auto            # resume from latest checkpoint
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path

import torch

# Add kokoro_training to path so we can import its modules
sys.path.insert(0, str(Path(__file__).resolve().parent / "kokoro_training"))

from igbo_tts.config import (
    IgboTrainingConfig,
    get_default_config,
    get_small_config,
    get_medium_config,
    get_cloud_config,
    get_large_config,
)
from igbo_tts.dataset import IgboDataset, collate_fn, LengthBasedBatchSampler
from igbo_tts.g2p import IgboG2P

# Kokoro model & training infrastructure
from kokoro.model import KokoroModel
from training.checkpoint_manager import (
    save_model_config,
    load_checkpoint,
    find_latest_checkpoint,
    save_checkpoint,
    save_final_model,
    cleanup_old_checkpoints,
    check_disk_space,
)

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*backend.*parameter is not used.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Igbo TTS model on WAXAL dataset")

    parser.add_argument("--corpus", "-c", default="data/waxal_igbo", help="Dataset directory")
    parser.add_argument("--output", "-o", default="./output_models_igbo", help="Output directory")
    parser.add_argument("--batch-size", "-b", type=int)
    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--learning-rate", "-lr", type=float)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--resume", "-r", type=str)
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "default", "cloud", "large"],
        default="default",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="igbo-tts")
    parser.add_argument("--wandb-entity", type=str)
    parser.add_argument("--wandb-name", type=str)
    parser.add_argument("--wandb-tags", nargs="*")

    return parser.parse_args()


def build_config(args) -> IgboTrainingConfig:
    """Create training config from CLI args."""
    if args.model_size == "small":
        config = get_small_config()
    elif args.model_size == "medium":
        config = get_medium_config()
    elif args.model_size == "cloud":
        config = get_cloud_config()
    elif args.model_size == "large":
        config = get_large_config()
    else:
        config = get_default_config()

    config.data_dir = args.corpus
    config.output_dir = args.output

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.save_every is not None:
        config.save_every = args.save_every
    if args.resume:
        config.resume_checkpoint = args.resume
    if args.device != "auto":
        config.device = args.device
    if args.no_gradient_checkpointing:
        config.gradient_checkpointing = False
    if args.no_mixed_precision:
        config.use_mixed_precision = False

    if args.wandb:
        config.use_wandb = True
        config.wandb_project = args.wandb_project
        config.wandb_entity = args.wandb_entity
        config.wandb_run_name = args.wandb_name
        config.wandb_tags = args.wandb_tags

    if args.test_mode:
        logger.warning("TEST MODE — limited data and epochs")
        config.num_epochs = 5
        config.save_every = 1
        config.batch_size = min(4, config.batch_size)

    return config


def validate_dataset(data_dir: str) -> bool:
    """Check that the dataset directory has required files."""
    p = Path(data_dir)
    if not p.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        logger.info("Run: python -m igbo_tts.setup_data -o data/waxal_igbo")
        return False
    if not (p / "metadata.csv").exists():
        logger.error(f"metadata.csv not found in {data_dir}")
        return False
    if not (p / "wavs").is_dir():
        logger.error(f"wavs/ directory not found in {data_dir}")
        return False
    tg = p / "TextGrid"
    if tg.exists():
        logger.info(f"Found MFA alignments at {tg}")
    else:
        logger.warning(f"No TextGrid/ found at {tg} — MFA alignment needed")
    return True


def save_phoneme_processor(processor: IgboG2P, output_dir: str):
    """Serialize the Igbo G2P processor alongside the model."""
    import pickle

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "phoneme_processor.pkl")
    with open(path, "wb") as f:
        pickle.dump(processor.to_dict(), f)
    logger.info(f"Saved phoneme processor to {path}")


def main():
    print("\n" + "=" * 70)
    print("Igbo TTS Training")
    print("WAXAL Igbo Dataset + Kokoro Architecture")
    print("=" * 70 + "\n")

    args = parse_arguments()

    # Device info + CUDA optimizations
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled: cudnn.benchmark, TF32 matmul")
    elif torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available")
    else:
        logger.warning("No GPU — training will be slow")

    config = build_config(args)

    # Validate dataset
    if not validate_dataset(config.data_dir):
        sys.exit(1)

    os.makedirs(config.output_dir, exist_ok=True)

    # Test G2P
    g2p = IgboG2P()
    test_texts = ["Nnọọ", "Kedụ ka ị mere?", "Igbo bụ asụsụ dị mma"]
    for t in test_texts:
        ph = g2p.text_to_phonemes(t)
        logger.info(f"G2P: '{t}' → {' '.join(ph)}")

    # Initialize trainer
    # We use the EnglishTrainer from kokoro_training but with our Igbo dataset/config.
    # The trainer expects the dataset to have a `.phoneme_processor` attribute.
    logger.info("Initializing Igbo trainer...")
    try:
        from training.english_trainer import EnglishTrainer

        # The EnglishTrainer creates the dataset internally from config.data_dir.
        # We need to monkey-patch it to use IgboDataset instead of LJSpeechDataset.
        # Strategy: subclass or patch the dataset creation.

        # Simpler approach: create a thin wrapper that reuses EnglishTrainer's
        # training loop but swaps in our dataset.
        trainer = _create_igbo_trainer(config)

        logger.info(f"Trainer initialized with {len(trainer.dataset)} samples")
    except Exception as e:
        logger.error(f"Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save artifacts
    save_phoneme_processor(trainer.dataset.phoneme_processor, config.output_dir)
    save_model_config(config, config.output_dir)

    # Train
    logger.info("\n" + "=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted — saving checkpoint")
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info(f"\nTraining complete. Models saved to: {config.output_dir}")


def _create_igbo_trainer(config: IgboTrainingConfig):
    """
    Create an EnglishTrainer but with IgboDataset swapped in.

    We monkeypatch the trainer module's reference to LJSpeechDataset,
    collate_fn, and LengthBasedBatchSampler so EnglishTrainer uses our
    Igbo versions instead of the English/LJSpeech ones.
    """
    import training.english_trainer as trainer_module

    # Save originals — the trainer imports these names directly
    orig_dataset_cls = trainer_module.LJSpeechDataset
    orig_collate = trainer_module.collate_fn
    orig_sampler = trainer_module.LengthBasedBatchSampler

    # Patch the trainer module's namespace
    trainer_module.LJSpeechDataset = IgboDataset
    trainer_module.collate_fn = collate_fn
    trainer_module.LengthBasedBatchSampler = LengthBasedBatchSampler

    try:
        trainer = trainer_module.EnglishTrainer(config)
    finally:
        # Restore
        trainer_module.LJSpeechDataset = orig_dataset_cls
        trainer_module.collate_fn = orig_collate
        trainer_module.LengthBasedBatchSampler = orig_sampler

    return trainer


if __name__ == "__main__":
    main()
