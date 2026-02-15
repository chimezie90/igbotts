#!/usr/bin/env python3
"""
Download and prepare the WAXAL Igbo TTS dataset.

Google's WAXAL dataset (WaxalNLP on HuggingFace) has 180+ hours of
single-speaker, studio-quality Igbo TTS audio with transcriptions (CC-BY-4.0).

This script:
  1. Downloads the Igbo TTS subset from HuggingFace
  2. Resamples audio to 22050 Hz
  3. Saves individual WAV files to data/waxal_igbo/wavs/
  4. Creates metadata.csv in LJSpeech format: id|transcription|normalized
  5. Optionally generates an MFA pronunciation dictionary
  6. Optionally runs MFA forced alignment
"""

import argparse
import csv
import os
import sys
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 22050


def download_waxal_igbo(output_dir: str, max_samples: int = None):
    """
    Download the WAXAL Igbo TTS subset and prepare it for training.

    Args:
        output_dir: root directory for the prepared dataset
        max_samples: limit number of samples (for testing)
    """
    from datasets import load_dataset, Audio

    out = Path(output_dir)
    wavs_dir = out / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading WAXAL Igbo TTS dataset from HuggingFace...")
    # Disable automatic audio decoding (avoids torchcodec dependency).
    # We'll decode audio manually with soundfile/torchaudio.
    ds = load_dataset("google/WaxalNLP", "ibo_tts", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    logger.info("Streaming WAXAL Igbo TTS dataset...")

    if max_samples:
        ds = ds.take(max_samples)
        logger.info(f"Limited to {max_samples} samples (--max-samples)")

    metadata_rows = []
    skipped = 0

    for idx, sample in enumerate(tqdm(ds, desc="Processing audio", total=max_samples)):
        sample_id = f"igbo_{idx:06d}"

        try:
            # Audio comes as {"bytes": ..., "path": ...} when decode=False
            audio_data = sample["audio"]
            audio_bytes = audio_data["bytes"]

            # Decode audio bytes with soundfile
            import io
            audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

            # Resample to target sample rate if necessary
            if sr != TARGET_SR:
                waveform = torch.from_numpy(audio_array).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                waveform = resampler(waveform)
                audio_array = waveform.squeeze(0).numpy()

            # Skip very short or silent clips
            if len(audio_array) < TARGET_SR * 0.5:  # < 0.5s
                skipped += 1
                continue
            if np.max(np.abs(audio_array)) < 0.01:
                skipped += 1
                continue

            # Save WAV
            wav_path = wavs_dir / f"{sample_id}.wav"
            sf.write(str(wav_path), audio_array, TARGET_SR)

            # Get transcription
            text = sample.get("text", sample.get("sentence", ""))
            if not text or not text.strip():
                skipped += 1
                os.remove(wav_path)
                continue

            # Replace newlines with spaces to keep metadata.csv one-line-per-entry
            text = " ".join(text.strip().split())
            metadata_rows.append((sample_id, text, text))

        except Exception as e:
            logger.warning(f"Skipping sample {idx}: {e}")
            skipped += 1

    # Write metadata.csv (LJSpeech format: id|text|normalized_text)
    metadata_path = out / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        for row in metadata_rows:
            f.write(f"{row[0]}|{row[1]}|{row[2]}\n")

    logger.info(f"Dataset prepared: {len(metadata_rows)} samples, {skipped} skipped")
    logger.info(f"Audio saved to: {wavs_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")

    return metadata_rows


def create_mfa_dictionary(data_dir: str):
    """
    Generate an MFA pronunciation dictionary from the dataset transcriptions.

    Reads metadata.csv, extracts all unique words, and produces a
    pronunciation dictionary using our Igbo G2P.
    """
    from .g2p import generate_mfa_dictionary, _normalize, _TOKEN_RE

    data_path = Path(data_dir)
    metadata_path = data_path / "metadata.csv"

    if not metadata_path.exists():
        logger.error(f"metadata.csv not found at {metadata_path}")
        return

    # Collect all words
    words = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                text = _normalize(parts[1])
                tokens = _TOKEN_RE.findall(text)
                for tok in tokens:
                    if tok not in ".,!?;:-":
                        words.add(tok)

    # Also write individual .txt files for MFA (one per wav)
    wavs_dir = data_path / "wavs"
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2 and parts[0].startswith("igbo_"):
                sample_id = parts[0]
                text = parts[1]
                txt_path = wavs_dir / f"{sample_id}.txt"
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(text)

    dict_path = data_path / "igbo_mfa_dictionary.txt"
    generate_mfa_dictionary(list(words), str(dict_path))
    logger.info(f"MFA dictionary written: {dict_path} ({len(words)} words)")
    logger.info(f"Text files written to: {wavs_dir}")


def run_mfa_alignment(data_dir: str):
    """
    Run Montreal Forced Aligner on the WAXAL Igbo dataset.

    Requires MFA installed: conda install -c conda-forge montreal-forced-aligner

    Produces TextGrid files in data_dir/TextGrid/wavs/
    """
    import subprocess

    data_path = Path(data_dir)
    dict_path = data_path / "igbo_mfa_dictionary.txt"
    wavs_dir = data_path / "wavs"
    output_dir = data_path / "TextGrid"

    if not dict_path.exists():
        logger.info("MFA dictionary not found, generating...")
        create_mfa_dictionary(data_dir)

    if not dict_path.exists():
        logger.error("Failed to create MFA dictionary")
        return

    logger.info("Running MFA forced alignment...")
    logger.info(f"  Corpus:     {wavs_dir}")
    logger.info(f"  Dictionary: {dict_path}")
    logger.info(f"  Output:     {output_dir}")

    # MFA command: align corpus dictionary acoustic_model output
    # Using --clean to start fresh
    cmd = [
        "mfa", "align",
        str(wavs_dir),
        str(dict_path),
        "english_mfa",  # acoustic model (closest available; Igbo-specific would be better)
        str(output_dir),
        "--clean",
        "--overwrite",
        "--num_jobs", "4",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode == 0:
            logger.info("MFA alignment completed successfully!")
            logger.info(f"TextGrid files at: {output_dir}")
        else:
            logger.error(f"MFA alignment failed:\n{result.stderr}")
    except FileNotFoundError:
        logger.error(
            "MFA not found. Install with:\n"
            "  conda install -c conda-forge montreal-forced-aligner"
        )
    except subprocess.TimeoutExpired:
        logger.error("MFA alignment timed out (>2 hours)")


def verify_dataset(data_dir: str):
    """Verify that the prepared dataset is ready for training."""
    data_path = Path(data_dir)

    checks = {
        "metadata.csv exists": (data_path / "metadata.csv").exists(),
        "wavs/ directory exists": (data_path / "wavs").is_dir(),
    }

    if checks["metadata.csv exists"]:
        with open(data_path / "metadata.csv", "r", encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)
        checks[f"metadata has {num_lines} entries"] = num_lines > 0

    if checks["wavs/ directory exists"]:
        wav_files = list((data_path / "wavs").glob("*.wav"))
        checks[f"wavs/ has {len(wav_files)} files"] = len(wav_files) > 0

        # Spot-check a few audio files
        if wav_files:
            sample_wav = wav_files[0]
            info = sf.info(str(sample_wav))
            checks[f"sample rate = {info.samplerate}"] = info.samplerate == TARGET_SR
            checks[f"sample duration = {info.duration:.1f}s"] = info.duration > 0.5

    textgrid_dir = data_path / "TextGrid"
    if textgrid_dir.exists():
        tg_files = list(textgrid_dir.rglob("*.TextGrid"))
        checks[f"TextGrid has {len(tg_files)} alignments"] = len(tg_files) > 0
    else:
        checks["TextGrid/ not found (alignment needed)"] = False

    dict_path = data_path / "igbo_mfa_dictionary.txt"
    if dict_path.exists():
        with open(dict_path, "r", encoding="utf-8") as f:
            num_words = sum(1 for _ in f)
        checks[f"MFA dictionary has {num_words} words"] = num_words > 0

    print("\n" + "=" * 50)
    print("Dataset Verification")
    print("=" * 50)
    all_ok = True
    for desc, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            all_ok = False
    print("=" * 50)

    if all_ok:
        print("Dataset is ready for training!")
    else:
        print("Some checks failed. See above for details.")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Prepare WAXAL Igbo TTS dataset")
    parser.add_argument(
        "--output-dir", "-o",
        default="data/waxal_igbo",
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )
    parser.add_argument(
        "--create-dict",
        action="store_true",
        help="Generate MFA pronunciation dictionary",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Run MFA forced alignment",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify prepared dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps: download, dict, align, verify",
    )
    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output_dir)
        return

    if args.all or not (args.create_dict or args.align or args.verify):
        download_waxal_igbo(args.output_dir, args.max_samples)
        create_mfa_dictionary(args.output_dir)
        if args.all:
            run_mfa_alignment(args.output_dir)
            verify_dataset(args.output_dir)
        return

    if args.create_dict:
        create_mfa_dictionary(args.output_dir)

    if args.align:
        run_mfa_alignment(args.output_dir)


if __name__ == "__main__":
    main()
