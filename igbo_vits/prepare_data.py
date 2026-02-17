#!/usr/bin/env python3
"""
Prepare WAXAL Igbo dataset for VITS training.

Reads metadata.csv, runs G2P, filters by duration, splits train/val,
and writes VITS filelists with pre-computed integer phoneme sequences.

Output format (per line):
    /absolute/path/to/wav.wav|3 4 12 7 2 15 ...

Usage:
    python -m igbo_vits.prepare_data \
        --data-dir data/waxal_igbo \
        --output-dir igbo_vits/filelists \
        --max-duration 15.0 \
        --val-ratio 0.05
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from igbo_tts.g2p import IgboG2P


def main():
    parser = argparse.ArgumentParser(description="Prepare WAXAL data for VITS")
    parser.add_argument("--data-dir", default="data/waxal_igbo")
    parser.add_argument("--output-dir", default="igbo_vits/filelists")
    parser.add_argument("--max-duration", type=float, default=15.0,
                        help="Max audio duration in seconds")
    parser.add_argument("--min-duration", type=float, default=1.0,
                        help="Min audio duration in seconds")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sr", type=int, default=22050)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = data_dir / "metadata.csv"
    if not metadata_file.exists():
        print(f"ERROR: {metadata_file} not found")
        sys.exit(1)

    g2p = IgboG2P()

    # Read metadata
    samples = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            audio_id = parts[0]
            text = parts[2] if len(parts) >= 3 else parts[1]
            wav_path = data_dir / "wavs" / f"{audio_id}.wav"
            if not wav_path.exists():
                continue
            samples.append((str(wav_path), text, audio_id))

    print(f"Found {len(samples)} samples in metadata")

    # Filter by duration and convert phonemes
    valid_samples = []
    skipped_short = 0
    skipped_long = 0
    skipped_empty = 0
    skipped_sr = 0

    for wav_path, text, audio_id in samples:
        try:
            info = sf.info(wav_path)
        except Exception as e:
            print(f"  Skip {audio_id}: can't read audio ({e})")
            continue

        if info.duration < args.min_duration:
            skipped_short += 1
            continue
        if info.duration > args.max_duration:
            skipped_long += 1
            continue

        # Check sample rate
        if info.samplerate != args.target_sr:
            # We'll resample during training, but flag it
            pass

        # G2P conversion
        indices = g2p.text_to_indices(text)
        if len(indices) < 3:
            skipped_empty += 1
            continue

        # Store as space-separated integers
        indices_str = " ".join(str(i) for i in indices)
        valid_samples.append((wav_path, indices_str, info.duration))

    print(f"\nFiltering results:")
    print(f"  Valid:        {len(valid_samples)}")
    print(f"  Too short:    {skipped_short} (< {args.min_duration}s)")
    print(f"  Too long:     {skipped_long} (> {args.max_duration}s)")
    print(f"  Empty G2P:    {skipped_empty}")

    if not valid_samples:
        print("ERROR: No valid samples after filtering!")
        sys.exit(1)

    # Duration stats
    durations = [d for _, _, d in valid_samples]
    durations.sort()
    total_hours = sum(durations) / 3600
    print(f"\nDuration stats ({len(valid_samples)} samples, {total_hours:.1f} hours):")
    print(f"  min: {durations[0]:.1f}s")
    print(f"  p50: {durations[len(durations)//2]:.1f}s")
    print(f"  p90: {durations[int(len(durations)*0.9)]:.1f}s")
    print(f"  max: {durations[-1]:.1f}s")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(valid_samples)

    val_size = max(1, int(len(valid_samples) * args.val_ratio))
    val_samples = valid_samples[:val_size]
    train_samples = valid_samples[val_size:]

    print(f"\nSplit: {len(train_samples)} train, {len(val_samples)} val")

    # Write filelists
    train_file = output_dir / "igbo_train.txt"
    val_file = output_dir / "igbo_val.txt"

    with open(train_file, "w", encoding="utf-8") as f:
        for wav_path, indices_str, _ in train_samples:
            f.write(f"{wav_path}|{indices_str}\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for wav_path, indices_str, _ in val_samples:
            f.write(f"{wav_path}|{indices_str}\n")

    print(f"\nWrote: {train_file}")
    print(f"Wrote: {val_file}")

    # Write a small sample for inspection
    print(f"\nSample entries:")
    for wav_path, indices_str, dur in train_samples[:3]:
        name = Path(wav_path).stem
        tokens = indices_str.split()
        print(f"  {name} ({dur:.1f}s): {len(tokens)} tokens → {indices_str[:60]}...")


if __name__ == "__main__":
    main()
