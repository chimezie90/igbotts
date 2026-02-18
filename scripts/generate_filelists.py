"""
Generate VITS multi-speaker filelists from African Voices metadata.

Output format: audio_path|speaker_id|normalized_text
Audio paths point to FLAC files (will be converted to WAV on training VM).

Usage:
    python generate_filelists.py \
        --metadata-dir /path/to/metadata/ \
        --audio-dir /path/to/extracted_audio/audio/ \
        --output-dir /path/to/output/
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import unicodedata

import openpyxl

# ── Igbo grapheme symbol set (must match igbo_vits/graphemes.py) ──
VALID_CHARS = set(
    " aeiouọịụbcdfghjklmnṅprstwvyz"
    "àáèéìíòóùú"
    ".,!?;:-'"
)


def normalize_igbo_text(text):
    """Normalize Igbo transcript for grapheme-level VITS training."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\[(?:um|uh|\?)\]', '', text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '').replace('\u201d', '')
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    for src, dst in {'ȧ': 'a', 'ė': 'e', 'ȯ': 'o'}.items():
        text = text.replace(src, dst)
    for src, dst in {'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u'}.items():
        text = text.replace(src, dst)
    text = text.replace('ǹ', 'n')
    cleaned = [ch for ch in text if ch in VALID_CHARS]
    text = ''.join(cleaned)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-dir', required=True)
    parser.add_argument('--audio-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-duration', type=float, default=20.0)
    parser.add_argument('--min-duration', type=float, default=1.5)
    parser.add_argument('--val-ratio', type=float, default=0.02)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all metadata
    print("Loading metadata...")
    all_rows = []
    header = None
    for xlsx_path in sorted(glob.glob(os.path.join(args.metadata_dir, "Batch_*_metadata.xlsx"))):
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if header is None:
            header = rows[0]
        for row in rows[1:]:
            all_rows.append(dict(zip(header, row)))
        wb.close()
    print(f"  Total samples: {len(all_rows)}")

    # Build speaker mapping
    speakers = sorted(set(row['speaker_id'] for row in all_rows))
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    print(f"  Speakers: {len(speakers)}")

    # Filter and build entries
    entries = []
    skipped = collections.Counter()
    for row in all_rows:
        # Duration filter
        try:
            dur = float(row['duration'])
        except (ValueError, TypeError):
            skipped['bad_duration'] += 1
            continue
        if dur < args.min_duration:
            skipped['too_short'] += 1
            continue
        if dur > args.max_duration:
            skipped['too_long'] += 1
            continue

        # Normalize text
        transcript = normalize_igbo_text(row.get('transcript', ''))
        if len(transcript) < 5:
            skipped['short_text'] += 1
            continue

        # Check audio exists
        audio_path = row.get('audio_path', '')
        if not audio_path:
            skipped['no_audio_path'] += 1
            continue
        full_audio_path = os.path.join(args.audio_dir, audio_path)
        if not os.path.exists(full_audio_path):
            skipped['audio_missing'] += 1
            continue

        speaker_idx = speaker_to_idx[row['speaker_id']]
        entries.append(f"{full_audio_path}|{speaker_idx}|{transcript}")

    print(f"  Valid entries: {len(entries)}")
    print(f"  Skipped: {dict(skipped)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(entries)
    val_size = max(1, int(len(entries) * args.val_ratio))
    val_entries = entries[:val_size]
    train_entries = entries[val_size:]

    # Write filelists
    train_path = os.path.join(args.output_dir, "train.txt")
    val_path = os.path.join(args.output_dir, "val.txt")
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_entries) + '\n')
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_entries) + '\n')

    # Save speaker mapping
    with open(os.path.join(args.output_dir, "speakers.json"), 'w') as f:
        json.dump(speaker_to_idx, f, indent=2)

    # Save config snippet
    config_info = {
        "n_speakers": len(speakers),
        "n_train": len(train_entries),
        "n_val": len(val_entries),
        "vocab_size": 49,  # matches igbo_vits/graphemes.py
    }
    with open(os.path.join(args.output_dir, "data_info.json"), 'w') as f:
        json.dump(config_info, f, indent=2)

    print(f"\n  Train: {len(train_entries)} → {train_path}")
    print(f"  Val:   {len(val_entries)} → {val_path}")
    print(f"  Speakers: {len(speakers)}")

    # Show sample entries
    print(f"\n  Sample entries:")
    for e in train_entries[:5]:
        parts = e.split('|', 2)
        print(f"    ...{parts[0][-40:]}|{parts[1]}|{parts[2][:60]}")

    # Speaker distribution
    speaker_counts = collections.Counter()
    for entry in entries:
        _, sid, _ = entry.split('|', 2)
        speaker_counts[int(sid)] += 1
    counts = list(speaker_counts.values())
    print(f"\n  Samples per speaker: min={min(counts)}, max={max(counts)}, mean={sum(counts)/len(counts):.0f}")


if __name__ == '__main__':
    main()
