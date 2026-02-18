"""Analyze African Voices metadata across all batches."""
import openpyxl, glob, collections, statistics, sys

metadata_dir = sys.argv[1] if len(sys.argv) > 1 else "."

all_rows = []
header = None
for xlsx_path in sorted(glob.glob(f"{metadata_dir}/Batch_*_metadata.xlsx")):
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if header is None:
        header = rows[0]
    all_rows.extend(rows[1:])
    wb.close()

print(f"Total samples across all batches: {len(all_rows)}")
print(f"Columns: {header}")

sid_idx = header.index("speaker_id")
gen_idx = header.index("gender")
dur_idx = header.index("duration")
snr_idx = header.index("snr")
trans_idx = header.index("transcript")

speaker_count = collections.Counter()
speaker_duration = collections.defaultdict(float)
speaker_gender = {}
for row in all_rows:
    sid = row[sid_idx]
    speaker_count[sid] += 1
    try:
        speaker_duration[sid] += float(row[dur_idx])
    except:
        pass
    speaker_gender[sid] = row[gen_idx]

n_female = sum(1 for s in speaker_gender if speaker_gender[s] == "female")
n_male = sum(1 for s in speaker_gender if speaker_gender[s] == "male")
print(f"\nTotal unique speakers: {len(speaker_count)}")
print(f"Female: {n_female}, Male: {n_male}")

print(f"\n{'='*80}")
print(f"TOP 30 SPEAKERS BY SAMPLE COUNT")
print(f"{'='*80}")
print(f"{'Rank':>4} {'Speaker':>15} {'Gender':>8} {'Samples':>8} {'Hours':>8} {'Avg(s)':>8}")
print("-" * 60)
for rank, (sid, count) in enumerate(speaker_count.most_common(30), 1):
    hours = speaker_duration[sid] / 3600
    avg_dur = speaker_duration[sid] / count if count > 0 else 0
    print(f"{rank:>4} {sid:>15} {speaker_gender[sid]:>8} {count:>8} {hours:>8.2f} {avg_dur:>8.1f}")

total_duration = sum(speaker_duration.values())
print(f"\nTotal dataset duration: {total_duration/3600:.1f} hours")

has_transcript = sum(1 for r in all_rows if r[trans_idx] and str(r[trans_idx]).strip())
print(f"Samples with transcripts: {has_transcript} / {len(all_rows)} ({100*has_transcript/len(all_rows):.1f}%)")

durations = []
for row in all_rows:
    try:
        durations.append(float(row[dur_idx]))
    except:
        pass

print(f"\nDuration stats:")
print(f"  Mean: {statistics.mean(durations):.1f}s")
print(f"  Median: {statistics.median(durations):.1f}s")
print(f"  Min: {min(durations):.1f}s, Max: {max(durations):.1f}s")
print(f"  <3s: {sum(1 for d in durations if d < 3)}")
print(f"  3-10s: {sum(1 for d in durations if 3 <= d < 10)}")
print(f"  10-20s: {sum(1 for d in durations if 10 <= d < 20)}")
print(f"  20-30s: {sum(1 for d in durations if 20 <= d < 30)}")
print(f"  >30s: {sum(1 for d in durations if d >= 30)}")

snrs = [row[snr_idx] for row in all_rows if row[snr_idx] is not None]
print(f"\nSNR distribution:")
snr_counts = collections.Counter(snrs)
for snr_val, cnt in sorted(snr_counts.items()):
    print(f"  SNR={snr_val}: {cnt} samples")

# Top 5 speakers: show sample transcripts
print(f"\n{'='*80}")
print("SAMPLE TRANSCRIPTS FROM TOP 5 SPEAKERS")
print(f"{'='*80}")
for sid, _ in speaker_count.most_common(5):
    samples = [r for r in all_rows if r[sid_idx] == sid][:3]
    print(f"\n{sid} ({speaker_gender[sid]}, {speaker_count[sid]} samples, {speaker_duration[sid]/3600:.1f}h):")
    for s in samples:
        print(f"  [{float(s[dur_idx]):.1f}s] {s[trans_idx]}")
