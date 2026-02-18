#!/bin/bash
# Pull training data from GCS to RunPod
# Run after gcloud auth login

set -e
export PATH="/workspace/google-cloud-sdk/bin:$PATH"

cd /workspace/igbotts
BUCKET="gs://igbotts-data"

echo "=== Pulling data from GCS ==="

# 1. Pull filelists
echo "$(date): Downloading filelists..."
mkdir -p data/multispeaker
gcloud storage cp "${BUCKET}/filelists/filelists/*" data/multispeaker/ 2>&1 | tail -3
echo "  Train: $(wc -l < data/multispeaker/train.txt) samples"
echo "  Val:   $(wc -l < data/multispeaker/val.txt) samples"

# 2. Pull African Voices FLAC audio (this is the big one ~45GB)
echo ""
echo "$(date): Downloading African Voices audio (45GB, ~10 min)..."
mkdir -p data/african_voices_audio
gcloud storage cp -r "${BUCKET}/african_voices/*.tar.zst" data/african_voices_audio/ 2>&1 | tail -3

# 3. Extract all batches
echo ""
echo "$(date): Extracting audio..."
mkdir -p data/extracted_audio
for f in data/african_voices_audio/Batch_*.tar.zst; do
    batch=$(basename "$f" .tar.zst)
    echo "  Extracting $batch..."
    tar --zstd -xf "$f" -C data/extracted_audio/ 2>&1
done
echo "$(date): Extraction complete"
echo "  FLAC files: $(find data/extracted_audio/ -name '*.flac' | wc -l)"

# 4. Convert FLAC to WAV 22050Hz (parallel with ffmpeg)
echo ""
echo "$(date): Converting FLAC to WAV 22050Hz..."
mkdir -p data/wavs
find data/extracted_audio/audio/ -name "*.flac" | while read flac; do
    wav="data/wavs/$(basename "${flac%.flac}.wav")"
    if [ ! -f "$wav" ]; then
        echo "$flac $wav"
    fi
done | xargs -P 16 -L 1 sh -c 'ffmpeg -y -i "$0" -ar 22050 -ac 1 -sample_fmt s16 "$1" 2>/dev/null'
echo "  WAV files: $(ls data/wavs/*.wav 2>/dev/null | wc -l)"

# 5. Update filelists to point to local WAV paths
echo ""
echo "$(date): Updating filelists with local paths..."
sed -i 's|extracted_audio/audio/\(.*\)\.flac|wavs/\1.wav|g' data/multispeaker/train.txt
sed -i 's|/home/emmanuelchimezie/data/extracted_audio/audio/\(.*\)\.flac|data/wavs/\1.wav|g' data/multispeaker/train.txt
sed -i 's|extracted_audio/audio/\(.*\)\.flac|wavs/\1.wav|g' data/multispeaker/val.txt
sed -i 's|/home/emmanuelchimezie/data/extracted_audio/audio/\(.*\)\.flac|data/wavs/\1.wav|g' data/multispeaker/val.txt

echo "  Sample entry: $(head -1 data/multispeaker/train.txt)"

# 6. Clean up compressed files to free disk
echo ""
echo "$(date): Cleaning up compressed files..."
rm -rf data/african_voices_audio/
rm -rf data/extracted_audio/
echo "  Disk usage: $(du -sh data/ | cut -f1)"

echo ""
echo "=== Data ready! ==="
echo "$(date): All done. Start training with:"
echo "  python train_ms_igbo.py -c igbo_vits/configs/igbo_multispeaker.json -m output_vits_igbo_ms"
