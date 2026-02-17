#!/bin/bash
# RunPod setup script for Igbo VITS training
# Usage: ssh into RunPod, then: bash scripts/cloud_setup_vits.sh
set -e

echo "=== Igbo VITS Cloud Setup ==="

# 1. System deps
apt-get update -qq && apt-get install -y -qq libsndfile1 ffmpeg git 2>/dev/null

# 2. Clone repo (or pull latest)
REPO_DIR="/workspace/igbotts"
if [ -d "$REPO_DIR" ]; then
    echo "Repo exists, pulling latest..."
    cd "$REPO_DIR" && git pull
else
    echo "Cloning repo..."
    git clone https://github.com/chimezie90/igbotts.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

# 3. Init submodules + clone VITS
git submodule update --init --recursive 2>/dev/null || true
if [ ! -d "vits_repo" ]; then
    git clone https://github.com/jaywalnut310/vits.git vits_repo
fi

# 4. Build monotonic alignment (Cython extension required by VITS)
echo "Building monotonic alignment..."
cd vits_repo/monotonic_align
python setup.py build_ext --inplace
cd "$REPO_DIR"

# 5. Python deps (torch/torchaudio already on RunPod images)
pip install -q soundfile textgrid tqdm tensorboard scipy matplotlib Cython librosa Unidecode

# 6. Extract MFA alignments (for reference — VITS doesn't need them)
if [ -f "textgrids.tar.gz" ] && [ ! -d "data/waxal_igbo/TextGrid" ]; then
    echo "Extracting TextGrid alignments..."
    tar xzf textgrids.tar.gz -C data/waxal_igbo/
fi

# 7. Download WAXAL data if not present
if [ ! -d "data/waxal_igbo/wavs" ]; then
    echo "Downloading WAXAL Igbo dataset (takes ~40 min)..."
    python -m igbo_tts.setup_data -o data/waxal_igbo
fi

# 8. Prepare VITS filelists
echo "Preparing VITS filelists..."
python -m igbo_vits.prepare_data \
    --data-dir data/waxal_igbo \
    --output-dir igbo_vits/filelists \
    --max-duration 15.0

# 9. Check GPU
echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 10. Print training commands
N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "=== Ready to Train ==="
echo ""
echo "Smoke test first (5k steps, ~15 min, stop if bad):"
echo "  python train_vits.py -c igbo_vits/configs/igbo_base.json -m output_vits_igbo --smoke"
echo ""
echo "Full training ($N_GPUS GPU(s)):"
echo "  python train_vits.py -c igbo_vits/configs/igbo_base.json -m output_vits_igbo --n-gpus $N_GPUS"
echo ""
echo "Resume training:"
echo "  python train_vits.py -c igbo_vits/configs/igbo_base.json -m output_vits_igbo --n-gpus $N_GPUS --resume"
echo ""
echo "Monitor (in another terminal):"
echo "  tensorboard --logdir output_vits_igbo --bind_all"
echo ""
