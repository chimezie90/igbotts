#!/bin/bash
# =============================================================
# Igbo TTS — RunPod Cloud Setup
# Run this on a fresh RunPod instance (PyTorch template)
# =============================================================
set -e

echo "================================================"
echo "  Igbo TTS — Cloud Environment Setup"
echo "================================================"

# 1. Clone repo
if [ ! -d "igbotts" ]; then
    echo "[1/5] Cloning repository..."
    git clone https://github.com/chimezie90/igbotts.git
    cd igbotts
else
    echo "[1/5] Repository already exists, updating..."
    cd igbotts
    git pull
fi

# 2. Install Python dependencies
echo "[2/5] Installing dependencies..."
pip install -q torch torchaudio librosa soundfile numpy tqdm datasets textgrid wandb psutil

# 3. Download WAXAL Igbo dataset (audio files)
echo "[3/5] Downloading WAXAL Igbo dataset..."
python3 -m igbo_tts.setup_data -o data/waxal_igbo

# 4. Extract TextGrid alignments (included in repo)
if [ -f "textgrids.tar.gz" ]; then
    echo "[4/5] Extracting MFA TextGrid alignments..."
    tar xzf textgrids.tar.gz -C data/waxal_igbo/
    echo "    Extracted $(ls data/waxal_igbo/TextGrid/*.TextGrid 2>/dev/null | wc -l) TextGrid files"
elif [ -d "data/waxal_igbo/TextGrid" ]; then
    echo "[4/5] TextGrid alignments already present"
else
    echo "[4/5] WARNING: No TextGrid alignments found!"
    echo "    Upload textgrids.tar.gz or run MFA alignment"
fi

# 5. Verify GPU
echo "[5/5] Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu} ({mem:.0f} GB)')
else:
    print('WARNING: No CUDA GPU detected!')
"

echo ""
echo "================================================"
echo "  Setup complete! To start training, run:"
echo ""
echo "  python3 train.py --model-size cloud --output output_models_igbo"
echo ""
echo "  Or with W&B logging:"
echo "  python3 train.py --model-size cloud --output output_models_igbo --wandb"
echo "================================================"
