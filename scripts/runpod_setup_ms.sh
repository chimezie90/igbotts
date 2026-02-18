#!/bin/bash
# RunPod setup for multi-speaker Igbo VITS training (4x A100)
# Run this after SSH into the pod

set -e
echo "=== Igbo Multi-Speaker VITS Setup ==="
echo "$(date): Starting setup..."

cd /workspace

# 1. Install gcloud CLI for pulling data from GCS
echo "$(date): Installing gcloud CLI..."
if ! command -v gcloud &> /dev/null; then
    curl -sSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz | tar -xz
    ./google-cloud-sdk/install.sh --quiet --path-update true
    export PATH="/workspace/google-cloud-sdk/bin:$PATH"
fi
echo 'export PATH="/workspace/google-cloud-sdk/bin:$PATH"' >> ~/.bashrc

# 2. Install Python deps
echo "$(date): Installing Python packages..."
pip install -q openpyxl soundfile librosa unidecode tensorboard

# 3. Clone repo
echo "$(date): Cloning repo..."
if [ ! -d "igbotts" ]; then
    git clone https://github.com/chimezie90/igbotts.git
fi
cd igbotts
git pull

# 4. Build monotonic_align Cython extension (needed for training MAS)
echo "$(date): Building monotonic_align..."
cd vits_repo/monotonic_align
if [ ! -f "monotonic_align/core.cpython-*.so" ]; then
    python setup.py build_ext --inplace 2>/dev/null || echo "WARNING: Cython build failed, will use numpy fallback"
fi
cd /workspace/igbotts

# 5. Pull data from GCS
echo "$(date): Authenticating with GCS..."
# User needs to run: gcloud auth login
echo "After this script, run: gcloud auth login"
echo "Then run: bash scripts/runpod_pull_data.sh"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. gcloud auth login"
echo "  2. bash scripts/runpod_pull_data.sh"
echo "  3. python train_ms_igbo.py -c igbo_vits/configs/igbo_multispeaker.json -m output_vits_igbo_ms"
