#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Recommended environment:
#   conda create -n semguard python=3.12 -y
#   conda activate semguard

conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 \
  -c pytorch -c nvidia -y

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install git+https://github.com/necla-ml/Diff-JPEG

python -m pip install ./4DGaussians/submodules/depth-diff-gaussian-rasterization
python -m pip install ./4DGaussians/submodules/simple-knn

echo "SemGuard dependencies installed successfully."
