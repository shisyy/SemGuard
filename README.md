# SemGuard: Semantics-Decoupled Watermarking for Gaussian Splatting

**Yingying Shi, Zhong Zhou, Bin Zhou**  
Beihang University, Zhongguancun Laboratory, and Peace-Bay Academy of Virtual Reality

Official implementation of **SemGuard**.

**TL;DR:** We train a message decoder with progressive visual semantic interference, then embed watermarks into static and dynamic Gaussian assets by optimizing spherical-harmonic offsets.

<p align="center"><img src="assets/intro.png" alt="SemGuard motivation and performance comparison" width="100%"></p>

## Framework

<p align="center"><img src="assets/method.png" alt="SemGuard framework" width="100%"></p>

We combine message-related CLIP text features with visual features from unwatermarked images to train a reusable message decoder. We then freeze the decoder and CLIP encoders and optimize SH offsets for watermark embedding.

## Get started

- Python 3.12
- CUDA 12.1
- PyTorch 2.5.1
- NVIDIA GPU

**Create a Python environment**

```bash
git clone https://github.com/shisyy/SemGuard.git
cd SemGuard
conda create -n semguard python=3.12 -y
conda activate semguard
```

**Install dependencies**

Follow the installation commands in [requirements.txt](requirements.txt). Set up [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for static scenes and [4DGaussians](https://github.com/hustvl/4DGaussians) for dynamic scenes.

## Prepare data

We use [NeRF Synthetic (Blender)](https://www.matthewtancik.com/nerf), [LLFF](https://bmild.github.io/llff/), [D-NeRF](https://github.com/albertpumarola/D-NeRF) and [HyperNeRF](https://github.com/google/hypernerf).

We generate Gaussian assets using modified 3D Gaussian Splatting and 4DGaussians. Download our [processed assets](https://github.com/shisyy/SemGuard/releases/tag/rendered-assets-v2) and extract them from the repository root to restore `output/`. See [extraction instructions](DATA_PREPARATION.md).

## Stage 1: Train the watermark decoder

See [setup_run_3d.sh](setup_run_3d.sh) and [setup_run_4d.sh](setup_run_4d.sh) for commands Stage 1.

## Stage 2: Train and evaluate the SemGuard 3DGS model and 4DGS model

See [setup_run_3d.sh](setup_run_3d.sh) and [setup_run_4d.sh](setup_run_4d.sh) for commands Stage 2.

## Qualitative results

<p align="center"><img src="assets/view.png" alt="Qualitative comparison with GuardSplat" width="100%"></p>

## Citation

```bibtex
@article{shi2026semguard,
  title={SemGuard: Semantics-Decoupled Watermarking for Gaussian Splatting},
  author={Shi, Yingying and Zhou, Zhong and Zhou, Bin},
  year={2026}
}
```

## Acknowledgements

We build our project on [GuardSplat](https://github.com/NarcissusEx/GuardSplat), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [4DGaussians](https://github.com/hustvl/4DGaussians), [CLIP](https://github.com/openai/CLIP) and [Diff-JPEG](https://github.com/necla-ml/Diff-JPEG). We thank the authors for their work and code release.

Contact: `yyshi@buaa.edu.cn`.
