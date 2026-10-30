# SemGuard: Semantics-Decoupled Watermarking for Gaussian Splatting

**Yingying Shi, Zhong Zhou, Bin Zhou**  
Beihang University, Zhongguancun Laboratory, and Peace-Bay Academy of Virtual Reality

Official implementation of **SemGuard**.

**TL;DR:** We train a message decoder with progressive visual semantic interference, then embed watermarks into static and dynamic Gaussian assets by optimizing spherical-harmonic offsets.

<p align="center"><img src="assets/intro.png" alt="SemGuard motivation and performance comparison" width="100%"></p>

## Framework

<p align="center"><img src="assets/method.png" alt="SemGuard framework" width="100%"></p>

We combine message-related CLIP text features with visual features from unwatermarked images to train a reusable message decoder. We then freeze the decoder and CLIP encoders and optimize SH offsets for watermark embedding.

## 1) Get started

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

## 2) Prepare data

We use [NeRF Synthetic (Blender)](https://www.matthewtancik.com/nerf), [LLFF](https://bmild.github.io/llff/), [D-NeRF](https://github.com/albertpumarola/D-NeRF) and [HyperNeRF](https://github.com/google/hypernerf).

We generate Gaussian assets using modified 3D Gaussian Splatting and 4DGaussians. Download our [processed assets](https://github.com/shisyy/SemGuard/releases/tag/rendered-assets-v2) and extract them from the repository root to restore `output/`. See [extraction instructions](DATA_PREPARATION.md).

## 3) Train and evaluate the message decoder

Set the dataset paths and `used_datasets` in [ms_utils_divid.py](ms_utils_divid.py).

```bash
EXP_NAME=${DECODER_BASE_NAME}_msg${msg_len}_${num_epochs}_${type_fusion_weight}

python make_decoder_divid.py --mode train --msg_len "$msg_len" --save \
    --num_epochs "$num_epochs" --sdir decoders/nerf_synthetic/${EXP_NAME} \
    --type_fusion_weight "$type_fusion_weight" \
    --type_fusion_weight_max 0.9 --type_fusion_weight_min 0.0

python make_decoder_divid.py --mode test --msg_len "$msg_len" \
    --sdir decoders/nerf_synthetic/${EXP_NAME}
```

We use 400 decoder epochs for 4/8/16-bit messages and 800 for 32/48-bit messages. Set `type_fusion_weight=continuous` for progressive semantic interference.

## 4) Watermark a 3DGS model

Set `root`, `out_root`, `exp_root` and `item` for the dataset, clean Gaussian model, watermark output and scene:

```bash
mdir=$out_root/$item
sdir=$exp_root/${EXP_NAME}/$item
decoder_path=decoders/nerf_synthetic/${EXP_NAME}

# Watermark embedding (Blender example)
python run_3Ddivid.py -s "$root/$item" -m "$mdir" -w \
    --mode train --msg_len "$msg_len" --decoder_path "$decoder_path" \
    --sdir "$sdir" --batch_size 16 --dtype blender --num_epochs2 "$num_epochs2"

# Evaluation and visualization
python run_3Ddivid.py -s "$root/$item" -m "$mdir" -w \
    --mode eval --msg_len "$msg_len" --decoder_path "$decoder_path" \
    --sdir "$sdir" --batch_size 16 --dtype blender --num_epochs2 100 \
    --test_cams 10 --resume_num "$resume_num" --save_map
```

Use `--dtype llff` for LLFF scenes. More settings are provided in [setup_run_3d.sh](setup_run_3d.sh).

## 5) Watermark a 4DGS model

```bash
# Watermark embedding
python run_4Ddivid.py -s "$root/$item" -m "$mdir" -w \
    --mode train --msg_len "$msg_len" --decoder_path "$decoder_path" \
    --sdir "$sdir" --batch_size 16 --num_epochs2 "$num_epochs2"

# Evaluation and visualization
python run_4Ddivid.py -s "$root/$item" -m "$mdir" -w \
    --mode eval --msg_len "$msg_len" --decoder_path "$decoder_path" \
    --sdir "$sdir" --batch_size 16 --num_epochs2 100 \
    --resume_num "$resume_num" --save_map
```

See [setup_run_4d.sh](setup_run_4d.sh) for the stage commands and experiment settings.

## 6) Visual distortions

Distortion settings and the Stage III command are provided in [setup_run_3d.sh](setup_run_3d.sh), with implementations in [attack_utils.py](attack_utils.py).

- Single distortions: GaussianNoise, Rotation, Resize, GaussianBlur, Crop, Brightness, JPEGCompress and VAEBmshj2018.
- Combined example: Crop, Brightness and JPEGCompress.

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
