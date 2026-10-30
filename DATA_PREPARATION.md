# Data preparation

We use [NeRF Synthetic (Blender)](https://www.matthewtancik.com/nerf), [LLFF](https://bmild.github.io/llff/), [D-NeRF](https://github.com/albertpumarola/D-NeRF) and [HyperNeRF](https://github.com/google/hypernerf).

We prepare static Gaussian assets with [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and dynamic assets with [4DGaussians](https://github.com/hustvl/4DGaussians). Our processed assets were generated with modified backbone code/settings; unmodified upstream code may produce different results.

## Download

Download our [processed Gaussian assets and SHA256SUMS](https://github.com/shisyy/SemGuard/releases/tag/rendered-assets-v2). The archives contain clean Gaussian models, dynamic deformation parameters, metadata and available rendered views. Download original image/camera datasets from the links above.

| Archive | Scenes |
| --- | --- |
| D-NeRF | bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex |
| HyperNeRF | 3dprinter, banana, broom2, chicken |
| NeRF Synthetic | chair, drums, ficus, hotdog, lego, materials, mic, ship |
| LLFF part 1 | fern, flower, fortress, horns |
| LLFF part 2 | leaves, orchids, room, trex |

## Extract

Run from the SemGuard root:

```bash
sha256sum -c SHA256SUMS
for archive in semguard-rendered-*.tar.zst; do
    tar --zstd -xf "$archive"
done
```

The assets are extracted to `output/<dataset>/<scene>/`. Extract both LLFF archives into the same directory; no concatenation is needed. Install `zstd` if required by your tar installation.

Set the dataset paths in [ms_utils_divid.py](ms_utils_divid.py) and the experiment paths in [setup_run_3d.sh](setup_run_3d.sh) or [setup_run_4d.sh](setup_run_4d.sh).
