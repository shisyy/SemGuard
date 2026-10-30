# Data preparation and provenance

## What is provided

SemGuard uses the first two benchmark families discussed by each backbone: NeRF Synthetic (Blender) and LLFF for 3D Gaussian Splatting, and D-NeRF and HyperNeRF for 4DGaussians. This refers to four dataset families, not just two scenes per family.

The release archives contain the author's existing `output/` assets. Depending on the family, these include PLY Gaussian models, configuration files, camera metadata, deformation weights and rendered views. They are derived assets, not new original datasets or a redistribution of every source training image.

| Archive family | Scenes |
| --- | --- |
| `dnerf` | bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex |
| `hypernerf` | 3dprinter, banana, broom2, chicken |
| `nerf_synthetic` | chair, drums, ficus, hotdog, lego, materials, mic, ship |
| `nerf_llff_data` part 1 | fern, flower, fortress, horns |
| `nerf_llff_data` part 2 | leaves, orchids, room, trex |

## Download the author's processed assets

Download the `.tar.zst` archives and `SHA256SUMS` from [GitHub Releases](https://github.com/shisyy/SemGuard/releases). LLFF is distributed in two independently extractable archives. Extract all archives from the repository root:

```bash
sha256sum -c SHA256SUMS
for archive in semguard-rendered-*.tar.zst; do
  tar --zstd -xf "$archive"
done
```

Install `zstd` if your tar installation requires it. Extraction restores `output/<dataset>/<scene>/`. Download original image/camera datasets separately using the links below. Saved `cfg_args` files may contain author-machine paths; use explicit `-s` and `-m` arguments for your local paths. Only load model/checkpoint files obtained from trusted sources.

## Original sources

- [NeRF Synthetic / Blender](https://www.matthewtancik.com/nerf): NeRF, Mildenhall et al.
- [LLFF](https://bmild.github.io/llff/): Local Light Field Fusion, Mildenhall et al.
- [D-NeRF](https://github.com/albertpumarola/D-NeRF): D-NeRF, Pumarola et al.
- [HyperNeRF](https://github.com/google/hypernerf): HyperNeRF, Park et al. Follow the [4DGaussians preparation instructions](https://github.com/hustvl/4DGaussians#data-preparation) for its Nerfies-style layout and point-cloud preprocessing.

## Backbone preparation and SemGuard stages

Static assets were constructed with [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting); dynamic assets were constructed with [4DGaussians](https://github.com/hustvl/4DGaussians). Refer to the original repositories for backbone installation, dataset preprocessing, training and rendering instructions.

The supplied processed assets were generated with author-modified code and settings. Unmodified upstream code is not guaranteed to generate identical assets. Downloading the supplied archives gives the author's actual processed assets.

For SemGuard decoder pretraining, watermark embedding, evaluation and distortions, follow the existing stage commands in [setup_run_3d.sh](setup_run_3d.sh) and [setup_run_4d.sh](setup_run_4d.sh). The variables `root`, `out_root`, `exp_root` and `item` in those scripts identify the source dataset, clean Gaussian assets, watermark output and scene. These source scripts are preserved unchanged.
