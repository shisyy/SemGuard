# Data preparation

We use [NeRF Synthetic](https://www.matthewtancik.com/nerf), [LLFF](https://bmild.github.io/llff/), [D-NeRF](https://github.com/albertpumarola/D-NeRF) and [HyperNeRF](https://github.com/google/hypernerf). We generate Gaussian assets using modified [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [4DGaussians](https://github.com/hustvl/4DGaussians).

Download our [processed assets and SHA256SUMS](https://github.com/shisyy/SemGuard/releases/tag/rendered-assets-v2), then run from the repository root:

```bash
sha256sum -c SHA256SUMS
for archive in semguard-rendered-*.tar.zst; do
    tar --zstd -xf "$archive"
done
```

Files are restored to `output/<dataset>/<scene>/`. Extract both LLFF parts into the same directory. Download original images/camera data from the dataset links above.

Set image paths for decoder training in [ms_utils_divid.py](ms_utils_divid.py). See [setup_run_3d.sh](setup_run_3d.sh) and [setup_run_4d.sh](setup_run_4d.sh) for watermarking paths.
