# Source acknowledgements

SemGuard builds on the following original projects:

- [GuardSplat](https://github.com/NarcissusEx/GuardSplat): CLIP-guided Gaussian watermarking implementation and project presentation reference.
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): static Gaussian representation, training and rendering.
- [4DGaussians](https://github.com/hustvl/4DGaussians): dynamic Gaussian representation, training and rendering.
- [CLIP](https://github.com/openai/CLIP): frozen image and text encoders.
- [Diff-JPEG](https://github.com/necla-ml/Diff-JPEG): differentiable JPEG distortion.

The 3DGS and 4DGaussians source trees are not bundled here. Obtain them from their original repositories and follow their licenses and setup instructions. The author's processed assets were generated using locally modified backbones; the original upstream code alone is not guaranteed to reproduce identical assets. The custom dynamic loader interfaces required by the SemGuard 4D entry point are not included in this release.

NeRF, LLFF, D-NeRF and HyperNeRF datasets remain the work of their original authors. See [data preparation and provenance](DATA_PREPARATION.md) for attribution and asset contents. No new end-to-end reproduction result is claimed.
