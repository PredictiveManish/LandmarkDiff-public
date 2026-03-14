# LandmarkDiff Benchmarks

Reproducible performance and quality benchmarks for LandmarkDiff.

## Inference Latency by GPU

End-to-end latency for a single 512x512 image, 30 denoising steps, FP16.

| GPU | ControlNet | ControlNet+IP | Img2Img | TPS Only |
|-----|------------|---------------|---------|----------|
| A100 80GB | -- | -- | -- | -- |
| A100 40GB | -- | -- | -- | -- |
| RTX 4090 | -- | -- | -- | -- |
| RTX 3090 | -- | -- | -- | -- |
| T4 16GB | -- | -- | -- | -- |
| M3 Pro (MPS) | -- | -- | -- | -- |
| CPU (i9-13900K) | N/A | N/A | N/A | -- |

## VRAM Usage by Mode

Peak GPU memory during inference at 512x512, FP16.

| Component | VRAM (GB) |
|-----------|-----------|
| SD 1.5 (FP16) | -- |
| ControlNet (FP16) | -- |
| IP-Adapter | -- |
| VAE (FP32 decode) | -- |
| CodeFormer | -- |
| ArcFace | -- |
| **ControlNet mode total** | **--** |
| **ControlNet+IP mode total** | **--** |
| **Img2Img mode total** | **--** |
| **TPS mode total** | **< 0.5** |

## Per-Stage Latency (A100 80GB)

Single image, averaged over 100 runs.

| Stage | Mean (ms) | Std (ms) |
|-------|-----------|----------|
| Landmark Extraction (MediaPipe) | -- | -- |
| Manipulation (Gaussian RBF) | -- | -- |
| Conditioning Generation | -- | -- |
| Mask Generation | -- | -- |
| TPS Warp | -- | -- |
| Diffusion (30 steps) | -- | -- |
| CodeFormer Restoration | -- | -- |
| Laplacian Blend | -- | -- |
| **Total** | **--** | **--** |

## Quality Metrics by Procedure

TPS mode, intensity=65, evaluated on held-out test set.

| Procedure | SSIM | LPIPS | NME | Identity Sim |
|-----------|------|-------|-----|-------------|
| Rhinoplasty | -- | -- | -- | -- |
| Blepharoplasty | -- | -- | -- | -- |
| Rhytidectomy | -- | -- | -- | -- |
| Orthognathic | -- | -- | -- | -- |
| Brow Lift | -- | -- | -- | -- |
| Mentoplasty | -- | -- | -- | -- |

## Quality Metrics by Mode

Rhinoplasty, intensity=65, same test set.

| Mode | SSIM | LPIPS | NME | Identity Sim | FID |
|------|------|-------|-----|-------------|-----|
| TPS Only | -- | -- | -- | -- | -- |
| Img2Img | -- | -- | -- | -- | -- |
| ControlNet | -- | -- | -- | -- | -- |
| ControlNet + IP | -- | -- | -- | -- | -- |

## Comparison with Related Work

| Method | SSIM | LPIPS | FID | Identity Sim | Notes |
|--------|------|-------|-----|-------------|-------|
| Alpha Blending | -- | -- | -- | -- | Trivial baseline |
| TPS Warp | -- | -- | -- | -- | Geometric only |
| Morphing (Beier-Neely) | -- | -- | -- | N/A | Classic approach |
| LandmarkDiff (Ours) | -- | -- | -- | -- | ControlNet mode |

## Fitzpatrick Equity Analysis

Metrics stratified by Fitzpatrick skin type (ITA-based classification).

| Skin Type | N | SSIM | LPIPS | NME | Identity Sim |
|-----------|---|------|-------|-----|-------------|
| I (very light) | -- | -- | -- | -- | -- |
| II (light) | -- | -- | -- | -- | -- |
| III (intermediate) | -- | -- | -- | -- | -- |
| IV (tan) | -- | -- | -- | -- | -- |
| V (brown) | -- | -- | -- | -- | -- |
| VI (dark) | -- | -- | -- | -- | -- |

## Reproducing These Benchmarks

```bash
# Per-stage + TPS benchmark (no GPU needed)
python scripts/benchmark_inference.py --modes tps --repeats 20

# Full mode comparison (GPU required)
python scripts/benchmark_inference.py \
    --modes tps img2img controlnet controlnet_ip \
    --repeats 10 --output results/benchmark

# Quality evaluation on test set
python scripts/evaluate_quality.py \
    --pred_dir results/predictions \
    --target_dir data/test_targets \
    --original_dir data/test_inputs \
    --compute-fid --output results/quality

# Generate paper figures
python scripts/generate_paper_figures.py \
    --input data/faces_all/000001.png \
    --output paper/figures/ --figure all
```
