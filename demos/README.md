# Demo Outputs

## Pipeline Overview

Abstract schematic of the five-stage LandmarkDiff pipeline.

| File | Description |
|------|-------------|
| `pipeline_abstract.png` | Five-stage pipeline: Input, Mesh Extraction, RBF Deformation, ControlNet + SD1.5, Result |

## Mesh Deformation

Side-by-side comparison of original and deformed face meshes showing procedure-specific Gaussian RBF displacement vectors.

| File | Description |
|------|-------------|
| `mesh_deformation.png` | Original vs. deformed mesh with displacement vectors and legend |

## Photorealistic Results

ControlNet-generated photorealistic demos will be added here once model training is complete. The pipeline is fully implemented -- we are currently training on synthetic data and will update this directory with proper results.

To generate your own demos:

```bash
# TPS mode (CPU, geometric only)
python examples/tps_only.py /path/to/face.jpg --procedure rhinoplasty --intensity 60

# ControlNet mode (GPU, photorealistic)
python scripts/run_inference.py /path/to/face.jpg --procedure rhinoplasty --intensity 60 --mode controlnet
```
