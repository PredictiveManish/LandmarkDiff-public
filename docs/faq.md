# FAQ and Troubleshooting

## General

**What input does LandmarkDiff need?**

A single 2D photograph of a face. No 3D scans, no depth sensors. A clear frontal photo with even lighting works best. The image gets resized to 512x512 internally.

**Can I run this without a GPU?**

Yes, in TPS mode. It does geometric warping on CPU in about 0.5 seconds. The photorealistic diffusion modes (controlnet, img2img) need a GPU with at least 6GB VRAM.

**What GPU do I need?**

For inference: anything with 6GB+ VRAM (RTX 3060 and up, T4, etc.). For training: 24GB minimum (RTX 3090), 80GB recommended (A100).

**Is this FDA approved for clinical use?**

No. This is a research tool. Predictions are AI-generated simulations, not medical advice. Any clinical deployment would need regulatory review.

**Can I use my own photos?**

Yes. The pipeline works on any photo with a detectable face. Upload through the Gradio demo or pass a file path to the CLI.

## Installation Issues

**`mediapipe` fails to install**

MediaPipe has specific Python version requirements. Make sure you're on Python 3.10 or 3.11. Python 3.12 support depends on the mediapipe version.

```bash
# check your python version
python --version

# if needed, create a fresh env
conda create -n landmarkdiff python=3.11
conda activate landmarkdiff
pip install -e .
```

**CUDA out of memory**

The full pipeline uses about 5.2GB VRAM. If you're running out of memory:

1. Close other GPU processes (`nvidia-smi` to check)
2. Use `--mode tps` for CPU-only inference
3. Reduce `--steps` (fewer diffusion steps = less memory)
4. Make sure you're using fp16 (the default)

**`torch` not finding CUDA**

```bash
# check if pytorch sees your GPU
python -c "import torch; print(torch.cuda.is_available())"

# if False, reinstall pytorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Models downloading every time**

The models cache to `~/.cache/huggingface/`. If that's getting cleared (e.g. on a cluster with temp home dirs), set `HF_HOME` to a persistent location:

```bash
export HF_HOME=/path/to/persistent/cache
```

## Pipeline Issues

**"No face detected"**

MediaPipe couldn't find a face. Common causes:
- Image is too dark or too bright
- Face is heavily occluded (sunglasses, mask)
- Face is at an extreme angle (>60 degrees from frontal)
- Image resolution is very low

Try a clearer, more frontal photo.

**Output looks like a different person**

This can happen when the ControlNet generates from the mesh without enough identity signal from the input. Try:
- Using `controlnet_ip` mode (adds IP-Adapter for identity preservation)
- Lowering `controlnet_conditioning_scale` (less mesh influence, more identity)
- Checking the ArcFace identity score in the output dict

**Output has visible seams or color mismatch**

The Laplacian pyramid blending should handle this, but if you see artifacts:
- Make sure post-processing is enabled (`postprocess=True`)
- The LAB histogram matching step corrects skin tone differences
- Check if the surgical mask is too large or too small for the procedure

**TPS warp looks distorted**

TPS mode is purely geometric - it pushes pixels around without any neural generation. At high intensity (>70%) the distortion becomes obvious. This is expected. Use ControlNet mode for photorealistic results.

## Training Issues

**Training loss is NaN**

Common causes:
- Using fp16 instead of bf16 (fp16 can overflow with diffusion models)
- Learning rate too high (start with 1e-5)
- Bad training data (corrupted images or landmarks)

Make sure `mixed_precision: bf16` is set in your config.

**Training is very slow**

- Check that you're actually using the GPU (`nvidia-smi` during training)
- Pre-compute TPS warps before training to avoid CPU bottleneck
- Use gradient accumulation to increase effective batch size without more VRAM
- See the [GPU training guide](GPU_TRAINING_GUIDE.md) for SLURM and multi-GPU setup

**Checkpoints are huge**

ControlNet checkpoints are about 1.4GB each in fp16. Use safetensors format (the default) which is slightly smaller and faster to load. Set `save_every_n_steps` in the config to avoid saving too frequently.

## Gradio Demo Issues

**Demo won't start**

```bash
# make sure gradio is installed
pip install -e ".[app]"

# check the port isn't in use
lsof -i :7860

# try a different port
python scripts/app.py --port 7861
```

**Demo is slow on first run**

The first inference downloads model weights (~6GB total). Subsequent runs use the cached models and should be much faster.

**File upload not working in Colab**

The Gradio demo uses `share=True` by default in Colab, which creates a public URL. If file upload fails, try running locally or using the notebook instead.
