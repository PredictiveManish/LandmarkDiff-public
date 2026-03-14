# Configuration

LandmarkDiff uses YAML-based experiment configs for reproducible training and evaluation. All configuration is handled through typed dataclasses in `landmarkdiff.config`.

## Loading Configuration

```python
from landmarkdiff.config import ExperimentConfig, load_config

# From YAML file
config = ExperimentConfig.from_yaml("configs/rhinoplasty_phaseA.yaml")

# With dot-notation overrides
config = load_config(
    "configs/phaseA.yaml",
    overrides={"training.learning_rate": 5e-6, "data.batch_size": 8},
)

# Programmatic
config = ExperimentConfig(
    experiment_name="rhino_v2",
    training=TrainingConfig(phase="A", learning_rate=1e-5),
)
config.to_yaml("configs/rhino_v2.yaml")
```

## YAML Schema

Below is the full schema with all defaults:

```yaml
experiment_name: "default"
description: ""
version: "0.3.0"
output_dir: "outputs"

model:
  base_model: "runwayml/stable-diffusion-v1-5"
  controlnet_conditioning_channels: 3
  controlnet_conditioning_scale: 1.0
  use_ema: true
  ema_decay: 0.9999
  gradient_checkpointing: true

training:
  phase: "A"                          # "A" = ControlNet-only, "B" = + identity loss
  learning_rate: 1.0e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  max_train_steps: 50000
  warmup_steps: 500
  mixed_precision: "fp16"
  seed: 42

  # Optimizer
  optimizer: "adamw"                  # "adamw", "adam8bit", "prodigy"
  adam_beta1: 0.9
  adam_beta2: 0.999
  weight_decay: 0.01
  max_grad_norm: 1.0

  # LR scheduler
  lr_scheduler: "cosine"
  lr_scheduler_kwargs: {}

  # Phase B specific
  identity_loss_weight: 0.1
  perceptual_loss_weight: 0.05
  use_differentiable_arcface: false
  arcface_weights_path: null

  # Checkpointing
  save_every_n_steps: 5000
  resume_from_checkpoint: null

  # Validation
  validate_every_n_steps: 2500
  num_validation_samples: 4

data:
  train_dir: "data/training"
  val_dir: "data/validation"
  test_dir: "data/test"
  image_size: 512
  num_workers: 4
  pin_memory: true

  # Augmentation
  random_flip: true
  random_rotation: 5.0                # degrees
  color_jitter: 0.1

  # Procedure filtering
  procedures:
    - rhinoplasty
    - blepharoplasty
    - rhytidectomy
    - orthognathic
  intensity_range: [30.0, 100.0]

  # Data-driven displacement
  displacement_model_path: null
  noise_scale: 0.1

inference:
  num_inference_steps: 30
  guidance_scale: 7.5
  scheduler: "dpmsolver++"            # "ddpm", "ddim", "dpmsolver++"
  controlnet_conditioning_scale: 1.0

  # Post-processing
  use_neural_postprocess: false
  restore_mode: "codeformer"
  codeformer_fidelity: 0.7
  use_realesrgan: true
  use_laplacian_blend: true
  sharpen_strength: 0.25

  # Identity verification
  verify_identity: true
  identity_threshold: 0.6

evaluation:
  compute_fid: true
  compute_lpips: true
  compute_nme: true
  compute_identity: true
  compute_ssim: true
  stratify_fitzpatrick: true
  stratify_procedure: true
  max_eval_samples: 0                 # 0 = all

wandb:
  enabled: true
  project: "landmarkdiff"
  entity: null
  run_name: null
  tags: []

safety:
  identity_threshold: 0.6
  max_displacement_fraction: 0.05
  watermark_enabled: true
  watermark_text: "AI-GENERATED PREDICTION"
  ood_detection_enabled: true
  ood_confidence_threshold: 0.3
  min_face_confidence: 0.5
  max_yaw_degrees: 45.0
```

## CLI Flags

The CLI (`python -m landmarkdiff`) accepts these flags:

### `infer` command

| Flag | Default | Description |
|------|---------|-------------|
| `image` | (required) | Path to input face image |
| `--procedure` | rhinoplasty | Surgical procedure |
| `--intensity` | 60.0 | Deformation intensity (0-100) |
| `--mode` | tps | Inference mode: tps, controlnet, img2img, controlnet_ip |
| `--output` | output/ | Output directory |
| `--steps` | 30 | Number of diffusion steps |
| `--seed` | None | Random seed for reproducibility |

### `landmarks` command

| Flag | Default | Description |
|------|---------|-------------|
| `image` | (required) | Path to input face image |
| `--output` | output/landmarks.png | Where to save the mesh visualization |

### `demo` command

Launches the Gradio web demo. No flags required.

## Extended CLI (`landmarkdiff.cli`)

The extended CLI has additional commands:

### `ensemble` command

| Flag | Default | Description |
|------|---------|-------------|
| `image` | (required) | Input face image |
| `--n-samples` | 5 | Number of ensemble members |
| `--strategy` | best_of_n | pixel_average, weighted_average, best_of_n, median |
| `--mode` | tps | Inference mode |

### `evaluate` command

| Flag | Default | Description |
|------|---------|-------------|
| `--test-dir` | (required) | Directory of test image pairs |
| `--output` | eval_results | Output directory for metrics |
| `--max-samples` | 0 | Max samples to evaluate (0 = all) |

### `config` command

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | None | YAML config to display/validate |
| `--validate` | false | Run validation checks on config |

## Config Validation

```python
from landmarkdiff.config import validate_config

warnings = validate_config(config)
for w in warnings:
    print(w)
```

Checks for common mistakes:
- Phase B without a Phase A checkpoint to resume from
- Effective batch size < 8 (may cause instability)
- Learning rate > 1e-4 (unusually high for fine-tuning)
- Image size != 512 (SD1.5 expects 512)
- Identity threshold < 0.3 (may pass poor quality outputs)

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `HF_TOKEN` | diffusers | HuggingFace auth for gated models |
| `CUDA_VISIBLE_DEVICES` | PyTorch | GPU device selection |
| `WANDB_API_KEY` | wandb | Weights & Biases logging |
| `TORCH_HOME` | PyTorch | Model cache directory |
