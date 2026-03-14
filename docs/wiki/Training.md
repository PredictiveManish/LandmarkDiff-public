# Training

LandmarkDiff training happens in two phases, with an optional data-driven displacement model.

## Training Phases

### Phase A: ControlNet Fine-Tuning

Phase A trains the ControlNet on face mesh conditioning images paired with target photographs. The ControlNet learns to generate photorealistic faces from wireframe conditioning while the SD1.5 base model stays frozen.

**Data format:** Pairs of (conditioning_image, target_image) where:
- conditioning_image: face mesh wireframe rendered on black background
- target_image: corresponding clinical photograph

**Key hyperparameters:**
```yaml
training:
  phase: "A"
  learning_rate: 1.0e-5
  batch_size: 4
  gradient_accumulation_steps: 4    # effective batch = 16
  max_train_steps: 50000
  warmup_steps: 500
  mixed_precision: "fp16"
  lr_scheduler: "cosine"
```

### Phase B: Identity-Aware Fine-Tuning

Phase B adds identity preservation losses to the training objective. It starts from a Phase A checkpoint and adds:

- **Identity loss** (weight 0.1): Cosine distance between ArcFace embeddings of generated and target images. Ensures the output preserves the patient's identity.
- **Perceptual loss** (weight 0.05): LPIPS perceptual similarity for texture quality.

```yaml
training:
  phase: "B"
  resume_from_checkpoint: "checkpoints/phaseA_final"
  identity_loss_weight: 0.1
  perceptual_loss_weight: 0.05
  use_differentiable_arcface: false
  learning_rate: 5.0e-6             # lower than Phase A
```

Phase B requires InsightFace for the identity loss:
```bash
pip install -e ".[train]"
```

## Data Preparation

### Training Data Structure

```
data/training/
    pair_001_before.png
    pair_001_after.png
    pair_002_before.png
    pair_002_after.png
    ...
```

Or with the `_input` / `_target` naming convention:
```
data/training/
    sample_001_input.png
    sample_001_target.png
    ...
```

Images should be:
- At least 512x512 pixels (will be resized during training)
- Well-lit clinical photographs
- Face centered and filling most of the frame
- Before and after images should have similar pose/angle

### Conditioning Generation

For each training pair, generate the ControlNet conditioning:

```python
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image

img = cv2.imread("pair_001_before.png")
img = cv2.resize(img, (512, 512))
face = extract_landmarks(img)
conditioning = render_landmark_image(face, 512, 512)
cv2.imwrite("pair_001_conditioning.png", conditioning)
```

Or use the batch script:
```bash
python scripts/build_training_dataset.py --input-dir data/raw --output-dir data/training
```

### Data Augmentation

LandmarkDiff includes augmentation strategies designed for clinical photography:

- Random horizontal flip (enabled by default)
- Random rotation up to 5 degrees
- Color jitter (brightness, contrast, saturation)
- TPS-based geometric augmentation (random non-rigid warps)

Configure in the experiment YAML:
```yaml
data:
  random_flip: true
  random_rotation: 5.0
  color_jitter: 0.1
```

## DisplacementModel Fitting

The `DisplacementModel` learns statistical displacement patterns from real before/after surgery pairs. Once fitted, it replaces the hand-tuned RBF displacement vectors with data-driven ones.

### Step 1: Extract Displacements

```python
from landmarkdiff.displacement_model import extract_from_directory

pairs = extract_from_directory(
    "data/surgery_pairs/",
    min_detection_confidence=0.5,
    min_quality=0.5,            # reject poorly-aligned pairs
)
print(f"Extracted {len(pairs)} pairs")
```

Each pair yields:
- `landmarks_before`: (478, 2) normalized coordinates
- `landmarks_after`: (478, 2) normalized coordinates
- `displacements`: (478, 2) displacement vectors
- `procedure`: auto-classified procedure name
- `quality_score`: alignment quality (0-1)

### Step 2: Fit the Model

```python
from landmarkdiff.displacement_model import DisplacementModel

model = DisplacementModel()
model.fit(pairs)

# Check what procedures were found
print(model.procedures)       # e.g., ['rhinoplasty', 'blepharoplasty']
print(model.n_samples)        # e.g., {'rhinoplasty': 42, 'blepharoplasty': 18}

# View summary statistics
summary = model.get_summary()
for proc, stats in summary["procedures"].items():
    print(f"{proc}: {stats['n_samples']} samples, "
          f"mean_mag={stats['global_mean_magnitude']:.5f}")
```

### Step 3: Save and Use

```python
model.save("data/displacement_model.npz")

# Later, use in pipeline:
pipe = LandmarkDiffPipeline(
    mode="controlnet",
    displacement_model_path="data/displacement_model.npz",
)
```

When a displacement model is loaded and the requested procedure exists in the model, `generate()` uses data-driven displacements instead of hand-tuned presets. The `manipulation_mode` key in the result dict indicates which was used.

### Procedure Classification

The displacement extraction automatically classifies which procedure was performed based on which anatomical region shows the most displacement. This uses the same `PROCEDURE_LANDMARKS` regions as the deformation presets. If no region shows significant displacement (mean < 0.002 in normalized coordinates), the pair is classified as "unknown".

## Training Commands

### Launch Training

```bash
python scripts/train_controlnet.py --config configs/phaseA.yaml
```

### Monitor Training

Weights & Biases logging is enabled by default:
```yaml
wandb:
  enabled: true
  project: "landmarkdiff"
  run_name: "phaseA_v3"
```

Or use the training dashboard:
```bash
python scripts/training_dashboard.py --log-dir outputs/phaseA_v3/
```

### Checkpointing

Checkpoints are saved every N steps (default 5000):
```yaml
training:
  save_every_n_steps: 5000
  resume_from_checkpoint: null      # path to resume from
```

Resume from a checkpoint:
```bash
python scripts/train_controlnet.py --config configs/phaseA.yaml \
    --resume checkpoints/phaseA_step30000
```

### Validation During Training

Validation runs every N steps with a small set of images:
```yaml
training:
  validate_every_n_steps: 2500
  num_validation_samples: 4
```

## Evaluation

After training, evaluate on a held-out test set:

```bash
landmarkdiff evaluate --test-dir data/test --output eval_results --mode controlnet
```

Available metrics:
- **FID** (Frechet Inception Distance): overall image quality
- **LPIPS** (Learned Perceptual Image Patch Similarity): perceptual similarity
- **NME** (Normalized Mean Error): landmark accuracy
- **Identity** (ArcFace cosine similarity): identity preservation
- **SSIM** (Structural Similarity): structural quality

Results can be stratified by Fitzpatrick skin type and by procedure.
