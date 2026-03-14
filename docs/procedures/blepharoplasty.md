# Blepharoplasty

Eyelid surgery targeting upper lid elevation, lower lid tightening, and periorbital reshaping.

## Anatomy

Blepharoplasty modifies three zones around each eye:

- **Upper lid**: central lid elevated upward (landmarks 159, 160, 161 left; 386, 385, 384 right)
- **Lid corners**: medial and lateral canthal regions with tapered displacement (landmarks 158, 157, 133, 33 left; 387, 388, 362, 263 right)
- **Lower lid**: subtle upward tightening to reduce laxity (landmarks 145, 153, 154 left; 374, 380, 381 right)

## Affected Landmarks

28 landmarks from the MediaPipe 478-point mesh:

```
[33, 7, 163, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 246,
 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
```

Default influence radius: **15 px** (at 512x512). This is the smallest radius of any procedure, reflecting the fine anatomy of the eyelids.

## Displacement Behavior

| Sub-region | Direction | Magnitude (at scale=1.0) | Radius factor |
|------------|-----------|--------------------------|---------------|
| Upper lid (central) | -Y (upward) | 2.0 px | 1.0x |
| Lid corners | -Y (upward) | 0.8 px | 0.7x |
| Lower lid | +Y (downward/tighten) | 0.5 px | 0.5x |

## Intensity Recommendations

| Level | Intensity | Clinical analog |
|-------|-----------|-----------------|
| Conservative | 20-35 | Mild ptosis correction |
| Moderate | 40-65 | Standard upper blepharoplasty |
| Aggressive | 70-100 | Combined upper + lower lid surgery |

## Clinical Considerations

- **Bell's palsy**: handles on the affected eye are removed entirely, since operating on a denervated eyelid carries significant risk. Only the healthy eye is deformed.
- **Ehlers-Danlos**: radius increases to 22.5 px (1.5x). Tissue laxity means the deformation effect spreads beyond the typical periorbital zone.
- The small default radius (15 px) keeps deformations tightly localized to the lid margins. If results look too pinched, consider a slight increase.

## Code Example

```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.conditioning import render_wireframe

face = extract_landmarks(image_bgr)
deformed = apply_procedure_preset(face, "blepharoplasty", intensity=50)
wireframe = render_wireframe(deformed, width=512, height=512)
```

Combine with rhinoplasty for a multi-procedure plan:

```python
step1 = apply_procedure_preset(face, "blepharoplasty", intensity=50)
step2 = apply_procedure_preset(step1, "rhinoplasty", intensity=40)
```
