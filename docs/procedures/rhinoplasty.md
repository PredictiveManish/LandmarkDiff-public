# Rhinoplasty

Nose reshaping targeting the nasal bridge, tip, and alar base.

## Anatomy

Rhinoplasty modifies three sub-regions of the nose:

- **Alar base**: nostrils narrowed toward the midline (landmarks 240, 236, 141, 363, 370 on the left; 460, 456, 274, 275, 278, 279 on the right)
- **Nasal tip**: upward rotation and refinement (landmarks 1, 2, 94, 19)
- **Dorsum**: bilateral narrowing of the nasal bridge (landmarks 195, 197, 236 left; 326, 327, 456 right)

## Affected Landmarks

24 landmarks from the MediaPipe 478-point mesh:

```
[1, 2, 4, 5, 6, 19, 94, 141, 168, 195, 197, 236, 240, 274, 275, 278, 279, 294, 326, 327, 360, 363, 370, 456, 460]
```

Default influence radius: **30 px** (at 512x512).

## Displacement Behavior

| Sub-region | Direction | Magnitude (at scale=1.0) | Radius factor |
|------------|-----------|--------------------------|---------------|
| Left alar  | +X (inward) | 2.5 px | 0.6x |
| Right alar | -X (inward) | 2.5 px | 0.6x |
| Tip        | -Y (upward) | 2.0 px | 0.5x |
| Left dorsum | +X (inward) | 1.5 px | 0.5x |
| Right dorsum | -X (inward) | 1.5 px | 0.5x |

All displacements scale linearly with `intensity / 100`.

## Intensity Recommendations

| Level | Intensity | Clinical analog |
|-------|-----------|-----------------|
| Conservative | 20-35 | Minor tip refinement |
| Moderate | 40-65 | Standard cosmetic rhinoplasty |
| Aggressive | 70-100 | Major reconstructive reshaping |

## Clinical Considerations

- **Ehlers-Danlos**: influence radius increases to 45 px (1.5x), producing a wider zone of deformation around the nose.
- **Keloid-prone**: if `keloid_regions` includes `"nose"`, post-compositing mask transitions are softened in the nasal region.
- **Bell's palsy**: not typically relevant for rhinoplasty, but if set, handles on the affected side are removed.

## Code Example

```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.conditioning import render_wireframe

face = extract_landmarks(image_bgr)
deformed = apply_procedure_preset(face, "rhinoplasty", intensity=50)
wireframe = render_wireframe(deformed, width=512, height=512)
```

For data-driven mode with a fitted displacement model:

```python
deformed = apply_procedure_preset(
    face,
    "rhinoplasty",
    intensity=60,
    displacement_model_path="data/hda_displacement_model.npz",
    noise_scale=0.2,
)
```
