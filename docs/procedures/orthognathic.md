# Orthognathic Surgery

Jaw repositioning targeting mandibular advancement, chin projection, and bilateral jaw narrowing.

## Anatomy

Orthognathic surgery modifies three structural zones:

- **Lower jaw (mandible)**: repositioned upward to simulate surgical advancement (landmarks 17, 18, 200, 201, 202, 204, 208, 211, 212, 214)
- **Chin point**: projected forward and upward (landmarks 175, 170, 169, 167, 396)
- **Lateral jaw**: bilateral inward pull for jaw narrowing. Left jaw (57, 61, 78, 91, 95, 146, 181) and right jaw (291, 311, 312, 321, 324, 325, 375, 405).

## Affected Landmarks

42 landmarks from the MediaPipe 478-point mesh:

```
[0, 17, 18, 36, 37, 39, 40, 57, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
 146, 167, 169, 170, 175, 181, 191, 200, 201, 202, 204, 208, 211, 212, 214,
 269, 270, 291, 311, 312, 317, 321, 324, 325, 375, 396, 405, 407, 415]
```

Default influence radius: **35 px** (at 512x512).

## Displacement Behavior

| Sub-region | Direction | Magnitude (at scale=1.0) | Radius factor |
|------------|-----------|--------------------------|---------------|
| Lower jaw | -Y (upward) | 3.0 px | 1.0x |
| Chin projection | -Y (upward) | 2.0 px | 0.7x |
| Left lateral jaw | +X, -Y (inward + up) | 1.5, 1.0 px | 0.8x |
| Right lateral jaw | -X, -Y (inward + up) | 1.5, 1.0 px | 0.8x |

## Intensity Recommendations

| Level | Intensity | Clinical analog |
|-------|-----------|-----------------|
| Conservative | 20-35 | Minor chin advancement |
| Moderate | 40-65 | Standard LeFort I or BSSO |
| Aggressive | 70-100 | Major skeletal repositioning |

## Clinical Considerations

- **Ehlers-Danlos**: radius increases to 52.5 px. The jaw deformation zone bleeds significantly into surrounding facial structures.
- **Bell's palsy**: has minimal effect on orthognathic surgery since jaw landmarks are mostly midline, but lateral handles on the affected side will be removed if the flag is set.
- Orthognathic surgery produces the largest structural changes of any procedure. At high intensities (>80), the jaw contour can shift noticeably.

## Code Example

```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset

face = extract_landmarks(image_bgr)
deformed = apply_procedure_preset(face, "orthognathic", intensity=55)

# combine with mentoplasty for jaw + chin work
combo = apply_procedure_preset(face, "orthognathic", intensity=50)
combo = apply_procedure_preset(combo, "mentoplasty", intensity=40)
```
