# Mentoplasty

Chin reshaping targeting chin tip advancement, lower contour refinement, and jaw angle transition.

## Anatomy

Mentoplasty modifies three zones of the lower face:

- **Chin tip**: primary advancement, pushing the chin forward/upward (landmarks 152, 175)
- **Lower chin contour**: follows the tip with softer displacement for a natural shape (landmarks 148, 149, 150, 176, 377)
- **Jaw angles**: minimal upward pull to blend the chin advancement into the jaw contour (landmarks 171, 396)

## Affected Landmarks

8 landmarks from the MediaPipe 478-point mesh (the most focused procedure):

```
[148, 149, 150, 152, 171, 175, 176, 377]
```

Default influence radius: **25 px** (at 512x512).

## Displacement Behavior

| Sub-region | Direction | Magnitude (at scale=1.0) | Radius factor |
|------------|-----------|--------------------------|---------------|
| Chin tip | -Y (upward/forward) | 4.0 px | 1.0x |
| Lower contour | -Y (upward) | 2.5 px | 0.8x |
| Jaw angles | -Y (upward) | 1.0 px | 0.6x |

The chin tip receives the strongest displacement of any single handle across all procedures (4.0 px at full scale). The graduated falloff from tip to contour to jaw angles produces a natural chin shape.

## Intensity Recommendations

| Level | Intensity | Clinical analog |
|-------|-----------|-----------------|
| Conservative | 20-35 | Minor chin augmentation (implant) |
| Moderate | 40-65 | Standard sliding genioplasty |
| Aggressive | 70-100 | Major advancement genioplasty |

## Clinical Considerations

- **Ehlers-Danlos**: radius increases to 37.5 px. The chin advancement zone spreads wider into the lower lip and jawline.
- **Keloid-prone**: if `keloid_regions` includes `"jawline"`, post-compositing transitions near the chin incision line are softened.
- Mentoplasty pairs well with orthognathic surgery for comprehensive lower face reshaping. Apply orthognathic first for structural jaw changes, then mentoplasty for chin refinement.
- At high intensities (>80), the chin can appear over-projected. For natural results, 40-60 is typically sufficient.

Community contribution by [P-r-e-m-i-u-m](https://github.com/P-r-e-m-i-u-m) (PR [#36](https://github.com/dreamlessx/LandmarkDiff-public/pull/36)).

## Code Example

```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset

face = extract_landmarks(image_bgr)

# standard chin advancement
deformed = apply_procedure_preset(face, "mentoplasty", intensity=50)

# combine with orthognathic for full lower face work
combo = apply_procedure_preset(face, "orthognathic", intensity=45)
combo = apply_procedure_preset(combo, "mentoplasty", intensity=40)
```
