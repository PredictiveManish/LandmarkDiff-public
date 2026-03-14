# Brow Lift

Forehead and brow elevation targeting the eyebrow arch and upper forehead for a refreshed appearance.

## Anatomy

Brow lift modifies two zones:

- **Brow arch**: primary elevation of the eyebrow, with the lateral brow lifted more than the medial brow (matching real surgical technique). Left brow (70, 63, 105, 66, 107) and right brow (300, 293, 334, 296, 336).
- **Forehead**: subtle smoothing lift across the upper face (landmarks 9, 8, 10, 109, 67, 103, 338, 297, 332)

## Affected Landmarks

19 landmarks from the MediaPipe 478-point mesh:

```
[70, 63, 105, 66, 107, 300, 293, 334, 296, 336, 9, 8, 10, 109, 67, 103, 338, 297, 332]
```

Default influence radius: **25 px** (at 512x512).

## Displacement Behavior

Brow landmarks use weighted displacement. The lateral brow receives a higher weight than the medial brow:

| Brow position | Weight | Displacement at scale=1.0 |
|---------------|--------|---------------------------|
| Most medial | 0.7 | -2.8 px (upward) |
| Inner-mid | 0.8 | -3.2 px |
| Mid | 0.9 | -3.6 px |
| Outer-mid | 1.0 | -4.0 px |
| Most lateral | 1.1 | -4.4 px |

| Sub-region | Direction | Magnitude | Radius factor |
|------------|-----------|-----------|---------------|
| Brow arch (weighted) | -Y (upward) | 2.8-4.4 px | 1.0x |
| Forehead | -Y (upward) | 1.5 px | 1.2x |

The forehead uses a wider radius (1.2x = 30 px) to produce a smooth, gradual lift across the upper face.

## Intensity Recommendations

| Level | Intensity | Clinical analog |
|-------|-----------|-----------------|
| Conservative | 20-35 | Mild brow ptosis correction |
| Moderate | 40-65 | Standard endoscopic brow lift |
| Aggressive | 70-100 | Coronal brow lift, significant elevation |

## Clinical Considerations

- **Bell's palsy**: if the affected side includes the brow region, handles on that side are removed. This produces an asymmetric lift.
- **Ehlers-Danlos**: radius increases to 37.5 px, with forehead radius at 45 px. The wide forehead influence zone means the lift extends well into the hairline region.
- Brow lift pairs naturally with blepharoplasty. The brow elevation can reduce upper lid hooding, sometimes reducing the blepharoplasty intensity needed.

Community contribution by [Deepak8858](https://github.com/Deepak8858) (PR [#35](https://github.com/dreamlessx/LandmarkDiff-public/pull/35)).

## Code Example

```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset

face = extract_landmarks(image_bgr)

# standard brow lift
deformed = apply_procedure_preset(face, "brow_lift", intensity=50)

# combine brow lift with blepharoplasty for upper face rejuvenation
step1 = apply_procedure_preset(face, "brow_lift", intensity=40)
step2 = apply_procedure_preset(step1, "blepharoplasty", intensity=35)
```
