# Rhytidectomy

Facelift surgery targeting the jowl area, submental region, and upper face for comprehensive tissue repositioning.

## Anatomy

Rhytidectomy modifies three anatomical zones:

- **Jowl area**: the strongest lift direction, pulling tissue upward and toward the ear. Left jowl (landmarks 132, 136, 172, 58, 150, 176) moves up and left; right jowl (361, 365, 397, 288, 379, 400) moves up and right.
- **Submental/chin**: upward-only lift without lateral pull (landmarks 152, 148, 377, 378)
- **Temple/upper face**: very mild lift in the forehead and temple region. Left temple (10, 21, 54, 67, 103, 109, 162, 127) and right temple (284, 297, 332, 338, 323, 356, 389, 454).

## Affected Landmarks

32 landmarks from the MediaPipe 478-point mesh:

```
[10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 150, 162, 172, 176, 187,
 207, 213, 234, 284, 297, 323, 332, 338, 356, 361, 365, 379, 389, 397, 400, 427, 454]
```

Default influence radius: **40 px** (at 512x512). The largest radius of any procedure, reflecting the broad tissue redistribution involved in a facelift.

## Displacement Behavior

| Sub-region | Direction | Magnitude (at scale=1.0) | Radius factor |
|------------|-----------|--------------------------|---------------|
| Left jowl | -X, -Y (up + toward left ear) | 2.5, 3.0 px | 1.0x |
| Right jowl | +X, -Y (up + toward right ear) | 2.5, 3.0 px | 1.0x |
| Chin | -Y (upward only) | 2.0 px | 0.8x |
| Left temple | -X, -Y (mild lateral + up) | 0.5, 1.0 px | 0.6x |
| Right temple | +X, -Y (mild lateral + up) | 0.5, 1.0 px | 0.6x |

## Intensity Recommendations

| Level | Intensity | Clinical analog |
|-------|-----------|-----------------|
| Conservative | 20-35 | Mini-lift, mild jowl reduction |
| Moderate | 40-65 | Standard SMAS facelift |
| Aggressive | 70-100 | Deep-plane facelift with submental work |

## Clinical Considerations

- **Bell's palsy**: this is the procedure most affected by the Bell's palsy flag. All handles on the paralyzed side (jowl + temple) are excluded, producing an asymmetric lift on only the healthy side.
- **Ehlers-Danlos**: radius increases to 60 px. The broad influence zone means almost the entire lower face is affected.
- **Keloid-prone**: if `keloid_regions` includes `"jawline"`, mask boundaries along the jawline incision line are softened.

## Code Example

```python
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.clinical import ClinicalFlags

face = extract_landmarks(image_bgr)

# standard bilateral facelift
deformed = apply_procedure_preset(face, "rhytidectomy", intensity=60)

# with Bell's palsy (right side affected, only lift left side)
flags = ClinicalFlags(bells_palsy=True, bells_palsy_side="right")
deformed_bp = apply_procedure_preset(face, "rhytidectomy", intensity=60, clinical_flags=flags)
```
