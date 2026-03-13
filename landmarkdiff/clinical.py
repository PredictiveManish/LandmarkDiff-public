"""Clinical edge cases: vitiligo, Bell's palsy, keloid, Ehlers-Danlos.

Each condition modifies the pipeline differently (mask exclusion,
asymmetric deformation, wider radii, etc).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks


@dataclass
class ClinicalFlags:
    """Flags that change how the pipeline handles this patient."""

    vitiligo: bool = False
    bells_palsy: bool = False
    bells_palsy_side: str = "left"  # affected side: "left" or "right"
    keloid_prone: bool = False
    keloid_regions: list[str] = field(default_factory=list)  # e.g. ["jawline", "nose"]
    ehlers_danlos: bool = False

    def has_any(self) -> bool:
        return self.vitiligo or self.bells_palsy or self.keloid_prone or self.ehlers_danlos


def detect_vitiligo_patches(
    image: np.ndarray,
    face: FaceLandmarks,
    l_threshold: float = 85.0,
    min_patch_area: int = 200,
) -> np.ndarray:
    """Detect depigmented (vitiligo) patches on face using LAB luminance."""
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Create face ROI mask from landmarks
    coords = face.pixel_coords.astype(np.int32)
    hull = cv2.convexHull(coords)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 255)

    # Get face-region luminance statistics
    l_channel = lab[:, :, 0]
    face_pixels = l_channel[face_mask > 0]
    if len(face_pixels) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    l_mean = np.mean(face_pixels)
    l_std = np.std(face_pixels)

    # Vitiligo patches: significantly brighter than mean skin
    threshold = min(l_threshold, l_mean + 2.0 * l_std)
    bright_mask = ((l_channel > threshold) & (face_mask > 0)).astype(np.uint8) * 255

    # Also check for low saturation (a,b channels close to 128)
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    low_sat = (
        (np.abs(a_channel - 128) < 15) & (np.abs(b_channel - 128) < 15)
    ).astype(np.uint8) * 255

    # Combined: bright AND low-saturation within face
    vitiligo_raw = cv2.bitwise_and(bright_mask, low_sat)

    # Filter small noise patches
    contours, _ = cv2.findContours(vitiligo_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_patch_area:
            cv2.fillPoly(result, [cnt], 255)

    return result


def adjust_mask_for_vitiligo(
    mask: np.ndarray,
    vitiligo_patches: np.ndarray,
    preservation_factor: float = 0.3,
) -> np.ndarray:
    """Reduce mask intensity over vitiligo patches to preserve them."""
    patches_f = vitiligo_patches.astype(np.float32) / 255.0
    reduction = patches_f * preservation_factor
    return np.clip(mask - reduction, 0.0, 1.0)


def get_bells_palsy_side_indices(
    side: str,
) -> dict[str, list[int]]:
    """Get landmark indices for the affected side in Bell's palsy."""
    if side == "left":
        return {
            "eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            "eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            "mouth_corner": [61, 146, 91, 181, 84],
            "jawline": [132, 136, 172, 58, 150, 176, 148, 149],
        }
    else:
        return {
            "eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            "eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            "mouth_corner": [291, 308, 324, 318, 402],
            "jawline": [361, 365, 397, 288, 379, 400, 377, 378],
        }


def get_keloid_exclusion_mask(
    face: FaceLandmarks,
    regions: list[str],
    width: int,
    height: int,
    margin_px: int = 10,
) -> np.ndarray:
    """Generate mask of keloid-prone regions to exclude from aggressive compositing."""
    from landmarkdiff.landmarks import LANDMARK_REGIONS

    mask = np.zeros((height, width), dtype=np.float32)
    coords = face.pixel_coords.astype(np.int32)

    for region in regions:
        indices = LANDMARK_REGIONS.get(region, [])
        if not indices:
            continue
        pts = coords[indices]
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1.0)

    # Dilate by margin
    if margin_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
        )
        mask = cv2.dilate(mask, kernel)

    return np.clip(mask, 0.0, 1.0)


def adjust_mask_for_keloid(
    mask: np.ndarray,
    keloid_mask: np.ndarray,
    reduction_factor: float = 0.5,
) -> np.ndarray:
    """Soften mask transitions in keloid-prone areas."""
    # Reduce mask intensity in keloid-prone areas
    keloid_reduction = keloid_mask * reduction_factor
    modified = mask * (1.0 - keloid_reduction)

    # Extra Gaussian blur in keloid regions for softer transitions
    blur_kernel = 31
    blurred = cv2.GaussianBlur(modified, (blur_kernel, blur_kernel), 10.0)

    # Use blurred version only in keloid regions
    result = modified * (1.0 - keloid_mask) + blurred * keloid_mask
    return np.clip(result, 0.0, 1.0)
