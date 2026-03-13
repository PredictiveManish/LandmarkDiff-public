"""4-term loss for ControlNet fine-tuning.

L_total = L_diff + w_lm * L_landmark + w_id * L_identity + w_perc * L_perceptual

Phase A (synthetic TPS data): diffusion loss only. No perceptual against
rubbery TPS warps - it would penalize realism.
Phase B (FEM/clinical data): all 4 terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    """Loss term weights."""

    diffusion: float = 1.0
    landmark: float = 0.1
    identity: float = 0.05
    perceptual: float = 0.1


class DiffusionLoss:
    """Epsilon-prediction MSE."""

    def __call__(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(noise_pred, noise_target)


class LandmarkLoss:
    """L2 landmark distance, IOD-normalized, inside surgical mask only.

    Requires re-extraction from generated image (eval only, too slow per step).
    """

    def __call__(
        self,
        pred_landmarks: torch.Tensor,  # (B, N, 2)
        target_landmarks: torch.Tensor,  # (B, N, 2)
        mask: torch.Tensor | None = None,  # (B, N) binary
        iod: torch.Tensor | None = None,  # (B,) inter-ocular distance
    ) -> torch.Tensor:
        diff = pred_landmarks - target_landmarks  # (B, N, 2)
        dist = torch.norm(diff, dim=-1)  # (B, N)

        if mask is not None:
            dist = dist * mask
            count = mask.sum(dim=-1).clamp(min=1)
            mean_dist = dist.sum(dim=-1) / count
        else:
            mean_dist = dist.mean(dim=-1)

        if iod is not None:
            mean_dist = mean_dist / iod.clamp(min=1.0)

        return mean_dist.mean()


class IdentityLoss:
    """ArcFace cosine sim loss, procedure-dependent crop.

    buffalo_l 512-dim embeddings, falls back to pixel cosine if unavailable.
    Disabled for orthognathic. Images MUST be [-1,1] at 112x112 for ArcFace.
    """

    def __init__(self, device: torch.device | None = None):
        self._model = None
        self._device = device
        self._has_arcface = None  # None = not checked yet

    def _ensure_loaded(self, device: torch.device) -> None:
        """Lazy-load ArcFace on first call."""
        if self._has_arcface is not None:
            return
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            ctx_id = device.index if device.type == "cuda" and device.index is not None else (0 if device.type == "cuda" else -1)
            self._app.prepare(ctx_id=ctx_id, det_size=(320, 320))
            self._has_arcface = True
        except Exception:
            self._has_arcface = False

    @torch.no_grad()
    def _extract_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """(B,3,112,112) in [-1,1] -> (B,512) embeddings (or pixel fallback)."""
        if self._has_arcface:
            import numpy as np
            embeddings = []
            valid_mask = []
            for i in range(image_tensor.shape[0]):
                # Convert to uint8 BGR for InsightFace
                img = ((image_tensor[i].permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255)
                img_np = img.cpu().numpy().astype(np.uint8)
                img_bgr = img_np[:, :, ::-1].copy()

                faces = self._app.get(img_bgr)
                if faces and hasattr(faces[0], "embedding") and faces[0].embedding is not None:
                    embeddings.append(torch.from_numpy(faces[0].embedding))
                    valid_mask.append(True)
                else:
                    embeddings.append(torch.zeros(512))
                    valid_mask.append(False)

            return torch.stack(embeddings).to(image_tensor.device), valid_mask
        else:
            # Fallback: pixel-level features
            return image_tensor.flatten(1), [True] * image_tensor.shape[0]

    def __call__(
        self,
        pred_image: torch.Tensor,  # (B, 3, H, W) in [0, 1]
        target_image: torch.Tensor,
        procedure: str = "rhinoplasty",
    ) -> torch.Tensor:
        if procedure == "orthognathic":
            return torch.tensor(0.0, device=pred_image.device)

        self._ensure_loaded(pred_image.device)

        # Crop based on procedure
        pred_crop = self._procedure_crop(pred_image, procedure)
        target_crop = self._procedure_crop(target_image, procedure)

        # Resize to 112x112 for ArcFace
        pred_112 = F.interpolate(pred_crop, size=(112, 112), mode="bilinear", align_corners=False)
        target_112 = F.interpolate(target_crop, size=(112, 112), mode="bilinear", align_corners=False)

        # Normalize to [-1, 1]
        pred_norm = pred_112 * 2 - 1
        target_norm = target_112 * 2 - 1

        # Extract embeddings (ArcFace or fallback)
        pred_emb, pred_valid = self._extract_embedding(pred_norm)
        target_emb, target_valid = self._extract_embedding(target_norm)

        # Only compute loss for samples where both faces were detected
        valid = [p and t for p, t in zip(pred_valid, target_valid)]
        if not any(valid):
            return torch.tensor(0.0, device=pred_image.device)

        valid_t = torch.tensor(valid, device=pred_image.device)

        # L2 normalize (safe, only valid embeddings have nonzero norm)
        pred_emb = F.normalize(pred_emb.float(), dim=1)
        target_emb = F.normalize(target_emb.float(), dim=1)

        cosine_sim = (pred_emb * target_emb).sum(dim=1)
        # Zero out invalid entries before averaging
        cosine_sim = cosine_sim * valid_t.float()
        return (1 - cosine_sim).sum() / valid_t.float().sum()

    def _procedure_crop(
        self,
        image: torch.Tensor,
        procedure: str,
    ) -> torch.Tensor:
        """Procedure-specific crop for identity comparison."""
        _, _, h, w = image.shape

        if procedure == "rhinoplasty":
            # Upper face crop (forehead to nose tip)
            return image[:, :, : h * 2 // 3, :]
        elif procedure == "blepharoplasty":
            # Full face
            return image
        elif procedure == "rhytidectomy":
            # Upper face (above jawline)
            return image[:, :, : h * 3 // 4, :]
        else:
            return image


class PerceptualLoss:
    """LPIPS outside surgical mask only. Remember: LPIPS wants [-1,1], VAE gives [0,1]."""

    def __init__(self):
        self._lpips = None

    def _ensure_loaded(self, device: torch.device) -> None:
        if self._lpips is None:
            try:
                import lpips
                self._lpips = lpips.LPIPS(net="alex").to(device)
                self._lpips.eval()
                for p in self._lpips.parameters():
                    p.requires_grad_(False)
            except ImportError:
                self._lpips = "unavailable"

    def __call__(
        self,
        pred: torch.Tensor,    # (B, 3, H, W) in [0, 1]
        target: torch.Tensor,
        mask: torch.Tensor,    # (B, 1, H, W) surgical mask [0, 1]
    ) -> torch.Tensor:
        self._ensure_loaded(pred.device)

        # Invert mask: we want loss OUTSIDE surgical region
        outside_mask = 1 - mask

        # Erode outside_mask by a few pixels to avoid artificial edge features
        # at the mask boundary (LPIPS VGG detects the hard 0->value transition)
        erode_kernel = 5
        if outside_mask.shape[-1] >= erode_kernel and outside_mask.shape[-2] >= erode_kernel:
            outside_mask = -F.max_pool2d(
                -outside_mask,
                kernel_size=erode_kernel,
                stride=1,
                padding=erode_kernel // 2,
            )

        # Normalize to [-1, 1] for LPIPS FIRST, then mask
        pred_norm = pred * 2 - 1
        target_norm = target * 2 - 1

        # Apply mask after normalization (masked regions become 0, not -1)
        pred_norm = pred_norm * outside_mask
        target_norm = target_norm * outside_mask

        if self._lpips == "unavailable":
            # Fallback: simple L1 loss
            return F.l1_loss(pred_norm, target_norm)

        return self._lpips(pred_norm, target_norm).mean()


class CombinedLoss:
    """4-term combined loss. phase='A' = diffusion only, phase='B' = all terms."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        phase: str = "A",
    ):
        self.weights = weights or LossWeights()
        self.phase = phase
        self.diffusion_loss = DiffusionLoss()
        self.landmark_loss = LandmarkLoss()
        self.identity_loss = IdentityLoss()
        self.perceptual_loss = PerceptualLoss()

    def __call__(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        losses = {}

        # Always compute diffusion loss
        losses["diffusion"] = self.weights.diffusion * self.diffusion_loss(noise_pred, noise_target)
        losses["total"] = losses["diffusion"]

        if self.phase == "B":
            # Phase B: add auxiliary losses
            if "pred_landmarks" in kwargs and "target_landmarks" in kwargs:
                losses["landmark"] = self.weights.landmark * self.landmark_loss(
                    kwargs["pred_landmarks"],
                    kwargs["target_landmarks"],
                    kwargs.get("landmark_mask"),
                    kwargs.get("iod"),
                )
                losses["total"] = losses["total"] + losses["landmark"]

            if "pred_image" in kwargs and "target_image" in kwargs:
                procedure = kwargs.get("procedure", "rhinoplasty")
                losses["identity"] = self.weights.identity * self.identity_loss(
                    kwargs["pred_image"],
                    kwargs["target_image"],
                    procedure,
                )
                losses["total"] = losses["total"] + losses["identity"]

            if "pred_image" in kwargs and "target_image" in kwargs and "mask" in kwargs:
                losses["perceptual"] = self.weights.perceptual * self.perceptual_loss(
                    kwargs["pred_image"],
                    kwargs["target_image"],
                    kwargs["mask"],
                )
                losses["total"] = losses["total"] + losses["perceptual"]

        return losses
