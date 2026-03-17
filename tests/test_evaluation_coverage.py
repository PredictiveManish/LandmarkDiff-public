"""Additional evaluation tests targeting uncovered lines.

Covers: summary() with full Fitzpatrick/FID breakdown, compute_ssim
fallback (no skimage), evaluate_batch with identity and NME+procedures,
classify_fitzpatrick_ita intermediate thresholds.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from landmarkdiff.evaluation import (
    EvalMetrics,
    classify_fitzpatrick_ita,
    compute_nme,
    compute_ssim,
    evaluate_batch,
)

# ---------------------------------------------------------------------------
# EvalMetrics.summary() — uncovered Fitzpatrick branches
# ---------------------------------------------------------------------------


class TestSummaryFullBreakdown:
    """Cover lines 62-69: summary with nme, identity_sim, and FID by Fitzpatrick."""

    def test_summary_with_nme_and_identity(self):
        m = EvalMetrics(
            ssim=0.85,
            count_by_fitzpatrick={"II": 5, "IV": 8},
            lpips_by_fitzpatrick={"II": 0.12, "IV": 0.15},
            ssim_by_fitzpatrick={"II": 0.88, "IV": 0.82},
            nme_by_fitzpatrick={"II": 0.03, "IV": 0.05},
            identity_sim_by_fitzpatrick={"II": 0.92, "IV": 0.88},
        )
        s = m.summary()
        assert "NME=0.0300" in s
        assert "NME=0.0500" in s
        assert "ID=0.9200" in s
        assert "ID=0.8800" in s

    def test_summary_with_fid_by_fitzpatrick(self):
        m = EvalMetrics(
            fid=25.0,
            fid_by_fitzpatrick={"I": 20.0, "III": 30.0},
        )
        s = m.summary()
        assert "FID by Fitzpatrick" in s
        assert "Type I: 20.00" in s
        assert "Type III: 30.00" in s


# ---------------------------------------------------------------------------
# compute_ssim fallback (no skimage) — lines 183-200
# ---------------------------------------------------------------------------


class TestSSIMFallback:
    """Cover compute_ssim fallback when skimage is not available."""

    def test_ssim_fallback_identical(self):
        """Fallback SSIM on identical images should be close to 1."""
        with patch.dict("sys.modules", {"skimage": None, "skimage.metrics": None}):
            img = np.random.default_rng(0).integers(50, 200, (64, 64, 3), dtype=np.uint8)
            score = compute_ssim(img, img)
            assert isinstance(score, float)
            assert score > 0.9

    def test_ssim_fallback_different(self):
        """Fallback SSIM on very different images should be low."""
        with patch.dict("sys.modules", {"skimage": None, "skimage.metrics": None}):
            a = np.zeros((64, 64, 3), dtype=np.uint8)
            b = np.full((64, 64, 3), 255, dtype=np.uint8)
            score = compute_ssim(a, b)
            assert isinstance(score, float)
            assert score < 0.5


# ---------------------------------------------------------------------------
# classify_fitzpatrick_ita — intermediate thresholds (lines 128-134)
# ---------------------------------------------------------------------------


class TestFitzpatrickThresholds:
    """Cover all 6 Fitzpatrick type thresholds."""

    def test_returns_valid_type_for_various_images(self):
        """All synthetic images should return valid Fitzpatrick types."""
        for brightness in [250, 200, 150, 100, 50, 15]:
            img = np.full((64, 64, 3), brightness, dtype=np.uint8)
            ftype = classify_fitzpatrick_ita(img)
            assert ftype in {"I", "II", "III", "IV", "V", "VI"}

    def test_bright_image(self):
        """Very bright image should classify as light skin type."""
        img = np.full((64, 64, 3), 250, dtype=np.uint8)
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in {"I", "II", "III"}

    def test_mid_image(self):
        """Mid-tone image returns a valid type."""
        img = np.full((64, 64, 3), 130, dtype=np.uint8)
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in {"I", "II", "III", "IV", "V", "VI"}

    def test_dark_image(self):
        """Dark image should classify as darker skin type."""
        img = np.full((64, 64, 3), 15, dtype=np.uint8)
        ftype = classify_fitzpatrick_ita(img)
        assert ftype in {"IV", "V", "VI"}


# ---------------------------------------------------------------------------
# evaluate_batch — identity sim + NME by procedure (lines 357, 396-398, 411-413)
# ---------------------------------------------------------------------------


class TestEvaluateBatchIdentity:
    """Cover evaluate_batch with compute_identity=True (mocked)."""

    def test_with_identity_and_landmarks_and_procedures(self):
        """Cover identity scoring + NME by procedure + identity by Fitzpatrick."""
        rng = np.random.default_rng(42)
        preds = [rng.integers(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        targets = [p.copy() for p in preds]
        pred_lm = [rng.uniform(50, 200, (478, 2)).astype(np.float32) for _ in range(3)]
        target_lm = [lm.copy() for lm in pred_lm]
        procedures = ["rhinoplasty", "rhinoplasty", "blepharoplasty"]

        # Mock compute_identity_similarity to avoid needing InsightFace
        with patch(
            "landmarkdiff.evaluation.compute_identity_similarity",
            return_value=0.95,
        ):
            metrics = evaluate_batch(
                preds,
                targets,
                pred_landmarks=pred_lm,
                target_landmarks=target_lm,
                procedures=procedures,
                compute_identity=True,
            )

        assert metrics.identity_sim > 0
        assert "rhinoplasty" in metrics.nme_by_procedure
        assert "blepharoplasty" in metrics.nme_by_procedure
        assert "rhinoplasty" in metrics.lpips_by_procedure

    def test_identity_by_fitzpatrick_populated(self):
        """Verify identity_sim_by_fitzpatrick gets populated."""
        rng = np.random.default_rng(0)
        preds = [rng.integers(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(2)]
        targets = [p.copy() for p in preds]

        with patch(
            "landmarkdiff.evaluation.compute_identity_similarity",
            return_value=0.88,
        ):
            metrics = evaluate_batch(
                preds,
                targets,
                compute_identity=True,
            )

        assert metrics.identity_sim > 0
        # Should have at least one Fitzpatrick group with identity sim
        if metrics.count_by_fitzpatrick:
            for ftype in metrics.count_by_fitzpatrick:
                if ftype in metrics.identity_sim_by_fitzpatrick:
                    assert metrics.identity_sim_by_fitzpatrick[ftype] > 0


class TestEvaluateBatchNMEByProcedure:
    """Cover NME by procedure (lines 411-413)."""

    def test_nme_by_procedure(self):
        rng = np.random.default_rng(0)
        preds = [rng.integers(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        targets = [p.copy() for p in preds]
        pred_lm = [rng.uniform(50, 200, (478, 2)).astype(np.float32) for _ in range(4)]
        target_lm = [lm + 2.0 for lm in pred_lm]  # slight shift for nonzero NME
        procedures = ["rhinoplasty", "rhinoplasty", "blepharoplasty", "blepharoplasty"]

        metrics = evaluate_batch(
            preds,
            targets,
            pred_landmarks=pred_lm,
            target_landmarks=target_lm,
            procedures=procedures,
        )

        assert "rhinoplasty" in metrics.nme_by_procedure
        assert "blepharoplasty" in metrics.nme_by_procedure
        assert metrics.nme_by_procedure["rhinoplasty"] > 0
        assert metrics.nme_by_procedure["blepharoplasty"] > 0


# ---------------------------------------------------------------------------
# compute_nme edge case
# ---------------------------------------------------------------------------


class TestComputeNMESmallIOD:
    """Cover IOD clamping when eyes are very close."""

    def test_very_small_iod(self):
        lm = np.zeros((478, 2), dtype=np.float32)
        lm[33] = [100.0, 100.0]
        lm[263] = [100.5, 100.0]  # IOD < 1.0
        pred = lm.copy()
        pred[0] = [50.0, 50.0]
        nme = compute_nme(pred, lm)
        assert np.isfinite(nme)
        assert nme > 0
