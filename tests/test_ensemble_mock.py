"""Mock-based tests for ensemble module -- aggregation and CLI entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _weighted_average
# ---------------------------------------------------------------------------


class TestWeightedAverage:
    """Tests for _weighted_average with mocked compute_ssim."""

    def test_basic_weighted(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        a = np.full((32, 32, 3), 100, dtype=np.uint8)
        b = np.full((32, 32, 3), 200, dtype=np.uint8)
        ref = np.full((32, 32, 3), 150, dtype=np.uint8)

        with patch("landmarkdiff.evaluation.compute_ssim", side_effect=[0.9, 0.1]):
            result, scores = ei._weighted_average([a, b], ref)

        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8
        assert len(scores) == 2
        assert scores[0] == 0.9
        assert scores[1] == 0.1
        # Weighted toward 'a' (score 0.9 vs 0.1) -> closer to 100
        assert result.mean() < 150

    def test_equal_weights(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        a = np.full((32, 32, 3), 0, dtype=np.uint8)
        b = np.full((32, 32, 3), 200, dtype=np.uint8)
        ref = np.full((32, 32, 3), 100, dtype=np.uint8)

        with patch("landmarkdiff.evaluation.compute_ssim", side_effect=[0.5, 0.5]):
            result, scores = ei._weighted_average([a, b], ref)

        assert abs(result.mean() - 100.0) < 2.0

    def test_zero_scores(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        a = np.full((32, 32, 3), 100, dtype=np.uint8)
        b = np.full((32, 32, 3), 200, dtype=np.uint8)
        ref = np.full((32, 32, 3), 150, dtype=np.uint8)

        # All zero scores -> total fallback to 1.0
        with patch("landmarkdiff.evaluation.compute_ssim", side_effect=[0.0, 0.0]):
            result, scores = ei._weighted_average([a, b], ref)

        assert result.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# _best_of_n
# ---------------------------------------------------------------------------


class TestBestOfN:
    """Tests for _best_of_n with mocked compute_identity_similarity."""

    def test_selects_best(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        a = np.full((32, 32, 3), 100, dtype=np.uint8)
        b = np.full((32, 32, 3), 200, dtype=np.uint8)
        c = np.full((32, 32, 3), 150, dtype=np.uint8)
        ref = np.full((32, 32, 3), 150, dtype=np.uint8)

        with patch(
            "landmarkdiff.evaluation.compute_identity_similarity",
            side_effect=[0.3, 0.9, 0.5],
        ):
            best, scores, idx = ei._best_of_n([a, b, c], ref)

        assert idx == 1  # b has highest score
        assert scores[1] == 0.9
        np.testing.assert_array_equal(best, b)

    def test_first_is_best(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference()
        a = np.full((32, 32, 3), 100, dtype=np.uint8)
        b = np.full((32, 32, 3), 200, dtype=np.uint8)
        ref = np.full((32, 32, 3), 100, dtype=np.uint8)

        with patch(
            "landmarkdiff.evaluation.compute_identity_similarity",
            side_effect=[0.95, 0.1],
        ):
            best, scores, idx = ei._best_of_n([a, b], ref)

        assert idx == 0
        np.testing.assert_array_equal(best, a)


# ---------------------------------------------------------------------------
# generate with unknown strategy
# ---------------------------------------------------------------------------


class TestGenerateUnknownStrategy:
    """Test generate raises for unknown strategy."""

    def test_unknown_strategy(self):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference(strategy="nonexistent")
        ei._pipeline = MagicMock()
        ei._pipeline.is_loaded = True
        ei._pipeline.generate.return_value = {"output": np.full((32, 32, 3), 128, dtype=np.uint8)}

        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown strategy"):
            ei.generate(img)


# ---------------------------------------------------------------------------
# generate with mocked pipeline (full path)
# ---------------------------------------------------------------------------


class TestGenerateFullPath:
    """Tests for generate with mocked pipeline."""

    def _make_ensemble(self, strategy="pixel_average"):
        from landmarkdiff.ensemble import EnsembleInference

        ei = EnsembleInference(n_samples=3, strategy=strategy, base_seed=42)
        mock_pipeline = MagicMock()
        mock_pipeline.is_loaded = True

        output = np.full((32, 32, 3), 128, dtype=np.uint8)
        mock_pipeline.generate.return_value = {"output": output, "metadata": {}}
        ei._pipeline = mock_pipeline
        return ei

    def test_pixel_average_strategy(self):
        ei = self._make_ensemble("pixel_average")
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = ei.generate(img)

        assert "output" in result
        assert "outputs" in result
        assert result["strategy"] == "pixel_average"
        assert result["n_samples"] == 3
        assert result["selected_idx"] == -1

    def test_median_strategy(self):
        ei = self._make_ensemble("median")
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = ei.generate(img)

        assert result["strategy"] == "median"
        assert result["selected_idx"] == -1

    def test_weighted_average_strategy(self):
        ei = self._make_ensemble("weighted_average")
        img = np.full((32, 32, 3), 128, dtype=np.uint8)

        with patch("landmarkdiff.evaluation.compute_ssim", return_value=0.8):
            result = ei.generate(img)

        assert result["strategy"] == "weighted_average"
        assert result["selected_idx"] == -1

    def test_best_of_n_strategy(self):
        ei = self._make_ensemble("best_of_n")
        img = np.full((32, 32, 3), 128, dtype=np.uint8)

        with patch(
            "landmarkdiff.evaluation.compute_identity_similarity",
            return_value=0.9,
        ):
            result = ei.generate(img)

        assert result["strategy"] == "best_of_n"
        assert result["selected_idx"] >= 0

    def test_custom_seed(self):
        ei = self._make_ensemble("pixel_average")
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        ei.generate(img, seed=100)

        # Pipeline should have been called with seeds 100, 101, 102
        calls = ei._pipeline.generate.call_args_list
        seeds = [c.kwargs.get("seed") or c[1].get("seed") for c in calls]
        assert seeds == [100, 101, 102]


# ---------------------------------------------------------------------------
# ensemble_inference CLI entry point
# ---------------------------------------------------------------------------


class TestEnsembleInferenceCLI:
    """Tests for ensemble_inference function."""

    def test_invalid_image(self, tmp_path):
        from landmarkdiff.ensemble import ensemble_inference

        with patch("landmarkdiff.ensemble.cv2") as mock_cv2:
            mock_cv2.imread.return_value = None
            ensemble_inference(
                str(tmp_path / "nonexistent.png"),
                output_dir=str(tmp_path / "out"),
            )

        # Should return early without error

    def test_valid_image(self, tmp_path):
        from landmarkdiff.ensemble import ensemble_inference

        output = np.full((512, 512, 3), 128, dtype=np.uint8)
        img = np.full((64, 64, 3), 128, dtype=np.uint8)

        mock_ensemble = MagicMock()
        mock_ensemble.generate.return_value = {
            "output": output,
            "outputs": [output, output],
            "scores": [0.8, 0.7],
            "selected_idx": 0,
        }

        with (
            patch("landmarkdiff.ensemble.cv2") as mock_cv2,
            patch(
                "landmarkdiff.ensemble.EnsembleInference",
                return_value=mock_ensemble,
            ),
        ):
            mock_cv2.imread.return_value = img
            mock_cv2.resize.return_value = np.full((512, 512, 3), 128, dtype=np.uint8)
            ensemble_inference(
                str(tmp_path / "face.png"),
                output_dir=str(tmp_path / "out"),
                n_samples=2,
                strategy="best_of_n",
            )
