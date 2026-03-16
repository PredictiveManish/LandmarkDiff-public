"""Extended tests for __main__.py -- mock-based success paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _run_inference TPS mode
# ---------------------------------------------------------------------------


class TestRunInferenceTPS:
    """Tests for _run_inference with mode=tps."""

    def test_tps_success(self, tmp_path):
        from landmarkdiff.__main__ import _run_inference

        img_path = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        mock_lm = MagicMock()
        mock_lm.pixel_coords = np.random.rand(478, 3).astype(np.float32)
        mock_lm.image_width = 512
        mock_lm.image_height = 512

        mock_deformed = MagicMock()
        mock_deformed.pixel_coords = np.random.rand(478, 3).astype(np.float32)
        mock_deformed.image_width = 512
        mock_deformed.image_height = 512

        warped = np.full((512, 512, 3), 200, dtype=np.uint8)

        args = MagicMock()
        args.image = str(img_path)
        args.procedure = "rhinoplasty"
        args.intensity = 60.0
        args.mode = "tps"
        args.output = str(tmp_path / "output")
        args.steps = 30
        args.seed = None

        with (
            patch("landmarkdiff.landmarks.extract_landmarks", return_value=mock_lm),
            patch(
                "landmarkdiff.manipulation.apply_procedure_preset",
                return_value=mock_deformed,
            ),
            patch(
                "landmarkdiff.synthetic.tps_warp.warp_image_tps",
                return_value=warped,
            ),
        ):
            _run_inference(args)

        assert (Path(args.output) / "prediction.png").exists()

    def test_no_face_detected(self, tmp_path):
        from landmarkdiff.__main__ import _run_inference

        img_path = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        args = MagicMock()
        args.image = str(img_path)
        args.procedure = "rhinoplasty"
        args.intensity = 60.0
        args.mode = "tps"
        args.output = str(tmp_path / "output")

        with (
            patch("landmarkdiff.landmarks.extract_landmarks", return_value=None),
            pytest.raises(SystemExit),
        ):
            _run_inference(args)

    def test_bad_intensity_low(self, tmp_path):
        from landmarkdiff.__main__ import _run_inference

        img_path = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        args = MagicMock()
        args.image = str(img_path)
        args.intensity = -5.0
        args.mode = "tps"
        args.output = str(tmp_path / "output")

        with pytest.raises(SystemExit):
            _run_inference(args)


# ---------------------------------------------------------------------------
# _run_inference non-TPS mode
# ---------------------------------------------------------------------------


class TestRunInferenceNonTPS:
    """Tests for _run_inference with non-TPS modes (mocked pipeline)."""

    def test_controlnet_mode(self, tmp_path):
        from landmarkdiff.__main__ import _run_inference

        img_path = tmp_path / "face.png"
        from PIL import Image as PILImage

        PILImage.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        mock_lm = MagicMock()
        mock_deformed = MagicMock()
        mock_pipeline = MagicMock()
        mock_result = {"output": PILImage.new("RGB", (512, 512))}
        mock_pipeline.generate.return_value = mock_result

        args = MagicMock()
        args.image = str(img_path)
        args.procedure = "rhinoplasty"
        args.intensity = 60.0
        args.mode = "controlnet"
        args.output = str(tmp_path / "output")
        args.steps = 30
        args.seed = 42

        mock_torch = MagicMock()
        mock_torch.device.return_value = MagicMock()

        with (
            patch("landmarkdiff.landmarks.extract_landmarks", return_value=mock_lm),
            patch(
                "landmarkdiff.manipulation.apply_procedure_preset",
                return_value=mock_deformed,
            ),
            patch.dict("sys.modules", {"torch": mock_torch}),
            patch(
                "landmarkdiff.inference.LandmarkDiffPipeline",
                return_value=mock_pipeline,
            ),
        ):
            _run_inference(args)

        mock_pipeline.load.assert_called_once()
        mock_pipeline.generate.assert_called_once()
        assert (Path(args.output) / "prediction.png").exists()


# ---------------------------------------------------------------------------
# _run_landmarks
# ---------------------------------------------------------------------------


class TestRunLandmarks:
    """Tests for _run_landmarks success path."""

    def test_landmarks_success(self, tmp_path):
        from landmarkdiff.__main__ import _run_landmarks

        img_path = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        mock_lm = MagicMock()
        mock_lm.landmarks = np.zeros((478, 3))
        mock_lm.confidence = 0.95
        mesh = np.full((512, 512, 3), 100, dtype=np.uint8)

        out_path = tmp_path / "output" / "landmarks.png"

        args = MagicMock()
        args.image = str(img_path)
        args.output = str(out_path)

        with (
            patch("landmarkdiff.landmarks.extract_landmarks", return_value=mock_lm),
            patch("landmarkdiff.landmarks.render_landmark_image", return_value=mesh),
        ):
            _run_landmarks(args)

        assert out_path.exists()

    def test_landmarks_no_face(self, tmp_path):
        from landmarkdiff.__main__ import _run_landmarks

        img_path = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        args = MagicMock()
        args.image = str(img_path)
        args.output = str(tmp_path / "lm.png")

        with (
            patch("landmarkdiff.landmarks.extract_landmarks", return_value=None),
            pytest.raises(SystemExit),
        ):
            _run_landmarks(args)


# ---------------------------------------------------------------------------
# _run_demo
# ---------------------------------------------------------------------------


class TestRunDemo:
    """Tests for _run_demo."""

    def test_demo_import_error(self):
        from landmarkdiff.__main__ import _run_demo

        with (
            patch.dict("sys.modules", {"scripts": None, "scripts.app": None}),
            pytest.raises(SystemExit),
        ):
            _run_demo()

    def test_demo_success(self):
        from landmarkdiff.__main__ import _run_demo

        mock_app = MagicMock()
        mock_module = MagicMock()
        mock_module.build_app.return_value = mock_app

        with patch.dict("sys.modules", {"scripts": MagicMock(), "scripts.app": mock_module}):
            _run_demo()

        mock_module.build_app.assert_called_once()
        mock_app.launch.assert_called_once()


# ---------------------------------------------------------------------------
# main() exception handling
# ---------------------------------------------------------------------------


class TestMainExceptionHandling:
    """Tests for main() exception paths."""

    def test_keyboard_interrupt(self, tmp_path):
        from landmarkdiff.__main__ import main

        img_path = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        with (
            patch(
                "sys.argv",
                ["landmarkdiff", "infer", str(img_path), "--mode", "tps"],
            ),
            patch(
                "landmarkdiff.__main__._run_inference",
                side_effect=KeyboardInterrupt,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 130

    def test_generic_exception(self, tmp_path):
        from landmarkdiff.__main__ import main

        img_path = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        with (
            patch(
                "sys.argv",
                ["landmarkdiff", "infer", str(img_path), "--mode", "tps"],
            ),
            patch(
                "landmarkdiff.__main__._run_inference",
                side_effect=RuntimeError("test error"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    def test_infer_dispatched(self, tmp_path):
        """Verify that 'infer' command dispatches to _run_inference."""
        from landmarkdiff.__main__ import main

        img_path = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        with (
            patch(
                "sys.argv",
                ["landmarkdiff", "infer", str(img_path), "--mode", "tps"],
            ),
            patch("landmarkdiff.__main__._run_inference") as mock_infer,
        ):
            main()

        mock_infer.assert_called_once()

    def test_landmarks_dispatched(self, tmp_path):
        """Verify that 'landmarks' command dispatches to _run_landmarks."""
        from landmarkdiff.__main__ import main

        img_path = tmp_path / "face.png"
        from PIL import Image

        Image.new("RGB", (64, 64), (128, 128, 128)).save(str(img_path))

        with (
            patch(
                "sys.argv",
                ["landmarkdiff", "landmarks", str(img_path)],
            ),
            patch("landmarkdiff.__main__._run_landmarks") as mock_lm,
        ):
            main()

        mock_lm.assert_called_once()

    def test_demo_dispatched(self):
        """Verify that 'demo' command dispatches to _run_demo."""
        from landmarkdiff.__main__ import main

        with (
            patch("sys.argv", ["landmarkdiff", "demo"]),
            patch("landmarkdiff.__main__._run_demo") as mock_demo,
        ):
            main()

        mock_demo.assert_called_once()
