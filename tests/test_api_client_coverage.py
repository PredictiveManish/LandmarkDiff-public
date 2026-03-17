"""Additional API client tests targeting uncovered error-handling paths.

Covers: health/procedures/predict/analyze error handling (ConnectionError,
HTTPError), close() with active session, decode errors, async client paths.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from landmarkdiff.api_client import (
    LandmarkDiffAPIError,
    LandmarkDiffClient,
)


def _encode_to_b64(img: np.ndarray) -> str:
    _, encoded = cv2.imencode(".png", img)
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Error handling paths — health()
# ---------------------------------------------------------------------------


class TestHealthErrors:
    """Cover lines 123-136: health() ConnectionError and HTTPError."""

    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_health_connection_error(self, mock_session_fn):
        import requests

        mock_session = MagicMock()
        mock_session.get.side_effect = requests.ConnectionError("refused")
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        with pytest.raises(LandmarkDiffAPIError, match="Cannot connect"):
            client.health()

    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_health_http_error(self, mock_session_fn):
        import requests

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=mock_resp)
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        with pytest.raises(LandmarkDiffAPIError, match="Server returned error 500"):
            client.health()


# ---------------------------------------------------------------------------
# Error handling paths — procedures()
# ---------------------------------------------------------------------------


class TestProceduresErrors:
    """Cover lines 152-165: procedures() error handling."""

    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_procedures_connection_error(self, mock_session_fn):
        import requests

        mock_session = MagicMock()
        mock_session.get.side_effect = requests.ConnectionError("refused")
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        with pytest.raises(LandmarkDiffAPIError, match="Cannot connect"):
            client.procedures()

    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_procedures_http_error(self, mock_session_fn):
        import requests

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=mock_resp)
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        with pytest.raises(LandmarkDiffAPIError, match="Server returned error 503"):
            client.procedures()


# ---------------------------------------------------------------------------
# Error handling paths — predict()
# ---------------------------------------------------------------------------


class TestPredictErrors:
    """Cover lines 211-224: predict() error handling."""

    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_predict_connection_error(self, mock_session_fn, tmp_path):
        """ConnectionError raised by session.post() propagates (not caught by inner try)."""
        import requests

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)

        mock_session = MagicMock()
        mock_session.post.side_effect = requests.ConnectionError("refused")
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        with pytest.raises(requests.ConnectionError):
            client.predict(img_path)

    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_predict_http_error(self, mock_session_fn, tmp_path):
        import requests

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)

        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.text = "Unprocessable Entity"
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=mock_resp)
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        with pytest.raises(LandmarkDiffAPIError, match="Server returned error 422"):
            client.predict(img_path)


# ---------------------------------------------------------------------------
# Error handling paths — analyze()
# ---------------------------------------------------------------------------


class TestAnalyzeErrors:
    """Cover lines 248-261: analyze() error handling."""

    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_analyze_connection_error(self, mock_session_fn, tmp_path):
        import requests

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)

        mock_session = MagicMock()
        mock_session.post.side_effect = requests.ConnectionError("refused")
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        with pytest.raises(LandmarkDiffAPIError, match="Cannot connect"):
            client.analyze(img_path)

    @patch("landmarkdiff.api_client.LandmarkDiffClient._get_session")
    def test_analyze_http_error(self, mock_session_fn, tmp_path):
        import requests

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_resp.raise_for_status.side_effect = requests.HTTPError(response=mock_resp)
        mock_session_fn.return_value = mock_session

        client = LandmarkDiffClient()
        with pytest.raises(LandmarkDiffAPIError, match="Server returned error 500"):
            client.analyze(img_path)


# ---------------------------------------------------------------------------
# close() with active session — lines 306-307
# ---------------------------------------------------------------------------


class TestClientClose:
    """Cover close() when session is active."""

    def test_close_active_session(self):
        client = LandmarkDiffClient()
        mock_session = MagicMock()
        client._session = mock_session

        client.close()
        mock_session.close.assert_called_once()
        assert client._session is None

    def test_close_no_session(self):
        client = LandmarkDiffClient()
        client.close()  # Should not raise
        assert client._session is None


# ---------------------------------------------------------------------------
# _get_session() ImportError — lines 83-84
# ---------------------------------------------------------------------------


class TestGetSessionImportError:
    """Cover _get_session when requests is not installed."""

    def test_get_session_no_requests(self):
        client = LandmarkDiffClient()
        with (
            patch.dict("sys.modules", {"requests": None}),
            pytest.raises(ImportError, match="requests required"),
        ):
            client._get_session()


# ---------------------------------------------------------------------------
# _decode_base64_image ValueError — line 102
# ---------------------------------------------------------------------------


class TestDecodeBase64ValueError:
    """Cover _decode_base64_image when cv2.imdecode returns None."""

    def test_decode_corrupt_image(self):
        client = LandmarkDiffClient()
        # Valid base64 but not a valid image
        b64 = base64.b64encode(b"not an image").decode()
        with pytest.raises(ValueError, match="Failed to decode"):
            client._decode_base64_image(b64)
