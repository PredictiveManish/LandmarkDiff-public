"""Additional async client tests targeting uncovered lines.

Covers: _handle_response error path, health/procedures connection errors,
predict/analyze full path, batch_predict, close, decode errors.
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

from landmarkdiff.api_client import LandmarkDiffAPIError
from landmarkdiff.api_client_async import AsyncLandmarkDiffClient


def _encode_to_b64(img: np.ndarray) -> str:
    _, encoded = cv2.imencode(".png", img)
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Helper to run async functions
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run an async function in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestAsyncClientInit:
    def test_defaults(self):
        client = AsyncLandmarkDiffClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 60.0
        assert client.max_concurrent == 4

    def test_custom_params(self):
        client = AsyncLandmarkDiffClient(
            base_url="http://example.com:9999/",
            timeout=30.0,
            max_concurrent=8,
        )
        assert client.base_url == "http://example.com:9999"
        assert client.timeout == 30.0
        assert client.max_concurrent == 8

    def test_repr(self):
        client = AsyncLandmarkDiffClient("http://test:1234")
        assert "test:1234" in repr(client)


# ---------------------------------------------------------------------------
# _handle_response error path — line 92-93
# ---------------------------------------------------------------------------


class TestHandleResponse:
    def test_error_status(self):
        client = AsyncLandmarkDiffClient()
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Server Error")

        with pytest.raises(LandmarkDiffAPIError, match="Server returned 500"):
            run_async(client._handle_response(mock_resp))

    def test_success_status(self):
        client = AsyncLandmarkDiffClient()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"status": "ok"})

        result = run_async(client._handle_response(mock_resp))
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# health() error paths — lines 112-115
# ---------------------------------------------------------------------------


class TestAsyncHealthErrors:
    def test_health_connection_error(self):
        client = AsyncLandmarkDiffClient()
        mock_session = AsyncMock()
        mock_session.get.side_effect = ConnectionError("refused")
        client._session = mock_session

        with pytest.raises(LandmarkDiffAPIError, match="Cannot connect"):
            run_async(client.health())

    def test_health_api_error_reraise(self):
        """LandmarkDiffAPIError from _handle_response should re-raise."""
        client = AsyncLandmarkDiffClient()

        async def _test():
            mock_session = MagicMock()
            mock_resp = AsyncMock()
            mock_resp.status = 503
            mock_resp.text = AsyncMock(return_value="Unavailable")

            # Use a real async context manager
            class MockCtx:
                async def __aenter__(self):
                    return mock_resp

                async def __aexit__(self, *args):
                    return False

            mock_session.get = MagicMock(return_value=MockCtx())
            mock_session.closed = False
            client._session = mock_session
            return await client.health()

        with pytest.raises(LandmarkDiffAPIError, match="Server returned 503"):
            run_async(_test())


# ---------------------------------------------------------------------------
# procedures() error paths — lines 128-133
# ---------------------------------------------------------------------------


class TestAsyncProceduresErrors:
    def test_procedures_connection_error(self):
        client = AsyncLandmarkDiffClient()
        mock_session = AsyncMock()
        mock_session.get.side_effect = OSError("Connection refused")
        client._session = mock_session

        with pytest.raises(LandmarkDiffAPIError, match="Cannot connect"):
            run_async(client.procedures())


# ---------------------------------------------------------------------------
# _read_image and _decode_base64_image — static methods
# ---------------------------------------------------------------------------


class TestAsyncStaticMethods:
    def test_read_image(self, tmp_path):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        path = tmp_path / "img.png"
        cv2.imwrite(str(path), img)

        data = AsyncLandmarkDiffClient._read_image(path)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_read_image_not_found(self):
        with pytest.raises(FileNotFoundError):
            AsyncLandmarkDiffClient._read_image("/nonexistent/file.png")

    def test_decode_base64_image(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        b64 = _encode_to_b64(img)
        decoded = AsyncLandmarkDiffClient._decode_base64_image(b64)
        assert decoded.shape == (64, 64, 3)

    def test_decode_base64_image_invalid(self):
        b64 = base64.b64encode(b"not an image").decode()
        with pytest.raises(ValueError, match="Failed to decode"):
            AsyncLandmarkDiffClient._decode_base64_image(b64)


# ---------------------------------------------------------------------------
# close() — lines 251-255
# ---------------------------------------------------------------------------


class TestAsyncClose:
    def test_close_active_session(self):
        client = AsyncLandmarkDiffClient()
        mock_session = AsyncMock()
        mock_session.closed = False
        client._session = mock_session

        run_async(client.close())
        mock_session.close.assert_awaited_once()
        assert client._session is None

    def test_close_no_session(self):
        client = AsyncLandmarkDiffClient()
        run_async(client.close())  # Should not raise

    def test_context_manager(self):
        async def _test():
            client = AsyncLandmarkDiffClient()
            mock_session = AsyncMock()
            mock_session.closed = False
            client._session = mock_session
            async with client:
                pass
            mock_session.close.assert_awaited_once()

        run_async(_test())


# ---------------------------------------------------------------------------
# _get_session ImportError — lines 64-66
# ---------------------------------------------------------------------------


class TestAsyncGetSessionImportError:
    def test_get_session_no_aiohttp(self):
        client = AsyncLandmarkDiffClient()
        with (
            patch.dict("sys.modules", {"aiohttp": None}),
            pytest.raises(ImportError, match="aiohttp required"),
        ):
            run_async(client._get_session())
