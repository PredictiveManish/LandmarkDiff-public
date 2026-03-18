"""Tests for landmarkdiff.synthetic.tps_onnx_runtime."""

from __future__ import annotations

import sys
import types

import numpy as np

import landmarkdiff.synthetic.tps_onnx_runtime as rt


class TestHelpers:
    def test_add_edge_anchors_appends_identity_points(self):
        src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dst = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

        src_out, dst_out = rt._add_edge_anchors(src, dst, width=10, height=8)

        assert src_out.shape == (10, 2)
        assert dst_out.shape == (10, 2)
        np.testing.assert_array_equal(src_out[-8:], dst_out[-8:])
        np.testing.assert_array_equal(src_out[-1], np.array([9, 4], dtype=np.float32))

    def test_compute_tps_weights_identity_near_zero(self):
        src = np.array(
            [
                [10.0, 10.0],
                [20.0, 10.0],
                [10.0, 20.0],
                [20.0, 20.0],
            ],
            dtype=np.float32,
        )
        weights_x, weights_y = rt._compute_tps_weights(src, src.copy())

        assert weights_x.shape == (7,)
        assert weights_y.shape == (7,)
        assert np.all(np.isfinite(weights_x))
        assert np.all(np.isfinite(weights_y))
        assert float(np.max(np.abs(weights_x))) < 1e-4
        assert float(np.max(np.abs(weights_y))) < 1e-4

    def test_nchw_hwc_conversion_helpers(self):
        image = np.array(
            [
                [[0, 10, 20], [30, 40, 50]],
                [[60, 70, 80], [90, 100, 110]],
            ],
            dtype=np.uint8,
        )
        nchw = rt._to_nchw_float32(image)
        recovered = rt._to_hwc_uint8(nchw)

        assert nchw.shape == (1, 3, 2, 2)
        assert nchw.dtype == np.float32
        assert recovered.shape == image.shape
        assert recovered.dtype == np.uint8
        np.testing.assert_array_equal(recovered, image)


class TestTPSONNXRuntime:
    def test_init_uses_cpu_execution_provider(self):
        calls: dict[str, object] = {}

        class FakeSession:
            def __init__(self, path, providers):
                calls["path"] = path
                calls["providers"] = providers

            def run(self, *_args, **_kwargs):
                return [np.zeros((1, 3, 2, 2), dtype=np.float32)]

        fake_ort = types.SimpleNamespace(InferenceSession=FakeSession)
        original = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = fake_ort
        try:
            runtime = rt.TPSONNXRuntime("model.onnx")
        finally:
            if original is None:
                del sys.modules["onnxruntime"]
            else:
                sys.modules["onnxruntime"] = original

        assert runtime.onnx_path == "model.onnx"
        assert calls["path"] == "model.onnx"
        assert calls["providers"] == ["CPUExecutionProvider"]

    def test_warp_builds_onnx_inputs_and_returns_image(self, monkeypatch):
        calls: dict[str, object] = {}

        class FakeSession:
            def __init__(self, path, providers):
                calls["init_path"] = path
                calls["init_providers"] = providers

            def run(self, output_names, onnx_input):
                calls["output_names"] = output_names
                calls["onnx_input"] = onnx_input
                return [onnx_input["image"]]

        fake_ort = types.SimpleNamespace(InferenceSession=FakeSession)
        original = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = fake_ort
        try:
            runtime = rt.TPSONNXRuntime("mock.onnx")
        finally:
            if original is None:
                del sys.modules["onnxruntime"]
            else:
                sys.modules["onnxruntime"] = original

        monkeypatch.setattr(
            rt,
            "_subsample_control_points",
            lambda src, dst: (src.copy(), dst.copy()),
        )
        monkeypatch.setattr(rt, "_add_edge_anchors", lambda src, dst, width, height: (src, dst))
        monkeypatch.setattr(
            rt,
            "_compute_tps_weights",
            lambda src, dst: (
                np.zeros((len(src) + 3,), dtype=np.float32),
                np.zeros((len(src) + 3,), dtype=np.float32),
            ),
        )

        image = np.full((4, 5, 3), 128, dtype=np.uint8)
        src = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
        dst = src + 0.5

        output = runtime.warp(image, src, dst)

        assert output.shape == image.shape
        assert output.dtype == np.uint8
        assert calls["output_names"] == ["warped"]

        onnx_input = calls["onnx_input"]
        assert onnx_input["image"].shape == (1, 3, 4, 5)
        assert onnx_input["control_points"].shape == (1, 3, 2)
        assert onnx_input["weights_x"].shape == (1, 6)
        assert onnx_input["weights_y"].shape == (1, 6)
