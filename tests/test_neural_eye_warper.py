"""Pure geometry tests for the optional ONNX full-eye renderer.

They intentionally do not require an ONNX model or onnxruntime installation.
The runtime-specific session contract is validated by ``NeuralEyeWarper`` when
the optional models are actually loaded.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from gaze_estimator import EyeGaze
from neural_eye_warper import (
    EYE_MODEL_SPECS,
    FEATURE_CHANNELS,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    NeuralEyeWarper,
    _ModelTensors,
)


def make_landmarks(*, mirrored: bool = False, width: int = 240) -> np.ndarray:
    """Make two level, 32-pixel-wide semantic eyes in a 240px frame."""

    points = np.zeros((478, 2), dtype=np.float32)
    # Anatomical R (model R) in canonical image space.
    r = {
        33: (84, 50), 160: (91, 44), 158: (102, 43), 133: (116, 50),
        153: (102, 57), 144: (91, 56),
    }
    # Anatomical L (model L).
    l = {
        263: (144, 50), 387: (151, 44), 385: (162, 43), 362: (176, 50),
        380: (162, 57), 373: (151, 56),
    }
    for index, value in {**r, **l}.items():
        points[index] = value
    if mirrored:
        points[:, 0] = width - 1 - points[:, 0]
    return points


def make_eye(center_x: float, *, reverse_axis: bool = False) -> EyeGaze:
    contour = np.asarray(
        [
            (center_x - 16, 50), (center_x - 11, 44), (center_x - 3, 43),
            (center_x + 8, 44), (center_x + 16, 50), (center_x + 8, 56),
            (center_x - 3, 57), (center_x - 11, 56),
        ],
        dtype=np.float32,
    )
    axis = np.asarray((-1.0, 0.0) if reverse_axis else (1.0, 0.0), dtype=np.float32)
    return EyeGaze(
        contour=contour,
        iris_center=np.asarray((center_x, 50.0), dtype=np.float32),
        ratio=np.zeros(2, dtype=np.float32),
        center=np.asarray((center_x, 50.0), dtype=np.float32),
        horizontal_axis=axis,
        vertical_axis=np.asarray((0.0, 1.0), dtype=np.float32),
        eye_width=32.0,
        eye_height=14.0,
        iris_radius=5.0,
    )


def make_response_curve() -> dict[str, object]:
    """Small measured-style curve fixture for geometry-only renderer tests."""

    # Horizontal output moves right as control angle rises; vertical output
    # moves upward (negative image y) as vertical control rises.  The curve
    # uses the same [angle, model-pixel] shape as the downloaded file.
    return {
        "L": {
            "h": [[-20.0, -2.0], [-10.0, -1.0], [0.0, 0.0], [10.0, 1.0], [20.0, 2.0]],
            "v": [[-12.0, 1.5], [-6.0, 0.8], [0.0, 0.0], [6.0, -0.8], [12.0, -1.5]],
        },
        "R": {
            "h": [[-20.0, -2.0], [-10.0, -1.0], [0.0, 0.0], [10.0, 1.0], [20.0, 2.0]],
            "v": [[-12.0, 1.5], [-6.0, 0.8], [0.0, 0.0], [6.0, -0.8], [12.0, -1.5]],
        },
    }


def make_renderer() -> NeuralEyeWarper:
    """Build a no-ONNX renderer whose response mapping is still real-path."""

    return NeuralEyeWarper(auto_load=False, response_curve_data=make_response_curve())


class NeuralEyeGeometryTests(unittest.TestCase):
    def test_documented_crop_dimensions_and_offsets(self) -> None:
        points = make_landmarks()
        crop = NeuralEyeWarper.crop_for_side(points, "R", 240, 120)
        assert crop is not None
        # 32px corner length: 1.5x wide and 1.125x high.
        self.assertEqual((crop.left, crop.top, crop.right, crop.bottom), (76, 29, 124, 65))
        self.assertEqual((crop.width, crop.height), (48, 36))
        anchors = NeuralEyeWarper.make_anchor_map(points, crop)
        self.assertEqual(anchors.shape, (MODEL_HEIGHT, MODEL_WIDTH, FEATURE_CHANNELS))
        # First R feature is landmark 33: crop-local (8, 21), model-local
        # (10.666..., 28). The published model convention truncates it to
        # (10, 28), then stores grid minus that anchor position.
        self.assertAlmostEqual(float(anchors[0, 0, 0]), -10.0, places=4)
        self.assertAlmostEqual(float(anchors[0, 0, 1]), -28.0, places=4)

    def test_model_angle_is_clamped_and_plan_reports_true_endpoint(self) -> None:
        renderer = make_renderer()
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        eye = make_eye(100)
        plan = renderer.plan(
            frame,
            make_landmarks(),
            (eye,),
            np.asarray(((0.45, 0.0),), dtype=np.float32),
            1.0,
        )[0]
        self.assertTrue(plan.applied)
        assert plan.model_angle is not None
        self.assertLessEqual(abs(float(plan.model_angle[1])), 10.0)
        self.assertLess(float(plan.destination[0]), float(plan.requested_target[0]))
        self.assertGreater(float(plan.destination[0]), float(plan.source[0]))
        self.assertEqual(plan.reason, "limited by model angle")

    def test_mirrored_input_is_canonicalized_for_model_direction(self) -> None:
        renderer = make_renderer()
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        # The semantic R eye is at x=139 after a 240px mirror.  A local
        # reversed axis means ratio -0.10 still asks for a visible rightward
        # movement, which canonicalizes to the correct model horizontal sign.
        eye = make_eye(139, reverse_axis=True)
        plan = renderer.plan(
            frame,
            make_landmarks(mirrored=True),
            (eye,),
            np.asarray(((-0.10, 0.0),), dtype=np.float32),
            1.0,
            mirrored=True,
        )[0]
        self.assertTrue(plan.applied)
        assert plan.model_angle is not None
        # In the mirrored preview the requested movement is visibly to the
        # right; the canonical model receives the opposite image-x sign.
        self.assertGreater(float(plan.destination[0]), float(plan.source[0]))
        self.assertLess(float(plan.model_angle[1]), 0.0)

    def test_aperture_alpha_never_covers_entire_rectangular_crop(self) -> None:
        points = make_landmarks()
        crop = NeuralEyeWarper.crop_for_side(points, "R", 240, 120)
        assert crop is not None
        alpha = NeuralEyeWarper.aperture_alpha(crop, make_eye(100).contour)
        assert alpha is not None
        self.assertEqual(alpha.shape, (crop.height, crop.width))
        self.assertEqual(float(alpha[0, 0]), 0.0)
        # A protected lid band deliberately keeps even the centre a little
        # below opaque, so a full-eye network cannot cover the real lid.
        self.assertGreater(float(alpha[crop.height // 2, crop.width // 2]), 0.85)
        self.assertLess(float(np.mean(alpha > 0.0)), 0.75)

    def test_response_curve_safely_removes_small_measurement_reversals(self) -> None:
        curve = make_response_curve()
        # A tiny upward bounce is normal in a measured response near the
        # endpoint.  It must not make inverse interpolation ambiguous.
        curve["L"]["v"] = [
            [-12.0, 1.5], [-6.0, 0.8], [0.0, 0.0], [6.0, -0.8],
            [9.0, -0.74], [12.0, -1.5],
        ]
        parsed = NeuralEyeWarper._parse_response_curve(curve)
        vertical = parsed["L"].vertical
        self.assertTrue(np.all(np.diff(vertical.angles) > 0.0))
        self.assertTrue(np.all(np.diff(vertical.pixels) < 0.0))
        self.assertLess(vertical.angle_for_pixels(-0.60), 6.0)

    def test_render_guard_rejects_pupil_pulled_into_lower_lid(self) -> None:
        points = make_landmarks()
        crop = NeuralEyeWarper.crop_for_side(points, "R", 240, 120)
        assert crop is not None
        source = np.full((crop.height, crop.width, 3), 180, dtype=np.uint8)
        original_local = (100 - crop.left, 50 - crop.top)
        cv2.circle(source, original_local, 3, (12, 12, 12), -1)
        corrected = source.copy()
        cv2.circle(corrected, original_local, 4, (180, 180, 180), -1)
        # The lower boundary of this synthetic eye is local y=28.  A dark
        # feature placed there is the same visually obvious failure shown in
        # the live preview: it looks as though an eyelid swallowed the pupil.
        cv2.circle(corrected, (original_local[0], 27), 3, (12, 12, 12), -1)
        self.assertFalse(
            NeuralEyeWarper._rendered_pupil_is_safe(
                source,
                corrected,
                crop,
                make_eye(100).contour,
                np.asarray((100.0, 50.0), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.asarray((0.0, 1.0), dtype=np.float32),
                14.0,
            )
        )

    def test_native_zero_flow_uses_export_grid_and_lighting(self) -> None:
        source = np.zeros((18, 24, 3), dtype=np.uint8)
        source[..., 0] = np.arange(24, dtype=np.uint8)[None, :] * 8
        source[..., 1] = 80
        source[..., 2] = 160
        flow = np.zeros((MODEL_HEIGHT, MODEL_WIDTH, 2), dtype=np.float32)
        lcm = np.zeros((MODEL_HEIGHT, MODEL_WIDTH, 2), dtype=np.float32)
        lcm[..., 0] = 1.0
        output = NeuralEyeWarper.apply_flow_and_lighting(source, flow, lcm)
        height, width = source.shape[:2]
        expected = cv2.remap(
            source.astype(np.float32) / 255.0,
            np.tile(
                np.arange(width, dtype=np.float32)[None, :] * MODEL_WIDTH / (MODEL_WIDTH - 1),
                (height, 1),
            ),
            np.tile(
                (np.arange(height, dtype=np.float32)[:, None] * MODEL_HEIGHT / (MODEL_HEIGHT - 1)),
                (1, width),
            ),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        self.assertTrue(np.array_equal(output, np.rint(expected * 255.0).astype(np.uint8)))
        self.assertFalse(np.array_equal(output, source))

    def test_apply_feeds_documented_tensors_and_blends_only_aperture(self) -> None:
        class BlackEyeSession:
            def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
                self.assertEqual(names, ["flow_raw:0", "lcm_map:0"])
                self.assertEqual(feeds["input_img:0"].shape, (1, 48, 64, 3))
                self.assertEqual(feeds["input_fp:0"].shape, (1, 48, 64, 12))
                self.assertEqual(feeds["input_ang:0"].shape, (1, 2))
                flow = np.zeros((1, 48, 64, 2), dtype=np.float32)
                # gain=0, bias=0 makes the neural crop black.  This lets the
                # test prove that aperture compositing, rather than an entire
                # rectangular paste, controls the affected pixels.
                lighting = np.zeros((1, 48, 64, 2), dtype=np.float32)
                return [flow, lighting]

            # ``self`` is deliberately a test helper, not unittest.TestCase.
            assertEqual = staticmethod(unittest.TestCase().assertEqual)

        renderer = make_renderer()
        fake = BlackEyeSession()
        renderer._sessions = {"L": fake, "R": fake}
        renderer._tensor_names = {"L": _ModelTensors(), "R": _ModelTensors()}
        # A saturated source keeps the deliberate sclera-preservation guard
        # out of this synthetic all-black-lighting test.
        frame = np.full((120, 240, 3), (200, 20, 20), dtype=np.uint8)
        eye = make_eye(100)
        points = make_landmarks()
        plans = renderer.plan(
            frame, points, (eye,), np.asarray(((0.10, 0.0),), dtype=np.float32), 1.0
        )
        output = renderer.apply(frame, points, (eye,), plans)
        self.assertTrue(np.array_equal(output[0, 0], frame[0, 0]))
        # Crop top is y=29 but the eyelid mask is zero at that boundary.
        self.assertTrue(np.array_equal(output[29, 100], frame[29, 100]))
        # The new safety policy deliberately preserves camera pixels beneath
        # the model output instead of allowing an opaque whole-eye repaint.
        self.assertLess(float(output[50, 100].mean()), 65.0)
        self.assertGreater(float(output[50, 100].mean()), 5.0)


if __name__ == "__main__":
    unittest.main()
