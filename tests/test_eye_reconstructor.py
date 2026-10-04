from __future__ import annotations

import unittest

import cv2
import numpy as np

from eye_reconstructor import IrisReconstructor
from gaze_estimator import EyeGaze


def make_eye() -> EyeGaze:
    """A visibly open synthetic eye with an iris in its centre."""

    contour = np.asarray(
        [
            (20, 50),
            (25, 44),
            (33, 40),
            (42, 38),
            (50, 38),
            (58, 38),
            (67, 40),
            (75, 44),
            (80, 50),
            (75, 56),
            (67, 60),
            (58, 62),
            (50, 62),
            (42, 62),
            (33, 60),
            (25, 56),
        ],
        dtype=np.float32,
    )
    return EyeGaze(
        contour=contour,
        iris_center=np.asarray((50.0, 50.0), dtype=np.float32),
        ratio=np.asarray((0.0, 0.0), dtype=np.float32),
        center=np.asarray((50.0, 50.0), dtype=np.float32),
        horizontal_axis=np.asarray((1.0, 0.0), dtype=np.float32),
        vertical_axis=np.asarray((0.0, 1.0), dtype=np.float32),
        eye_width=60.0,
        eye_height=24.0,
        iris_radius=6.0,
    )


def make_frame() -> np.ndarray:
    frame = np.full((100, 110, 3), 120, dtype=np.uint8)
    cv2.ellipse(frame, (50, 50), (30, 12), 0, 0, 360, (224, 224, 224), -1)
    cv2.circle(frame, (50, 50), 6, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(frame, (50, 50), 2, (2, 2, 2), -1, cv2.LINE_AA)
    return frame


class IrisReconstructorTests(unittest.TestCase):
    def test_erases_old_pupil_then_places_one_at_destination(self) -> None:
        eye = make_eye()
        frame = make_frame()
        output = IrisReconstructor().correct(
            frame,
            (eye,),
            np.asarray(((0.18, 0.0),), dtype=np.float32),
            strength=1.0,
        )

        # The old centre is inpainted toward surrounding sclera, while the
        # corrected destination receives the original dark iris texture.
        self.assertGreater(float(output[50, 50].mean()), 80.0)
        self.assertLess(float(output[50, 61].mean()), 90.0)
        self.assertTrue(np.array_equal(output[5, 5], frame[5, 5]))
        self.assertTrue(np.array_equal(frame, make_frame()))

    def test_zero_strength_is_identity(self) -> None:
        eye = make_eye()
        frame = make_frame()
        output = IrisReconstructor().correct(
            frame,
            (eye,),
            np.asarray(((0.18, 0.0),), dtype=np.float32),
            strength=0.0,
        )
        self.assertTrue(np.array_equal(output, frame))

    def test_large_move_is_clipped_by_visible_eyelid_region(self) -> None:
        eye = make_eye()
        plan = IrisReconstructor(max_horizontal_fraction=0.50).plan(
            (eye,),
            np.asarray(((0.60, 0.0),), dtype=np.float32),
            strength=1.0,
        )[0]
        self.assertTrue(plan.applied)
        self.assertEqual(plan.reason, "limited by eyelid")
        self.assertLess(float(plan.destination[0]), float(plan.requested_target[0]))
        self.assertGreater(float(plan.destination[0]), float(plan.source[0]))

    def test_invalid_target_returns_non_drawing_plan(self) -> None:
        eye = make_eye()
        plans = IrisReconstructor().plan(
            (eye,), np.asarray(((np.nan, 0.0),), dtype=np.float32), strength=1.0
        )
        self.assertEqual(len(plans), 1)
        self.assertFalse(plans[0].applied)
        self.assertEqual(plans[0].reason, "invalid target")


if __name__ == "__main__":
    unittest.main()
