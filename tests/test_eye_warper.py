from __future__ import annotations

import unittest

import cv2
import numpy as np

from eye_warper import EyeWarper
from gaze_estimator import EyeGaze


def make_warp_eye() -> EyeGaze:
    contour = np.asarray(
        [
            (20, 40),
            (24, 34),
            (31, 31),
            (40, 30),
            (49, 31),
            (56, 34),
            (60, 40),
            (56, 46),
            (49, 49),
            (40, 50),
            (31, 49),
            (24, 46),
            (20, 40),
            (24, 34),
            (31, 31),
            (40, 30),
        ],
        dtype=np.float32,
    )
    return EyeGaze(
        contour=contour,
        iris_center=np.asarray((40.0, 40.0), dtype=np.float32),
        ratio=np.asarray((0.0, 0.0), dtype=np.float32),
        center=np.asarray((40.0, 40.0), dtype=np.float32),
        horizontal_axis=np.asarray((1.0, 0.0), dtype=np.float32),
        vertical_axis=np.asarray((0.0, 1.0), dtype=np.float32),
        eye_width=40.0,
        eye_height=20.0,
        iris_radius=4.0,
    )


class EyeWarperTests(unittest.TestCase):
    def test_moves_real_dark_texture_toward_target(self) -> None:
        eye = make_warp_eye()
        frame = np.full((90, 90, 3), 220, dtype=np.uint8)
        frame[38:43, 38:43] = 10
        output = EyeWarper().correct(
            frame,
            (eye,),
            np.asarray(((0.10, 0.0),), dtype=np.float32),
            strength=1.0,
        )
        self.assertLess(float(output[40, 44].mean()), float(frame[40, 44].mean()))
        self.assertTrue(np.array_equal(output[5, 5], frame[5, 5]))
        self.assertTrue(np.array_equal(output[30, 40], frame[30, 40]))

    def test_zero_strength_is_identity(self) -> None:
        eye = make_warp_eye()
        frame = np.full((90, 90, 3), 220, dtype=np.uint8)
        output = EyeWarper().correct(
            frame,
            (eye,),
            np.asarray(((0.10, 0.0),), dtype=np.float32),
            strength=0.0,
        )
        self.assertTrue(np.array_equal(output, frame))

    def test_strength_is_applied_before_eyelid_safety_check(self) -> None:
        eye = make_warp_eye()
        # At 100% this target is near the upper lid. A gentle 25% request is
        # still safe and should not be discarded because of the hypothetical
        # full-strength endpoint.
        plan = EyeWarper().plan(
            (eye,),
            np.asarray(((0.0, -0.25),), dtype=np.float32),
            strength=0.25,
        )
        self.assertTrue(plan[0].applied)
        self.assertLess(float(plan[0].destination[1]), float(eye.iris_center[1]))

    def test_large_motion_is_clamped_at_eyelid_boundary(self) -> None:
        eye = make_warp_eye()
        warper = EyeWarper(max_vertical_fraction=0.40)
        plan = warper.plan(
            (eye,),
            np.asarray(((0.0, -0.30),), dtype=np.float32),
            strength=1.0,
        )[0]
        self.assertTrue(plan.applied)
        self.assertEqual(plan.reason, "limited by eyelid")
        self.assertGreater(float(plan.destination[1]), float(plan.requested_target[1]))
        polygon = eye.contour.astype(np.float32).reshape((-1, 1, 2))
        distance = cv2.pointPolygonTest(
            polygon, tuple(float(value) for value in plan.destination), True
        )
        self.assertGreater(distance, 0.0)


if __name__ == "__main__":
    unittest.main()
