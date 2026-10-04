from __future__ import annotations

import unittest

import numpy as np

from gaze_estimator import EyeLandmarks, GazeEstimator


def make_eye(x: float, iris_offset: tuple[float, float] = (0.0, 0.0)) -> EyeLandmarks:
    """Create a horizontally aligned synthetic eye ring in pixel coordinates."""

    contour = np.asarray(
        [
            (x + 0, 20),  # corner A
            (x + 3, 17),
            (x + 6, 16),
            (x + 10, 15),
            (x + 14, 15),
            (x + 18, 16),
            (x + 22, 17),
            (x + 26, 18),
            (x + 30, 20),  # corner B
            (x + 26, 23),
            (x + 22, 24),
            (x + 18, 25),
            (x + 14, 25),
            (x + 10, 25),
            (x + 6, 24),
            (x + 3, 23),
        ],
        dtype=np.float32,
    )
    iris_center = np.asarray((x + 15, 20), dtype=np.float32) + iris_offset
    iris = np.asarray(
        [
            iris_center,
            iris_center + (1.0, 0.0),
            iris_center + (0.0, 1.0),
            iris_center + (-1.0, 0.0),
            iris_center + (0.0, -1.0),
        ],
        dtype=np.float32,
    )
    return EyeLandmarks(contour=contour, iris=iris)


class GazeEstimatorTests(unittest.TestCase):
    def test_estimates_two_eyes_in_preview_order(self) -> None:
        estimate = GazeEstimator().estimate(
            (make_eye(50, (3.0, 1.5)), make_eye(5, (-3.0, -1.5)))
        )
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertLess(estimate.eyes[0].iris_center[0], estimate.eyes[1].iris_center[0])
        self.assertAlmostEqual(float(estimate.eyes[0].ratio[0]), -0.07, places=2)
        self.assertAlmostEqual(float(estimate.eyes[0].ratio[1]), -0.06, places=2)

    def test_local_target_round_trip(self) -> None:
        estimate = GazeEstimator().estimate((make_eye(5), make_eye(50)))
        self.assertIsNotNone(estimate)
        assert estimate is not None
        eye = estimate.eyes[0]
        np.testing.assert_allclose(eye.point_for_ratio(eye.ratio), eye.iris_center, atol=0.01)


if __name__ == "__main__":
    unittest.main()
