from __future__ import annotations

import unittest

import numpy as np

from calibration import GazeCalibration
from gaze_estimator import EyeGaze, GazeEstimate, GazeEstimator
from test_gaze_estimator import make_eye


class CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        estimate = GazeEstimator().estimate((make_eye(5), make_eye(50)))
        lens_estimate = GazeEstimator().estimate(
            (make_eye(5, (1.2, -1.8)), make_eye(50, (1.2, -1.8)))
        )
        assert estimate is not None and lens_estimate is not None
        self.estimate = estimate
        self.lens_estimate = lens_estimate

    def _complete_two_step_calibration(self, calibration: GazeCalibration) -> None:
        calibration.start(now=10.0)
        # First usable frame establishes stability; the next three are samples.
        for now in (12.0, 12.1, 12.2, 12.3):
            screen_update = calibration.update(self.estimate, now=now)
        self.assertEqual(screen_update.phase, "lens_countdown")
        self.assertTrue(calibration.active)
        for now in (14.3, 14.4, 14.5, 14.6):
            lens_update = calibration.update(self.lens_estimate, now=now)
        self.assertTrue(lens_update.completed_now)

    def test_collects_distinct_screen_and_lens_references(self) -> None:
        calibration = GazeCalibration(countdown_seconds=2.0, sample_count=3)
        calibration.start(now=10.0)
        countdown = calibration.update(self.estimate, now=10.5)
        self.assertEqual(countdown.phase, "screen_countdown")
        self.assertEqual(countdown.stage, "screen")
        self.assertEqual(countdown.countdown_remaining, 2)

        self._complete_two_step_calibration(calibration)
        self.assertTrue(calibration.calibrated)
        np.testing.assert_allclose(calibration.screen_baseline, self.estimate.ratios, atol=0.001)
        np.testing.assert_allclose(calibration.camera_target, self.lens_estimate.ratios, atol=0.001)

    def test_same_lens_and_screen_sample_retries_lens_stage_once(self) -> None:
        calibration = GazeCalibration(countdown_seconds=0.0, sample_count=2)
        calibration.start(now=0.0)
        # Screen stage: first stable candidate then two samples.
        for now in (0.0, 0.1, 0.2):
            screen_update = calibration.update(self.estimate, now=now)
        self.assertEqual(screen_update.phase, "lens_countdown")
        # Lens stage uses the same gaze by mistake and gets an explicit retry.
        for now in (0.3, 0.4, 0.5):
            retry_update = calibration.update(self.estimate, now=now)
        self.assertEqual(retry_update.phase, "lens_retry")
        self.assertTrue(retry_update.retrying_lens)
        self.assertFalse(calibration.calibrated)

    def test_measured_lens_target_and_screen_nudges_are_applied(self) -> None:
        calibration = GazeCalibration(countdown_seconds=2.0, sample_count=3)
        self._complete_two_step_calibration(calibration)
        targets = calibration.targets_for(
            self.lens_estimate,
            camera_lift=-0.08,
            horizontal_nudge=0.02,
            vertical_nudge=0.01,
        )
        assert targets is not None
        for eye, target, lens_ratio in zip(
            self.lens_estimate.eyes, targets, self.lens_estimate.ratios
        ):
            observed_screen_delta = eye.point_for_ratio(target) - eye.point_for_ratio(
                lens_ratio
            )
            np.testing.assert_allclose(
                observed_screen_delta,
                [eye.width * 0.02, eye.width * -0.07],
                atol=0.001,
            )

    def test_screen_space_nudge_moves_opposite_local_axes_together(self) -> None:
        contour = np.asarray(
            [(0, 0), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1), (6, -1), (7, -1),
             (8, 0), (7, 1), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1)],
            dtype=np.float32,
        )
        left_to_right = EyeGaze(
            contour=contour,
            iris_center=np.asarray((20.0, 20.0), dtype=np.float32),
            ratio=np.zeros(2, dtype=np.float32),
            center=np.asarray((20.0, 20.0), dtype=np.float32),
            horizontal_axis=np.asarray((1.0, 0.0), dtype=np.float32),
            vertical_axis=np.asarray((0.0, 1.0), dtype=np.float32),
            eye_width=40.0,
            eye_height=12.0,
            iris_radius=3.0,
        )
        right_to_left = EyeGaze(
            contour=contour + np.asarray((80.0, 0.0), dtype=np.float32),
            iris_center=np.asarray((100.0, 20.0), dtype=np.float32),
            ratio=np.zeros(2, dtype=np.float32),
            center=np.asarray((100.0, 20.0), dtype=np.float32),
            horizontal_axis=np.asarray((-1.0, 0.0), dtype=np.float32),
            vertical_axis=np.asarray((0.0, 1.0), dtype=np.float32),
            eye_width=40.0,
            eye_height=12.0,
            iris_radius=3.0,
        )
        estimate = GazeEstimate((left_to_right, right_to_left))
        calibration = GazeCalibration(countdown_seconds=0.0, sample_count=1)
        calibration.camera_target = np.zeros((2, 2), dtype=np.float32)
        targets = calibration.targets_for(
            estimate,
            horizontal_nudge=0.03,
            vertical_nudge=-0.02,
        )
        assert targets is not None
        for eye, target in zip(estimate.eyes, targets):
            np.testing.assert_allclose(
                eye.point_for_ratio(target) - eye.center,
                [eye.width * 0.03, -eye.width * 0.02],
                atol=0.001,
            )

    def test_center_test_has_a_geometric_zero_target(self) -> None:
        calibration = GazeCalibration(countdown_seconds=2.0, sample_count=3)
        self._complete_two_step_calibration(calibration)
        targets = calibration.targets_for(
            self.lens_estimate,
            use_eye_center=True,
            horizontal_nudge=0.02,
            vertical_nudge=-0.01,
        )
        assert targets is not None
        for eye, target in zip(self.lens_estimate.eyes, targets):
            np.testing.assert_allclose(
                eye.point_for_ratio(target) - eye.center,
                [eye.width * 0.02, -eye.width * 0.01],
                atol=0.001,
            )

    def test_has_no_target_before_lens_stage_completes(self) -> None:
        calibration = GazeCalibration(countdown_seconds=0.0, sample_count=1)
        calibration.start(now=0.0)
        calibration.update(self.estimate, now=0.0)
        calibration.update(self.estimate, now=0.1)
        self.assertIsNone(calibration.targets_for(self.estimate))


if __name__ == "__main__":
    unittest.main()
