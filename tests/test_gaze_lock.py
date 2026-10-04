from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from gaze_estimator import GazeEstimate, GazeEstimator
from gaze_lock import DynamicGazeLockCalibration, HeadPose
try:  # Works both under ``unittest discover`` and ``python -m unittest``.
    from .test_gaze_estimator import make_eye
except ImportError:  # pragma: no cover - discovery loads tests as top-level modules.
    from test_gaze_estimator import make_eye


def estimate_for_target(target: np.ndarray) -> GazeEstimate:
    """Synthetic pair whose gaze ratios match the supplied (2, 2) target."""

    estimate = GazeEstimator().estimate((make_eye(5), make_eye(50)))
    assert estimate is not None
    eyes = tuple(
        replace(
            eye,
            ratio=target[index].astype(np.float32),
            iris_center=eye.point_for_ratio(target[index]),
        )
        for index, eye in enumerate(estimate.eyes)
    )
    return GazeEstimate(eyes)


def target_for_pose(pose: np.ndarray) -> np.ndarray:
    """A known affine user-specific lens-target relation for tests."""

    yaw, pitch = pose
    return np.asarray(
        (
            (0.015 + 0.110 * yaw - 0.045 * pitch, -0.030 + 0.035 * yaw + 0.085 * pitch),
            (-0.010 + 0.080 * yaw - 0.025 * pitch, -0.020 + 0.050 * yaw + 0.065 * pitch),
        ),
        dtype=np.float32,
    )


class DynamicGazeLockCalibrationTests(unittest.TestCase):
    _stage_poses = (
        np.asarray((0.00, 0.00), dtype=np.float32),
        np.asarray((-0.18, 0.01), dtype=np.float32),
        np.asarray((0.17, -0.01), dtype=np.float32),
        np.asarray((0.00, -0.14), dtype=np.float32),
        np.asarray((0.01, 0.16), dtype=np.float32),
    )

    def _complete(self, calibration: DynamicGazeLockCalibration) -> None:
        calibration.start(now=0.0)
        now = 0.0
        for pose in self._stage_poses:
            gaze = estimate_for_target(target_for_pose(pose))
            # First frame starts the stability reference; the next frames are
            # retained samples.  The fifth retained frame completes a stage.
            calibration.update(gaze, HeadPose(*pose), now=now)
            for _ in range(calibration.sample_count):
                now += 0.01
                update = calibration.update(gaze, pose, now=now)
            now += 0.01
        self.assertTrue(update.completed_now)

    def test_fits_pose_aware_two_eye_target(self) -> None:
        calibration = DynamicGazeLockCalibration(countdown_seconds=0.0, sample_count=4)
        self._complete(calibration)

        self.assertTrue(calibration.calibrated)
        self.assertTrue(calibration.coverage_ok)
        self.assertEqual(len(calibration.anchors), 5)
        wanted_pose = np.asarray((0.08, -0.06), dtype=np.float32)
        predicted = calibration.predict_target(wanted_pose)
        assert predicted is not None
        np.testing.assert_allclose(predicted, target_for_pose(wanted_pose), atol=0.003)

    def test_out_of_range_pose_and_target_are_safely_clamped(self) -> None:
        calibration = DynamicGazeLockCalibration(
            countdown_seconds=0.0,
            sample_count=3,
            pose_padding=0.01,
            target_padding=0.02,
        )
        self._complete(calibration)
        predicted = calibration.predict_target((100.0, -100.0))
        limits = calibration.target_limits
        assert predicted is not None and limits is not None
        self.assertTrue(np.all(predicted >= limits[0] - 1e-6))
        self.assertTrue(np.all(predicted <= limits[1] + 1e-6))
        self.assertTrue(np.all(np.abs(predicted) <= calibration.max_target))

    def test_low_pose_coverage_is_reported_but_does_not_crash(self) -> None:
        calibration = DynamicGazeLockCalibration(countdown_seconds=0.0, sample_count=2)
        calibration.start(now=0.0)
        gaze = estimate_for_target(target_for_pose(np.zeros(2, dtype=np.float32)))
        now = 0.0
        for _ in range(5):
            calibration.update(gaze, (0.0, 0.0), now=now)
            for _ in range(2):
                now += 0.01
                update = calibration.update(gaze, (0.0, 0.0), now=now)
            now += 0.01
        self.assertTrue(update.completed_now)
        self.assertTrue(calibration.calibrated)
        self.assertFalse(calibration.coverage_ok)
        self.assertIn("too small", calibration.coverage_message)
        self.assertIsNotNone(calibration.predict_target((0.4, -0.4)))

    def test_targets_for_keeps_manual_nudges_in_screen_space(self) -> None:
        calibration = DynamicGazeLockCalibration(countdown_seconds=0.0, sample_count=2)
        self._complete(calibration)
        base_pose = np.asarray((0.0, 0.0), dtype=np.float32)
        raw = estimate_for_target(target_for_pose(base_pose))
        # Give the second eye the opposite horizontal local axis, as can occur
        # after contour ordering/mirroring.  The same image-space nudge must
        # still move both eyes in the same displayed direction.
        first = raw.eyes[0]
        second = raw.eyes[1]
        reversed_second = replace(second, horizontal_axis=-second.horizontal_axis)
        gaze = GazeEstimate((first, reversed_second))
        targets = calibration.targets_for(
            gaze,
            base_pose,
            horizontal_nudge=0.03,
            vertical_nudge=-0.02,
        )
        assert targets is not None
        base = calibration.predict_target(base_pose)
        assert base is not None
        for eye, target, reference in zip(gaze.eyes, targets, base):
            np.testing.assert_allclose(
                eye.point_for_ratio(target) - eye.point_for_ratio(reference),
                (eye.width * 0.03, eye.width * -0.02),
                atol=0.001,
            )

    def test_invalid_pose_never_adds_a_sample(self) -> None:
        calibration = DynamicGazeLockCalibration(countdown_seconds=0.0, sample_count=2)
        calibration.start(now=0.0)
        gaze = estimate_for_target(target_for_pose(np.zeros(2, dtype=np.float32)))
        update = calibration.update(gaze, (float("nan"), 0.0), now=0.0)
        self.assertEqual(update.collected, 0)
        update = calibration.update(gaze, None, now=0.1)
        self.assertEqual(update.collected, 0)


if __name__ == "__main__":
    unittest.main()
