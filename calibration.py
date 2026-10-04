"""A visible two-step screen-and-lens calibration workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from config import MIN_LENS_TARGET_DELTA
from gaze_estimator import GazeEstimate


@dataclass
class CalibrationUpdate:
    """One UI-friendly snapshot of the calibration wizard."""

    active: bool
    phase: str
    stage: str
    countdown_remaining: int
    collected: int
    required: int
    completed_now: bool = False
    accepted_frame: bool = False
    retrying_lens: bool = False
    target_near_screen: bool = False


@dataclass
class GazeCalibration:
    """Record both normal screen gaze and the real camera-lens target.

    A screen-centre sample alone cannot tell us where a particular laptop's
    physical camera sits. The second step records the actual iris position
    while the person looks at that lens, so it includes horizontal as well as
    vertical offset without guessing a universal camera-lift constant.
    """

    countdown_seconds: float
    sample_count: int
    minimum_lens_delta: float = MIN_LENS_TARGET_DELTA
    _started_at: float | None = None
    _stage: str | None = None
    _samples: list[np.ndarray] = field(default_factory=list)
    _last_candidate: np.ndarray | None = None
    screen_baseline: np.ndarray | None = None
    camera_target: np.ndarray | None = None
    _lens_retrying: bool = False
    _lens_retry_count: int = 0

    @property
    def active(self) -> bool:
        return self._started_at is not None and self._stage is not None

    @property
    def calibrated(self) -> bool:
        return self.camera_target is not None

    @property
    def baseline(self) -> np.ndarray | None:
        """Compatibility alias for callers that want the screen reference."""

        return self.screen_baseline

    def start(self, now: float | None = None) -> None:
        """Start over at the screen-centre reference stage."""

        self.screen_baseline = None
        self.camera_target = None
        self._lens_retrying = False
        self._lens_retry_count = 0
        self._begin_stage("screen", time.monotonic() if now is None else float(now))

    def reset(self) -> None:
        self._started_at = None
        self._stage = None
        self._samples.clear()
        self._last_candidate = None
        self.screen_baseline = None
        self.camera_target = None
        self._lens_retrying = False
        self._lens_retry_count = 0

    def _begin_stage(self, stage: str, now: float, *, retrying_lens: bool = False) -> None:
        self._stage = stage
        self._started_at = float(now)
        self._samples.clear()
        self._last_candidate = None
        self._lens_retrying = retrying_lens if stage == "lens" else False

    def update(
        self,
        gaze: GazeEstimate | None,
        now: float | None = None,
    ) -> CalibrationUpdate:
        if not self.active:
            return CalibrationUpdate(False, "idle", "idle", 0, 0, self.sample_count)

        current = time.monotonic() if now is None else float(now)
        assert self._started_at is not None
        assert self._stage is not None
        stage = self._stage
        elapsed = current - self._started_at
        if elapsed < self.countdown_seconds:
            remaining = max(1, int(np.ceil(self.countdown_seconds - elapsed)))
            return CalibrationUpdate(
                True,
                f"{stage}_countdown",
                stage,
                remaining,
                len(self._samples),
                self.sample_count,
                retrying_lens=self._lens_retrying,
            )

        accepted = False
        if self._is_usable_sample(gaze):
            assert gaze is not None
            candidate = gaze.ratios.copy()
            # A sudden iris jump usually means a blink, partial occlusion, or
            # head movement. Do not let a bad frame enter either reference.
            stable = (
                self._last_candidate is not None
                and float(np.max(np.abs(candidate - self._last_candidate))) < 0.060
            )
            self._last_candidate = candidate
            if stable:
                self._samples.append(candidate)
                accepted = True
        else:
            self._last_candidate = None

        if len(self._samples) < self.sample_count:
            return CalibrationUpdate(
                True,
                f"{stage}_collecting",
                stage,
                0,
                len(self._samples),
                self.sample_count,
                accepted_frame=accepted,
                retrying_lens=self._lens_retrying,
            )

        value = np.median(np.asarray(self._samples, dtype=np.float32), axis=0).astype(
            np.float32
        )
        if stage == "screen":
            self.screen_baseline = value
            # Give the person a separate countdown to move their gaze from the
            # screen centre to the physical webcam lens.
            self._begin_stage("lens", current, retrying_lens=False)
            return CalibrationUpdate(
                True,
                "lens_countdown",
                "lens",
                max(1, int(np.ceil(self.countdown_seconds))),
                0,
                self.sample_count,
                accepted_frame=accepted,
            )

        target_near_screen = bool(
            self.screen_baseline is not None
            and np.all(
                np.linalg.norm(value - self.screen_baseline, axis=1)
                < float(self.minimum_lens_delta)
            )
        )
        if target_near_screen and self._lens_retry_count < 1:
            # If the second sample is indistinguishable from screen centre,
            # the usual cause is that the person kept watching the screen.
            # Retry once with a persistent, explicit instruction rather than
            # silently accepting a calibration that has no visible effect.
            self._lens_retry_count += 1
            self._begin_stage("lens", current, retrying_lens=True)
            return CalibrationUpdate(
                True,
                "lens_retry",
                "lens",
                max(1, int(np.ceil(self.countdown_seconds))),
                0,
                self.sample_count,
                accepted_frame=accepted,
                retrying_lens=True,
            )

        self.camera_target = value
        self._started_at = None
        self._stage = None
        self._samples.clear()
        self._last_candidate = None
        self._lens_retrying = False
        return CalibrationUpdate(
            False,
            "complete",
            "complete",
            0,
            self.sample_count,
            self.sample_count,
            completed_now=True,
            accepted_frame=accepted,
            target_near_screen=target_near_screen,
        )

    @staticmethod
    def _is_usable_sample(gaze: GazeEstimate | None) -> bool:
        if gaze is None or len(gaze.eyes) != 2:
            return False
        first, second = gaze.eyes
        width_ratio = first.width / max(second.width, 1e-6)
        if not 0.60 <= width_ratio <= 1.65:
            return False
        for eye in gaze.eyes:
            if eye.width < 24.0 or eye.openness < 0.09:
                return False
            if float(np.max(np.abs(eye.ratio))) > 0.28:
                return False
        return True

    def targets_for(
        self,
        gaze: GazeEstimate,
        camera_lift: float = 0.0,
        *,
        horizontal_nudge: float = 0.0,
        vertical_nudge: float = 0.0,
        use_eye_center: bool = False,
    ) -> np.ndarray | None:
        """Return the desired position for each iris in the current eye frame.

        ``use_eye_center`` is deliberately a visible geometry test rather than
        a claim that a centred pupil equals eye contact. Normal camera mode
        instead uses the measured lens target from stage two.
        """

        if self.camera_target is None or self.camera_target.shape != gaze.ratios.shape:
            return None

        if use_eye_center:
            target = np.zeros_like(self.camera_target, dtype=np.float32)
        else:
            target = self.camera_target.copy()

        # Manual nudges describe a single direction on the displayed image,
        # not a local eye coordinate. The two eye rings have opposite local
        # horizontal axes, so adding the same local X value would pull the
        # eyes apart. Project a screen-space vector into each eye's local axes.
        vertical_screen_nudge = float(vertical_nudge)
        if not use_eye_center:
            vertical_screen_nudge += float(camera_lift)
        for index, eye in enumerate(gaze.eyes):
            screen_delta = np.asarray(
                (
                    float(horizontal_nudge) * eye.width,
                    vertical_screen_nudge * eye.width,
                ),
                dtype=np.float32,
            )
            target[index, 0] += float(np.dot(screen_delta, eye.horizontal_axis)) / eye.width
            target[index, 1] += float(np.dot(screen_delta, eye.vertical_axis)) / eye.width
        return np.clip(target, -0.30, 0.30)
