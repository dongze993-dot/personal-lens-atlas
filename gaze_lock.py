"""Personal, CPU-only head-pose calibration for a stronger gaze-lock mode.

This module deliberately does *not* claim to estimate a person's 3-D gaze.
Instead it learns a small, per-user mapping:

``head pose in the camera image -> iris position when that user looks at a target``.

At run time the renderer can use the learned iris position as its destination.
That makes the desired destination move with the face instead of pinning a
single pupil position to the video frame.  It is a useful calibration layer
for a local reconstruction renderer, while remaining NumPy-only and safe to
run on a CPU.

The class is UI-agnostic.  The application supplies a two-value pose vector
(``yaw, pitch``) and the existing :class:`gaze_estimator.GazeEstimate`.
``FaceObservation`` may expose a richer pose object; callers can pass
``HeadPose`` or simply ``(yaw, pitch)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Sequence

import numpy as np

from gaze_estimator import GazeEstimate


# The physical labels are only instructions for the person being calibrated.
# The regression learns the signs from the collected data, so it continues to
# work whether a mirrored preview makes ``left`` look reversed on screen.
_STAGES: tuple[tuple[str, str], ...] = (
    ("center", "Keep looking at the camera lens. Keep your head centered."),
    ("left", "Keep looking at the camera lens. Move your head to your left."),
    ("right", "Keep looking at the camera lens. Move your head to your right."),
    ("up", "Keep looking at the camera lens. Move your head slightly up."),
    ("down", "Keep looking at the camera lens. Move your head slightly down."),
)


@dataclass(frozen=True)
class HeadPose:
    """A lightweight, translation-independent head-pose signal.

    ``yaw`` and ``pitch`` are intentionally unitless.  They can be landmark
    ratios rather than degrees, which avoids requiring camera intrinsics or a
    GPU pose model.  Only their relative variation during calibration matters.
    """

    yaw: float
    pitch: float

    def as_array(self) -> np.ndarray:
        return np.asarray((self.yaw, self.pitch), dtype=np.float32)


@dataclass(frozen=True)
class GazeLockAnchor:
    """One median pose/target pair captured during a calibration stage."""

    stage: str
    pose: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class GazeLockUpdate:
    """A UI-friendly state snapshot returned by :meth:`update`.

    ``prompt`` is intentionally plain text so a UI can either show it as-is
    or replace it with a localized equivalent keyed by ``stage``.
    """

    active: bool
    phase: str
    stage: str
    prompt: str
    countdown_remaining: int
    collected: int
    required: int
    completed_now: bool = False
    accepted_frame: bool = False
    coverage_ok: bool = False
    coverage_message: str = ""


@dataclass
class DynamicGazeLockCalibration:
    """Collect five camera-target samples and predict a pose-aware target.

    The wizard records the user while they *continue looking at the actual
    camera lens* in five comfortable head poses.  Each stage is reduced to a
    median anchor, then a small ridge-regularized affine model is fit to both
    eyes' two normalized iris coordinates.  A linear model is a deliberate
    safety choice here: five poses cannot support an unconstrained neural or
    high-order model without unstable extrapolation.

    Predictions are clamped in two places:

    * the input pose is kept near the range the user demonstrated;
    * target ratios stay near observed target ratios and inside ``max_target``.

    This means an abrupt tracking failure cannot command an implausibly large
    reconstructed iris move.  The renderer should still apply its own eyelid
    and image-boundary safety checks.
    """

    countdown_seconds: float = 1.5
    sample_count: int = 18
    min_pose_span: float = 0.025
    pose_padding: float = 0.025
    target_padding: float = 0.035
    max_target: float = 0.32
    ridge: float = 1e-4
    max_gaze_jump: float = 0.070
    max_pose_jump: float = 0.120
    _started_at: float | None = None
    _stage_index: int | None = None
    _samples: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    _last_pose: np.ndarray | None = None
    _last_target: np.ndarray | None = None
    _anchors: list[GazeLockAnchor] = field(default_factory=list)
    _weights: np.ndarray | None = None
    _pose_center: np.ndarray | None = None
    _pose_scale: np.ndarray | None = None
    _pose_min: np.ndarray | None = None
    _pose_max: np.ndarray | None = None
    _target_min: np.ndarray | None = None
    _target_max: np.ndarray | None = None
    _coverage_ok: bool = False
    _coverage_message: str = ""

    @property
    def active(self) -> bool:
        return self._started_at is not None and self._stage_index is not None

    @property
    def calibrated(self) -> bool:
        return self._weights is not None

    @property
    def coverage_ok(self) -> bool:
        """Whether the recorded poses span both yaw and pitch usefully."""

        return self.calibrated and self._coverage_ok

    @property
    def coverage_message(self) -> str:
        return self._coverage_message

    @property
    def anchors(self) -> tuple[GazeLockAnchor, ...]:
        """Copies of captured anchors, suitable for a diagnostics panel."""

        return tuple(
            GazeLockAnchor(anchor.stage, anchor.pose.copy(), anchor.target.copy())
            for anchor in self._anchors
        )

    @property
    def pose_limits(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Inclusive runtime pose clamp used by :meth:`predict_target`."""

        if self._pose_min is None or self._pose_max is None:
            return None
        lower = self._pose_min - float(self.pose_padding)
        upper = self._pose_max + float(self.pose_padding)
        return lower.astype(np.float32), upper.astype(np.float32)

    @property
    def target_limits(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Inclusive target-ratio clamp used by :meth:`predict_target`."""

        if self._target_min is None or self._target_max is None:
            return None
        lower = np.maximum(self._target_min - float(self.target_padding), -self.max_target)
        upper = np.minimum(self._target_max + float(self.target_padding), self.max_target)
        return lower.astype(np.float32), upper.astype(np.float32)

    def start(self, now: float | None = None) -> None:
        """Discard the previous model and begin the five-position wizard."""

        self._anchors.clear()
        self._weights = None
        self._pose_center = None
        self._pose_scale = None
        self._pose_min = None
        self._pose_max = None
        self._target_min = None
        self._target_max = None
        self._coverage_ok = False
        self._coverage_message = ""
        self._begin_stage(0, time.monotonic() if now is None else float(now))

    def reset(self) -> None:
        """Clear all calibration state without beginning another wizard."""

        self._started_at = None
        self._stage_index = None
        self._samples.clear()
        self._last_pose = None
        self._last_target = None
        self._anchors.clear()
        self._weights = None
        self._pose_center = None
        self._pose_scale = None
        self._pose_min = None
        self._pose_max = None
        self._target_min = None
        self._target_max = None
        self._coverage_ok = False
        self._coverage_message = ""

    def _begin_stage(self, index: int, now: float) -> None:
        self._stage_index = int(index)
        self._started_at = float(now)
        self._samples.clear()
        self._last_pose = None
        self._last_target = None

    def update(
        self,
        gaze: GazeEstimate | None,
        pose: HeadPose | Sequence[float] | np.ndarray | None,
        now: float | None = None,
    ) -> GazeLockUpdate:
        """Advance the wizard with one observed gaze and head pose.

        The first usable frame after each countdown establishes a stability
        reference.  Only subsequent near-by frames are counted, avoiding a
        partial blink or a head that is still moving into position.
        """

        if not self.active:
            return self._idle_update()

        current = time.monotonic() if now is None else float(now)
        assert self._stage_index is not None
        assert self._started_at is not None
        stage, prompt = _STAGES[self._stage_index]
        elapsed = current - self._started_at
        if elapsed < self.countdown_seconds:
            return GazeLockUpdate(
                True,
                "countdown",
                stage,
                prompt,
                max(1, int(np.ceil(self.countdown_seconds - elapsed))),
                len(self._samples),
                self.sample_count,
            )

        candidate_pose = _coerce_pose(pose)
        candidate_target = _coerce_target(gaze)
        accepted = False
        if candidate_pose is not None and candidate_target is not None and self._is_usable(gaze):
            stable = (
                self._last_pose is not None
                and self._last_target is not None
                and float(np.max(np.abs(candidate_target - self._last_target)))
                <= float(self.max_gaze_jump)
                and float(np.max(np.abs(candidate_pose - self._last_pose)))
                <= float(self.max_pose_jump)
            )
            self._last_pose = candidate_pose
            self._last_target = candidate_target
            if stable:
                self._samples.append((candidate_pose, candidate_target))
                accepted = True
        else:
            self._last_pose = None
            self._last_target = None

        if len(self._samples) < self.sample_count:
            return GazeLockUpdate(
                True,
                "collecting",
                stage,
                prompt,
                0,
                len(self._samples),
                self.sample_count,
                accepted_frame=accepted,
            )

        poses = np.asarray([value[0] for value in self._samples], dtype=np.float32)
        targets = np.asarray([value[1] for value in self._samples], dtype=np.float32)
        self._anchors.append(
            GazeLockAnchor(
                stage=stage,
                pose=np.median(poses, axis=0).astype(np.float32),
                target=np.median(targets, axis=0).astype(np.float32),
            )
        )

        next_index = self._stage_index + 1
        if next_index < len(_STAGES):
            self._begin_stage(next_index, current)
            next_stage, next_prompt = _STAGES[next_index]
            return GazeLockUpdate(
                True,
                "next_stage",
                next_stage,
                next_prompt,
                max(1, int(np.ceil(self.countdown_seconds))),
                0,
                self.sample_count,
                accepted_frame=accepted,
            )

        self._fit_model()
        self._started_at = None
        self._stage_index = None
        self._samples.clear()
        self._last_pose = None
        self._last_target = None
        phase = "complete" if self._coverage_ok else "complete_low_coverage"
        return GazeLockUpdate(
            False,
            phase,
            "complete",
            "Personal gaze-lock calibration complete.",
            0,
            self.sample_count,
            self.sample_count,
            completed_now=True,
            accepted_frame=accepted,
            coverage_ok=self._coverage_ok,
            coverage_message=self._coverage_message,
        )

    def predict_target(
        self, pose: HeadPose | Sequence[float] | np.ndarray | None
    ) -> np.ndarray | None:
        """Return the learned two-eye target ratios for a current pose.

        ``None`` means the pose is invalid or the wizard has not completed.
        Returned arrays are always shaped ``(2, 2)`` and are safe-clamped.
        """

        current_pose = _coerce_pose(pose)
        if (
            current_pose is None
            or self._weights is None
            or self._pose_center is None
            or self._pose_scale is None
        ):
            return None

        limits = self.pose_limits
        if limits is not None:
            current_pose = np.clip(current_pose, limits[0], limits[1])
        normalized = (current_pose - self._pose_center) / self._pose_scale
        row = np.asarray((1.0, normalized[0], normalized[1]), dtype=np.float32)
        predicted = (row @ self._weights).reshape(2, 2).astype(np.float32)
        target_limits = self.target_limits
        if target_limits is not None:
            predicted = np.clip(predicted, target_limits[0], target_limits[1])
        return np.clip(predicted, -float(self.max_target), float(self.max_target)).astype(
            np.float32
        )

    # ``predict`` makes the common runtime call terse while preserving the
    # explicit, descriptive ``predict_target`` API for diagnostics/tests.
    predict = predict_target

    def targets_for(
        self,
        gaze: GazeEstimate,
        pose: HeadPose | Sequence[float] | np.ndarray | None,
        camera_lift: float = 0.0,
        *,
        horizontal_nudge: float = 0.0,
        vertical_nudge: float = 0.0,
        use_eye_center: bool = False,
    ) -> np.ndarray | None:
        """Return a pose-aware target and optional screen-space nudges.

        This mirrors :meth:`calibration.GazeCalibration.targets_for`, except
        that the destination is predicted from the person's current head pose.
        It is the intended direct integration API for the OpenCV preview:

        ``targets = gaze_lock.targets_for(gaze, observation_pose, ...)``.
        """

        if len(gaze.eyes) != 2:
            return None
        if use_eye_center:
            target = np.zeros((2, 2), dtype=np.float32)
        else:
            target = self.predict_target(pose)
            if target is None or target.shape != gaze.ratios.shape:
                return None
            target = target.copy()

        # Nudge in preview image axes, not per-eye local axes.  The two eye
        # contours can have opposing horizontal axes, so a shared local x
        # value would erroneously pull the reconstructed irises apart.
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
        return np.clip(target, -float(self.max_target), float(self.max_target)).astype(
            np.float32
        )

    def _fit_model(self) -> None:
        if len(self._anchors) != len(_STAGES):
            raise RuntimeError("Cannot fit gaze lock before all calibration stages complete.")

        poses = np.asarray([anchor.pose for anchor in self._anchors], dtype=np.float32)
        targets = np.asarray([anchor.target for anchor in self._anchors], dtype=np.float32)
        self._pose_min = np.min(poses, axis=0)
        self._pose_max = np.max(poses, axis=0)
        self._target_min = np.min(targets, axis=0)
        self._target_max = np.max(targets, axis=0)
        self._pose_center = np.median(poses, axis=0).astype(np.float32)
        # Normalize the two pose components to similar scales.  A minimum
        # avoids division by zero if a user completes a direction too subtly.
        self._pose_scale = np.maximum(
            (self._pose_max - self._pose_min) * 0.5,
            float(self.min_pose_span),
        ).astype(np.float32)
        normalized = (poses - self._pose_center) / self._pose_scale
        design = np.column_stack(
            (np.ones(len(poses), dtype=np.float32), normalized[:, 0], normalized[:, 1])
        )
        outputs = targets.reshape(len(targets), 4)
        regularizer = np.diag((0.0, float(self.ridge), float(self.ridge))).astype(np.float32)
        # solve is deterministic and cheaper/more stable than forming a
        # pseudo-inverse in a per-frame path.  Ridge keeps near-flat captures
        # well behaved; fall back to lstsq for a pathological numeric input.
        try:
            self._weights = np.linalg.solve(design.T @ design + regularizer, design.T @ outputs)
        except np.linalg.LinAlgError:
            self._weights = np.linalg.lstsq(design, outputs, rcond=None)[0]
        self._weights = self._weights.astype(np.float32)

        span = self._pose_max - self._pose_min
        missing: list[str] = []
        if float(span[0]) < float(self.min_pose_span):
            missing.append("left/right")
        if float(span[1]) < float(self.min_pose_span):
            missing.append("up/down")
        self._coverage_ok = not missing
        self._coverage_message = (
            "Good head-pose coverage."
            if self._coverage_ok
            else "Head movement was too small for " + " and ".join(missing) + "; recalibrate for stronger tracking."
        )

    @staticmethod
    def _is_usable(gaze: GazeEstimate | None) -> bool:
        if gaze is None or len(gaze.eyes) != 2:
            return False
        first, second = gaze.eyes
        width_ratio = first.width / max(second.width, 1e-6)
        if not 0.55 <= width_ratio <= 1.80:
            return False
        for eye in gaze.eyes:
            if eye.width < 20.0 or eye.openness < 0.075:
                return False
            if not np.isfinite(eye.ratio).all() or float(np.max(np.abs(eye.ratio))) > 0.34:
                return False
        return True

    def _idle_update(self) -> GazeLockUpdate:
        if self.calibrated:
            return GazeLockUpdate(
                False,
                "ready",
                "ready",
                "Personal gaze-lock calibration is ready.",
                0,
                0,
                self.sample_count,
                coverage_ok=self._coverage_ok,
                coverage_message=self._coverage_message,
            )
        return GazeLockUpdate(
            False,
            "idle",
            "idle",
            "Start personal gaze-lock calibration.",
            0,
            0,
            self.sample_count,
        )


def _coerce_pose(
    pose: HeadPose | Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    """Accept a small named pose or a finite ``(yaw, pitch)`` sequence."""

    if pose is None:
        return None
    if isinstance(pose, HeadPose):
        value = pose.as_array()
    elif hasattr(pose, "yaw") and hasattr(pose, "pitch"):
        value = np.asarray((getattr(pose, "yaw"), getattr(pose, "pitch")), dtype=np.float32)
    else:
        try:
            value = np.asarray(pose, dtype=np.float32)
        except (TypeError, ValueError):
            return None
    if value.shape != (2,) or not np.isfinite(value).all():
        return None
    return value.astype(np.float32, copy=True)


def _coerce_target(gaze: GazeEstimate | None) -> np.ndarray | None:
    if gaze is None:
        return None
    try:
        value = np.asarray(gaze.ratios, dtype=np.float32)
    except (AttributeError, TypeError, ValueError):
        return None
    if value.shape != (2, 2) or not np.isfinite(value).all():
        return None
    return value.astype(np.float32, copy=True)
