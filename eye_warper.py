"""Conservative local eye-region warp, planning, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import MAX_HORIZONTAL_SHIFT_FRACTION, MAX_VERTICAL_SHIFT_FRACTION
from gaze_estimator import EyeGaze


@dataclass(frozen=True)
class EyeWarpPlan:
    """The exact per-eye movement that is safe to apply to this frame."""

    source: np.ndarray
    requested_target: np.ndarray
    destination: np.ndarray
    delta: np.ndarray
    applied: bool
    reason: str

    @property
    def pixels(self) -> float:
        return float(np.linalg.norm(self.delta))


class EyeWarper:
    """Move real eye texture with a dense, masked inverse remap.

    Planning is separated from drawing so the preview can report exactly why a
    frame changed or was safely held. It never paints a replacement pupil.
    """

    def __init__(
        self,
        max_horizontal_fraction: float = MAX_HORIZONTAL_SHIFT_FRACTION,
        max_vertical_fraction: float = MAX_VERTICAL_SHIFT_FRACTION,
    ) -> None:
        self.max_horizontal_fraction = float(max_horizontal_fraction)
        self.max_vertical_fraction = float(max_vertical_fraction)

    def plan(
        self,
        eyes: tuple[EyeGaze, ...],
        target_ratios: np.ndarray,
        strength: float,
    ) -> tuple[EyeWarpPlan, ...]:
        """Calculate the *actual* safe destination for each eye.

        Safety intentionally tests the strength-scaled, clipped destination.
        Testing the full target before strength made a gentle request get
        skipped merely because a hypothetical 100% request was too close to a
        lid.
        """

        if len(eyes) != len(target_ratios):
            return ()
        if len(eyes) == 2:
            width_ratio = eyes[0].width / max(eyes[1].width, 1e-6)
            if not 0.60 <= width_ratio <= 1.65:
                return tuple(
                    self._skipped_plan(eye, target, "profile angle")
                    for eye, target in zip(eyes, target_ratios)
                )

        return tuple(
            self._plan_one_eye(eye, target_ratio, strength)
            for eye, target_ratio in zip(eyes, target_ratios)
        )

    def correct(
        self,
        frame_bgr: np.ndarray,
        eyes: tuple[EyeGaze, ...],
        target_ratios: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Backward-compatible convenience wrapper used by tests and callers."""

        return self.apply(frame_bgr, eyes, self.plan(eyes, target_ratios, strength))

    def apply(
        self,
        frame_bgr: np.ndarray,
        eyes: tuple[EyeGaze, ...],
        plans: tuple[EyeWarpPlan, ...],
    ) -> np.ndarray:
        output = frame_bgr.copy()
        if len(eyes) != len(plans):
            return output
        for eye, plan in zip(eyes, plans):
            if plan.applied:
                self._warp_one_eye(output, eye, plan.delta)
        return output

    def _plan_one_eye(
        self,
        eye: EyeGaze,
        target_ratio: np.ndarray,
        strength: float,
    ) -> EyeWarpPlan:
        source = np.asarray(eye.iris_center, dtype=np.float32).copy()
        requested = eye.point_for_ratio(np.asarray(target_ratio, dtype=np.float32))
        if strength <= 0.0:
            return EyeWarpPlan(source, requested, source, np.zeros(2, dtype=np.float32), False, "zero strength")
        if eye.width < 24.0 or eye.height < 4.0 or eye.openness < 0.09:
            return EyeWarpPlan(source, requested, source, np.zeros(2, dtype=np.float32), False, "eye not open")

        delta = (requested - source) * float(strength)
        delta[0] = np.clip(
            delta[0],
            -eye.width * self.max_horizontal_fraction,
            eye.width * self.max_horizontal_fraction,
        )
        delta[1] = np.clip(
            delta[1],
            -eye.width * self.max_vertical_fraction,
            eye.width * self.max_vertical_fraction,
        )
        if float(np.linalg.norm(delta)) < 0.15:
            return EyeWarpPlan(source, requested, source, np.zeros(2, dtype=np.float32), False, "target already reached")

        destination = source + delta
        safe_destination, limited, safe = self._safe_destination(eye, source, destination)
        applied_delta = (safe_destination - source).astype(np.float32)
        if not safe or float(np.linalg.norm(applied_delta)) < 0.15:
            return EyeWarpPlan(source, requested, source, np.zeros(2, dtype=np.float32), False, "iris near eyelid")
        return EyeWarpPlan(
            source,
            requested,
            safe_destination,
            applied_delta,
            True,
            "limited by eyelid" if limited else "applied",
        )

    @staticmethod
    def _skipped_plan(
        eye: EyeGaze,
        target_ratio: np.ndarray,
        reason: str,
    ) -> EyeWarpPlan:
        source = np.asarray(eye.iris_center, dtype=np.float32).copy()
        requested = eye.point_for_ratio(np.asarray(target_ratio, dtype=np.float32))
        return EyeWarpPlan(source, requested, source, np.zeros(2, dtype=np.float32), False, reason)

    def _safe_destination(
        self,
        eye: EyeGaze,
        source: np.ndarray,
        destination: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool]:
        """Clamp a requested movement at the visible eyelid margin if needed."""

        if not self._has_safe_iris_margin(eye, source, source):
            return source, False, False
        if self._has_safe_iris_margin(eye, source, destination):
            return destination.astype(np.float32), False, True

        # The source is safe but the requested endpoint is not. Keep as much
        # movement as possible instead of silently disabling the whole eye.
        direction = destination - source
        low, high = 0.0, 1.0
        for _ in range(12):
            middle = (low + high) / 2.0
            candidate = source + direction * middle
            if self._has_safe_iris_margin(eye, source, candidate):
                low = middle
            else:
                high = middle
        if low <= 0.02:
            return source, True, False
        return (source + direction * low).astype(np.float32), True, True

    @staticmethod
    def _has_safe_iris_margin(
        eye: EyeGaze,
        source: np.ndarray,
        destination: np.ndarray,
    ) -> bool:
        """Reject occlusion cases before moving iris texture toward a lid."""

        polygon = eye.contour.astype(np.float32).reshape((-1, 1, 2))
        # The five refined iris points can span most of a partially open eye,
        # so using the raw iris radius as an eyelid clearance overestimates the
        # needed margin and limits real webcam frames to about two pixels. Cap
        # it by measured lid opening; the remap's distance field still fades
        # all displacement to zero at the exact eyelid boundary.
        margin = max(
            1.0,
            min(eye.iris_radius * 0.38, eye.height * 0.25),
            eye.width * 0.018,
        )
        source_distance = cv2.pointPolygonTest(
            polygon, tuple(float(value) for value in source), True
        )
        destination_distance = cv2.pointPolygonTest(
            polygon, tuple(float(value) for value in destination), True
        )
        return source_distance >= margin and destination_distance >= margin

    def _warp_one_eye(
        self,
        frame: np.ndarray,
        eye: EyeGaze,
        delta: np.ndarray,
    ) -> None:
        frame_height, frame_width = frame.shape[:2]
        padding_x = max(4, int(round(eye.width * 0.30 + abs(float(delta[0])))))
        padding_y = max(3, int(round(eye.height * 0.55 + abs(float(delta[1])))))
        min_xy = np.floor(np.min(eye.contour, axis=0)).astype(int)
        max_xy = np.ceil(np.max(eye.contour, axis=0)).astype(int)
        x0 = max(0, int(min_xy[0]) - padding_x)
        y0 = max(0, int(min_xy[1]) - padding_y)
        x1 = min(frame_width, int(max_xy[0]) + padding_x + 1)
        y1 = min(frame_height, int(max_xy[1]) + padding_y + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return

        roi = frame[y0:y1, x0:x1]
        roi_height, roi_width = roi.shape[:2]
        local_contour = np.rint(eye.contour - np.asarray((x0, y0))).astype(np.int32)
        hard_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        cv2.fillPoly(hard_mask, [local_contour], 255)
        if cv2.countNonZero(hard_mask) < 12:
            return

        grid_y, grid_x = np.mgrid[0:roi_height, 0:roi_width].astype(np.float32)
        destination = eye.iris_center + delta - np.asarray((x0, y0), dtype=np.float32)
        sigma_x = max(2.5, eye.width * 0.33)
        sigma_y = max(1.8, eye.height * 0.42)
        radial = np.exp(
            -0.5
            * (
                ((grid_x - destination[0]) / sigma_x) ** 2
                + ((grid_y - destination[1]) / sigma_y) ** 2
            )
        ).astype(np.float32)

        # Erode one pixel before distance transform. OpenCV fillPoly includes
        # contour pixels, so the actual eyelid boundary remains unchanged.
        inner_mask = cv2.erode(
            hard_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        if cv2.countNonZero(inner_mask) < 12:
            return
        distance_to_edge = cv2.distanceTransform(inner_mask, cv2.DIST_L2, 3)
        feather_distance = max(1.5, min(5.0, eye.height * 0.40))
        interior_weight = np.clip(
            distance_to_edge / feather_distance, 0.0, 1.0
        ).astype(np.float32)
        displacement = radial * interior_weight
        map_x = grid_x - float(delta[0]) * displacement
        map_y = grid_y - float(delta[1]) * displacement
        warped = cv2.remap(
            roi,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        alpha = interior_weight[..., None] * 0.96
        blended = warped.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
        frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
