"""Estimate a stable, local image-space gaze signal from refined eye landmarks.

This is deliberately a 2-D image correction signal, not a biometric 3-D gaze
tracker and not a mouse-control system.  Its local axes follow the eye corners
and eyelids, so head roll affects it much less than a raw image bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EyeLandmarks:
    """One ordered eyelid contour and its five refined iris landmarks."""

    contour: np.ndarray
    iris: np.ndarray

    @property
    def iris_center(self) -> np.ndarray:
        # The Tasks model emits centre plus four iris-ring points.  Their mean
        # is less jumpy than relying on only one landmark.
        return np.mean(self.iris, axis=0, dtype=np.float32)


@dataclass(frozen=True)
class EyeGaze:
    """Eye geometry and a normalized gaze offset in a local U/V coordinate frame."""

    contour: np.ndarray
    iris_center: np.ndarray
    ratio: np.ndarray
    center: np.ndarray
    horizontal_axis: np.ndarray
    vertical_axis: np.ndarray
    eye_width: float
    eye_height: float
    iris_radius: float
    # Elliptical radii are useful to a reconstruction renderer.  They default
    # to None so external callers that construct EyeGaze in small tests remain
    # source-compatible with the first MVP.
    iris_radius_x: float | None = None
    iris_radius_y: float | None = None

    @property
    def width(self) -> float:
        return self.eye_width

    @property
    def height(self) -> float:
        return self.eye_height

    @property
    def openness(self) -> float:
        return self.eye_height / max(self.eye_width, 1e-6)

    def point_for_ratio(self, ratio: np.ndarray) -> np.ndarray:
        """Map a local gaze coordinate back to image pixels."""

        value = np.asarray(ratio, dtype=np.float32)
        return (
            self.center
            + self.horizontal_axis * float(value[0]) * self.eye_width
            + self.vertical_axis * float(value[1]) * self.eye_width
        ).astype(np.float32)


@dataclass(frozen=True)
class GazeEstimate:
    """A pair of eye estimates ordered from left to right in the preview."""

    eyes: tuple[EyeGaze, ...]

    @property
    def ratios(self) -> np.ndarray:
        return np.asarray([eye.ratio for eye in self.eyes], dtype=np.float32)


class GazeEstimator:
    """Use corners and lids to normalize the iris position inside each eye."""

    def estimate(self, eyes: Sequence[EyeLandmarks]) -> GazeEstimate | None:
        estimated: list[EyeGaze] = []
        for eye in eyes:
            result = self._estimate_eye(eye)
            if result is not None:
                estimated.append(result)

        if len(estimated) != 2:
            return None
        estimated.sort(key=lambda item: float(item.iris_center[0]))
        return GazeEstimate(eyes=tuple(estimated))

    @staticmethod
    def _estimate_eye(eye: EyeLandmarks) -> EyeGaze | None:
        contour = np.asarray(eye.contour, dtype=np.float32)
        iris = np.asarray(eye.iris, dtype=np.float32)
        if contour.shape[0] < 16 or contour.shape[-1] != 2 or iris.shape[-1] != 2:
            return None

        # Face tracker keeps the ring ordered as:
        # outer corner, upper lid (7), inner corner, lower lid (7).
        corner_a = contour[0]
        corner_b = contour[8]
        upper_lid = contour[1:8]
        lower_lid = contour[9:16]
        horizontal = corner_b - corner_a
        eye_width = float(np.linalg.norm(horizontal))
        if eye_width < 12.0:
            return None
        horizontal_axis = (horizontal / eye_width).astype(np.float32)

        # Lower-to-upper geometry gives a vertical direction.  Orthogonalising
        # it against the corner axis prevents roll from leaking into horizontal
        # gaze.  Its sign is set to point from upper lid toward lower lid.
        vertical_raw = np.mean(lower_lid, axis=0) - np.mean(upper_lid, axis=0)
        vertical_raw = vertical_raw - np.dot(vertical_raw, horizontal_axis) * horizontal_axis
        eye_height = float(np.linalg.norm(vertical_raw))
        if eye_height < 2.0:
            return None
        vertical_axis = (vertical_raw / eye_height).astype(np.float32)

        center = (
            np.mean(upper_lid, axis=0) + np.mean(lower_lid, axis=0)
        ).astype(np.float32) / 2.0
        iris_center = np.mean(iris, axis=0, dtype=np.float32)
        iris_radius = float(np.median(np.linalg.norm(iris - iris_center, axis=1)))
        local_iris = iris - iris_center
        iris_radius_x = float(
            np.quantile(np.abs(local_iris @ horizontal_axis), 0.80)
        )
        iris_radius_y = float(
            np.quantile(np.abs(local_iris @ vertical_axis), 0.80)
        )
        # With only five refined landmarks one axis can become close to zero
        # on a noisy / partly occluded frame.  Keep a conservative fallback
        # rather than producing a line-shaped reconstructed pupil.
        iris_radius_x = max(1.0, iris_radius_x, iris_radius * 0.45)
        iris_radius_y = max(1.0, iris_radius_y, iris_radius * 0.45)
        offset = iris_center - center
        ratio = np.asarray(
            (
                float(np.dot(offset, horizontal_axis)) / eye_width,
                float(np.dot(offset, vertical_axis)) / eye_width,
            ),
            dtype=np.float32,
        )
        # A lost iris landmark should not turn into an implausibly large warp.
        ratio = np.clip(ratio, -0.35, 0.35)

        return EyeGaze(
            contour=contour,
            iris_center=iris_center,
            ratio=ratio,
            center=center,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
            eye_width=eye_width,
            eye_height=eye_height,
            iris_radius=iris_radius,
            iris_radius_x=iris_radius_x,
            iris_radius_y=iris_radius_y,
        )


def smooth_gaze(
    estimate: GazeEstimate,
    smoother: "ArrayEMALike",
) -> GazeEstimate:
    """EMA-smooth only the gaze offset after landmark geometry is smoothed."""

    eyes: list[EyeGaze] = []
    for index, eye in enumerate(estimate.eyes):
        ratio = np.clip(
            smoother.update(f"eye-{index}-gaze-ratio", eye.ratio), -0.35, 0.35
        )
        # This source point drives the inverse remap.  Using its smoothed
        # position trades a tiny amount of precision for visibly steadier eyes.
        iris_center = eye.point_for_ratio(ratio)
        eyes.append(
            EyeGaze(
                contour=eye.contour,
                iris_center=iris_center,
                ratio=ratio,
                center=eye.center,
                horizontal_axis=eye.horizontal_axis,
                vertical_axis=eye.vertical_axis,
                eye_width=eye.eye_width,
                eye_height=eye.eye_height,
                iris_radius=eye.iris_radius,
                iris_radius_x=eye.iris_radius_x,
                iris_radius_y=eye.iris_radius_y,
            )
        )
    return GazeEstimate(eyes=tuple(eyes))


class ArrayEMALike:
    """Protocol-shaped base kept runtime-light for Python 3.11."""

    def update(self, key: str, value: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError
