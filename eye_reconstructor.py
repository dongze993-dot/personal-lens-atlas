"""CPU-only iris reconstruction for the strong gaze-lock preview mode.

The earlier MVP used a dense ``remap`` around an eye.  That works for a very
small move, but a large move can leave the old pupil behind while also drawing
the moved one.  This module deliberately uses a different operation:

1. sample the currently visible iris/pupil texture;
2. inpaint its old visible location; and
3. blend the sampled texture at the planned target behind the current eyelid
   contour.

It is still a 2-D local renderer, not a generative 3-D eye model.  In
particular, it cannot invent an iris portion that is hidden by an eyelid or
the side of the face.  The explicit erase-and-replace flow does, however,
avoid the fixed dark-dot / double-pupil artifact caused by trying to drag the
same pixels through the original eye region.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gaze_estimator import EyeGaze


@dataclass(frozen=True)
class IrisReconstructionPlan:
    """The exact per-eye placement accepted for one video frame.

    Its public fields intentionally mirror :class:`eye_warper.EyeWarpPlan` so
    the preview/debug UI can use either renderer without special cases.
    """

    source: np.ndarray
    requested_target: np.ndarray
    destination: np.ndarray
    delta: np.ndarray
    applied: bool
    reason: str

    @property
    def pixels(self) -> float:
        """Euclidean movement actually rendered, in camera pixels."""

        return float(np.linalg.norm(self.delta))


@dataclass(frozen=True)
class _RenderAssets:
    """Source texture and masks captured before any eye is altered."""

    texture: np.ndarray
    source_alpha: np.ndarray
    patch_width: int
    patch_height: int
    radius_x: float
    radius_y: float


class IrisReconstructor:
    """Erase the visible old iris and composite it at a safe new location.

    ``max_horizontal_fraction`` is a fraction of the measured eye width.
    ``max_vertical_fraction`` is a fraction of the measured visible eye
    height.  The latter is deliberately expressed differently: vertical space
    is controlled by eyelid opening, not by the much wider eye-corner span.
    The endpoint is also clipped against the measured eyelid contour.
    """

    def __init__(
        self,
        max_horizontal_fraction: float = 0.30,
        max_vertical_fraction: float = 0.72,
        iris_scale: float = 1.10,
        min_visible_fraction: float = 0.42,
    ) -> None:
        if max_horizontal_fraction <= 0.0 or max_vertical_fraction <= 0.0:
            raise ValueError("movement fractions must be positive")
        if not 0.75 <= iris_scale <= 1.50:
            raise ValueError("iris_scale must be between 0.75 and 1.50")
        if not 0.15 <= min_visible_fraction <= 0.90:
            raise ValueError("min_visible_fraction must be between 0.15 and 0.90")
        self.max_horizontal_fraction = float(max_horizontal_fraction)
        self.max_vertical_fraction = float(max_vertical_fraction)
        self.iris_scale = float(iris_scale)
        self.min_visible_fraction = float(min_visible_fraction)

    def plan(
        self,
        eyes: tuple[EyeGaze, ...],
        target_ratios: np.ndarray,
        strength: float,
    ) -> tuple[IrisReconstructionPlan, ...]:
        """Plan a strong but eyelid-safe movement for each supplied eye.

        The supplied targets are normalized local eye coordinates, matching
        ``EyeGaze.point_for_ratio``.  The resulting plan is independently
        usable by :meth:`apply`, which makes visual diagnostics truthful.
        """

        targets = np.asarray(target_ratios, dtype=np.float32)
        if targets.ndim != 2 or targets.shape != (len(eyes), 2):
            return ()
        if not np.isfinite(targets).all():
            return tuple(
                self._skipped_plan(eye, np.asarray((0.0, 0.0), dtype=np.float32), "invalid target")
                for eye in eyes
            )
        return tuple(
            self._plan_one_eye(eye, target, strength)
            for eye, target in zip(eyes, targets)
        )

    def correct(
        self,
        frame_bgr: np.ndarray,
        eyes: tuple[EyeGaze, ...],
        target_ratios: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Convenience wrapper that plans and applies reconstruction."""

        return self.apply(frame_bgr, eyes, self.plan(eyes, target_ratios, strength))

    def apply(
        self,
        frame_bgr: np.ndarray,
        eyes: tuple[EyeGaze, ...],
        plans: tuple[IrisReconstructionPlan, ...],
    ) -> np.ndarray:
        """Return a corrected copy of ``frame_bgr`` without changing it.

        Every source patch is captured from the unmodified frame first.  Thus
        inpainting the first eye cannot accidentally corrupt the source used
        for the second eye, even in an unusually tight crop.
        """

        output = frame_bgr.copy()
        if (
            frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or len(eyes) != len(plans)
        ):
            return output

        assets: list[tuple[EyeGaze, IrisReconstructionPlan, _RenderAssets]] = []
        for eye, plan in zip(eyes, plans):
            if not plan.applied:
                continue
            asset = self._make_assets(frame_bgr, eye, plan)
            if asset is not None:
                assets.append((eye, plan, asset))

        # Remove every old pupil before placing either replacement.  Keeping
        # these two phases separate prevents a replacement from being erased
        # when a target happens to overlap the other eye's source ROI.
        for eye, plan, asset in assets:
            self._erase_source(output, eye, plan, asset)
        for eye, plan, asset in assets:
            self._composite_destination(output, eye, plan, asset)
        return output

    def _plan_one_eye(
        self,
        eye: EyeGaze,
        target_ratio: np.ndarray,
        strength: float,
    ) -> IrisReconstructionPlan:
        source = np.asarray(eye.iris_center, dtype=np.float32).copy()
        requested = eye.point_for_ratio(np.asarray(target_ratio, dtype=np.float32))
        if not np.isfinite(source).all() or not np.isfinite(requested).all():
            return IrisReconstructionPlan(
                source, requested, source, np.zeros(2, dtype=np.float32), False, "invalid landmarks"
            )
        if strength <= 0.0:
            return IrisReconstructionPlan(
                source, requested, source, np.zeros(2, dtype=np.float32), False, "zero strength"
            )
        if eye.width < 20.0 or eye.height < 3.0 or eye.openness < 0.055:
            return IrisReconstructionPlan(
                source, requested, source, np.zeros(2, dtype=np.float32), False, "eye not open"
            )

        amount = float(np.clip(strength, 0.0, 1.0))
        delta = (requested - source) * amount
        delta[0] = np.clip(
            delta[0],
            -eye.width * self.max_horizontal_fraction,
            eye.width * self.max_horizontal_fraction,
        )
        delta[1] = np.clip(
            delta[1],
            -eye.height * self.max_vertical_fraction,
            eye.height * self.max_vertical_fraction,
        )
        if float(np.linalg.norm(delta)) < 0.35:
            return IrisReconstructionPlan(
                source, requested, source, np.zeros(2, dtype=np.float32), False, "target already reached"
            )

        radius_x, radius_y = self._visible_iris_radii(eye)
        if self._visible_fraction(eye, source, radius_x, radius_y) < self.min_visible_fraction:
            return IrisReconstructionPlan(
                source, requested, source, np.zeros(2, dtype=np.float32), False, "iris hidden by eyelid"
            )
        destination = source + delta
        safe_destination, limited, safe = self._safe_destination(
            eye, source, destination, radius_x, radius_y
        )
        applied_delta = (safe_destination - source).astype(np.float32)
        if not safe or float(np.linalg.norm(applied_delta)) < 0.35:
            return IrisReconstructionPlan(
                source, requested, source, np.zeros(2, dtype=np.float32), False, "target outside visible eye"
            )
        return IrisReconstructionPlan(
            source,
            requested,
            safe_destination,
            applied_delta,
            True,
            "limited by eyelid" if limited else "reconstructed",
        )

    @staticmethod
    def _skipped_plan(
        eye: EyeGaze,
        target_ratio: np.ndarray,
        reason: str,
    ) -> IrisReconstructionPlan:
        source = np.asarray(eye.iris_center, dtype=np.float32).copy()
        requested = eye.point_for_ratio(target_ratio)
        return IrisReconstructionPlan(
            source, requested, source, np.zeros(2, dtype=np.float32), False, reason
        )

    def _safe_destination(
        self,
        eye: EyeGaze,
        source: np.ndarray,
        destination: np.ndarray,
        radius_x: float,
        radius_y: float,
    ) -> tuple[np.ndarray, bool, bool]:
        if self._visible_fraction(eye, destination, radius_x, radius_y) >= self.min_visible_fraction:
            return destination.astype(np.float32), False, True

        # Preserve as much useful movement as possible.  Testing the complete
        # elliptical area rather than only its centre avoids a pupil half
        # hanging over an eyelid at a large gaze correction.
        direction = destination - source
        low, high = 0.0, 1.0
        for _ in range(13):
            middle = (low + high) * 0.5
            candidate = source + direction * middle
            if self._visible_fraction(eye, candidate, radius_x, radius_y) >= self.min_visible_fraction:
                low = middle
            else:
                high = middle
        if low < 0.04:
            return source.astype(np.float32), True, False
        return (source + direction * low).astype(np.float32), True, True

    def _visible_iris_radii(self, eye: EyeGaze) -> tuple[float, float]:
        """Estimate only the iris area that can plausibly be shown this frame."""

        landmark_radius = max(1.2, float(eye.iris_radius) * self.iris_scale)
        # The refined iris ring often continues beneath a partially closed lid.
        # Capping the vertical radius by visible opening makes the resulting
        # patch look like an iris behind eyelids rather than a floating circle.
        radius_x = min(landmark_radius, eye.width * 0.22)
        radius_y = min(landmark_radius, eye.height * 0.58, eye.width * 0.22)
        return max(1.2, float(radius_x)), max(1.1, float(radius_y))

    @staticmethod
    def _visible_fraction(
        eye: EyeGaze,
        center: np.ndarray,
        radius_x: float,
        radius_y: float,
    ) -> float:
        """Approximate which portion of an iris ellipse is inside the eyelids."""

        polygon = eye.contour.astype(np.float32).reshape((-1, 1, 2))
        # A compact area sampling pattern, not just perimeter points.  It
        # remains fast enough to run during planning for every webcam frame.
        samples: list[tuple[float, float]] = [(0.0, 0.0)]
        for radius in (0.45, 0.78):
            for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
                samples.append((radius * float(np.cos(angle)), radius * float(np.sin(angle))))
        inside = 0
        for local_x, local_y in samples:
            point = (
                center
                + eye.horizontal_axis * (local_x * radius_x)
                + eye.vertical_axis * (local_y * radius_y)
            )
            if cv2.pointPolygonTest(
                polygon, (float(point[0]), float(point[1])), False
            ) >= 0:
                inside += 1
        return inside / len(samples)

    def _make_assets(
        self,
        source_frame: np.ndarray,
        eye: EyeGaze,
        plan: IrisReconstructionPlan,
    ) -> _RenderAssets | None:
        radius_x, radius_y = self._visible_iris_radii(eye)
        feather = max(1.2, min(3.0, min(radius_x, radius_y) * 0.34))
        patch_width = int(np.ceil((radius_x + feather) * 2.0)) + 3
        patch_height = int(np.ceil((radius_y + feather) * 2.0)) + 3
        frame_height, frame_width = source_frame.shape[:2]
        half_w = patch_width * 0.5
        half_h = patch_height * 0.5
        source = plan.source
        if (
            source[0] - half_w < 0.0
            or source[1] - half_h < 0.0
            or source[0] + half_w >= frame_width
            or source[1] + half_h >= frame_height
        ):
            return None

        texture = cv2.getRectSubPix(
            source_frame,
            (patch_width, patch_height),
            (float(source[0]), float(source[1])),
        )
        local_center = np.asarray(
            ((patch_width - 1) * 0.5, (patch_height - 1) * 0.5), dtype=np.float32
        )
        soft_mask = self._soft_ellipse_mask(
            (patch_height, patch_width),
            local_center,
            radius_x,
            radius_y,
            eye.horizontal_axis,
            eye.vertical_axis,
            feather,
        )
        source_eye_mask = self._eye_mask(
            (patch_height, patch_width), eye.contour - source + local_center
        )
        source_alpha = soft_mask * (source_eye_mask.astype(np.float32) / 255.0)
        if float(source_alpha.max()) < 0.25:
            return None
        return _RenderAssets(
            texture=texture,
            source_alpha=source_alpha,
            patch_width=patch_width,
            patch_height=patch_height,
            radius_x=radius_x,
            radius_y=radius_y,
        )

    def _erase_source(
        self,
        output: np.ndarray,
        eye: EyeGaze,
        plan: IrisReconstructionPlan,
        asset: _RenderAssets,
    ) -> None:
        """Inpaint just the visible old iris, never an entire eye rectangle."""

        roi = self._roi_for_center(output.shape[:2], plan.source, asset.patch_width, asset.patch_height)
        if roi is None:
            return
        x0, y0, x1, y1, patch_x0, patch_y0 = roi
        patch_alpha = asset.source_alpha[
            patch_y0 : patch_y0 + (y1 - y0), patch_x0 : patch_x0 + (x1 - x0)
        ]
        # The hard inpaint region intentionally reaches the dark iris edge;
        # the replacement itself supplies a soft outer edge at its destination.
        erase_mask = (patch_alpha >= 0.18).astype(np.uint8) * 255
        if cv2.countNonZero(erase_mask) < 4:
            return
        inpaint_radius = max(1.0, min(4.0, min(asset.radius_x, asset.radius_y) * 0.45))
        region = output[y0:y1, x0:x1]
        output[y0:y1, x0:x1] = cv2.inpaint(
            region, erase_mask, inpaint_radius, cv2.INPAINT_TELEA
        )

    def _composite_destination(
        self,
        output: np.ndarray,
        eye: EyeGaze,
        plan: IrisReconstructionPlan,
        asset: _RenderAssets,
    ) -> None:
        roi = self._roi_for_center(
            output.shape[:2], plan.destination, asset.patch_width, asset.patch_height
        )
        if roi is None:
            return
        x0, y0, x1, y1, patch_x0, patch_y0 = roi
        destination = plan.destination
        local_center = np.asarray(
            ((asset.patch_width - 1) * 0.5, (asset.patch_height - 1) * 0.5), dtype=np.float32
        )
        # The original texture was sampled about the source centre.  Translate
        # it by the fractional part of the desired target, so landmark motion
        # does not visibly jump between integer pixels.
        base_x0 = int(np.floor(float(destination[0]) - asset.patch_width * 0.5))
        base_y0 = int(np.floor(float(destination[1]) - asset.patch_height * 0.5))
        placed_center = destination - np.asarray((base_x0, base_y0), dtype=np.float32)
        shift = placed_center - local_center
        transform = np.asarray(
            ((1.0, 0.0, float(shift[0])), (0.0, 1.0, float(shift[1]))), dtype=np.float32
        )
        texture = cv2.warpAffine(
            asset.texture,
            transform,
            (asset.patch_width, asset.patch_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        alpha = cv2.warpAffine(
            asset.source_alpha,
            transform,
            (asset.patch_width, asset.patch_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        # Clip the replacement behind the eyelids at its *new* location.  This
        # is what makes a partial blink look like an occlusion rather than a
        # black circular sticker on top of the lid.
        dest_eye_mask = self._eye_mask(
            (asset.patch_height, asset.patch_width), eye.contour - destination + local_center
        ).astype(np.float32) / 255.0
        alpha *= dest_eye_mask

        local_x0 = patch_x0
        local_y0 = patch_y0
        local_x1 = local_x0 + (x1 - x0)
        local_y1 = local_y0 + (y1 - y0)
        alpha = alpha[local_y0:local_y1, local_x0:local_x1]
        if alpha.size == 0 or float(alpha.max()) < 0.02:
            return
        texture = texture[local_y0:local_y1, local_x0:local_x1]
        region = output[y0:y1, x0:x1]
        blend = alpha[..., None]
        output[y0:y1, x0:x1] = np.clip(
            texture.astype(np.float32) * blend
            + region.astype(np.float32) * (1.0 - blend),
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def _roi_for_center(
        frame_shape: tuple[int, int],
        center: np.ndarray,
        patch_width: int,
        patch_height: int,
    ) -> tuple[int, int, int, int, int, int] | None:
        """Return clipped frame ROI plus matching coordinates in the patch."""

        frame_height, frame_width = frame_shape
        x0_unclipped = int(np.floor(float(center[0]) - patch_width * 0.5))
        y0_unclipped = int(np.floor(float(center[1]) - patch_height * 0.5))
        x1_unclipped = x0_unclipped + patch_width
        y1_unclipped = y0_unclipped + patch_height
        x0 = max(0, x0_unclipped)
        y0 = max(0, y0_unclipped)
        x1 = min(frame_width, x1_unclipped)
        y1 = min(frame_height, y1_unclipped)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1, x0 - x0_unclipped, y0 - y0_unclipped

    @staticmethod
    def _eye_mask(shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        polygon = np.rint(contour).astype(np.int32).reshape((-1, 1, 2))
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 255, lineType=cv2.LINE_AA)
        return mask

    @staticmethod
    def _soft_ellipse_mask(
        shape: tuple[int, int],
        center: np.ndarray,
        radius_x: float,
        radius_y: float,
        horizontal_axis: np.ndarray,
        vertical_axis: np.ndarray,
        feather: float,
    ) -> np.ndarray:
        """Return an eye-axis-aligned, antialiased 0..1 ellipse alpha mask."""

        height, width = shape
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        dx = grid_x - float(center[0])
        dy = grid_y - float(center[1])
        local_x = dx * float(horizontal_axis[0]) + dy * float(horizontal_axis[1])
        local_y = dx * float(vertical_axis[0]) + dy * float(vertical_axis[1])
        radial = np.sqrt(
            (local_x / max(radius_x, 1e-4)) ** 2
            + (local_y / max(radius_y, 1e-4)) ** 2
        )
        feather_ratio = feather / max(min(radius_x, radius_y), 1e-4)
        edge_start = max(0.0, 1.0 - feather_ratio)
        alpha = np.clip((1.0 - radial) / max(1.0 - edge_start, 1e-4), 0.0, 1.0)
        alpha[radial <= edge_start] = 1.0
        return alpha.astype(np.float32)
