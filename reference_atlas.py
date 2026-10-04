"""CPU-only personal camera-lens eye reference atlas.

The old neural renderer predicts a small 48x64 flow field from the current
eye.  This module takes a different approach that is practical on a modest
Windows CPU: it uses a locally recorded frame of *this same person* looking at
the physical camera lens at a similar head pose.  Only the inside of the
current eye aperture is warped and blended; the live eyelids, lashes, skin and
closed-eye frames always remain in charge.

It is intentionally a personal, pose-covered renderer rather than a claim of
unbounded generative gaze reconstruction.  When the current pose or eye
opening has no reliable reference, it returns the original frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from gaze_estimator import EyeGaze


ATLAS_FORMAT = "personal-eye-atlas-v1"
ACTIVE_ATLAS_FILENAME = "active.json"
ATLAS_ROOT_NAME = "reference_atlas"

# Face Landmarker source groups, kept in their contour order.  A mirrored
# input can cause MediaPipe's *named* left/right groups to exchange even after
# their coordinates are flipped back.  ``canonical_eye_geometries`` below
# therefore assigns the two resulting eyes to stable canonical image cells by
# their X positions, rather than trusting a named landmark group at runtime.
EYE_RINGS: Mapping[str, tuple[int, ...]] = {
    "R": (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7),
    "L": (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249),
}
IRIS_GROUPS: Mapping[str, tuple[int, ...]] = {
    "R": (468, 469, 470, 471, 472),
    "L": (473, 474, 475, 476, 477),
}

MIN_EYE_WIDTH = 24.0
MIN_OPENNESS = 0.075
MIN_APERTURE_PIXELS = 28


class ReferenceAtlasError(RuntimeError):
    """Raised for an atlas that cannot be used safely."""


@dataclass(frozen=True)
class EyeGeometry:
    """Current or recorded geometry for one semantic eye."""

    side: str
    contour: np.ndarray
    iris_center: np.ndarray
    center: np.ndarray
    horizontal_axis: np.ndarray
    vertical_axis: np.ndarray
    width: float
    height: float
    openness: float
    iris_ratio: np.ndarray


@dataclass(frozen=True)
class AtlasRecord:
    """One real camera-lens eye crop selected during offline atlas building."""

    record_id: str
    pose_key: str
    side: str
    frame_index: int
    pose: np.ndarray  # [yaw, pitch, roll]
    image: np.ndarray
    contour: np.ndarray  # local to ``image``
    iris_center: np.ndarray  # local to ``image``
    center: np.ndarray  # local to ``image``
    horizontal_axis: np.ndarray
    vertical_axis: np.ndarray
    width: float
    height: float
    openness: float
    sharpness: float
    brightness: float
    aperture_mask: np.ndarray


@dataclass(frozen=True)
class AtlasWarpPlan:
    """A pose-matched donor and its safe local affine transform."""

    source: np.ndarray
    requested_target: np.ndarray
    destination: np.ndarray
    delta: np.ndarray
    applied: bool
    reason: str
    record: AtlasRecord | None = None
    matrix: np.ndarray | None = None  # record crop coordinates -> canonical frame
    roi: tuple[int, int, int, int] | None = None
    contour: np.ndarray | None = None  # canonical full-frame current contour
    eye_index: int = -1
    composite_alpha: float = 0.0
    match_score: float = float("inf")
    # The local vertical eye-axis length, carried from the live geometry.
    # It must not be reconstructed from raw image-Y span when the head rolls.
    live_eye_height: float = 0.0

    @property
    def pixels(self) -> float:
        return float(np.linalg.norm(self.delta))

    @property
    def model_side(self) -> str | None:
        """Compatibility alias used by the existing debug overlay."""

        return self.record.side if self.record is not None else None

    @property
    def model_angle(self) -> None:
        """Atlas records are real pixels, not a neural angle model."""

        return None


def default_atlas_root() -> Path:
    return Path(__file__).resolve().parent / ATLAS_ROOT_NAME


def _validated_points(landmarks: np.ndarray) -> np.ndarray | None:
    try:
        points = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 478 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    return points if np.isfinite(points).all() else None


def eye_geometry(landmarks: np.ndarray, side: str) -> EyeGeometry | None:
    """Build stable eye axes from semantic Face Landmarker points."""

    points = _validated_points(landmarks)
    ring = EYE_RINGS.get(side)
    iris_group = IRIS_GROUPS.get(side)
    if points is None or ring is None or iris_group is None:
        return None
    contour = points[list(ring)].astype(np.float32, copy=True)
    iris = points[list(iris_group)].astype(np.float32, copy=True)
    corner_a, corner_b = contour[0], contour[8]
    horizontal_raw = corner_b - corner_a
    width = float(np.linalg.norm(horizontal_raw))
    if width < 1e-4:
        return None
    horizontal_axis = (horizontal_raw / width).astype(np.float32)
    upper = contour[1:8]
    lower = contour[9:16]
    vertical_raw = np.mean(lower, axis=0) - np.mean(upper, axis=0)
    vertical_raw = vertical_raw - float(np.dot(vertical_raw, horizontal_axis)) * horizontal_axis
    height = float(np.linalg.norm(vertical_raw))
    if height < 1e-4:
        return None
    vertical_axis = (vertical_raw / height).astype(np.float32)
    center = ((np.mean(upper, axis=0) + np.mean(lower, axis=0)) * 0.5).astype(np.float32)
    iris_center = np.mean(iris, axis=0).astype(np.float32)
    iris_ratio = np.asarray(
        (
            float(np.dot(iris_center - center, horizontal_axis)) / width,
            float(np.dot(iris_center - center, vertical_axis)) / width,
        ),
        dtype=np.float32,
    )
    return EyeGeometry(
        side=side,
        contour=contour,
        iris_center=iris_center,
        center=center,
        horizontal_axis=horizontal_axis,
        vertical_axis=vertical_axis,
        width=width,
        height=height,
        openness=height / max(width, 1e-6),
        iris_ratio=iris_ratio,
    )


def canonical_eye_geometries(landmarks: np.ndarray) -> dict[str, EyeGeometry]:
    """Return stable left/right *reference cells* in canonical raw orientation.

    Atlas records use ``R`` for the canonical image-left cell and ``L`` for
    the canonical image-right cell, matching the raw unmirrored capture.  The
    names retain the original MediaPipe group strings for compatibility, but
    they are not trusted as anatomical identity at live runtime: FaceLandmarker
    can exchange them if detection ran on a mirrored preview frame.
    """

    candidates = [eye_geometry(landmarks, side) for side in EYE_RINGS]
    if any(candidate is None for candidate in candidates):
        return {}
    left, right = sorted(
        (candidate for candidate in candidates if candidate is not None),
        key=lambda geometry: float(geometry.center[0]),
    )
    if float(right.center[0] - left.center[0]) < max(8.0, 0.25 * (left.width + right.width)):
        return {}
    return {"R": left, "L": right}


def eye_crop_bounds(
    geometry: EyeGeometry,
    frame_width: int,
    frame_height: int,
    *,
    horizontal_margin: float = 0.34,
    vertical_margin: float = 0.38,
) -> tuple[int, int, int, int] | None:
    """Return a modest padded crop that includes the full visible aperture."""

    minimum = np.min(geometry.contour, axis=0)
    maximum = np.max(geometry.contour, axis=0)
    horizontal_pad = max(3.0, geometry.width * horizontal_margin)
    vertical_pad = max(3.0, geometry.width * vertical_margin)
    left = max(0, int(np.floor(float(minimum[0] - horizontal_pad))))
    right = min(frame_width, int(np.ceil(float(maximum[0] + horizontal_pad))))
    top = max(0, int(np.floor(float(minimum[1] - vertical_pad))))
    bottom = min(frame_height, int(np.ceil(float(maximum[1] + vertical_pad))))
    if right - left < 10 or bottom - top < 10:
        return None
    return left, top, right, bottom


def aperture_alpha(
    shape: tuple[int, int],
    contour_local: np.ndarray,
    eye_height: float,
    *,
    lid_guard_fraction: float = 0.11,
    feather_fraction: float = 0.18,
) -> np.ndarray | None:
    """Make a live-lid-preserving alpha mask inside an eye ring only."""

    height, width = shape
    contour = np.asarray(contour_local, dtype=np.float32)
    if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] < 2:
        return None
    polygon = np.rint(contour[:, :2]).astype(np.int32).reshape((-1, 1, 2))
    hard = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(hard, [polygon], 255, lineType=cv2.LINE_AA)
    if cv2.countNonZero(hard) < MIN_APERTURE_PIXELS:
        return None
    distance = cv2.distanceTransform(hard, cv2.DIST_L2, 3)
    guard = max(0.70, min(1.55, float(eye_height) * lid_guard_fraction))
    feather = max(0.85, min(1.85, float(eye_height) * feather_fraction))
    interior = np.clip((distance - guard) / feather, 0.0, 1.0).astype(np.float32)
    return (interior * interior * (3.0 - 2.0 * interior)).astype(np.float32)


def eye_sharpness(image: np.ndarray) -> float:
    """Cheap local focus score used only for offline reference selection."""

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def eye_brightness(image: np.ndarray) -> float:
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return 0.0
    return float(np.median(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))


def _canonicalize_points(points: np.ndarray, frame_width: int, mirrored: bool) -> np.ndarray:
    result = np.asarray(points, dtype=np.float32).copy()
    if mirrored:
        result[:, 0] = float(frame_width - 1) - result[:, 0]
    return result


def _canonicalize_point(point: np.ndarray, frame_width: int, mirrored: bool) -> np.ndarray:
    result = np.asarray(point, dtype=np.float32).copy()
    if mirrored:
        result[0] = float(frame_width - 1) - result[0]
    return result


def _decanonicalize_point(point: np.ndarray, frame_width: int, mirrored: bool) -> np.ndarray:
    return _canonicalize_point(point, frame_width, mirrored)


def _basis_from_geometry(geometry: EyeGeometry) -> tuple[np.ndarray, np.ndarray] | None:
    """Return an origin and eye-coordinate basis robust to head roll."""

    origin = ((geometry.contour[0] + geometry.contour[8]) * 0.5).astype(np.float32)
    horizontal = (geometry.contour[8] - geometry.contour[0]).astype(np.float32)
    vertical = (
        np.mean(geometry.contour[9:16], axis=0) - np.mean(geometry.contour[1:8], axis=0)
    ).astype(np.float32)
    basis = np.column_stack((horizontal, vertical)).astype(np.float32)
    if abs(float(np.linalg.det(basis))) < 1e-4:
        return None
    return origin, basis


def affine_from_geometries(
    reference: AtlasRecord,
    live: EyeGeometry,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Map a recorded eye crop to a current eye without using iris position.

    The transform uses only corners and lid shape.  The iris is deliberately
    *not* a control point: it is the content being redirected toward the lens.
    """

    reference_geometry = EyeGeometry(
        side=reference.side,
        contour=reference.contour,
        iris_center=reference.iris_center,
        center=reference.center,
        horizontal_axis=reference.horizontal_axis,
        vertical_axis=reference.vertical_axis,
        width=reference.width,
        height=reference.height,
        openness=reference.openness,
        iris_ratio=np.zeros(2, dtype=np.float32),
    )
    reference_basis = _basis_from_geometry(reference_geometry)
    live_basis = _basis_from_geometry(live)
    if reference_basis is None or live_basis is None:
        return None
    reference_origin, reference_axes = reference_basis
    live_origin, live_axes = live_basis
    try:
        transform = live_axes @ np.linalg.inv(reference_axes)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(transform).all():
        return None
    determinant = float(np.linalg.det(transform))
    condition = float(np.linalg.cond(transform))
    if determinant <= 0.0 or not 0.38 <= determinant <= 2.70 or condition > 3.8:
        return None
    translation = live_origin - transform @ reference_origin
    matrix = np.column_stack((transform, translation)).astype(np.float32)
    predicted_iris = (transform @ reference.iris_center + translation).astype(np.float32)
    return matrix, predicted_iris


def _point_clearance(contour: np.ndarray, point: np.ndarray) -> float:
    polygon = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    return float(cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True))


class ReferenceAtlasRenderer:
    """Load real camera-lens eye references and composite them conservatively."""

    def __init__(
        self,
        atlas_path: str | Path | None = None,
        *,
        auto_load: bool = True,
    ) -> None:
        self.atlas_path = Path(atlas_path) if atlas_path else self.default_atlas_path()
        self._records: dict[str, list[AtlasRecord]] = {"R": [], "L": []}
        self._pose_scale = np.asarray((0.030, 0.030, 0.10), dtype=np.float32)
        self._last_records: dict[str, str] = {}
        self._record_by_id: dict[str, AtlasRecord] = {}
        self.source_description = ""
        if auto_load:
            self.load()

    @staticmethod
    def default_atlas_path() -> Path:
        root = default_atlas_root()
        active = root / ACTIVE_ATLAS_FILENAME
        if active.is_file():
            try:
                document = json.loads(active.read_text(encoding="utf-8"))
                relative = document.get("atlas_file") if isinstance(document, Mapping) else None
                if isinstance(relative, str) and relative:
                    candidate = root / relative
                    if candidate.is_file():
                        return candidate
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        return root / "atlas.json"

    @property
    def ready(self) -> bool:
        return bool(self._records["R"] and self._records["L"])

    @property
    def record_count(self) -> int:
        return len(self._records["R"]) + len(self._records["L"])

    def close(self) -> None:
        self._records = {"R": [], "L": []}
        self._record_by_id.clear()
        self._last_records.clear()

    def load(self) -> None:
        """Load a checked local atlas without opening a neural runtime."""

        try:
            document: Any = json.loads(self.atlas_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ReferenceAtlasError(
                "No personal eye reference atlas was found. Run build_reference_atlas.bat "
                "after completing capture_reference.bat."
            ) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReferenceAtlasError(f"Could not read personal eye atlas: {exc}") from exc
        if not isinstance(document, Mapping) or document.get("format") != ATLAS_FORMAT:
            raise ReferenceAtlasError("Personal eye atlas has an incompatible format.")
        raw_records = document.get("records")
        if not isinstance(raw_records, list):
            raise ReferenceAtlasError("Personal eye atlas has no records.")
        root = self.atlas_path.parent
        records: dict[str, list[AtlasRecord]] = {"R": [], "L": []}
        record_by_id: dict[str, AtlasRecord] = {}
        for raw in raw_records:
            record = self._parse_record(raw, root)
            if record is None:
                continue
            records[record.side].append(record)
            record_by_id[record.record_id] = record
        if not records["R"] or not records["L"]:
            raise ReferenceAtlasError(
                "Personal eye atlas has no usable references for both eyes. "
                "Run build_reference_atlas.bat again."
            )
        all_pose = np.asarray(
            [record.pose for side_records in records.values() for record in side_records],
            dtype=np.float32,
        )
        span = np.ptp(all_pose, axis=0) * 0.5
        self._pose_scale = np.maximum(span, np.asarray((0.025, 0.025, 0.08), dtype=np.float32))
        self._records = records
        self._record_by_id = record_by_id
        self._last_records.clear()
        source = document.get("source_capture")
        self.source_description = str(source) if source else self.atlas_path.name

    @staticmethod
    def _parse_record(raw: Any, root: Path) -> AtlasRecord | None:
        if not isinstance(raw, Mapping):
            return None
        try:
            side = str(raw["side"])
            if side not in EYE_RINGS:
                return None
            image_file = raw["image_file"]
            if not isinstance(image_file, str):
                return None
            image = cv2.imread(str(root / image_file), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                return None
            contour = np.asarray(raw["contour"], dtype=np.float32)
            iris_center = np.asarray(raw["iris_center"], dtype=np.float32)
            center = np.asarray(raw["center"], dtype=np.float32)
            horizontal_axis = np.asarray(raw["horizontal_axis"], dtype=np.float32)
            vertical_axis = np.asarray(raw["vertical_axis"], dtype=np.float32)
            pose = np.asarray(raw["pose"], dtype=np.float32)
            width = float(raw["eye_width"])
            height = float(raw["eye_height"])
            openness = float(raw["openness"])
            sharpness = float(raw.get("sharpness", 0.0))
            brightness = float(raw.get("brightness", 0.0))
            frame_index = int(raw.get("frame_index", -1))
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if (
            contour.shape != (16, 2)
            or iris_center.shape != (2,)
            or center.shape != (2,)
            or horizontal_axis.shape != (2,)
            or vertical_axis.shape != (2,)
            or pose.shape != (3,)
            or not np.isfinite(contour).all()
            or not np.isfinite(iris_center).all()
            or not np.isfinite(center).all()
            or not np.isfinite(horizontal_axis).all()
            or not np.isfinite(vertical_axis).all()
            or not np.isfinite(pose).all()
            or not np.isfinite((width, height, openness, sharpness, brightness)).all()
            or width < MIN_EYE_WIDTH
            or height <= 0.0
            or openness < MIN_OPENNESS
        ):
            return None
        if _point_clearance(contour, iris_center) < max(0.8, height * 0.12):
            return None
        mask = aperture_alpha(image.shape[:2], contour, height)
        if mask is None:
            return None
        return AtlasRecord(
            record_id=str(raw.get("id", image_file)),
            pose_key=str(raw.get("pose_key", "unknown")),
            side=side,
            frame_index=frame_index,
            pose=pose,
            image=image,
            contour=contour,
            iris_center=iris_center,
            center=center,
            horizontal_axis=horizontal_axis,
            vertical_axis=vertical_axis,
            width=width,
            height=height,
            openness=openness,
            sharpness=sharpness,
            brightness=brightness,
            aperture_mask=mask,
        )

    @staticmethod
    def _valid_frame(frame_bgr: np.ndarray) -> tuple[int, int] | None:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None
        height, width = frame_bgr.shape[:2]
        return (height, width) if height >= 2 and width >= 2 else None

    @staticmethod
    def _assign_sides(
        eyes: Sequence[EyeGaze],
        semantic_geometries: Mapping[str, EyeGeometry],
    ) -> dict[int, str]:
        if not eyes or not semantic_geometries:
            return {}
        centres = {side: geometry.iris_center for side, geometry in semantic_geometries.items()}
        if len(eyes) == 1:
            return {
                0: min(centres, key=lambda side: float(np.linalg.norm(eyes[0].iris_center - centres[side])))
            }
        pairings = (("R", "L"), ("L", "R"))
        choice = min(
            pairings,
            key=lambda sides: sum(
                float(np.linalg.norm(eyes[index].iris_center - centres[side]))
                for index, side in enumerate(sides)
            ),
        )
        return {index: side for index, side in enumerate(choice[: len(eyes)])}

    def _select_record(
        self,
        side: str,
        pose: np.ndarray,
        live: EyeGeometry,
    ) -> tuple[AtlasRecord, float] | None:
        candidates: list[tuple[AtlasRecord, float]] = []
        for record in self._records.get(side, []):
            opening_ratio = live.openness / max(record.openness, 1e-5)
            if not 0.62 <= opening_ratio <= 1.48:
                continue
            distance = float(np.linalg.norm((pose - record.pose) / self._pose_scale))
            score = distance + 0.70 * abs(float(np.log(opening_ratio)))
            candidates.append((record, score))
        if not candidates:
            return None
        selected, score = min(candidates, key=lambda item: item[1])
        # Outside the demonstrated pose range the system deliberately returns
        # the live eye.  A real reference from the wrong pose is worse than no
        # correction and is exactly how a pasted-eye effect starts.
        if score > 2.25:
            return None
        old_id = self._last_records.get(side)
        if old_id is not None:
            old = self._record_by_id.get(old_id)
            if old is not None:
                old_opening_ratio = live.openness / max(old.openness, 1e-5)
                if 0.62 <= old_opening_ratio <= 1.48:
                    old_score = float(np.linalg.norm((pose - old.pose) / self._pose_scale)) + 0.70 * abs(
                        float(np.log(old_opening_ratio))
                    )
                    # Retain a nearly-as-good record to avoid visible toggling
                    # when pose landmarks jiggle near two reference cells.
                    if old_score <= score * 1.16 and old_score <= 2.25:
                        selected, score = old, old_score
        self._last_records[side] = selected.record_id
        return selected, score

    @staticmethod
    def _skipped_plan(
        index: int,
        eye: EyeGaze,
        reason: str,
        requested: np.ndarray | None = None,
    ) -> AtlasWarpPlan:
        source = np.asarray(eye.iris_center, dtype=np.float32).copy()
        target = np.asarray(requested, dtype=np.float32).copy() if requested is not None else source.copy()
        return AtlasWarpPlan(
            source=source,
            requested_target=target,
            destination=source.copy(),
            delta=np.zeros(2, dtype=np.float32),
            applied=False,
            reason=reason,
            eye_index=index,
        )

    def plan(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        eyes: Sequence[EyeGaze],
        target_ratios: np.ndarray,
        strength: float,
        pose: Sequence[float] | np.ndarray,
        *,
        mirrored: bool = False,
        force: bool = False,
    ) -> tuple[AtlasWarpPlan, ...]:
        """Select pose-matched direct-camera eyes and create safe warp plans."""

        shape = self._valid_frame(frame_bgr)
        eye_tuple = tuple(eyes)
        targets = np.asarray(target_ratios, dtype=np.float32)
        pose_value = np.asarray(pose, dtype=np.float32)
        if (
            shape is None
            or not self.ready
            or targets.shape != (len(eye_tuple), 2)
            or pose_value.shape != (2,)
            or not np.isfinite(targets).all()
            or not np.isfinite(pose_value).all()
        ):
            return tuple(self._skipped_plan(index, eye, "atlas not ready") for index, eye in enumerate(eye_tuple))
        height, width = shape
        validated_landmarks = _validated_points(landmarks)
        if validated_landmarks is None:
            return tuple(self._skipped_plan(index, eye, "invalid landmarks") for index, eye in enumerate(eye_tuple))
        canonical_points = _canonicalize_points(validated_landmarks, width, mirrored)
        semantic_geometries = canonical_eye_geometries(canonical_points)
        if len(semantic_geometries) != 2:
            return tuple(self._skipped_plan(index, eye, "invalid eye geometry") for index, eye in enumerate(eye_tuple))
        assignments = self._assign_sides(
            tuple(
                EyeGaze(
                    contour=_canonicalize_points(eye.contour, width, mirrored),
                    iris_center=_canonicalize_point(eye.iris_center, width, mirrored),
                    ratio=eye.ratio,
                    center=_canonicalize_point(eye.center, width, mirrored),
                    horizontal_axis=np.asarray(
                        (-eye.horizontal_axis[0], eye.horizontal_axis[1]) if mirrored else eye.horizontal_axis,
                        dtype=np.float32,
                    ),
                    vertical_axis=np.asarray(
                        (-eye.vertical_axis[0], eye.vertical_axis[1]) if mirrored else eye.vertical_axis,
                        dtype=np.float32,
                    ),
                    eye_width=eye.width,
                    eye_height=eye.height,
                    iris_radius=eye.iris_radius,
                    iris_radius_x=eye.iris_radius_x,
                    iris_radius_y=eye.iris_radius_y,
                )
                for eye in eye_tuple
            ),
            semantic_geometries,
        )
        amount = float(np.clip(strength, 0.0, 1.0))
        plans: list[AtlasWarpPlan] = []
        for index, (display_eye, target) in enumerate(zip(eye_tuple, targets)):
            requested_display = display_eye.point_for_ratio(target)
            source_display = np.asarray(display_eye.iris_center, dtype=np.float32)
            if amount <= 0.0:
                plans.append(self._skipped_plan(index, display_eye, "zero strength", requested_display))
                continue
            side = assignments.get(index)
            live = semantic_geometries.get(side) if side is not None else None
            if live is None:
                plans.append(self._skipped_plan(index, display_eye, "eye/model mismatch", requested_display))
                continue
            if live.width < MIN_EYE_WIDTH or live.openness < MIN_OPENNESS:
                plans.append(self._skipped_plan(index, display_eye, "eye not open", requested_display))
                continue
            if _point_clearance(live.contour, live.iris_center) < max(0.8, live.height * 0.12):
                plans.append(self._skipped_plan(index, display_eye, "iris near eyelid", requested_display))
                continue
            source_canonical = _canonicalize_point(source_display, width, mirrored)
            roll = float(np.arctan2(live.horizontal_axis[1], live.horizontal_axis[0]))
            selection = self._select_record(
                side,
                np.asarray((pose_value[0], pose_value[1], roll), dtype=np.float32),
                live,
            )
            if selection is None:
                plans.append(self._skipped_plan(index, display_eye, "pose outside reference", requested_display))
                continue
            record, match_score = selection
            affine = affine_from_geometries(record, live)
            if affine is None:
                plans.append(self._skipped_plan(index, display_eye, "eye shape mismatch", requested_display))
                continue
            matrix, donor_iris = affine
            if _point_clearance(live.contour, donor_iris) < max(0.85, live.height * 0.14):
                plans.append(self._skipped_plan(index, display_eye, "donor iris near eyelid", requested_display))
                continue
            # The legacy gaze-lock target is only a UI/diagnostic hint in the
            # atlas renderer.  The actual endpoint is the iris from a real
            # camera-lens reference.  Checking the old target here would
            # repeat the former 0.9px failure even when a donor can provide a
            # meaningful redirection.
            if not force and float(np.linalg.norm(donor_iris - source_canonical)) < 1.25:
                plans.append(
                    self._skipped_plan(index, display_eye, "lens reference already reached", requested_display)
                )
                continue
            roi = eye_crop_bounds(live, width, height, horizontal_margin=0.10, vertical_margin=0.16)
            if roi is None:
                plans.append(self._skipped_plan(index, display_eye, "eye crop outside frame", requested_display))
                continue
            destination_display = _decanonicalize_point(donor_iris, width, mirrored)
            plans.append(
                AtlasWarpPlan(
                    source=source_display.copy(),
                    requested_target=requested_display.astype(np.float32),
                    destination=destination_display.astype(np.float32),
                    delta=(destination_display - source_display).astype(np.float32),
                    applied=True,
                    reason="personal lens reference",
                    record=record,
                    matrix=matrix,
                    roi=roi,
                    contour=live.contour.copy(),
                    eye_index=index,
                    composite_alpha=0.92 * amount,
                    match_score=match_score,
                    live_eye_height=live.height,
                )
            )
        return tuple(plans)

    @staticmethod
    def _match_brightness(
        donor_bgr: np.ndarray,
        current_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray | None:
        """Apply a small global luminance correction only when it is credible."""

        support = mask > 0.30
        if int(np.count_nonzero(support)) < 24:
            return None
        donor_gray = cv2.cvtColor(donor_bgr, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        donor_median = float(np.median(donor_gray[support]))
        current_median = float(np.median(current_gray[support]))
        if donor_median < 8.0:
            return None
        if abs(current_median - donor_median) > 58.0:
            return None
        gain = float(np.clip(current_median / donor_median, 0.82, 1.18))
        return np.clip(donor_bgr.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)

    def apply(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        plans: Sequence[AtlasWarpPlan],
        *,
        mirrored: bool = False,
    ) -> np.ndarray:
        """Apply the plan while never modifying live pixels outside eye openings."""

        shape = self._valid_frame(frame_bgr)
        if shape is None or not self.ready:
            return frame_bgr.copy()
        height, width = shape
        canonical = cv2.flip(frame_bgr, 1) if mirrored else frame_bgr.copy()
        output = canonical.copy()
        for plan in plans:
            if (
                not plan.applied
                or plan.record is None
                or plan.matrix is None
                or plan.roi is None
                or plan.contour is None
            ):
                continue
            left, top, right, bottom = plan.roi
            if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
                continue
            local_contour = plan.contour - np.asarray((left, top), dtype=np.float32)
            live_eye_height = plan.live_eye_height
            if not np.isfinite(live_eye_height) or live_eye_height <= 0.0:
                # Compatibility fallback for callers that constructed an old
                # plan themselves.  Runtime plans always carry the true
                # local-axis height above.
                live_eye_height = float(np.ptp(plan.contour[:, 1]))
            live_mask = aperture_alpha(
                (bottom - top, right - left), local_contour, live_eye_height
            )
            if live_mask is None:
                continue
            matrix = np.asarray(plan.matrix, dtype=np.float32).copy()
            matrix[:, 2] -= np.asarray((left, top), dtype=np.float32)
            roi_size = (right - left, bottom - top)
            donor = cv2.warpAffine(
                plan.record.image,
                matrix,
                roi_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            donor_mask = cv2.warpAffine(
                plan.record.aperture_mask,
                matrix,
                roi_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            region = output[top:bottom, left:right]
            alpha = np.minimum(live_mask, donor_mask)
            adjusted = self._match_brightness(donor, region, alpha)
            if adjusted is None:
                continue
            alpha *= float(np.clip(plan.composite_alpha, 0.0, 0.95))
            if float(alpha.max()) < 0.08:
                continue
            blend = alpha[..., None]
            output[top:bottom, left:right] = np.clip(
                adjusted.astype(np.float32) * blend
                + region.astype(np.float32) * (1.0 - blend),
                0.0,
                255.0,
            ).astype(np.uint8)
        return cv2.flip(output, 1) if mirrored else output

    def correct(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        eyes: Sequence[EyeGaze],
        target_ratios: np.ndarray,
        strength: float,
        pose: Sequence[float] | np.ndarray,
        *,
        mirrored: bool = False,
        force: bool = False,
    ) -> np.ndarray:
        plans = self.plan(
            frame_bgr,
            landmarks,
            eyes,
            target_ratios,
            strength,
            pose,
            mirrored=mirrored,
            force=force,
        )
        return self.apply(frame_bgr, landmarks, plans, mirrored=mirrored)
