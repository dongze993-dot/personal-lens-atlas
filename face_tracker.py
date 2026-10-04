"""Current MediaPipe Tasks Face Landmarker wrapper for streamed camera frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from gaze_estimator import EyeLandmarks


# Ordered eyelid boundary rings.  The 478-point Face Landmarker model provides
# both these ordinary eye points and five refined points for each iris.
_EYE_RINGS: tuple[tuple[int, ...], tuple[int, ...]] = (
    (
        33,
        246,
        161,
        160,
        159,
        158,
        157,
        173,
        133,
        155,
        154,
        153,
        145,
        144,
        163,
        7,
    ),
    (
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
    ),
)
_IRIS_GROUPS: tuple[tuple[int, ...], tuple[int, ...]] = (
    (468, 469, 470, 471, 472),
    (473, 474, 475, 476, 477),
)


@dataclass(frozen=True)
class HeadPose:
    """Small, translation-invariant head-pose signal for personal calibration.

    This is deliberately *not* advertised as a degree-accurate 3-D pose.
    The calibration model only needs a repeatable signal that changes as the
    person turns or nods.  It learns the user's own relationship between that
    signal and the iris target, so the sign also remains correct for mirrored
    preview images.
    """

    yaw: float
    pitch: float

    @property
    def vector(self) -> np.ndarray:
        return np.asarray((self.yaw, self.pitch), dtype=np.float32)


@dataclass(frozen=True)
class FaceObservation:
    """Tracked eye features plus full-frame pixel landmarks.

    ``eyes`` stay ordered from left to right in the displayed image so the
    existing calibration code is mirror-safe.  ``landmarks`` intentionally
    retains the full 478-point mesh in original-frame pixel coordinates: the
    neural full-eye renderer needs the semantic corner and eyelid points for
    each eye rather than a cropped iris-only estimate.
    """

    eyes: tuple[EyeLandmarks, EyeLandmarks]
    pose: HeadPose
    landmarks: np.ndarray


class FaceLandmarkerTracker:
    """Run the modern MediaPipe Tasks FaceLandmarker in VIDEO mode.

    VIDEO mode returns a result for each submitted frame and enables
    MediaPipe's own temporal tracking when num_faces is one.  It avoids the
    asynchronous frame dropping semantics of LIVE_STREAM for this MVP.
    """

    def __init__(
        self,
        model_path: str | Path,
        processing_scale: float = 0.67,
        detection_confidence: float = 0.55,
        tracking_confidence: float = 0.55,
    ) -> None:
        if not 0.35 <= processing_scale <= 1.0:
            raise ValueError("processing_scale must be between 0.35 and 1.0")
        self.processing_scale = float(processing_scale)
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model was not found: {self.model_path}\n"
                "Run setup.bat once to download the official face-landmarker model."
            )

        try:
            base_options = mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=float(detection_confidence),
                min_face_presence_confidence=float(detection_confidence),
                min_tracking_confidence=float(tracking_confidence),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except (AttributeError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Could not create MediaPipe Tasks FaceLandmarker. "
                "Run setup.bat again to restore the pinned dependencies and model."
            ) from exc

        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray) -> FaceObservation | None:
        height, width = frame_bgr.shape[:2]
        if self.processing_scale != 1.0:
            small = cv2.resize(
                frame_bgr,
                None,
                fx=self.processing_scale,
                fy=self.processing_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(
            self._last_timestamp_ms + 1,
            int(time.monotonic_ns() // 1_000_000),
        )
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        if len(landmarks) < 478:
            # Iris points are expected from the official Face Landmarker model.
            return None

        # Points are normalized to the processed image.  Multiplying by the
        # original dimensions maps them back to the unscaled camera frame.
        points = np.asarray(
            [(landmark.x * width, landmark.y * height) for landmark in landmarks],
            dtype=np.float32,
        )
        contour_candidates = [points[list(ring)] for ring in _EYE_RINGS]
        iris_candidates = [points[list(group)] for group in _IRIS_GROUPS]

        # Pair by image-space distance rather than anatomical left/right labels,
        # since the preview can be mirrored and labels would otherwise invert.
        available = set(range(len(iris_candidates)))
        paired: list[EyeLandmarks] = []
        for contour in contour_candidates:
            contour_center = np.mean(contour, axis=0)
            nearest = min(
                available,
                key=lambda index: float(
                    np.linalg.norm(np.mean(iris_candidates[index], axis=0) - contour_center)
                ),
            )
            available.remove(nearest)
            paired.append(EyeLandmarks(contour=contour, iris=iris_candidates[nearest]))

        paired.sort(key=lambda eye: float(eye.iris_center[0]))
        return FaceObservation(
            eyes=(paired[0], paired[1]),
            pose=self._estimate_head_pose(points),
            landmarks=points,
        )

    @staticmethod
    def _estimate_head_pose(points: np.ndarray) -> HeadPose:
        """Build a stable two-number pose cue from the same Face Landmarks.

        Ratios are measured inside the face instead of against the image, so
        ordinary camera framing changes do not look like a head turn.  The
        values are intentionally modest and later clamped to the range the
        user actually covered during their own calibration.
        """

        face_left = points[234]
        face_right = points[454]
        eye_mid = (points[33] + points[263]) * 0.5
        mouth_mid = (points[13] + points[14]) * 0.5
        nose = points[1]

        horizontal = face_right - face_left
        face_width = float(np.linalg.norm(horizontal))
        vertical = mouth_mid - eye_mid
        face_height = float(np.linalg.norm(vertical))
        if face_width < 1e-4 or face_height < 1e-4:
            return HeadPose(0.0, 0.0)

        horizontal_axis = horizontal / face_width
        vertical_axis = vertical / face_height
        # Relative to each facial midline, not image centre.  A person who
        # simply leans left/right remains close to the same pose value.
        yaw = float(np.dot(nose - (face_left + face_right) * 0.5, horizontal_axis)) / face_width
        pitch = float(np.dot(nose - (eye_mid + mouth_mid) * 0.5, vertical_axis)) / face_height
        return HeadPose(
            float(np.clip(yaw, -0.60, 0.60)),
            float(np.clip(pitch, -0.60, 0.60)),
        )

    def set_processing_scale(self, value: float) -> None:
        """Lower inference resolution live when sustained FPS is too low."""

        if not 0.35 <= value <= 1.0:
            raise ValueError("processing_scale must be between 0.35 and 1.0")
        self.processing_scale = float(value)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkerTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
