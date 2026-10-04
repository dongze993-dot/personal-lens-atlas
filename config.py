"""Runtime defaults for the CPU-only gaze-correction MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


APP_NAME = "Neural Eye Contact Camera MVP"
WINDOW_NAME = APP_NAME
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Camera probing is deliberately narrow: Windows commonly exposes the internal
# camera at index 0, but a USB camera can take one of the next slots.
CAMERA_INDICES = (0, 1, 2, 3)
# The first real-world run showed that 1280x720 could fall to 10 FPS in dim
# light.  960x540 keeps enough eye pixels for a local warp, fits two panels in
# a 1920px-wide display without downscaling the text, and is a better quality /
# speed compromise than 640x480.  The two common fallback modes remain.
PREFERRED_RESOLUTIONS = ((960, 540), (640, 480), (1280, 720))
PREFERRED_FPS = 30

# Face Mesh is run on a smaller image while OpenCV keeps the original camera
# resolution for display and the final local eye warp.
DEFAULT_PROCESSING_SCALE = 0.55
MIN_PROCESSING_SCALE = 0.45

# A two-step calibration now records the user's actual "look at the camera
# lens" iris position, so it no longer needs to guess a fixed camera lift.
# This remains an optional fine vertical nudge after calibration.
DEFAULT_CAMERA_LIFT = 0.00
# The learned full-eye model has a useful but finite angle range.  Starting a
# little below maximum keeps its first live preview natural; the on-screen
# slider still deliberately reaches 100% for a visual test.
DEFAULT_CORRECTION_STRENGTH = 0.85
STRENGTH_STEP = 0.10
MAX_CORRECTION_STRENGTH = 1.00
MIN_CORRECTION_STRENGTH = 0.00
TARGET_NUDGE_STEP = 0.025
MAX_TARGET_NUDGE = 0.20

LANDMARK_EMA_ALPHA = 0.62
GAZE_EMA_ALPHA = 0.52

CALIBRATION_COUNTDOWN_SECONDS = 2.0
CALIBRATION_SAMPLE_COUNT = 30
# Normalized by eye width.  This is deliberately tiny: it only catches the
# common mistake of completing the second stage while still looking at the
# screen, not a webcam that genuinely sits near the screen centre.
MIN_LENS_TARGET_DELTA = 0.012

# Strong gaze-lock mode records the iris target while the person looks at the
# physical camera in five comfortable head poses.  At the measured ~10 FPS
# this takes roughly 12-18 seconds, rather than asking for a long motion
# capture or relying on a GPU model.
GAZE_LOCK_COUNTDOWN_SECONDS = 1.5
GAZE_LOCK_SAMPLE_COUNT = 12

# Conservative local deformation limits.  Values are fractions of the eye
# bounding box, not frame dimensions, so the warp remains subtle at 720p.
MAX_HORIZONTAL_SHIFT_FRACTION = 0.18
MAX_VERTICAL_SHIFT_FRACTION = 0.13


@dataclass
class RuntimeSettings:
    """Settings changed interactively while the app is running."""

    correction_strength: float = DEFAULT_CORRECTION_STRENGTH
    camera_lift: float = DEFAULT_CAMERA_LIFT
    show_side_by_side: bool = True
    # A clean corrected image is the default.  Press D only when you need to
    # inspect source/target markers during calibration or troubleshooting.
    debug: bool = False
    aim_at_eye_center: bool = False
    target_nudge_x: float = 0.0
    target_nudge_y: float = 0.0

    def set_strength(self, value: float) -> None:
        self.correction_strength = max(
            MIN_CORRECTION_STRENGTH,
            min(MAX_CORRECTION_STRENGTH, float(value)),
        )

    def change_strength(self, delta: float) -> None:
        self.set_strength(self.correction_strength + float(delta))

    def nudge_target(self, horizontal: float = 0.0, vertical: float = 0.0) -> None:
        self.target_nudge_x = max(
            -MAX_TARGET_NUDGE,
            min(MAX_TARGET_NUDGE, self.target_nudge_x + float(horizontal)),
        )
        self.target_nudge_y = max(
            -MAX_TARGET_NUDGE,
            min(MAX_TARGET_NUDGE, self.target_nudge_y + float(vertical)),
        )

    def reset_target_nudge(self) -> None:
        self.target_nudge_x = 0.0
        self.target_nudge_y = 0.0
