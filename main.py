"""Stage 1 local webcam preview for CPU-only gaze correction.

Run ``python main.py``. If the project environment exists, this file safely
relaunches itself there instead of changing the user's system Python.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / "gaze-env" / "Scripts" / "python.exe"
MPL_CONFIG_DIR = PROJECT_DIR / "work" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


def _maybe_relaunch_in_project_environment() -> int | None:
    """Make ``python main.py`` work without global package installation."""

    if not VENV_PYTHON.is_file():
        return None
    try:
        same_interpreter = Path(sys.executable).resolve() == VENV_PYTHON.resolve()
    except OSError:
        same_interpreter = False
    if same_interpreter:
        return None
    print("Using the isolated project environment: gaze-env")
    return subprocess.call([str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


def _dependency_error() -> int:
    print(
        "\nThe optional legacy ONNX experiment is not installed.\n"
        "Use atlas_preview.bat for the current Personal Lens Atlas path, or read "
        "THIRD_PARTY_NOTICES.md and explicitly run setup.bat --experimental-neural "
        "before starting main.py.\n"
        "The isolated setup does not modify or uninstall your Python 3.11 installation."
    )
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional legacy CPU-only MediaPipe + ONNX eye-contact experiment."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Open one camera index directly (default: safely try 0, 1, 2, 3).",
    )
    parser.add_argument("--width", type=int, default=960, help="Preferred capture width.")
    parser.add_argument("--height", type=int, default=540, help="Preferred capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Preferred capture FPS.")
    parser.add_argument(
        "--processing-scale",
        type=float,
        default=0.55,
        help="Face Landmarker scale 0.35-1.0 (smaller is faster; default 0.55).",
    )
    parser.add_argument(
        "--camera-lift",
        type=float,
        default=0.0,
        help="Optional fine vertical target nudge after personal calibration (default: 0).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Show the camera's non-mirrored orientation.",
    )
    return parser.parse_args()


def _put_text(
    cv2: Any,
    frame: Any,
    message: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    """Draw outlined text that remains legible over bright webcam video."""

    cv2.putText(
        frame,
        message,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (10, 10, 10),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_panel_header(cv2: Any, frame: Any, label: str) -> None:
    cv2.rectangle(frame, (0, 0), (220, 38), (18, 18, 18), -1)
    _put_text(cv2, frame, label, (14, 27), 0.68, (255, 255, 255), 2)


def _draw_cross(cv2: Any, frame: Any, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = point
    cv2.line(frame, (x - 4, y - 4), (x + 4, y + 4), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x - 4, y + 4), (x + 4, y - 4), color, 2, cv2.LINE_AA)


def _draw_debug(
    cv2: Any,
    frame: Any,
    gaze: Any,
    plans: Any,
    *,
    corrected: bool,
) -> None:
    """Show source, logical target, and the actual planned destination.

    The old MVP drew the original orange point on both panels, including the
    corrected one. That made a successful local warp look like it had failed.
    """

    if gaze is None:
        return
    for index, eye in enumerate(gaze.eyes):
        contour = eye.contour.round().astype("int32")
        cv2.polylines(frame, [contour], True, (0, 220, 255), 1, cv2.LINE_AA)
        if index >= len(plans):
            iris = tuple(eye.iris_center.round().astype(int))
            cv2.circle(frame, iris, 4, (0, 90, 255), -1, cv2.LINE_AA)
            continue

        plan = plans[index]
        source = tuple(plan.source.round().astype(int))
        requested = tuple(plan.requested_target.round().astype(int))
        destination = tuple(plan.destination.round().astype(int))
        # Cyan ring is the full logical target; green is the exact endpoint
        # after strength, clipping, and eyelid safety checks.
        cv2.circle(frame, requested, 5, (255, 230, 70), 1, cv2.LINE_AA)
        if corrected:
            if plan.applied:
                cv2.circle(frame, destination, 4, (70, 255, 80), -1, cv2.LINE_AA)
            else:
                _draw_cross(cv2, frame, source, (30, 30, 255))
        else:
            cv2.circle(frame, source, 4, (0, 90, 255), -1, cv2.LINE_AA)
            cv2.line(
                frame,
                source,
                destination,
                (70, 255, 80) if plan.applied else (30, 30, 255),
                1,
                cv2.LINE_AA,
            )
        label_position = tuple((eye.center + (0, -8)).round().astype(int))
        _put_text(cv2, frame, str(index + 1), label_position, 0.45, (0, 220, 255), 1)


def _make_preview(cv2: Any, original: Any, corrected: Any, side_by_side: bool) -> Any:
    left = original.copy()
    right = corrected.copy()
    _draw_panel_header(cv2, left, "Original")
    _draw_panel_header(cv2, right, "Corrected")
    preview = cv2.hconcat([left, right]) if side_by_side else right

    # 960x540 is the default on this CPU, so two panels remain at native pixels
    # on a 1920px-wide display. Higher-resolution input is still fit to a
    # laptop screen.
    height, width = preview.shape[:2]
    scale = min(1.0, 1920.0 / width, 920.0 / height)
    if scale < 1.0:
        preview = cv2.resize(
            preview,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return preview


def _blend_box(
    cv2: Any,
    frame: Any,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    alpha: float = 0.78,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (10, 10, 10), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _plan_summary(plans: Any) -> str:
    if not plans:
        return "Neural eye flow: waiting for personal calibration"
    applied = [plan for plan in plans if plan.applied]
    if len(applied) == len(plans):
        shifts = " / ".join(f"{plan.pixels:.1f}px" for plan in plans)
        limits = " limited" if any(plan.reason == "limited by eyelid" for plan in plans) else ""
        model_limits = " model-limited" if any(
            plan.reason == "limited by model angle" for plan in plans
        ) else ""
        return (
            f"Neural eye flow: {len(applied)}/{len(plans)} eyes, "
            f"target shift {shifts}{limits}{model_limits}"
        )
    reasons = ", ".join(plan.reason for plan in plans if not plan.applied)
    return f"Neural eye flow: {len(applied)}/{len(plans)} eyes ({reasons})"


def _draw_calibration_banner(cv2: Any, preview: Any, update: Any) -> None:
    height, width = preview.shape[:2]
    box_width = min(width - 36, 820)
    x0 = max(18, (width - box_width) // 2)
    y0 = 58
    x1 = x0 + box_width
    y1 = min(height - 118, y0 + 112)
    _blend_box(cv2, preview, (x0, y0), (x1, y1), 0.82)

    stage_text = {
        "center": "1/5  LOOK AT CAMERA - HEAD CENTERED",
        "left": "2/5  LOOK AT CAMERA - TURN HEAD LEFT",
        "right": "3/5  LOOK AT CAMERA - TURN HEAD RIGHT",
        "up": "4/5  LOOK AT CAMERA - HEAD SLIGHTLY UP",
        "down": "5/5  LOOK AT CAMERA - HEAD SLIGHTLY DOWN",
    }.get(update.stage, "PERSONAL GAZE-LOCK CALIBRATION")
    if update.phase in {"countdown", "next_stage"}:
        detail = f"Get into position. Starting in {update.countdown_remaining} ..."
    else:
        detail = f"Keep looking at the physical lens: stable frames {update.collected}/{update.required}"
    _put_text(cv2, preview, stage_text, (x0 + 22, y0 + 42), 0.78, (90, 255, 255), 2)
    _put_text(cv2, preview, detail, (x0 + 22, y0 + 82), 0.62, (255, 255, 255), 2)


def _draw_footer(
    cv2: Any,
    preview: Any,
    *,
    fps: float,
    strength: float,
    calibrated: bool,
    calibration_update: Any,
    status_message: str,
    debug: bool,
    aim_label: str,
    target_nudge_x: float,
    target_nudge_y: float,
    plans: Any,
    processing_scale: float,
    inference_stride: int,
    capture_ms: float,
    tracker_ms: float,
    render_ms: float,
) -> None:
    """Draw one crisp global UI after panel composition and any downscaling."""

    height, width = preview.shape[:2]
    footer_height = min(118, max(92, int(height * 0.25)))
    footer_y = height - footer_height
    row_one = footer_y + max(28, int(footer_height * 0.26))
    row_two = footer_y + max(51, int(footer_height * 0.51))
    row_three = footer_y + max(70, int(footer_height * 0.75))
    row_four = height - 8
    _blend_box(cv2, preview, (0, footer_y), (width, height), 0.88)
    readiness = "READY" if calibrated else "NEEDS 5-POSE CALIBRATION"
    headline = (
        f"NEURAL EYE CONTACT {int(round(strength * 100)):3d}%   FPS {fps:4.1f}   "
        f"{readiness}   AIM {aim_label}"
    )
    _put_text(cv2, preview, headline, (18, row_one), 0.74, (255, 255, 255), 2)
    tracker_text = (
        f"{_plan_summary(plans)}   Read {capture_ms:.0f}ms | "
        f"Track {tracker_ms:.0f}ms | Neural {render_ms:.0f}ms | {processing_scale:.2f}x"
    )
    if inference_stride > 1:
        tracker_text += f" / every {inference_stride} frames"
    _put_text(cv2, preview, tracker_text, (18, row_two), 0.50, (120, 245, 160), 1)
    _put_text(
        cv2,
        preview,
        "Drag Strength, then click video before keys  |  1/2 or -/+ strength  |  Arrows/I/J/K/L fine tune",
        (18, row_three),
        0.44,
        (205, 225, 230),
        1,
    )
    _put_text(
        cv2,
        preview,
        "C: 5-pose calibrate  G: centre test  S: compare  D: guide  R: reset  T: reset aim  Q/Esc: quit",
        (18, row_four),
        0.44,
        (205, 225, 230),
        1,
    )

    if calibration_update.active:
        _draw_calibration_banner(cv2, preview, calibration_update)
    elif status_message:
        _blend_box(cv2, preview, (18, 52), (min(width - 18, 990), 98), 0.76)
        _put_text(cv2, preview, status_message, (34, 83), 0.60, (105, 255, 150), 2)

    if debug and not calibration_update.active and not status_message:
        debug_text = (
            "Guide: orange=source, cyan=full target, green=actual endpoint "
            f"| aim nudge x={target_nudge_x:+.3f}, y={target_nudge_y:+.3f}"
        )
        _put_text(cv2, preview, debug_text, (18, 62), 0.43, (80, 255, 255), 1)


_ARROW_DIRECTIONS: dict[int, tuple[float, float]] = {
    0x250000: (-1.0, 0.0),  # Windows HighGUI left
    0x260000: (0.0, -1.0),  # up
    0x270000: (1.0, 0.0),   # right
    0x280000: (0.0, 1.0),   # down
    65361: (-1.0, 0.0),     # Qt/X11-style fallbacks
    65362: (0.0, -1.0),
    65363: (1.0, 0.0),
    65364: (0.0, 1.0),
}


def _target_direction(raw_key: int, key: int) -> tuple[float, float] | None:
    if raw_key in _ARROW_DIRECTIONS:
        return _ARROW_DIRECTIONS[raw_key]
    mapping = {
        ord("j"): (-1.0, 0.0),
        ord("J"): (-1.0, 0.0),
        ord("l"): (1.0, 0.0),
        ord("L"): (1.0, 0.0),
        ord("i"): (0.0, -1.0),
        ord("I"): (0.0, -1.0),
        ord("k"): (0.0, 1.0),
        ord("K"): (0.0, 1.0),
    }
    return mapping.get(key)


def _is_strength_decrease(key: int) -> bool:
    return key in {ord("["), ord("-"), ord("_"), ord(","), ord("1")}


def _is_strength_increase(key: int) -> bool:
    return key in {ord("]"), ord("="), ord("+"), ord("."), ord("2")}


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError:
        return _dependency_error()

    from camera import Camera, CameraOpenError
    from config import (
        APP_NAME,
        DEFAULT_CORRECTION_STRENGTH,
        GAZE_EMA_ALPHA,
        GAZE_LOCK_COUNTDOWN_SECONDS,
        GAZE_LOCK_SAMPLE_COUNT,
        LANDMARK_EMA_ALPHA,
        MIN_PROCESSING_SCALE,
        MODEL_PATH,
        RuntimeSettings,
        STRENGTH_STEP,
        TARGET_NUDGE_STEP,
        WINDOW_NAME,
    )
    from face_tracker import FaceLandmarkerTracker
    from gaze_estimator import EyeLandmarks, GazeEstimator, smooth_gaze
    from gaze_lock import DynamicGazeLockCalibration
    from neural_eye_warper import NeuralEyeModelError, NeuralEyeWarper
    from smoothing import ArrayEMA, ScalarEMA

    if not 0.35 <= args.processing_scale <= 1.0:
        print("--processing-scale must be between 0.35 and 1.0.")
        return 2

    settings = RuntimeSettings(
        correction_strength=DEFAULT_CORRECTION_STRENGTH,
        camera_lift=float(args.camera_lift),
    )
    gaze_lock = DynamicGazeLockCalibration(
        countdown_seconds=GAZE_LOCK_COUNTDOWN_SECONDS,
        sample_count=GAZE_LOCK_SAMPLE_COUNT,
    )
    landmark_smoother = ArrayEMA(LANDMARK_EMA_ALPHA)
    gaze_smoother = ArrayEMA(GAZE_EMA_ALPHA)
    pose_smoother = ArrayEMA(0.58)
    fps_smoother = ScalarEMA(0.16)
    gaze_estimator = GazeEstimator()
    try:
        # Two CPU threads keep the renderer responsive on the target 4-core
        # laptop while MediaPipe continues to track the face.  No CUDA or
        # NVIDIA provider is requested here.
        neural_renderer = NeuralEyeWarper(cpu_threads=2)
    except NeuralEyeModelError as exc:
        print(
            "\nThe full-eye neural renderer is not ready.\n"
            f"{exc}\n\n"
            "Run setup.bat once, then start python main.py again.\n"
        )
        return 4

    try:
        camera = Camera.open(
            requested_index=args.camera,
            requested_width=args.width,
            requested_height=args.height,
            requested_fps=args.fps,
        )
    except CameraOpenError as exc:
        print(f"\n{exc}\n")
        return 3

    print(f"\n{APP_NAME} - Stage 1 local neural full-eye preview (no virtual camera)")
    print(f"Camera index: {camera.info.index}")
    print(f"Camera name: {camera.info.name}")
    print(f"Resolution: {camera.info.width}x{camera.info.height}")
    print(f"FPS reported by camera: {camera.info.fps:.1f}")
    print(f"Backend: {camera.info.backend}")
    print("Keys: C five-pose calibration, drag Strength slider or 1/2/-/+, arrows/IJKL fine tune, Q/Esc quit.\n")

    status_message = "Press C for personal five-pose calibration. Keep looking at the physical camera lens."
    status_until = time.monotonic() + 8.0
    last_time = time.perf_counter()
    slow_tracker_frames = 0
    current_processing_scale = float(args.processing_scale)
    inference_stride = 1
    frame_number = 0
    cached_observation: Any = None
    capture_ms_smoother = ScalarEMA(0.20)
    tracker_ms_smoother = ScalarEMA(0.20)
    render_ms_smoother = ScalarEMA(0.20)
    tracker_ms = 0.0
    render_ms = 0.0
    neural_renderer_available = True
    consecutive_frame_failures = 0
    last_calibration_update: Any = type(
        "IdleCalibration",
        (),
        {
            "active": False,
            "phase": "idle",
            "stage": "idle",
            "collected": 0,
            "required": GAZE_LOCK_SAMPLE_COUNT,
            "coverage_ok": False,
            "coverage_message": "",
        },
    )()

    try:
        with camera, FaceLandmarkerTracker(
            model_path=MODEL_PATH,
            processing_scale=args.processing_scale,
        ) as tracker:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            def _on_strength_slider(value: int) -> None:
                settings.set_strength(float(value) / 100.0)

            cv2.createTrackbar(
                "Strength % (drag mouse)",
                WINDOW_NAME,
                int(round(settings.correction_strength * 100)),
                100,
                _on_strength_slider,
            )

            def _set_strength(value: float) -> None:
                settings.set_strength(value)
                cv2.setTrackbarPos(
                    "Strength % (drag mouse)",
                    WINDOW_NAME,
                    int(round(settings.correction_strength * 100)),
                )

            while True:
                capture_started = time.perf_counter()
                frame = camera.read()
                capture_ms = capture_ms_smoother.update(
                    (time.perf_counter() - capture_started) * 1000.0
                )
                if frame is None:
                    status_message = "Camera frame unavailable - waiting..."
                    status_until = time.monotonic() + 2.0
                    consecutive_frame_failures += 1
                    raw_key = cv2.waitKeyEx(20)
                    key = raw_key & 0xFF if raw_key >= 0 else -1
                    if key in (27, ord("q"), ord("Q")):
                        break
                    if consecutive_frame_failures >= 90:
                        print("Camera stopped returning frames. Closing preview.")
                        break
                    continue
                consecutive_frame_failures = 0
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                original = frame.copy()
                corrected = frame.copy()
                frame_number += 1
                must_detect = (
                    gaze_lock.active
                    or cached_observation is None
                    or frame_number % inference_stride == 0
                )
                if must_detect:
                    tracker_started = time.perf_counter()
                    cached_observation = tracker.process(frame)
                    tracker_ms = tracker_ms_smoother.update(
                        (time.perf_counter() - tracker_started) * 1000.0
                    )
                observation = cached_observation
                gaze = None
                targets = None
                plans: Any = ()

                if observation is not None:
                    smoothed_eyes = tuple(
                        EyeLandmarks(
                            contour=landmark_smoother.update(
                                f"eye-{index}-contour", eye.contour
                            ),
                            iris=landmark_smoother.update(
                                f"eye-{index}-iris-points", eye.iris
                            ),
                        )
                        for index, eye in enumerate(observation.eyes)
                    )
                    estimate = gaze_estimator.estimate(smoothed_eyes)
                    if estimate is not None:
                        gaze = smooth_gaze(estimate, gaze_smoother)
                        pose = pose_smoother.update(
                            "head-pose", observation.pose.vector
                        )
                        last_calibration_update = gaze_lock.update(gaze, pose)
                        if last_calibration_update.completed_now:
                            coverage_detail = (
                                " Head movement coverage is small; press C again and turn/nod farther."
                                if not last_calibration_update.coverage_ok
                                else ""
                            )
                            renderer_state = (
                                "Neural full-eye correction is active."
                                if neural_renderer_available
                                else "Neural renderer is paused; original video remains visible."
                            )
                            status_message = (
                                "Personal camera-lens targets saved. "
                                + renderer_state
                                + coverage_detail
                            )
                            status_until = time.monotonic() + 6.0

                        targets = gaze_lock.targets_for(
                            gaze,
                            pose,
                            settings.camera_lift,
                            horizontal_nudge=settings.target_nudge_x,
                            vertical_nudge=settings.target_nudge_y,
                            use_eye_center=settings.aim_at_eye_center,
                        )
                        # Do not alter the preview while the person is moving
                        # into any of the five calibration poses.
                        if (
                            not gaze_lock.active
                            and targets is not None
                            and neural_renderer_available
                        ):
                            plans = neural_renderer.plan(
                                frame,
                                observation.landmarks,
                                gaze.eyes,
                                targets,
                                settings.correction_strength,
                                mirrored=not args.no_mirror,
                            )
                            if any(plan.applied for plan in plans):
                                rendered_started = time.perf_counter()
                                try:
                                    corrected = neural_renderer.apply(
                                        frame,
                                        observation.landmarks,
                                        gaze.eyes,
                                        plans,
                                        mirrored=not args.no_mirror,
                                    )
                                    render_ms = render_ms_smoother.update(
                                        (time.perf_counter() - rendered_started) * 1000.0
                                    )
                                except NeuralEyeModelError as exc:
                                    # A bad model or runtime is never replaced
                                    # with the old pupil-patch renderer.  Keep
                                    # the original frame visible and hold the
                                    # error until the user restarts after setup.
                                    neural_renderer_available = False
                                    plans = ()
                                    corrected = frame.copy()
                                    status_message = (
                                        "Neural rendering paused; original video is shown. "
                                        f"Restart after setup. ({str(exc).splitlines()[0]})"
                                    )
                                    status_until = time.monotonic() + 12.0
                    else:
                        last_calibration_update = gaze_lock.update(None, None)
                else:
                    last_calibration_update = gaze_lock.update(None, None)

                now = time.perf_counter()
                elapsed = max(now - last_time, 1e-6)
                last_time = now
                fps = fps_smoother.update(1.0 / elapsed)
                # Do not lower landmarks just because the webcam itself is
                # delivering 10 FPS in low light. The Read and Track timings
                # make that distinction visible, and only a slow tracker gets
                # an automatic inference-scale change.
                if tracker_ms > 55.0:
                    slow_tracker_frames += 1
                    if slow_tracker_frames >= 30:
                        if current_processing_scale > MIN_PROCESSING_SCALE:
                            current_processing_scale = max(
                                MIN_PROCESSING_SCALE,
                                round(current_processing_scale - 0.05, 2),
                            )
                            tracker.set_processing_scale(current_processing_scale)
                            status_message = (
                                "Performance safeguard: tracker scale reduced to "
                                f"{current_processing_scale:.2f}"
                            )
                            status_until = time.monotonic() + 5.0
                        elif inference_stride < 2 and not gaze_lock.active and tracker_ms > 65.0:
                            inference_stride = 2
                            status_message = (
                                "Performance safeguard: tracker now runs every 2 frames."
                            )
                            status_until = time.monotonic() + 5.0
                        slow_tracker_frames = 0
                else:
                    slow_tracker_frames = 0
                visible_status = status_message if time.monotonic() < status_until else ""

                display_original = original.copy()
                display_corrected = corrected.copy()
                if settings.debug:
                    _draw_debug(cv2, display_original, gaze, plans, corrected=False)
                    _draw_debug(cv2, display_corrected, gaze, plans, corrected=True)
                preview = _make_preview(
                    cv2,
                    display_original,
                    display_corrected,
                    settings.show_side_by_side,
                )
                _draw_footer(
                    cv2,
                    preview,
                    fps=fps,
                    strength=settings.correction_strength,
                    calibrated=gaze_lock.calibrated,
                    calibration_update=last_calibration_update,
                    status_message=visible_status,
                    debug=settings.debug,
                    aim_label="CENTER TEST" if settings.aim_at_eye_center else "CAMERA LENS",
                    target_nudge_x=settings.target_nudge_x,
                    target_nudge_y=settings.target_nudge_y,
                    plans=plans,
                    processing_scale=current_processing_scale,
                    inference_stride=inference_stride,
                    capture_ms=capture_ms,
                    tracker_ms=tracker_ms,
                    render_ms=render_ms,
                )
                cv2.imshow(WINDOW_NAME, preview)
                raw_key = cv2.waitKeyEx(1)
                key = raw_key & 0xFF if raw_key >= 0 else -1

                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("c"), ord("C")):
                    gaze_lock.start()
                    settings.aim_at_eye_center = False
                    settings.reset_target_nudge()
                    status_message = ""
                    landmark_smoother.clear()
                    gaze_smoother.clear()
                    pose_smoother.clear()
                    cached_observation = None
                elif key in (ord("r"), ord("R")):
                    gaze_lock.reset()
                    settings.aim_at_eye_center = False
                    settings.reset_target_nudge()
                    landmark_smoother.clear()
                    gaze_smoother.clear()
                    pose_smoother.clear()
                    cached_observation = None
                    status_message = "Calibration reset. Press C for the five-pose camera-lens calibration."
                    status_until = time.monotonic() + 5.0
                elif _is_strength_decrease(key):
                    _set_strength(settings.correction_strength - STRENGTH_STEP)
                    status_message = f"Strength set to {settings.correction_strength:.0%}"
                    status_until = time.monotonic() + 2.0
                elif _is_strength_increase(key):
                    _set_strength(settings.correction_strength + STRENGTH_STEP)
                    status_message = f"Strength set to {settings.correction_strength:.0%}"
                    status_until = time.monotonic() + 2.0
                elif key in (ord("g"), ord("G")):
                    settings.aim_at_eye_center = not settings.aim_at_eye_center
                    mode = "geometric pupil centre test" if settings.aim_at_eye_center else "camera lens"
                    status_message = f"Aim mode: {mode}"
                    status_until = time.monotonic() + 3.0
                elif key in (ord("t"), ord("T")):
                    settings.reset_target_nudge()
                    status_message = "Target nudge reset."
                    status_until = time.monotonic() + 2.0
                else:
                    direction = _target_direction(raw_key, key)
                    if direction is not None:
                        settings.nudge_target(
                            direction[0] * TARGET_NUDGE_STEP,
                            direction[1] * TARGET_NUDGE_STEP,
                        )
                        status_message = (
                            "Target nudge: "
                            f"x={settings.target_nudge_x:+.3f}, y={settings.target_nudge_y:+.3f}"
                        )
                        status_until = time.monotonic() + 2.5
                    elif key in (ord("s"), ord("S")):
                        settings.show_side_by_side = not settings.show_side_by_side
                    elif key in (ord("d"), ord("D")):
                        settings.debug = not settings.debug

    except FileNotFoundError as exc:
        print(f"\n{exc}\n")
        return 4
    except (RuntimeError, cv2.error) as exc:
        print(f"\nCould not start the local preview:\n{exc}\n")
        return 5
    finally:
        neural_renderer.close()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    relaunch_exit_code = _maybe_relaunch_in_project_environment()
    if relaunch_exit_code is not None:
        return relaunch_exit_code
    if not (
        importlib.util.find_spec("cv2")
        and importlib.util.find_spec("mediapipe")
        and importlib.util.find_spec("onnxruntime")
    ):
        return _dependency_error()
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
