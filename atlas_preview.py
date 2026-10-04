"""Live, CPU-only preview for the personal camera-lens eye reference atlas.

This deliberately runs as a separate validation window before it replaces the
older neural MVP.  Its right panel is built from real, locally captured frames
where this person was looking at the physical camera lens.  It never paints a
new pupil, and it keeps the live eyelids, lashes, skin and blink frames.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / "gaze-env" / "Scripts" / "python.exe"
MPL_CONFIG_DIR = PROJECT_DIR / "work" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


def _maybe_relaunch_in_project_environment() -> int | None:
    """Make a direct ``python atlas_preview.py`` command beginner-safe."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview real local camera-lens eye references against your live webcam."
    )
    parser.add_argument("--camera", type=int, default=None, help="Open one camera index directly.")
    parser.add_argument("--width", type=int, default=960, help="Preferred capture width.")
    parser.add_argument("--height", type=int, default=540, help="Preferred capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Preferred capture FPS.")
    parser.add_argument(
        "--processing-scale",
        type=float,
        default=0.55,
        help="Face Landmarker scale 0.35-1.0 (smaller is faster).",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Show the non-mirrored camera view.")
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


def _header(cv2: Any, image: Any, label: str) -> None:
    cv2.rectangle(image, (0, 0), (255, 42), (18, 18, 18), -1)
    _put_text(cv2, image, label, (14, 29), 0.68, (255, 255, 255), 2)


def _fit_preview(cv2: Any, image: Any) -> Any:
    height, width = image.shape[:2]
    scale = min(1.0, 1920.0 / width, 920.0 / height)
    if scale < 1.0:
        return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def _draw_footer(
    cv2: Any,
    image: Any,
    *,
    fps: float,
    strength: int,
    plans: Sequence[Any],
    status: str,
    show_debug: bool,
) -> None:
    height, width = image.shape[:2]
    footer_height = min(112, max(88, height // 5))
    overlay = image.copy()
    cv2.rectangle(overlay, (0, height - footer_height), (width, height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.84, image, 0.16, 0.0, image)
    applied = sum(1 for plan in plans if plan.applied)
    plan_text = f"{applied}/{len(plans)} real reference eyes" if plans else "waiting for eyes"
    _put_text(
        cv2,
        image,
        f"PERSONAL LENS ATLAS  {strength}%   FPS {fps:.1f}   {plan_text}",
        (16, height - footer_height + 28),
        0.59,
        (100, 255, 140) if applied else (255, 220, 90),
        2,
    )
    _put_text(
        cv2,
        image,
        status or "Look at a point left / right / above the screen: right panel should keep lens-looking eyes.",
        (16, height - footer_height + 57),
        0.43,
        (236, 236, 236),
        1,
    )
    help_text = "Drag Strength % | S compare/full view | D markers | Q / Esc quit"
    if show_debug:
        help_text += " | orange=current iris, green=real lens reference"
    _put_text(cv2, image, help_text, (16, height - 16), 0.42, (220, 220, 220), 1)


def _draw_debug(cv2: Any, original: Any, corrected: Any, gaze: Any, plans: Sequence[Any]) -> None:
    if gaze is None:
        return
    for index, eye in enumerate(gaze.eyes):
        contour = eye.contour.round().astype("int32")
        cv2.polylines(original, [contour], True, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.polylines(corrected, [contour], True, (0, 220, 255), 1, cv2.LINE_AA)
        source = tuple(eye.iris_center.round().astype(int))
        cv2.circle(original, source, 4, (0, 90, 255), -1, cv2.LINE_AA)
        if index < len(plans) and plans[index].applied:
            destination = tuple(plans[index].destination.round().astype(int))
            cv2.circle(corrected, destination, 4, (70, 255, 80), -1, cv2.LINE_AA)
        else:
            cv2.circle(corrected, source, 4, (0, 90, 255), -1, cv2.LINE_AA)


def run(args: argparse.Namespace) -> int:
    if not 0.35 <= args.processing_scale <= 1.0:
        print("--processing-scale must be between 0.35 and 1.0.")
        return 2

    import cv2
    import numpy as np

    from camera import Camera, CameraOpenError
    from config import MODEL_PATH
    from face_tracker import FaceLandmarkerTracker
    from gaze_estimator import GazeEstimator
    from reference_atlas import ReferenceAtlasError, ReferenceAtlasRenderer

    window_name = "Personal Lens Eye Contact Preview"
    try:
        atlas = ReferenceAtlasRenderer()
    except ReferenceAtlasError as exc:
        print(f"\n{exc}\nRun build_reference_atlas.bat first.\n")
        return 3
    if not atlas.ready:
        print("\nThe personal atlas does not contain both eyes. Run build_reference_atlas.bat again.\n")
        return 3

    gaze_estimator = GazeEstimator()
    show_side_by_side = True
    show_debug = False
    status = ""
    last_time = time.perf_counter()
    fps_value = 0.0
    try:
        with Camera.open(
            requested_index=args.camera,
            requested_width=args.width,
            requested_height=args.height,
            requested_fps=args.fps,
        ) as camera, FaceLandmarkerTracker(
            MODEL_PATH, processing_scale=args.processing_scale
        ) as tracker:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Strength %", window_name, 100, 100, lambda _value: None)
            print("Personal camera-lens atlas preview started. Press Q or Esc to exit.")
            while True:
                raw = camera.read()
                if raw is None:
                    raw_key = cv2.waitKeyEx(20)
                    if raw_key & 0xFF in (27, ord("q"), ord("Q")):
                        break
                    continue
                frame = raw if args.no_mirror else cv2.flip(raw, 1)
                original = frame.copy()
                corrected = frame.copy()
                observation = tracker.process(frame)
                gaze = None
                plans: tuple[Any, ...] = ()
                strength = cv2.getTrackbarPos("Strength %", window_name)
                if observation is None:
                    status = "Face not found: show both open eyes to the camera."
                else:
                    gaze = gaze_estimator.estimate(observation.eyes)
                    if gaze is None:
                        status = "Eyes not found clearly: use even front lighting and face the camera."
                    else:
                        # ``force=True`` is intentional in this preview.  It
                        # makes the real camera-lens reference visible even
                        # when the old 2-D gaze signal considers a target close
                        # to centre.  It is a visual validation mode, not the
                        # eventual meeting-camera policy.
                        targets = np.zeros((len(gaze.eyes), 2), dtype=np.float32)
                        plans = atlas.plan(
                            frame,
                            observation.landmarks,
                            gaze.eyes,
                            targets,
                            strength / 100.0,
                            observation.pose.vector,
                            mirrored=not args.no_mirror,
                            force=True,
                        )
                        if any(plan.applied for plan in plans):
                            corrected = atlas.apply(
                                frame,
                                observation.landmarks,
                                plans,
                                mirrored=not args.no_mirror,
                            )
                            status = "Using real local reference eyes recorded while you looked at the lens."
                        else:
                            reasons = ", ".join(plan.reason for plan in plans) or "no safe match"
                            status = "Safety fallback: " + reasons

                now = time.perf_counter()
                elapsed = max(1e-6, now - last_time)
                last_time = now
                instantaneous = 1.0 / elapsed
                fps_value = instantaneous if fps_value == 0.0 else fps_value * 0.82 + instantaneous * 0.18

                left = original.copy()
                right = corrected.copy()
                _header(cv2, left, "Original")
                _header(cv2, right, "Lens Reference")
                if show_debug:
                    _draw_debug(cv2, left, right, gaze, plans)
                preview = cv2.hconcat([left, right]) if show_side_by_side else right
                preview = _fit_preview(cv2, preview)
                _draw_footer(
                    cv2,
                    preview,
                    fps=fps_value,
                    strength=strength,
                    plans=plans,
                    status=status,
                    show_debug=show_debug,
                )
                cv2.imshow(window_name, preview)
                raw_key = cv2.waitKeyEx(1)
                key = raw_key & 0xFF if raw_key >= 0 else -1
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    show_side_by_side = not show_side_by_side
                elif key in (ord("d"), ord("D")):
                    show_debug = not show_debug
    except CameraOpenError as exc:
        print(f"\n{exc}\n")
        return 4
    except (RuntimeError, cv2.error) as exc:
        print(f"\nCould not start the personal lens preview:\n{exc}\n")
        return 5
    finally:
        atlas.close()
        try:
            cv2.destroyAllWindows()
        except UnboundLocalError:
            pass
    return 0


def main() -> int:
    relaunch_code = _maybe_relaunch_in_project_environment()
    if relaunch_code is not None:
        return relaunch_code
    if not all(importlib.util.find_spec(name) for name in ("cv2", "mediapipe", "numpy")):
        print("Missing project dependencies. Run setup.bat once, then start atlas_preview.bat.")
        return 1
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
