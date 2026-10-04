"""Record a private, local camera-lens reference sweep for the atlas prototype.

This tool does not upload, analyse, or send the video anywhere.  It records
the user's own natural eyes while they look at the physical webcam lens across
several comfortable head poses.  A later CPU-only reference-atlas renderer can
select a real, matching-pose eye frame instead of painting a pupil or asking a
small 2-D flow model to invent an eyelid.

Run through ``capture_reference.bat``.  Press Q or Escape to keep an explicitly
labelled partial recording and exit safely.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Final

import cv2

from camera import Camera, CameraOpenError
from config import APP_NAME, PROJECT_DIR


REFERENCE_DIR: Final[Path] = PROJECT_DIR / "reference_samples"
WINDOW_NAME: Final[str] = f"{APP_NAME} - Camera Lens Reference Capture"
MIN_COMPLETION_FRAMES_PER_POSE: Final[int] = 12


class ReferencePose:
    """One short, comfortable head-pose segment in the local capture sweep."""

    def __init__(self, key: str, instruction: str, chinese_hint: str) -> None:
        self.key = key
        self.instruction = instruction
        self.chinese_hint = chinese_hint


# The reference content is intentionally a pose sweep, not a request to move
# the eyes.  The person looks at the physical camera lens during every stage.
POSES: Final[tuple[ReferencePose, ...]] = (
    ReferencePose("center", "HEAD CENTERED", "头部居中"),
    ReferencePose("left", "TURN HEAD LEFT (YOUR LEFT)", "头向你自己的左侧"),
    ReferencePose("right", "TURN HEAD RIGHT (YOUR RIGHT)", "头向你自己的右侧"),
    ReferencePose("up", "TILT HEAD SLIGHTLY UP", "轻微抬头"),
    ReferencePose("down", "TILT HEAD SLIGHTLY DOWN", "轻微低头"),
    ReferencePose("upper_left", "UP AND LEFT (GENTLY)", "轻微抬头并向自己的左侧"),
    ReferencePose("upper_right", "UP AND RIGHT (GENTLY)", "轻微抬头并向自己的右侧"),
    ReferencePose("lower_left", "DOWN AND LEFT (GENTLY)", "轻微低头并向自己的左侧"),
    ReferencePose("lower_right", "DOWN AND RIGHT (GENTLY)", "轻微低头并向自己的右侧"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a local, private camera-lens reference sweep for the eye atlas."
    )
    parser.add_argument("--camera", type=int, default=None, help="Open one camera index.")
    parser.add_argument("--width", type=int, default=960, help="Preferred capture width.")
    parser.add_argument("--height", type=int, default=540, help="Preferred capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Preferred camera/video FPS.")
    parser.add_argument(
        "--countdown",
        type=float,
        default=2.0,
        help="Seconds to get into each pose before recording it (default: 2).",
    )
    parser.add_argument(
        "--seconds-per-pose",
        type=float,
        default=4.0,
        help="Seconds recorded at each pose while looking at the lens (default: 4).",
    )
    return parser.parse_args()


def _put_text(
    image: cv2.typing.MatLike,
    message: str,
    origin: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw high-contrast text over a webcam preview."""

    cv2.putText(
        image,
        message,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (12, 12, 12),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        message,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _preview(
    frame: cv2.typing.MatLike,
    pose: ReferencePose,
    pose_index: int,
    *,
    seconds_remaining: float,
    recording: bool,
    total_poses: int,
) -> cv2.typing.MatLike:
    """Show an instruction-only mirrored preview; raw frames remain unmodified."""

    preview = cv2.flip(frame, 1)
    height, width = preview.shape[:2]
    overlay = preview.copy()
    cv2.rectangle(overlay, (0, 0), (width, 115), (10, 10, 10), -1)
    cv2.rectangle(overlay, (0, height - 56), (width, height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.78, preview, 0.22, 0.0, preview)

    phase = "RECORDING" if recording else "GET READY"
    phase_colour = (80, 255, 120) if recording else (70, 230, 255)
    _put_text(
        preview,
        f"{phase}  {pose_index + 1}/{total_poses}",
        (22, 37),
        0.78,
        phase_colour,
        2,
    )
    _put_text(preview, "LOOK AT THE PHYSICAL CAMERA LENS", (22, 70), 0.64, (255, 255, 255), 2)
    _put_text(preview, pose.instruction, (22, 103), 0.58, (255, 245, 100), 2)
    _put_text(
        preview,
        f"{seconds_remaining:0.1f}s   Q / Esc: stop and keep partial recording",
        (22, height - 20),
        0.48,
        (235, 235, 235),
        1,
    )
    return preview


def _open_writer(
    stem: str,
    frame_size: tuple[int, int],
    fps: float,
) -> tuple[cv2.VideoWriter, Path]:
    """Create a broadly compatible local video writer with an AVI fallback."""

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    candidates = (
        (REFERENCE_DIR / f"{stem}.mp4", "mp4v"),
        (REFERENCE_DIR / f"{stem}.avi", "MJPG"),
    )
    for path, codec in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            max(1.0, float(fps)),
            frame_size,
        )
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError(
        "OpenCV could not create a local MP4 or AVI recording. "
        "Update the camera/OpenCV driver, then try capture_reference.bat again."
    )


def _write_metadata(
    path: Path,
    *,
    complete: bool,
    created_at: str,
    video_path: Path | None,
    camera_info: object | None,
    args: argparse.Namespace,
    manifest: list[dict[str, object]],
) -> Path:
    """Save pose boundaries and frame indexes beside the private local video."""

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".json")
    document = {
        "format": "gaze-reference-sweep-v1",
        "complete": complete,
        "created_at": created_at,
        "video_file": video_path.name if video_path is not None else None,
        "preview_mirrored": True,
        "recorded_video_mirrored": False,
        "camera": asdict(camera_info) if camera_info is not None else None,
        "settings": {
            "countdown_seconds": args.countdown,
            "seconds_per_pose": args.seconds_per_pose,
            "requested_fps": args.fps,
            "requested_width": args.width,
            "requested_height": args.height,
        },
        "poses": [
            {"key": pose.key, "instruction": pose.instruction, "hint_zh": pose.chinese_hint}
            for pose in POSES
        ],
        # Frame indexes make the sweep useful even if the webcam's true frame
        # rate differs from its reported media rate.
        "frames": manifest,
    }
    metadata_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata_path


def record(args: argparse.Namespace) -> int:
    if args.width < 160 or args.height < 120 or args.fps < 1:
        print("Width/height must be at least 160x120 and FPS must be positive.")
        return 2
    if args.countdown < 0.0 or args.seconds_per_pose <= 0.0:
        print("--countdown must be non-negative and --seconds-per-pose must be positive.")
        return 2

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"camera_lens_reference_{stamp}"
    writer: cv2.VideoWriter | None = None
    video_path: Path | None = None
    camera_info: object | None = None
    manifest: list[dict[str, object]] = []
    complete = False
    frame_index = 0
    stopped_by_user = False

    try:
        with Camera.open(
            requested_index=args.camera,
            requested_width=args.width,
            requested_height=args.height,
            requested_fps=args.fps,
        ) as camera:
            camera_info = camera.info
            print("\nPrivate local reference capture started.")
            print("Look at the PHYSICAL camera lens throughout every pose.")
            print("Do not follow your eyes in the preview. Press Q or Esc to stop safely.\n")
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            sweep_started = time.monotonic()

            for pose_index, pose in enumerate(POSES):
                for recording, phase_seconds in (
                    (False, float(args.countdown)),
                    (True, float(args.seconds_per_pose)),
                ):
                    phase_started = time.monotonic()
                    while True:
                        now = time.monotonic()
                        elapsed = now - phase_started
                        remaining = max(0.0, phase_seconds - elapsed)
                        if elapsed >= phase_seconds:
                            break
                        frame = camera.read()
                        if frame is None:
                            # Keep the stage timer moving; a brief bad webcam
                            # frame should not turn a guided capture into an
                            # endless loop.
                            raw_key = cv2.waitKeyEx(10)
                            if raw_key & 0xFF in (27, ord("q"), ord("Q")):
                                stopped_by_user = True
                                break
                            continue
                        if recording:
                            if writer is None:
                                height, width = frame.shape[:2]
                                capture_fps = (
                                    camera.info.fps
                                    if 1.0 <= camera.info.fps <= 120.0
                                    else float(args.fps)
                                )
                                writer, video_path = _open_writer(
                                    stem, (width, height), capture_fps
                                )
                            writer.write(frame)
                            manifest.append(
                                {
                                    "frame": frame_index,
                                    "pose": pose.key,
                                    "time_since_start_seconds": round(now - sweep_started, 3),
                                }
                            )
                            frame_index += 1
                        cv2.imshow(
                            WINDOW_NAME,
                            _preview(
                                frame,
                                pose,
                                pose_index,
                                seconds_remaining=remaining,
                                recording=recording,
                                total_poses=len(POSES),
                            ),
                        )
                        raw_key = cv2.waitKeyEx(1)
                        if raw_key & 0xFF in (27, ord("q"), ord("Q")):
                            stopped_by_user = True
                            break
                    if stopped_by_user:
                        break
                if stopped_by_user:
                    break
            pose_counts = {
                pose.key: sum(1 for item in manifest if item.get("pose") == pose.key)
                for pose in POSES
            }
            complete = not stopped_by_user and all(
                count >= MIN_COMPLETION_FRAMES_PER_POSE for count in pose_counts.values()
            )
    except CameraOpenError as exc:
        print(f"\n{exc}\n")
        return 3
    except (RuntimeError, cv2.error) as exc:
        print(f"\nCould not record the local reference sweep:\n{exc}\n")
        return 4
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    metadata_anchor = video_path or (REFERENCE_DIR / stem)
    metadata_path = _write_metadata(
        metadata_anchor,
        complete=complete,
        created_at=created_at,
        video_path=video_path,
        camera_info=camera_info,
        args=args,
        manifest=manifest,
    )
    if complete and video_path is not None:
        print("\nReference sweep completed locally.")
        print(f"Video:    {video_path}")
        print(f"Pose map: {metadata_path}")
        print("The video stays on this computer. Do not upload it unless you choose to share it.")
        return 0

    print("\nReference sweep stopped before completion or did not contain enough frames in every pose.")
    print("The available partial files were kept and will not be selected for the atlas.")
    if video_path is not None:
        print(f"Partial video: {video_path}")
    print(f"Pose map:      {metadata_path}")
    return 1


def main() -> int:
    return record(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
