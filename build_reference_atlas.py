"""Build a local, personal camera-lens eye atlas from a reference sweep.

The recorder stores every frame together with its guided head-pose label.  It
does not store the actual recording FPS as a trustworthy timing signal, so
this builder deliberately uses the JSON frame indexes rather than video time.
All output stays below ``reference_atlas`` in this project folder.

This is an offline preparation step.  The later preview uses only a small set
of selected eye crops and never needs to read the private source video again.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from capture_reference import POSES
from config import DEFAULT_PROCESSING_SCALE, MODEL_PATH, PROJECT_DIR
from face_tracker import FaceLandmarkerTracker
from reference_atlas import (
    ACTIVE_ATLAS_FILENAME,
    ATLAS_FORMAT,
    MIN_EYE_WIDTH,
    MIN_OPENNESS,
    aperture_alpha,
    canonical_eye_geometries,
    default_atlas_root,
    eye_brightness,
    eye_crop_bounds,
    eye_sharpness,
    _point_clearance,
)


REFERENCE_DIR = PROJECT_DIR / "reference_samples"
REQUIRED_POSE_KEYS = tuple(pose.key for pose in POSES)
ACTIVE_FORMAT = "personal-eye-atlas-active-v1"


class AtlasBuildError(RuntimeError):
    """A local recording cannot safely become a lens-reference atlas."""


@dataclass(frozen=True)
class ReferenceSweep:
    """A checked capture JSON and the frame-to-pose table beside its video."""

    metadata_path: Path
    video_path: Path
    frame_poses: Mapping[int, str]
    pose_counts: Mapping[str, int]


@dataclass(frozen=True)
class EyeCandidate:
    """One reliable eye crop from the private reference recording."""

    pose_key: str
    side: str
    frame_index: int
    pose: np.ndarray
    image: np.ndarray
    contour: np.ndarray
    iris_center: np.ndarray
    center: np.ndarray
    horizontal_axis: np.ndarray
    vertical_axis: np.ndarray
    eye_width: float
    eye_height: float
    openness: float
    iris_ratio: np.ndarray
    sharpness: float
    brightness: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a private personal lens-eye atlas from a completed local capture."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional capture JSON file. By default the newest complete capture is used.",
    )
    parser.add_argument(
        "--keep-per-eye",
        type=int,
        default=3,
        help="Number of real reference crops retained for each pose and eye (1-5).",
    )
    parser.add_argument(
        "--processing-scale",
        type=float,
        default=DEFAULT_PROCESSING_SCALE,
        help="MediaPipe scale used while building, between 0.35 and 1.0.",
    )
    parser.add_argument(
        "--min-frames-per-pose",
        type=int,
        default=12,
        help="Minimum recorded frames required in every guided pose.",
    )
    return parser.parse_args()


def _read_document(path: Path) -> Mapping[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, Mapping) else None


def _sweep_from_metadata(path: Path, min_frames_per_pose: int) -> ReferenceSweep | None:
    document = _read_document(path)
    if document is None or document.get("format") != "gaze-reference-sweep-v1":
        return None
    if document.get("complete") is not True:
        return None
    video_file = document.get("video_file")
    frames = document.get("frames")
    if not isinstance(video_file, str) or not video_file or not isinstance(frames, list):
        return None
    video_path = path.parent / video_file
    if not video_path.is_file() or video_path.stat().st_size < 100_000:
        return None

    frame_poses: dict[int, str] = {}
    pose_counts = {key: 0 for key in REQUIRED_POSE_KEYS}
    for item in frames:
        if not isinstance(item, Mapping):
            continue
        frame_index = item.get("frame")
        pose_key = item.get("pose")
        if (
            isinstance(frame_index, int)
            and frame_index >= 0
            and isinstance(pose_key, str)
            and pose_key in pose_counts
            and frame_index not in frame_poses
        ):
            frame_poses[frame_index] = pose_key
            pose_counts[pose_key] += 1
    if not frame_poses or any(count < min_frames_per_pose for count in pose_counts.values()):
        return None
    return ReferenceSweep(path, video_path, frame_poses, pose_counts)


def discover_sweep(source: Path | None, min_frames_per_pose: int) -> ReferenceSweep:
    """Pick the newest complete usable local sweep, never a partial test."""

    if source is not None:
        source_path = source.expanduser().resolve()
        if source_path.suffix.casefold() != ".json":
            raise AtlasBuildError("--source must name the capture JSON file next to its video.")
        result = _sweep_from_metadata(source_path, min_frames_per_pose)
        if result is None:
            raise AtlasBuildError(
                "That capture is incomplete or lacks enough frames in every one of the nine poses."
            )
        return result

    if not REFERENCE_DIR.is_dir():
        raise AtlasBuildError("No reference_samples folder exists. Run capture_reference.bat first.")
    metadata_files = sorted(
        REFERENCE_DIR.glob("camera_lens_reference_*.json"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for candidate in metadata_files:
        result = _sweep_from_metadata(candidate, min_frames_per_pose)
        if result is not None:
            return result
    raise AtlasBuildError(
        "No complete reference recording was found. Run capture_reference.bat again and finish all nine poses."
    )


def _localize_geometry(
    frame: np.ndarray,
    geometry: Any,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Extract a crop and serializable eye geometry in its local coordinates."""

    if geometry.width < MIN_EYE_WIDTH or geometry.openness < MIN_OPENNESS:
        return None
    height, width = frame.shape[:2]
    bounds = eye_crop_bounds(geometry, width, height)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    crop = frame[top:bottom, left:right].copy()
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 18:
        return None
    origin = np.asarray((left, top), dtype=np.float32)
    contour = geometry.contour - origin
    iris_center = geometry.iris_center - origin
    center = geometry.center - origin
    mask = aperture_alpha(crop.shape[:2], contour, geometry.height)
    if mask is None:
        return None
    if _point_clearance(contour, iris_center) < max(0.8, geometry.height * 0.12):
        return None
    sharpness = eye_sharpness(crop)
    brightness = eye_brightness(crop)
    # Very dark, blown-out, or nearly featureless frames are bad donors even
    # if MediaPipe managed to find landmarks on them.
    if sharpness < 10.0 or not 28.0 <= brightness <= 228.0:
        return None
    return crop, {
        "contour": contour,
        "iris_center": iris_center,
        "center": center,
        "horizontal_axis": geometry.horizontal_axis,
        "vertical_axis": geometry.vertical_axis,
        "eye_width": geometry.width,
        "eye_height": geometry.height,
        "openness": geometry.openness,
        "iris_ratio": geometry.iris_ratio,
        "sharpness": sharpness,
        "brightness": brightness,
    }


def extract_candidates(sweep: ReferenceSweep, processing_scale: float) -> dict[tuple[str, str], list[EyeCandidate]]:
    """Run MediaPipe once over the local video and keep only trustworthy eyes."""

    capture = cv2.VideoCapture(str(sweep.video_path))
    if not capture.isOpened():
        raise AtlasBuildError(f"Could not open recorded video: {sweep.video_path.name}")
    grouped: dict[tuple[str, str], list[EyeCandidate]] = {
        (pose_key, side): [] for pose_key in REQUIRED_POSE_KEYS for side in ("R", "L")
    }
    processed = 0
    detected = 0
    try:
        with FaceLandmarkerTracker(MODEL_PATH, processing_scale=processing_scale) as tracker:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    break
                pose_key = sweep.frame_poses.get(frame_index)
                if pose_key is not None:
                    # The capture preview was mirrored only for the person;
                    # the saved frame is deliberately raw.  Keep it raw here:
                    # the live preview is canonicalized back to this same
                    # orientation before a donor crop is aligned.
                    processed += 1
                    observation = tracker.process(frame)
                    if observation is not None:
                        detected += 1
                        geometries = canonical_eye_geometries(observation.landmarks)
                        for side, geometry in geometries.items():
                            localized = _localize_geometry(frame, geometry)
                            if localized is None:
                                continue
                            crop, data = localized
                            roll = float(np.arctan2(geometry.horizontal_axis[1], geometry.horizontal_axis[0]))
                            grouped[(pose_key, side)].append(
                                EyeCandidate(
                                    pose_key=pose_key,
                                    side=side,
                                    frame_index=frame_index,
                                    pose=np.asarray(
                                        (observation.pose.yaw, observation.pose.pitch, roll),
                                        dtype=np.float32,
                                    ),
                                    image=crop,
                                    contour=np.asarray(data["contour"], dtype=np.float32),
                                    iris_center=np.asarray(data["iris_center"], dtype=np.float32),
                                    center=np.asarray(data["center"], dtype=np.float32),
                                    horizontal_axis=np.asarray(data["horizontal_axis"], dtype=np.float32),
                                    vertical_axis=np.asarray(data["vertical_axis"], dtype=np.float32),
                                    eye_width=float(data["eye_width"]),
                                    eye_height=float(data["eye_height"]),
                                    openness=float(data["openness"]),
                                    iris_ratio=np.asarray(data["iris_ratio"], dtype=np.float32),
                                    sharpness=float(data["sharpness"]),
                                    brightness=float(data["brightness"]),
                                )
                            )
                frame_index += 1
    except FileNotFoundError as exc:
        raise AtlasBuildError(str(exc)) from exc
    except (RuntimeError, ValueError, cv2.error) as exc:
        raise AtlasBuildError(f"MediaPipe could not analyse the local recording: {exc}") from exc
    finally:
        capture.release()

    print(f"Analysed {processed} labelled frames; MediaPipe found a face in {detected}.")
    return grouped


def _candidate_score(candidate: EyeCandidate, candidates: Sequence[EyeCandidate]) -> float:
    """Prefer clear, stable lens-looking frames without inventing eye pixels."""

    iris_median = np.median(np.asarray([item.iris_ratio for item in candidates]), axis=0)
    pose_median = np.median(np.asarray([item.pose for item in candidates]), axis=0)
    open_median = float(np.median([item.openness for item in candidates]))
    brightness_median = float(np.median([item.brightness for item in candidates]))
    sharpness_values = np.asarray([item.sharpness for item in candidates], dtype=np.float32)
    sharpness_low = float(np.percentile(sharpness_values, 15))
    sharpness_high = max(sharpness_low + 1e-4, float(np.percentile(sharpness_values, 90)))
    sharpness_bonus = float(np.clip((candidate.sharpness - sharpness_low) / (sharpness_high - sharpness_low), 0.0, 1.0))
    iris_distance = float(np.linalg.norm((candidate.iris_ratio - iris_median) / np.asarray((0.040, 0.040))))
    pose_distance = float(
        np.linalg.norm((candidate.pose - pose_median) / np.asarray((0.045, 0.045, 0.16)))
    )
    open_distance = abs(candidate.openness - open_median) / 0.070
    brightness_distance = abs(candidate.brightness - brightness_median) / 32.0
    return (
        1.10 * iris_distance
        + 0.26 * pose_distance
        + 0.28 * open_distance
        + 0.18 * brightness_distance
        - 0.62 * sharpness_bonus
    )


def choose_candidates(candidates: Sequence[EyeCandidate], keep: int) -> list[EyeCandidate]:
    """Select a few temporally distinct real eye frames for one cell."""

    if not candidates:
        return []
    scored = sorted(((_candidate_score(candidate, candidates), candidate) for candidate in candidates), key=lambda item: item[0])
    chosen: list[EyeCandidate] = []
    for _, candidate in scored:
        # Adjacent frames contain nearly identical webcam pixels.  Distinct
        # samples give the runtime selector a little natural variation while
        # still retaining only genuine camera-lens frames.
        if all(abs(candidate.frame_index - old.frame_index) >= 6 for old in chosen):
            chosen.append(candidate)
        if len(chosen) >= keep:
            break
    if not chosen:
        chosen.append(scored[0][1])
    return chosen


def _serializable_record(candidate: EyeCandidate, record_id: str, image_file: str) -> dict[str, Any]:
    return {
        "id": record_id,
        "pose_key": candidate.pose_key,
        "side": candidate.side,
        "frame_index": candidate.frame_index,
        "pose": [round(float(value), 7) for value in candidate.pose],
        "image_file": image_file,
        "contour": [[round(float(x), 5), round(float(y), 5)] for x, y in candidate.contour],
        "iris_center": [round(float(value), 5) for value in candidate.iris_center],
        "center": [round(float(value), 5) for value in candidate.center],
        "horizontal_axis": [round(float(value), 7) for value in candidate.horizontal_axis],
        "vertical_axis": [round(float(value), 7) for value in candidate.vertical_axis],
        "eye_width": round(candidate.eye_width, 5),
        "eye_height": round(candidate.eye_height, 5),
        "openness": round(candidate.openness, 7),
        "sharpness": round(candidate.sharpness, 4),
        "brightness": round(candidate.brightness, 4),
    }


def write_atlas(
    sweep: ReferenceSweep,
    grouped: Mapping[tuple[str, str], Sequence[EyeCandidate]],
    keep_per_eye: int,
) -> tuple[Path, dict[tuple[str, str], int]]:
    """Write selected local PNG crops, manifest, and atomically switch active atlas."""

    root = default_atlas_root()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "runs" / f"atlas_{run_stamp}"
    records_dir = run_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    coverage: dict[tuple[str, str], int] = {}

    for pose_key in REQUIRED_POSE_KEYS:
        for side in ("R", "L"):
            selected = choose_candidates(tuple(grouped.get((pose_key, side), ())), keep_per_eye)
            coverage[(pose_key, side)] = len(selected)
            for rank, candidate in enumerate(selected, start=1):
                record_id = f"{pose_key}_{side}_{rank}"
                image_name = f"{record_id}.png"
                image_path = records_dir / image_name
                if not cv2.imwrite(str(image_path), candidate.image):
                    raise AtlasBuildError(f"Could not save local eye crop: {image_path.name}")
                records.append(
                    _serializable_record(candidate, record_id, f"records/{image_name}")
                )

    if any(count == 0 for count in coverage.values()):
        missing = ", ".join(
            f"{pose}/{side}" for (pose, side), count in coverage.items() if count == 0
        )
        raise AtlasBuildError(
            "The recording did not yield a usable open eye in: " + missing + ". "
            "Use even front lighting, look at the lens, then record another sweep."
        )

    manifest = {
        "format": ATLAS_FORMAT,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_capture": sweep.metadata_path.name,
        "source_video": sweep.video_path.name,
        "source_frame_counts": dict(sweep.pose_counts),
        "records": records,
    }
    atlas_path = run_dir / "atlas.json"
    atlas_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    active_path = root / ACTIVE_ATLAS_FILENAME
    active_tmp = active_path.with_suffix(".tmp")
    pointer = {
        "format": ACTIVE_FORMAT,
        "atlas_file": atlas_path.relative_to(root).as_posix(),
        "created_at": manifest["created_at"],
        "source_capture": sweep.metadata_path.name,
    }
    active_tmp.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    active_tmp.replace(active_path)
    return atlas_path, coverage


def build(args: argparse.Namespace) -> int:
    if not 1 <= args.keep_per_eye <= 5:
        print("--keep-per-eye must be between 1 and 5.")
        return 2
    if not 0.35 <= args.processing_scale <= 1.0:
        print("--processing-scale must be between 0.35 and 1.0.")
        return 2
    if args.min_frames_per_pose < 4:
        print("--min-frames-per-pose must be at least 4.")
        return 2
    try:
        sweep = discover_sweep(args.source, args.min_frames_per_pose)
        pose_text = ", ".join(f"{key}={sweep.pose_counts[key]}" for key in REQUIRED_POSE_KEYS)
        print("Using local reference recording:", sweep.video_path.name)
        print("Frames by pose:", pose_text)
        print("Building personal lens-eye atlas locally. This normally takes under a minute...")
        grouped = extract_candidates(sweep, args.processing_scale)
        atlas_path, coverage = write_atlas(sweep, grouped, args.keep_per_eye)
    except AtlasBuildError as exc:
        print(f"\nCould not build the personal eye atlas:\n{exc}\n")
        return 3
    except (OSError, cv2.error) as exc:
        print(f"\nCould not save the personal eye atlas:\n{exc}\n")
        return 4

    selected = sum(coverage.values())
    print("\nPersonal camera-lens atlas is ready.")
    print(f"Selected {selected} real eye crops from this computer only.")
    print(f"Atlas: {atlas_path}")
    print("Next: run atlas_preview.bat to compare Original with the real-reference correction.")
    return 0


def main() -> int:
    return build(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
