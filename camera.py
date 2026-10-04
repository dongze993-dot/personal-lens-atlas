"""Windows-friendly OpenCV camera probing with a small, predictable scope."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Iterable

import cv2
import numpy as np

from config import CAMERA_INDICES, PREFERRED_FPS, PREFERRED_RESOLUTIONS


class CameraOpenError(RuntimeError):
    """Raised when none of the safe camera-index attempts yields a frame."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    width: int
    height: int
    fps: float
    backend: str


def _windows_camera_names() -> list[str]:
    """Ask Windows for a best-effort friendly device name, without a dependency.

    OpenCV's Python API exposes an index but no portable camera-name function.
    PnPUtil is bundled with current Windows versions.  If its localized output
    cannot be parsed, returning an empty list is fine; capture still works.
    """

    if os.name != "nt":
        return []
    try:
        completed = subprocess.run(
            ["pnputil", "/enum-devices", "/class", "Camera"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return []

    labels = (
        "device description",
        "friendly name",
        "device name",
        "设备描述",
        "友好名称",
        "设备名称",
    )
    names: list[str] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        lowered = line.casefold()
        if ":" not in line or not any(label in lowered for label in labels):
            continue
        value = line.split(":", 1)[1].strip()
        if value and value not in names:
            names.append(value)
    return names


class Camera:
    """Owns an OpenCV VideoCapture and releases it reliably."""

    def __init__(self, capture: cv2.VideoCapture, info: CameraInfo) -> None:
        self._capture = capture
        self.info = info

    @classmethod
    def open(
        cls,
        requested_index: int | None = None,
        requested_width: int = PREFERRED_RESOLUTIONS[0][0],
        requested_height: int = PREFERRED_RESOLUTIONS[0][1],
        requested_fps: int = PREFERRED_FPS,
    ) -> "Camera":
        """Open the first camera that can actually return a frame.

        DirectShow avoids the long first-frame delay frequently seen with
        Media Foundation on older integrated cameras.  We still fall back to
        OpenCV's default backend in case DirectShow is unavailable.
        """

        indices: Iterable[int] = (
            (requested_index,) if requested_index is not None else CAMERA_INDICES
        )
        friendly_names = _windows_camera_names()
        attempts: list[str] = []

        backend_candidates: list[tuple[str, int]] = []
        if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
            backend_candidates.append(("DirectShow", cv2.CAP_DSHOW))
        backend_candidates.append(("OpenCV default", cv2.CAP_ANY))

        for index in indices:
            for backend_name, backend in backend_candidates:
                capture = cv2.VideoCapture(int(index), backend)
                if not capture.isOpened():
                    attempts.append(f"index {index} via {backend_name}: cannot open")
                    capture.release()
                    continue

                frame, selected_resolution = cls._configure_and_read(
                    capture,
                    requested_width,
                    requested_height,
                    requested_fps,
                )
                if frame is None:
                    attempts.append(f"index {index} via {backend_name}: no frames")
                    capture.release()
                    continue

                actual_height, actual_width = frame.shape[:2]
                reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                # Names are not reliably linked to OpenCV indices.  Expose the
                # list as informational rather than falsely claiming a match.
                display_name = (
                    f"OpenCV index {index}; Windows camera list: "
                    + ", ".join(friendly_names)
                    if friendly_names
                    else f"OpenCV camera at index {index}"
                )
                info = CameraInfo(
                    index=int(index),
                    name=display_name,
                    width=int(actual_width),
                    height=int(actual_height),
                    fps=reported_fps,
                    backend=f"{backend_name}; requested {selected_resolution[0]}x{selected_resolution[1]}",
                )
                return cls(capture, info)

        details = "\n  ".join(attempts) or "No OpenCV backend could be started."
        raise CameraOpenError(
            "Could not open a webcam. Attempts:\n"
            f"  {details}\n"
            "Check that another app is not using the camera, then enable "
            "Settings > Privacy & security > Camera > Let desktop apps access "
            "your camera."
        )

    @staticmethod
    def _configure_and_read(
        capture: cv2.VideoCapture,
        requested_width: int,
        requested_height: int,
        requested_fps: int,
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        # First honour a user-requested resolution, then the documented 720p
        # and 480p fallbacks.  Duplicates are removed without changing order.
        candidates: list[tuple[int, int]] = []
        for resolution in (
            (int(requested_width), int(requested_height)),
            *PREFERRED_RESOLUTIONS,
        ):
            if resolution not in candidates:
                candidates.append(resolution)

        capture.set(cv2.CAP_PROP_FPS, float(requested_fps))
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for width, height in candidates:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            frame = Camera._read_warm_frame(capture)
            if frame is not None and frame.size:
                return frame, (width, height)
        return None, candidates[-1]

    @staticmethod
    def _read_warm_frame(capture: cv2.VideoCapture) -> np.ndarray | None:
        # A couple of discarded frames let inexpensive integrated webcams
        # settle after resolution negotiation.
        frame: np.ndarray | None = None
        for _ in range(5):
            ok, candidate = capture.read()
            if ok and candidate is not None and candidate.size:
                frame = candidate
        return frame

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        return frame if ok and frame is not None and frame.size else None

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
