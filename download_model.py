"""Download the official MediaPipe Face Landmarker model atomically."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

from config import MODEL_PATH, MODEL_URL


def ensure_model(force: bool = False) -> Path:
    """Return a usable model path, downloading only when it is missing."""

    destination = MODEL_PATH
    if destination.is_file() and destination.stat().st_size > 1_000_000 and not force:
        print(f"MediaPipe model already present: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the official MediaPipe Face Landmarker model...")
    request = Request(MODEL_URL, headers={"User-Agent": "gaze-correction-mvp/1.0"})
    temporary_name: str | None = None
    try:
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=destination.parent,
                prefix="face_landmarker.",
                suffix=".part",
            ) as temporary:
                temporary_name = temporary.name
                shutil.copyfileobj(response, temporary)
        downloaded = Path(temporary_name)
        if downloaded.stat().st_size <= 1_000_000:
            raise RuntimeError("The downloaded model is unexpectedly small.")
        downloaded.replace(destination)
    except Exception:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
        raise

    print(f"Model saved locally: {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download again.")
    args = parser.parse_args()
    ensure_model(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
