"""Download local full-eye gaze models during optional legacy setup only.

The Personal Lens Atlas application never calls this module at normal startup.
It is invoked only by ``setup.bat --experimental-neural`` after the user has
reviewed the third-party notice. Once acquired, both ONNX files live under
``models/neural`` and the legacy preview can start without an Internet
connection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import ssl
import tempfile
import time
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "neural"
# Current upstream files are about 1.06 MB.  One megabyte rejects HTML/JSON
# error bodies and most interrupted transfers without depending on a brittle
# file hash from an unversioned public branch.
MIN_MODEL_BYTES = 1_000_000
# The response curve is tiny JSON, but it must contain both model digests and
# four non-empty axis curves.  The structural validator below is the real
# integrity check; this lower bound just rejects a short proxy error page.
MIN_RESPONSE_CURVE_BYTES = 200
RESPONSE_CURVE_FILENAME = "response_curve.json"
USER_AGENT = "gaze-correction-mvp/1.0"
# The files are about 1 MB; a short timeout gets the user to the fallback
# source quickly if a corporate network blocks one GitHub hostname.
REQUEST_TIMEOUT_SECONDS = 30
ATTEMPTS_PER_SOURCE = 2
RETRY_DELAY_SECONDS = 1.0
NETWORK_ERRORS = (
    URLError,
    TimeoutError,
    ConnectionError,
    ssl.SSLError,
    http.client.HTTPException,
)


@dataclass(frozen=True)
class NeuralModel:
    """One required locally cached ONNX model."""

    filename: str
    urls: tuple[str, ...]

    @property
    def destination(self) -> Path:
        return MODEL_DIR / self.filename


MODELS = (
    NeuralModel(
        "gaze_L.onnx",
        (
            "https://raw.githubusercontent.com/KypMon/coreml-eye-contact/"
            "refs/heads/master/neural/gaze_L.onnx",
            "https://github.com/KypMon/coreml-eye-contact/raw/"
            "refs/heads/master/neural/gaze_L.onnx",
            "https://api.github.com/repos/KypMon/coreml-eye-contact/contents/"
            "neural/gaze_L.onnx",
        ),
    ),
    NeuralModel(
        "gaze_R.onnx",
        (
            "https://raw.githubusercontent.com/KypMon/coreml-eye-contact/"
            "refs/heads/master/neural/gaze_R.onnx",
            "https://github.com/KypMon/coreml-eye-contact/raw/"
            "refs/heads/master/neural/gaze_R.onnx",
            "https://api.github.com/repos/KypMon/coreml-eye-contact/contents/"
            "neural/gaze_R.onnx",
        ),
    ),
)

RESPONSE_CURVE_URLS = (
    "https://raw.githubusercontent.com/KypMon/coreml-eye-contact/"
    "refs/heads/master/neural/response_curve.json",
    "https://github.com/KypMon/coreml-eye-contact/raw/"
    "refs/heads/master/neural/response_curve.json",
    "https://api.github.com/repos/KypMon/coreml-eye-contact/contents/"
    "neural/response_curve.json",
)


class NeuralModelDownloadError(RuntimeError):
    """A neural model could not be downloaded or validated."""


def is_usable_model(path: Path) -> bool:
    """Return whether a file is large enough to be an ONNX model, not an error page."""

    try:
        return path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES
    except OSError:
        return False


def _file_md5(path: Path) -> str:
    """Return a streaming MD5 for the provenance metadata supplied upstream."""

    digest = hashlib.md5()  # nosec B324 - provenance check, not a password.
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_response_curve(
    path: Path,
    model_paths: Mapping[str, Path],
) -> str | None:
    """Return ``None`` only for a curve tied to these exact local models.

    The public curve is calibrated for a specific pair of converted ONNX
    files.  Accepting it with a merely same-named model could drive an
    unrelated model with unsafe angles, so the metadata MD5 values are checked
    before this file is ever made the local cache.
    """

    try:
        if not path.is_file() or path.stat().st_size < MIN_RESPONSE_CURVE_BYTES:
            return "response is too small to be a response curve"
        parsed: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"response is not valid UTF-8 JSON ({exc})"
    if not isinstance(parsed, Mapping):
        return "curve root must be a JSON object"

    meta = parsed.get("meta")
    digests = meta.get("model_digests") if isinstance(meta, Mapping) else None
    if not isinstance(digests, Mapping):
        return "curve metadata has no model_digests object"
    for side, model_path in model_paths.items():
        expected = digests.get(model_path.name)
        if not isinstance(expected, str) or len(expected) != 32:
            return f"curve metadata has no MD5 for {model_path.name}"
        try:
            actual = _file_md5(model_path)
        except OSError as exc:
            return f"could not read {model_path.name} for MD5 validation ({exc})"
        if actual.lower() != expected.lower():
            return (
                f"curve MD5 for {model_path.name} does not match the local model "
                f"({actual} != {expected})"
            )

        # ``side`` is intentionally unused above only for the digest name; it
        # is used here to require the axis curves this model needs.
        side_curves = parsed.get(side)
        if not isinstance(side_curves, Mapping):
            return f"curve has no {side}-eye object"
        for axis in ("h", "v"):
            entries = side_curves.get(axis)
            if not isinstance(entries, list) or len(entries) < 3:
                return f"curve {side}.{axis} needs at least three [angle, pixel] pairs"
            try:
                pairs = [(float(item[0]), float(item[1])) for item in entries]
            except (IndexError, TypeError, ValueError):
                return f"curve {side}.{axis} has an invalid [angle, pixel] pair"
            if not all(
                abs(value) < float("inf")
                for pair in pairs
                for value in pair
            ):
                return f"curve {side}.{axis} contains a non-finite value"
            angles = [pair[0] for pair in pairs]
            if any(right <= left for left, right in zip(angles, angles[1:])):
                return f"curve {side}.{axis} angles are not strictly increasing"
            # These values were measured from real neural renders.  A few
            # samples near the end of an otherwise one-direction response can
            # differ by a tiny amount of measurement noise, so the official
            # calibration file is not required to be mathematically strictly
            # monotonic.  ``NeuralEyeWarper`` turns it into a safe monotonic
            # envelope before inverse interpolation.  Here we only reject a
            # completely flat/non-responsive axis.
            pixels = [pair[1] for pair in pairs]
            if max(pixels) - min(pixels) <= 1e-5:
                return f"curve {side}.{axis} has no measurable response"
    return None


def _temporary_path(destination: Path) -> Path:
    """Create an empty, same-directory temporary path for an atomic replace."""

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=destination.parent,
            prefix=f"{destination.stem}.",
            suffix=".part",
        ) as temporary:
            return Path(temporary.name)
    except OSError as exc:
        raise NeuralModelDownloadError(
            f"Could not create a temporary file in {destination.parent}. ({exc})"
        ) from exc


def _download_once(url: str, destination: Path) -> Path:
    """Download one URL to a temporary file without changing ``destination``."""

    temporary = _temporary_path(destination)
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
    except NETWORK_ERRORS:
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise NeuralModelDownloadError(
            f"Could not write the downloaded model in {destination.parent}. ({exc})"
        ) from exc
    return temporary


def _download_one(model: NeuralModel, force: bool) -> Path:
    destination = model.destination
    if is_usable_model(destination) and not force:
        print(f"Neural model already present: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading neural model: {model.filename}...")
    failures: list[str] = []
    for source_index, url in enumerate(model.urls):
        if source_index:
            print("  Trying a GitHub fallback source...")
        for attempt in range(1, ATTEMPTS_PER_SOURCE + 1):
            downloaded: Path | None = None
            try:
                downloaded = _download_once(url, destination)
                if not is_usable_model(downloaded):
                    size = downloaded.stat().st_size if downloaded.exists() else 0
                    failures.append(
                        f"source {source_index + 1}, attempt {attempt}: "
                        f"unexpectedly small response ({size} bytes)"
                    )
                    continue
                # replace() is atomic on the same volume. A failed download
                # never overwrites an older usable local model.
                downloaded.replace(destination)
                print(f"Neural model saved locally: {destination}")
                return destination
            except NETWORK_ERRORS as exc:
                failures.append(
                    f"source {source_index + 1}, attempt {attempt}: {exc}"
                )
            except NeuralModelDownloadError:
                # This is a local filesystem failure, for which trying another
                # network URL cannot help.
                raise
            except OSError as exc:
                raise NeuralModelDownloadError(
                    f"Could not save {model.filename} in {destination.parent}. ({exc})"
                ) from exc
            finally:
                if downloaded is not None:
                    downloaded.unlink(missing_ok=True)

            if attempt < ATTEMPTS_PER_SOURCE:
                print(f"  Network retry {attempt + 1}/{ATTEMPTS_PER_SOURCE}...")
                time.sleep(RETRY_DELAY_SECONDS)

    details = "; ".join(failures[-3:]) or "no usable response"
    raise NeuralModelDownloadError(
        f"Could not download {model.filename} after trying all GitHub sources. "
        "Check the Internet connection and try setup again. The existing local "
        f"model was kept unchanged. Last details: {details}"
    )


def _response_curve_destination() -> Path:
    return MODEL_DIR / RESPONSE_CURVE_FILENAME


def _download_response_curve(model_paths: Mapping[str, Path], force: bool) -> Path:
    """Cache a curve only after it validates against the installed ONNX pair."""

    destination = _response_curve_destination()
    if not force:
        existing_error = validate_response_curve(destination, model_paths)
        if existing_error is None:
            print(f"Neural response curve already present: {destination}")
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading neural response curve...")
    failures: list[str] = []
    for source_index, url in enumerate(RESPONSE_CURVE_URLS):
        if source_index:
            print("  Trying a GitHub fallback source...")
        for attempt in range(1, ATTEMPTS_PER_SOURCE + 1):
            downloaded: Path | None = None
            try:
                downloaded = _download_once(url, destination)
                validation_error = validate_response_curve(downloaded, model_paths)
                if validation_error is not None:
                    failures.append(
                        f"source {source_index + 1}, attempt {attempt}: {validation_error}"
                    )
                    continue
                # Keep a known-good cached curve if the network returns a stale
                # or incompatible JSON document.  A replace happens only after
                # the exact installed ONNX pair has been verified.
                downloaded.replace(destination)
                print(f"Neural response curve saved locally: {destination}")
                return destination
            except NETWORK_ERRORS as exc:
                failures.append(
                    f"source {source_index + 1}, attempt {attempt}: {exc}"
                )
            except NeuralModelDownloadError:
                raise
            except OSError as exc:
                raise NeuralModelDownloadError(
                    "Could not save the neural response curve in "
                    f"{destination.parent}. ({exc})"
                ) from exc
            finally:
                if downloaded is not None:
                    downloaded.unlink(missing_ok=True)

            if attempt < ATTEMPTS_PER_SOURCE:
                print(f"  Network retry {attempt + 1}/{ATTEMPTS_PER_SOURCE}...")
                time.sleep(RETRY_DELAY_SECONDS)

    details = "; ".join(failures[-3:]) or "no valid response"
    raise NeuralModelDownloadError(
        "Could not download a response curve that matches the installed neural "
        "models. The prior local curve was kept unchanged. Run "
        "python download_neural_models.py --force to refresh the complete "
        f"model pair. Last details: {details}"
    )


def ensure_models(force: bool = False) -> tuple[Path, ...]:
    """Ensure models and their matched local response curve are present."""

    paths = tuple(_download_one(model, force=force) for model in MODELS)
    by_side = {"L": paths[0], "R": paths[1]}
    _download_response_curve(by_side, force=force)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download both neural models again, replacing local copies only after success.",
    )
    args = parser.parse_args()
    ensure_models(force=args.force)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NeuralModelDownloadError as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
