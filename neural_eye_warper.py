"""CPU ONNX full-eye gaze renderer.

This module is intentionally separate from :mod:`eye_warper` and
:mod:`eye_reconstructor`.  Those modules are useful diagnostic fallbacks for
small geometric moves, but they cannot make an eye look naturally redirected
at a large angle.  The two ONNX models used here instead predict a dense flow
field and a light-colour modulation map for the *complete visible eye*.

The converted model contract is deliberately kept in one place:

``input_img:0``
    ``float32 [1, 48, 64, 3]``, BGR image in the 0..1 range.
``input_fp:0``
    ``float32 [1, 48, 64, 12]``, six landmark x/y offset maps.
``input_ang:0``
    ``float32 [1, 2]``, ``[vertical_degrees, horizontal_degrees]``.
``flow_raw:0`` and ``lcm_map:0``
    Dense ``[1, 48, 64, 2]`` fields used to rebuild the crop at its native
    camera resolution.  ``output:0`` is checked during model loading as a
    model-integrity signal, but is not blindly resized and pasted.

The native-resolution reconstruction is important: it keeps fine eyelid and
iris texture from the live camera frame, while the neural network controls
where every eye pixel comes from and how it is relit.  The blend is clipped to
the current eye aperture, so skin, eyebrows, hair, and glasses are not
repainted.

This is a renderer, not a gaze tracker.  It receives the measured
``EyeGaze`` objects and target normalized eye ratios from the personal
calibration layer.  It does not move a cursor or expose biometric data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from gaze_estimator import EyeGaze


# The model was trained at this eye crop resolution.  The final remap below
# happens at each camera crop's *native* resolution, not at 48 x 64.
MODEL_HEIGHT = 48
MODEL_WIDTH = 64
FEATURE_POINT_COUNT = 6
FEATURE_CHANNELS = FEATURE_POINT_COUNT * 2

IMAGE_INPUT_NAME = "input_img:0"
FEATURE_INPUT_NAME = "input_fp:0"
ANGLE_INPUT_NAME = "input_ang:0"
MODEL_OUTPUT_NAME = "output:0"
FLOW_OUTPUT_NAME = "flow_raw:0"
LIGHTING_OUTPUT_NAME = "lcm_map:0"

# These are the trained range, not arbitrary UX limits.  Keeping inference in
# range is much more useful than asking a neural eye renderer to hallucinate
# an unseen profile view.
MAX_VERTICAL_DEGREES = 12.0
MAX_HORIZONTAL_DEGREES = 20.0

# The published calibration curve was measured at the trained 64 x 48 model
# resolution.  It is deliberately used instead of treating an iris pixel as a
# spherical eyeball angle.  This conservative preview guard is important on a
# CPU-only MVP: a full-eye flow model can otherwise make a very small noisy
# landmark request look like a large eyelid deformation.
RESPONSE_CURVE_FILENAME = "response_curve.json"
RESPONSE_TARGET_GAIN = 0.35
SAFE_VERTICAL_DEGREES = 5.0
SAFE_HORIZONTAL_DEGREES = 10.0
MIN_AXIS_MODEL_PIXELS = 0.10
MIN_RENDER_MODEL_PIXELS = 0.25
MAX_COMPOSITE_ALPHA = 0.45
CROP_WIDTH_PER_EYE_LENGTH = 1.5
CROP_HEIGHT_PER_EYE_LENGTH = 1.125

# A whole-eye flow model has no way to reveal iris texture that is already
# covered by a lid in the input frame.  A raw "iris centre is in the contour"
# test is not enough: a centre can be technically inside while its iris is
# only a couple of pixels from the lid.  The values below apply a small safety
# envelope around the measured iris before a neural request is accepted.
#
# They intentionally favour a stable original eye over a large correction. A
# missed correction is reversible on the next frame; a learned flow that pulls
# lid/skin texture over an iris is visually much worse.
MIN_SOURCE_IRIS_VISIBLE_FRACTION = 0.82
MIN_DESTINATION_IRIS_VISIBLE_FRACTION = 0.88
# A learned full-eye renderer can visibly alter eyelashes even when the
# requested endpoint is only a few tenths of a pixel.  Do not invoke it for a
# change that a webcam preview cannot usefully show.
MIN_VISIBLE_MOVE_PIXELS = 0.50


def _smoothstep(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return a compact float32 smooth threshold without a SciPy dependency."""

    fraction = np.clip((value - lower) / (upper - lower), 0.0, 1.0)
    return (fraction * fraction * (3.0 - 2.0 * fraction)).astype(np.float32)


class NeuralEyeModelError(RuntimeError):
    """Raised when the optional neural model/runtime cannot be used safely."""


@dataclass(frozen=True)
class _ResponseAxis:
    """One monotonic model-control response, stored as angle -> model pixels."""

    angles: np.ndarray
    pixels: np.ndarray

    def angle_for_pixels(self, desired_pixels: float) -> float:
        """Invert the measured curve without extrapolating beyond its samples."""

        if self.pixels[-1] > self.pixels[0]:
            return float(np.interp(desired_pixels, self.pixels, self.angles))
        return float(
            np.interp(desired_pixels, self.pixels[::-1], self.angles[::-1])
        )

    def pixels_for_angle(self, angle: float) -> float:
        """Evaluate the measured curve at an in-range model control angle."""

        return float(np.interp(angle, self.angles, self.pixels))


@dataclass(frozen=True)
class _EyeResponseCurve:
    """Horizontal and vertical controls for one anatomical ONNX eye model."""

    horizontal: _ResponseAxis
    vertical: _ResponseAxis


@dataclass(frozen=True)
class EyeModelSpec:
    """Semantic MediaPipe points expected by one separately trained model."""

    side: str
    corner_indices: tuple[int, int]
    feature_indices: tuple[int, int, int, int, int, int]


# ``R`` and ``L`` describe the model's anatomical training side.  They are
# selected from semantic MediaPipe indices, never from the tuple order of the
# detected eyes, which may change when the preview is mirrored.
EYE_MODEL_SPECS: Mapping[str, EyeModelSpec] = {
    "R": EyeModelSpec(
        side="R",
        corner_indices=(33, 133),
        feature_indices=(33, 160, 158, 133, 153, 144),
    ),
    "L": EyeModelSpec(
        side="L",
        corner_indices=(362, 263),
        feature_indices=(263, 387, 385, 362, 380, 373),
    ),
}


@dataclass(frozen=True)
class NativeEyeCrop:
    """An axis-aligned, in-frame crop in canonical (unmirrored) pixels."""

    left: int
    top: int
    right: int
    bottom: int
    side: str

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> np.ndarray:
        return np.asarray(
            ((self.left + self.right - 1) * 0.5, (self.top + self.bottom - 1) * 0.5),
            dtype=np.float32,
        )


@dataclass(frozen=True)
class NeuralEyeWarpPlan:
    """Per-eye request and the exact model-safe endpoint for this frame.

    The leading fields mirror ``IrisReconstructionPlan`` so the preview can
    display either renderer without a special case.  Extra fields preserve
    the neural request used by :class:`NeuralEyeWarper.apply`.
    """

    source: np.ndarray
    requested_target: np.ndarray
    destination: np.ndarray
    delta: np.ndarray
    applied: bool
    reason: str
    model_side: str | None = None
    model_angle: np.ndarray | None = None
    crop: NativeEyeCrop | None = None
    eye_index: int = -1
    composite_alpha: float = 1.0

    @property
    def pixels(self) -> float:
        """Actual planned pixel movement, in the caller's frame orientation."""

        return float(np.linalg.norm(self.delta))


@dataclass(frozen=True)
class _ModelTensors:
    """Named graph tensors confirmed during ONNX session validation."""

    image_input: str = IMAGE_INPUT_NAME
    feature_input: str = FEATURE_INPUT_NAME
    angle_input: str = ANGLE_INPUT_NAME
    flow_output: str = FLOW_OUTPUT_NAME
    lighting_output: str = LIGHTING_OUTPUT_NAME


class NeuralEyeWarper:
    """Run learned flow-and-lighting gaze correction on a CPU ONNX Runtime.

    Public use is intentionally close to the previous renderers::

        renderer = NeuralEyeWarper()
        plans = renderer.plan(frame, landmarks, gaze.eyes, targets, strength,
                              mirrored=True)
        corrected = renderer.apply(frame, landmarks, gaze.eyes, plans,
                                   mirrored=True)

    Or use ``correct(...)`` for both steps.  ``landmarks`` must contain the
    MediaPipe Face Landmarker 478 pixel coordinates in the same orientation as
    ``frame``.  If the application mirror-flipped both before tracking, pass
    ``mirrored=True``: this renderer canonicalizes them internally for the
    separately trained left/right models, then flips the rendered result back.
    """

    def __init__(
        self,
        left_model_path: str | Path | None = None,
        right_model_path: str | Path | None = None,
        *,
        auto_load: bool = True,
        cpu_threads: int | None = None,
        # The documented conversion convention is [vertical, horizontal]
        # after the signs in ``_angle_from_canonical_delta``.  This optional
        # hook is only for advanced experimentation with a differently
        # converted model; ordinary users should leave it alone.
        angle_sign: tuple[float, float] = (1.0, 1.0),
        session_factory: Callable[..., Any] | None = None,
        response_curve_path: str | Path | None = None,
        # Test and diagnostic hook.  Normal application startup always uses
        # the digest-checked on-disk curve downloaded by setup.bat.
        response_curve_data: Mapping[str, Any] | None = None,
    ) -> None:
        default_left, default_right = self.default_model_paths()
        self.left_model_path = Path(left_model_path) if left_model_path else default_left
        self.right_model_path = Path(right_model_path) if right_model_path else default_right
        default_curve = self.left_model_path.parent / RESPONSE_CURVE_FILENAME
        self.response_curve_path = (
            Path(response_curve_path) if response_curve_path else default_curve
        )
        angle_sign_array = np.asarray(angle_sign, dtype=np.float32)
        if (
            angle_sign_array.shape != (2,)
            or not np.isfinite(angle_sign_array).all()
            or np.any(np.abs(angle_sign_array) < 1e-6)
        ):
            raise ValueError("angle_sign must contain two non-zero finite values")
        self.angle_sign = angle_sign_array
        self.cpu_threads = cpu_threads
        if cpu_threads is not None and cpu_threads < 1:
            raise ValueError("cpu_threads must be at least 1")
        self._session_factory = session_factory
        self._sessions: dict[str, Any] = {}
        self._tensor_names: dict[str, _ModelTensors] = {}
        self._response_curve_data = response_curve_data
        self._response_curves: dict[str, _EyeResponseCurve] = {}
        if not auto_load and response_curve_data is not None:
            self._response_curves = self._parse_response_curve(response_curve_data)
        if auto_load:
            self.load_models()

    @staticmethod
    def default_model_paths() -> tuple[Path, Path]:
        """Return the project-local paths installed by the model setup step."""

        root = Path(__file__).resolve().parent / "models" / "neural"
        return root / "gaze_L.onnx", root / "gaze_R.onnx"

    @classmethod
    def models_present(
        cls,
        left_model_path: str | Path | None = None,
        right_model_path: str | Path | None = None,
    ) -> bool:
        """Check model files without importing ONNX Runtime or opening a GPU."""

        default_left, default_right = cls.default_model_paths()
        left = Path(left_model_path) if left_model_path else default_left
        right = Path(right_model_path) if right_model_path else default_right
        return left.is_file() and right.is_file()

    @property
    def ready(self) -> bool:
        """Whether models and their matching measured response are available."""

        return set(self._sessions) == {"L", "R"} and set(self._response_curves) == {"L", "R"}

    def load_models(self) -> None:
        """Open only CPUExecutionProvider sessions and validate model I/O."""

        missing = [
            str(path)
            for path in (self.left_model_path, self.right_model_path)
            if not path.is_file()
        ]
        if missing:
            raise NeuralEyeModelError(
                "Neural gaze model files were not found:\n  "
                + "\n  ".join(missing)
                + "\nRun setup.bat to install the optional neural models."
            )

        self._response_curves = self._load_response_curves()

        try:
            if self._session_factory is None:
                import onnxruntime as ort  # type: ignore[import-not-found]

                options = ort.SessionOptions()
                if self.cpu_threads is not None:
                    options.intra_op_num_threads = int(self.cpu_threads)
                    options.inter_op_num_threads = 1

                def factory(path: Path) -> Any:
                    return ort.InferenceSession(
                        str(path),
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )

            else:
                factory = self._session_factory
        except ImportError as exc:
            raise NeuralEyeModelError(
                "onnxruntime is not installed in the project environment. "
                "Run setup.bat, then start the camera again."
            ) from exc

        sessions: dict[str, Any] = {}
        tensors: dict[str, _ModelTensors] = {}
        for side, path in (("L", self.left_model_path), ("R", self.right_model_path)):
            try:
                session = factory(path)
                tensors[side] = self._validate_session(session, side, path)
                sessions[side] = session
            except NeuralEyeModelError:
                raise
            except Exception as exc:  # pragma: no cover - runtime-specific text.
                raise NeuralEyeModelError(
                    f"Could not load the neural {side}-eye model: {path}\n{exc}"
                ) from exc
        self._sessions = sessions
        self._tensor_names = tensors

    def _load_response_curves(self) -> dict[str, _EyeResponseCurve]:
        """Load only a curve calibrated for the exact local ONNX file bytes."""

        if self._response_curve_data is not None:
            document: Any = self._response_curve_data
        else:
            try:
                document = json.loads(self.response_curve_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise NeuralEyeModelError(
                    "The neural response curve is missing: "
                    f"{self.response_curve_path}\nRun setup.bat once to download the "
                    "matching calibration file."
                ) from exc
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NeuralEyeModelError(
                    "The neural response curve cannot be read safely: "
                    f"{self.response_curve_path}\nRun setup.bat again. ({exc})"
                ) from exc
        self._validate_response_curve_digests(document)
        return self._parse_response_curve(document)

    def _validate_response_curve_digests(self, document: Any) -> None:
        """Reject a same-named but differently converted ONNX model pair."""

        if not isinstance(document, Mapping):
            raise NeuralEyeModelError("Neural response curve root must be a JSON object")
        meta = document.get("meta")
        digests = meta.get("model_digests") if isinstance(meta, Mapping) else None
        if not isinstance(digests, Mapping):
            raise NeuralEyeModelError(
                "Neural response curve has no model_digests metadata. Run setup.bat again."
            )
        for side, path in (("L", self.left_model_path), ("R", self.right_model_path)):
            expected = digests.get(path.name)
            if not isinstance(expected, str) or len(expected) != 32:
                raise NeuralEyeModelError(
                    f"Neural response curve has no MD5 for {path.name}. Run setup.bat again."
                )
            try:
                actual = self._file_md5(path)
            except OSError as exc:
                raise NeuralEyeModelError(
                    f"Could not validate local {side}-eye model against its response curve: {exc}"
                ) from exc
            if actual.lower() != expected.lower():
                raise NeuralEyeModelError(
                    "Neural response curve does not match the installed ONNX model "
                    f"{path.name}. Run setup.bat again to refresh the matched files."
                )

    @staticmethod
    def _file_md5(path: Path) -> str:
        digest = hashlib.md5()  # nosec B324 - upstream model provenance only.
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _parse_response_curve(document: Mapping[str, Any]) -> dict[str, _EyeResponseCurve]:
        """Parse the small public angle-to-pixel document without extrapolation.

        The curve is measured from real model renders, so a value near the
        end of an axis can occasionally move backwards by a few hundredths of
        a model pixel.  Treating that normal measurement noise as a hard error
        would make setup reject the publisher's own curve.  Instead, make a
        one-direction envelope and remove plateaus before using a curve for an
        inverse lookup.  It cannot invent extra motion: it only removes a
        noisy reversal that would otherwise make ``np.interp`` ambiguous.
        """

        if not isinstance(document, Mapping):
            raise NeuralEyeModelError("Neural response curve root must be a JSON object")

        def parse_axis(side: str, axis: str) -> _ResponseAxis:
            side_document = document.get(side)
            raw_pairs = side_document.get(axis) if isinstance(side_document, Mapping) else None
            if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
                raise NeuralEyeModelError(f"Neural response curve lacks {side}.{axis}")
            try:
                values = np.asarray(raw_pairs, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise NeuralEyeModelError(
                    f"Neural response curve {side}.{axis} is not numeric"
                ) from exc
            if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 2:
                raise NeuralEyeModelError(
                    f"Neural response curve {side}.{axis} must contain [angle, pixel] pairs"
                )
            angles, pixels = values[:, 0], values[:, 1]
            if not np.isfinite(values).all():
                raise NeuralEyeModelError(f"Neural response curve {side}.{axis} is non-finite")
            if np.any(np.diff(angles) <= 0.0):
                raise NeuralEyeModelError(
                    f"Neural response curve {side}.{axis} angles are not strictly increasing"
                )
            direction = 1.0 if float(pixels[-1] - pixels[0]) > 0.0 else -1.0
            signed_pixels = direction * pixels
            envelope = np.maximum.accumulate(signed_pixels)
            # ``np.interp`` requires the inverse lookup x coordinates to be
            # increasing.  Keep only strictly advancing samples; at least two
            # unique values are needed for a meaningful gaze response.
            keep = [0]
            for sample_index in range(1, len(envelope)):
                if envelope[sample_index] > envelope[keep[-1]] + 1e-5:
                    keep.append(sample_index)
            if len(keep) < 2:
                raise NeuralEyeModelError(
                    f"Neural response curve {side}.{axis} has no measurable response"
                )
            compact_angles = angles[keep].astype(np.float32)
            compact_pixels = (direction * envelope[keep]).astype(np.float32)
            return _ResponseAxis(compact_angles, compact_pixels)

        return {
            side: _EyeResponseCurve(
                horizontal=parse_axis(side, "h"),
                vertical=parse_axis(side, "v"),
            )
            for side in ("L", "R")
        }

    def close(self) -> None:
        """Release session references.  ONNX Runtime owns its native cleanup."""

        self._sessions.clear()
        self._tensor_names.clear()
        self._response_curves.clear()

    def plan(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        eyes: Sequence[EyeGaze],
        target_ratios: np.ndarray,
        strength: float,
        *,
        mirrored: bool = False,
    ) -> tuple[NeuralEyeWarpPlan, ...]:
        """Build model-safe full-eye requests without running inference.

        Strength is applied to the real image-space target movement *before*
        angle conversion.  The network's trained angle range then bounds the
        actual endpoint reported in each plan.
        """

        image_shape = self._valid_frame_shape(frame_bgr)
        if image_shape is None:
            return ()
        height, width = image_shape
        points = self._validated_landmarks(landmarks)
        eye_tuple = tuple(eyes)
        targets = np.asarray(target_ratios, dtype=np.float32)
        if points is None or targets.shape != (len(eye_tuple), 2):
            return tuple(
                self._skipped_plan(index, eye, "invalid landmarks or target")
                for index, eye in enumerate(eye_tuple)
            )
        if not np.isfinite(targets).all():
            return tuple(
                self._skipped_plan(index, eye, "invalid target")
                for index, eye in enumerate(eye_tuple)
            )
        # Never fall back to a guessed trigonometric response.  The neural
        # model is only allowed to run when it has a response measurement for
        # the exact ONNX files installed beside it.  This is especially
        # important for narrow eye openings, where a one-pixel mismatch can
        # look as though an eyelid has covered the pupil.
        if set(self._response_curves) != {"L", "R"}:
            return tuple(
                self._skipped_plan(index, eye, "response curve unavailable")
                for index, eye in enumerate(eye_tuple)
            )

        assignments = self._assign_model_sides(eye_tuple, points)
        canonical_points = self._canonicalize_points(points, width, mirrored)
        # A strong profile pose makes a model's square eye crop untrustworthy.
        profile = self._is_profile_or_degenerate(canonical_points)
        amount = float(np.clip(strength, 0.0, 1.0))
        plans: list[NeuralEyeWarpPlan] = []
        for index, (eye, target) in enumerate(zip(eye_tuple, targets)):
            source = np.asarray(eye.iris_center, dtype=np.float32).copy()
            requested = eye.point_for_ratio(target)
            if not np.isfinite(source).all() or not np.isfinite(requested).all():
                plans.append(self._skipped_plan(index, eye, "invalid landmarks", requested))
                continue
            if amount <= 0.0:
                plans.append(self._skipped_plan(index, eye, "zero strength", requested))
                continue
            if eye.width < 18.0 or eye.height < 2.5 or eye.openness < 0.052:
                plans.append(self._skipped_plan(index, eye, "eye not open", requested))
                continue
            if profile:
                plans.append(self._skipped_plan(index, eye, "profile angle", requested))
                continue

            side = assignments.get(index)
            if side is None:
                plans.append(self._skipped_plan(index, eye, "eye/model mismatch", requested))
                continue
            crop = self.crop_for_side(canonical_points, side, width, height)
            if crop is None:
                plans.append(self._skipped_plan(index, eye, "eye crop outside frame", requested, side=side))
                continue

            # The direction submitted to the model must be measured in its
            # unmirrored canonical pixel frame.  In a mirrored preview that
            # correctly flips the horizontal sign while leaving caller-facing
            # plan coordinates in the preview's orientation.
            desired_display_delta = (requested - source) * amount
            source_canonical = self._canonicalize_point(source, width, mirrored)
            desired_canonical_delta = desired_display_delta.copy()
            if mirrored:
                desired_canonical_delta[0] *= -1.0

            eye_length = self._eye_length(canonical_points, EYE_MODEL_SPECS[side])
            if eye_length < 12.0:
                plans.append(self._skipped_plan(index, eye, "eye too small", requested, side=side, crop=crop))
                continue

            angle, applied_canonical_delta, limited = self._angle_from_canonical_delta(
                desired_canonical_delta, eye_length, side
            )
            destination_canonical = source_canonical + applied_canonical_delta
            destination = self._decanonicalize_point(destination_canonical, width, mirrored)

            # The model synthesises a whole visible eye, not only an iris.  Do
            # not submit a request when the current iris is partly hidden by
            # an eyelid, and do not aim it into the lid margin.  A full
            # safety-envelope test is deliberately stricter than the final
            # compositing mask below: the latter protects pixels at the lid
            # boundary, whereas this prevents a learned eyelid-like texture
            # from being generated inside the opening in the first place.
            safe_destination, eyelid_limited, eyelid_safe = self._safe_visible_destination(
                eye, source, destination
            )
            if not eyelid_safe:
                plans.append(
                    self._skipped_plan(
                        index,
                        eye,
                        "iris near eyelid",
                        requested,
                        side=side,
                        crop=crop,
                    )
                )
                continue
            if eyelid_limited:
                # Recompute the model angle after clipping.  Keeping the old
                # angle would still ask the network for the unsafe endpoint
                # even if preview diagnostics displayed a safe one.
                safe_display_delta = safe_destination - source
                safe_canonical_delta = safe_display_delta.astype(np.float32).copy()
                if mirrored:
                    safe_canonical_delta[0] *= -1.0
                angle, applied_canonical_delta, model_limited_after_lid = (
                    self._angle_from_canonical_delta(
                        safe_canonical_delta, eye_length, side
                    )
                )
                limited = limited or model_limited_after_lid
                destination_canonical = source_canonical + applied_canonical_delta
                destination = self._decanonicalize_point(
                    destination_canonical, width, mirrored
                )

                # The forward/inverse angle mapping is near-exact but not
                # mathematically identical at float precision.  Refuse the
                # frame rather than letting a rounding difference cross the
                # lid safety envelope.
                if not self._destination_is_visible(eye, destination):
                    plans.append(
                        self._skipped_plan(
                            index,
                            eye,
                            "iris near eyelid",
                            requested,
                            side=side,
                            crop=crop,
                        )
                    )
                    continue
            actual_delta = (destination - source).astype(np.float32)
            if float(np.linalg.norm(actual_delta)) < MIN_VISIBLE_MOVE_PIXELS:
                plans.append(
                    NeuralEyeWarpPlan(
                        source, requested, source, np.zeros(2, dtype=np.float32), False,
                        "target already reached", side, angle, crop, index,
                    )
                )
                continue
            plans.append(
                NeuralEyeWarpPlan(
                    source=source,
                    requested_target=requested.astype(np.float32),
                    destination=destination.astype(np.float32),
                    delta=actual_delta,
                    applied=True,
                    reason=(
                        "limited by eyelid"
                        if eyelid_limited
                        else "limited by model angle"
                        if limited
                        else "neural flow"
                    ),
                    model_side=side,
                    model_angle=angle,
                    crop=crop,
                    eye_index=index,
                    # Keep a visible band of the real eye underneath the
                    # learned output.  This is deliberately capped even at
                    # 100% UI strength: the slider changes the requested
                    # correction, not permission to repaint an eyelid.
                    composite_alpha=min(
                        MAX_COMPOSITE_ALPHA,
                        0.20 + 0.30 * amount,
                        0.32 if eyelid_limited else MAX_COMPOSITE_ALPHA,
                    ),
                )
            )
        return tuple(plans)

    def correct(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        eyes: Sequence[EyeGaze],
        target_ratios: np.ndarray,
        strength: float,
        *,
        mirrored: bool = False,
    ) -> np.ndarray:
        """Plan then render learned full-eye correction in one call."""

        plans = self.plan(frame_bgr, landmarks, eyes, target_ratios, strength, mirrored=mirrored)
        return self.apply(frame_bgr, landmarks, eyes, plans, mirrored=mirrored)

    def apply(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        eyes: Sequence[EyeGaze],
        plans: Sequence[NeuralEyeWarpPlan],
        *,
        mirrored: bool = False,
    ) -> np.ndarray:
        """Apply preplanned neural reconstructions, never an iris-only fallback.

        This method raises :class:`NeuralEyeModelError` when callers forgot to
        load the neural models.  Silently substituting a painted pupil would
        hide the exact failure mode this renderer exists to avoid.
        """

        if not self.ready:
            raise NeuralEyeModelError(
                "Neural eye models are not loaded. Create NeuralEyeWarper() after "
                "setup.bat has installed gaze_L.onnx and gaze_R.onnx."
            )
        shape = self._valid_frame_shape(frame_bgr)
        points = self._validated_landmarks(landmarks)
        eye_tuple = tuple(eyes)
        plan_tuple = tuple(plans)
        if shape is None or points is None or len(eye_tuple) != len(plan_tuple):
            return frame_bgr.copy()
        height, width = shape
        canonical_frame = cv2.flip(frame_bgr, 1) if mirrored else frame_bgr.copy()
        canonical_points = self._canonicalize_points(points, width, mirrored)

        # Each source crop is read from the same unmodified canonical frame;
        # altering the first eye therefore cannot contaminate the second eye's
        # network input even if an unusual camera crop makes them very close.
        renders: list[tuple[NativeEyeCrop, np.ndarray, np.ndarray, float]] = []
        for index, (eye, plan) in enumerate(zip(eye_tuple, plan_tuple)):
            if not plan.applied or plan.model_side not in EYE_MODEL_SPECS:
                continue
            if plan.eye_index not in (-1, index):
                continue
            crop = plan.crop or self.crop_for_side(
                canonical_points, plan.model_side, width, height
            )
            if crop is None or crop.width < 8 or crop.height < 8:
                continue
            try:
                corrected_crop = self._render_crop(
                    canonical_frame,
                    canonical_points,
                    crop,
                    plan.model_side,
                    plan.model_angle,
                )
            except NeuralEyeModelError:
                raise
            except Exception as exc:  # pragma: no cover - defensive runtime guard.
                raise NeuralEyeModelError(
                    f"Neural {plan.model_side}-eye inference failed: {exc}"
                ) from exc
            canonical_contour = self._canonicalize_points(
                np.asarray(eye.contour, dtype=np.float32), width, mirrored
            )
            alpha = self.aperture_alpha(crop, canonical_contour)
            if alpha is None or float(alpha.max()) < 0.10:
                continue
            source_crop = canonical_frame[crop.top : crop.bottom, crop.left : crop.right]
            canonical_source = self._canonicalize_point(plan.source, width, mirrored)
            canonical_delta = np.asarray(plan.delta, dtype=np.float32).copy()
            canonical_vertical_axis = np.asarray(eye.vertical_axis, dtype=np.float32).copy()
            if mirrored:
                canonical_delta[0] *= -1.0
                canonical_vertical_axis[0] *= -1.0
            # A model can be mathematically within its angle range yet still
            # move the darkest iris feature into an eyelid on a particular
            # shallow-open eye.  Verify the rendered pixels themselves before
            # they are allowed into the corrected camera frame.
            if not self._rendered_pupil_is_safe(
                source_crop,
                corrected_crop,
                crop,
                canonical_contour,
                canonical_source,
                canonical_delta,
                canonical_vertical_axis,
                eye.height,
            ):
                continue
            composite_alpha = float(
                np.clip(plan.composite_alpha, 0.0, MAX_COMPOSITE_ALPHA)
            )
            if composite_alpha <= 0.0:
                continue
            renders.append((crop, corrected_crop, alpha, composite_alpha))

        output = canonical_frame.copy()
        for crop, corrected_crop, alpha, composite_alpha in renders:
            region = output[crop.top : crop.bottom, crop.left : crop.right]
            if region.shape[:2] != corrected_crop.shape[:2]:
                continue
            blend = (alpha * composite_alpha)[..., None]
            output[crop.top : crop.bottom, crop.left : crop.right] = np.clip(
                corrected_crop.astype(np.float32) * blend
                + region.astype(np.float32) * (1.0 - blend),
                0.0,
                255.0,
            ).astype(np.uint8)
        return cv2.flip(output, 1) if mirrored else output

    @staticmethod
    def crop_for_side(
        landmarks: np.ndarray,
        side: str,
        frame_width: int,
        frame_height: int,
    ) -> NativeEyeCrop | None:
        """Build the documented 1.5 x 1.125 eye crop from semantic points.

        The top margin is intentionally larger than the bottom margin.  It
        leaves enough eyebrow/upper-lid context for the learned flow while the
        final aperture alpha still prevents those surrounding pixels from being
        composited back to the camera frame.
        """

        spec = EYE_MODEL_SPECS.get(side)
        points = NeuralEyeWarper._validated_landmarks(landmarks)
        if spec is None or points is None or frame_width < 1 or frame_height < 1:
            return None
        length = NeuralEyeWarper._eye_length(points, spec)
        if not np.isfinite(length) or length < 1.0:
            return None
        center = np.mean(points[list(spec.corner_indices), :2], axis=0)
        # Keep the conventional top-left-aligned integer crop convention used
        # by the model's training/export pipeline.  Landmark feature maps keep
        # the remaining subpixel position instead of discarding it.
        left = int(np.floor(float(center[0] - 0.75 * length)))
        right = int(np.floor(float(center[0] + 0.75 * length)))
        top = int(np.floor(float(center[1] - 0.65625 * length)))
        bottom = int(np.floor(float(center[1] + 0.46875 * length)))
        # The network cannot safely infer from reflected border pixels.  This
        # is a deliberate frame hold near the camera edge, not a crop/paste
        # artifact that could spill onto skin.
        if left < 0 or top < 0 or right > frame_width or bottom > frame_height:
            return None
        if right - left < 8 or bottom - top < 8:
            return None
        return NativeEyeCrop(left, top, right, bottom, side)

    @staticmethod
    def make_anchor_map(landmarks: np.ndarray, crop: NativeEyeCrop) -> np.ndarray:
        """Create the six x/y landmark offset maps consumed by the ONNX graph.

        Every feature pair stores ``grid_x - landmark_x`` then
        ``grid_y - landmark_y`` in model-crop coordinates.  The published
        export quantises feature positions to this grid before creating the
        maps, so the live path intentionally does the same.
        """

        points = NeuralEyeWarper._validated_landmarks(landmarks)
        spec = EYE_MODEL_SPECS.get(crop.side)
        if points is None or spec is None:
            raise ValueError("A valid 478-point landmark array and eye side are required")
        if crop.width < 1 or crop.height < 1:
            raise ValueError("crop dimensions must be positive")
        grid_y, grid_x = np.mgrid[0:MODEL_HEIGHT, 0:MODEL_WIDTH].astype(np.float32)
        channels: list[np.ndarray] = []
        x_scale = MODEL_WIDTH / float(crop.width)
        y_scale = MODEL_HEIGHT / float(crop.height)
        for point_index in spec.feature_indices:
            point = points[point_index, :2]
            landmark_x = int((float(point[0]) - crop.left) * x_scale)
            landmark_y = int((float(point[1]) - crop.top) * y_scale)
            channels.extend((grid_x - landmark_x, grid_y - landmark_y))
        return np.stack(channels, axis=2).astype(np.float32)

    @staticmethod
    def aperture_alpha(crop: NativeEyeCrop, contour: np.ndarray) -> np.ndarray | None:
        """Return a feathered 0..1 mask for the currently visible eye opening."""

        points = np.asarray(contour, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            return None
        local = points[:, :2] - np.asarray((crop.left, crop.top), dtype=np.float32)
        polygon = np.rint(local).astype(np.int32).reshape((-1, 1, 2))
        hard = np.zeros((crop.height, crop.width), dtype=np.uint8)
        cv2.fillPoly(hard, [polygon], 255, lineType=cv2.LINE_AA)
        if cv2.countNonZero(hard) < max(12, int(crop.width * crop.height * 0.025)):
            return None
        # Preserve a real, unmodified lid band before feathering inside the
        # opening.  A conventional one-pixel feather still lets a dense model
        # pull dark upper-lid texture into a narrow eye; keeping this inset
        # band protects the observed lid contour rather than merely hiding a
        # rectangular crop seam.
        distance = cv2.distanceTransform(hard, cv2.DIST_L2, 3)
        aperture_height = float(np.ptp(points[:, 1]))
        protected_lid_band = max(1.0, min(2.2, aperture_height * 0.15))
        feather = max(1.4, min(4.0, aperture_height * 0.30))
        interior = np.clip(
            (distance - protected_lid_band) / feather, 0.0, 1.0
        ).astype(np.float32)
        # Smooth the transition and retain a small amount of genuine camera
        # texture even at the centre.  This is not an iris patch fallback: it
        # is a whole-aperture alpha guard against opaque, lid-like model
        # output on a live frame.
        alpha = (interior * interior * (3.0 - 2.0 * interior) * 0.90).astype(np.float32)
        return alpha

    @staticmethod
    def _dark_feature_centroid(
        crop_bgr: np.ndarray,
        centre_local: np.ndarray,
        eye_height: float,
    ) -> tuple[np.ndarray, float] | None:
        """Find a stable dark iris/pupil feature in a small central eye area.

        This is deliberately a *rejection* check, not a second gaze tracker.
        If lighting, makeup, glasses, or exposure make the feature ambiguous,
        it returns ``None`` and lets the other safety guards decide.  It only
        reports a point when there is enough local dark contrast to catch the
        common neural failure where a pupil is pulled into an eyelid.
        """

        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            return None
        centre = np.asarray(centre_local, dtype=np.float32)
        if centre.shape != (2,) or not np.isfinite(centre).all():
            return None
        height, width = crop_bgr.shape[:2]
        half_width = max(4, int(round(min(width * 0.30, 12.0))))
        half_height = max(3, int(round(min(max(eye_height * 0.62, 3.0), 10.0))))
        left = max(0, int(np.floor(centre[0] - half_width)))
        right = min(width, int(np.ceil(centre[0] + half_width + 1.0)))
        top = max(0, int(np.floor(centre[1] - half_height)))
        bottom = min(height, int(np.ceil(centre[1] + half_height + 1.0)))
        if right - left < 5 or bottom - top < 5:
            return None
        gray = cv2.cvtColor(crop_bgr[top:bottom, left:right], cv2.COLOR_BGR2GRAY).astype(
            np.float32
        )
        # A small webcam pupil can occupy well below 15% of this deliberately
        # generous central window, so use the low tail rather than the 15th
        # percentile when deciding whether a dark feature is present.
        low = float(np.percentile(gray, 5.0))
        high = float(np.percentile(gray, 85.0))
        if high - low < 14.0:
            return None
        threshold = float(np.percentile(gray, 58.0))
        weights = np.clip(threshold - gray, 0.0, None)
        mass = float(weights.sum())
        if mass < 25.0:
            return None
        grid_y, grid_x = np.mgrid[top:bottom, left:right].astype(np.float32)
        centroid = np.asarray(
            (
                float((grid_x * weights).sum() / mass),
                float((grid_y * weights).sum() / mass),
            ),
            dtype=np.float32,
        )
        return centroid, mass

    @classmethod
    def _rendered_pupil_is_safe(
        cls,
        source_crop: np.ndarray,
        corrected_crop: np.ndarray,
        crop: NativeEyeCrop,
        canonical_contour: np.ndarray,
        canonical_source: np.ndarray,
        canonical_delta: np.ndarray,
        canonical_vertical_axis: np.ndarray,
        eye_height: float,
    ) -> bool:
        """Reject a rendered crop when its dark iris detail enters a lid band."""

        local_source = np.asarray(canonical_source, dtype=np.float32) - np.asarray(
            (crop.left, crop.top), dtype=np.float32
        )
        before = cls._dark_feature_centroid(source_crop, local_source, eye_height)
        after = cls._dark_feature_centroid(corrected_crop, local_source, eye_height)
        # A clear rejection needs reliable evidence in both images.  Unknown
        # lighting is handled by the geometric lid guards rather than creating
        # a false negative for a dark-eyed or glasses-wearing user.
        if before is None or after is None:
            return True
        before_point, before_mass = before
        after_point, after_mass = after
        if after_mass < before_mass * 0.42:
            return False

        axis = np.asarray(canonical_vertical_axis, dtype=np.float32)
        axis_length = float(np.linalg.norm(axis))
        if axis_length < 1e-4 or not np.isfinite(axis).all():
            return True
        axis /= axis_length
        observed_vertical = float(np.dot(after_point - before_point, axis))
        expected_vertical = float(np.dot(canonical_delta, axis))
        allowed_error = max(0.90, min(1.80, float(eye_height) * 0.16))
        if abs(observed_vertical - expected_vertical) > allowed_error:
            return False

        full_point = after_point + np.asarray((crop.left, crop.top), dtype=np.float32)
        polygon = np.asarray(canonical_contour, dtype=np.float32).reshape((-1, 1, 2))
        lid_clearance = float(
            cv2.pointPolygonTest(polygon, (float(full_point[0]), float(full_point[1])), True)
        )
        return lid_clearance >= max(0.85, min(1.65, float(eye_height) * 0.15))

    @staticmethod
    def _iris_safety_radii(eye: EyeGaze) -> tuple[float, float]:
        """Return an iris ellipse enlarged by a conservative lid clearance.

        Face Landmarker iris points can continue behind an eyelid.  The
        visible opening, rather than that full ring, must decide whether it is
        safe to ask a generative flow model for a new gaze direction.
        """

        radius_x = (
            float(eye.iris_radius_x)
            if eye.iris_radius_x is not None and eye.iris_radius_x > 0.0
            else float(eye.iris_radius)
        )
        radius_y = (
            float(eye.iris_radius_y)
            if eye.iris_radius_y is not None and eye.iris_radius_y > 0.0
            else float(eye.iris_radius)
        )
        # Face Landmarker may report the full iris ring even when a portion is
        # behind a real eyelid.  Using that full reported radius would reject
        # normal, open eyes with a shallow webcam aperture.  Sample a compact
        # inner iris instead, then separately require a centre-to-lid margin.
        # That catches an actually covered pupil without classifying every
        # naturally narrow eye as closed.
        radius_x = min(max(0.9, radius_x * 0.58), eye.width * 0.14)
        radius_y = min(max(0.75, radius_y * 0.48), radius_x, eye.height * 0.20)
        # The vertical addition stays larger than horizontal, but is only a
        # small band.  The destination safety test below is the primary lid
        # guard; this envelope only detects an iris already partly hidden.
        lid_clearance = max(0.55, min(1.15, eye.height * 0.10))
        corner_clearance = max(0.35, min(0.75, eye.width * 0.025))
        return radius_x + corner_clearance, radius_y + lid_clearance

    @classmethod
    def _iris_visible_fraction(cls, eye: EyeGaze, center: np.ndarray) -> float:
        """Estimate how much of an iris plus lid-safety envelope is visible."""

        ellipse_center = np.asarray(center, dtype=np.float32)
        if ellipse_center.shape != (2,) or not np.isfinite(ellipse_center).all():
            return 0.0
        polygon = np.asarray(eye.contour, dtype=np.float32).reshape((-1, 1, 2))
        if polygon.shape[0] < 3:
            return 0.0
        radius_x, radius_y = cls._iris_safety_radii(eye)
        # A small area sampling pattern avoids treating an iris centre as a
        # point.  It is inexpensive at webcam rates and catches the common
        # "centre inside, rim behind lid" failure case.
        samples: list[tuple[float, float]] = [(0.0, 0.0)]
        for radius in (0.42, 0.72, 1.0):
            for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
                samples.append((radius * float(np.cos(angle)), radius * float(np.sin(angle))))
        inside = 0
        for local_x, local_y in samples:
            point = (
                ellipse_center
                + eye.horizontal_axis * (local_x * radius_x)
                + eye.vertical_axis * (local_y * radius_y)
            )
            if cv2.pointPolygonTest(
                polygon, (float(point[0]), float(point[1])), False
            ) >= 0.0:
                inside += 1
        return inside / len(samples)

    @staticmethod
    def _iris_center_clearance(eye: EyeGaze, center: np.ndarray) -> float:
        """Return signed distance from an iris centre to the observed lid ring."""

        polygon = np.asarray(eye.contour, dtype=np.float32).reshape((-1, 1, 2))
        if polygon.shape[0] < 3:
            return float("-inf")
        return float(
            cv2.pointPolygonTest(
                polygon, (float(center[0]), float(center[1])), True
            )
        )

    @classmethod
    def _source_is_visible(cls, eye: EyeGaze, source: np.ndarray) -> bool:
        center_margin = max(0.75, min(1.65, eye.height * 0.15))
        return (
            cls._iris_center_clearance(eye, source) >= center_margin
            and cls._iris_visible_fraction(eye, source) >= MIN_SOURCE_IRIS_VISIBLE_FRACTION
        )

    @classmethod
    def _destination_is_visible(cls, eye: EyeGaze, destination: np.ndarray) -> bool:
        center_margin = max(0.85, min(1.85, eye.height * 0.17))
        return (
            cls._iris_center_clearance(eye, destination) >= center_margin
            and cls._iris_visible_fraction(eye, destination)
            >= MIN_DESTINATION_IRIS_VISIBLE_FRACTION
        )

    @classmethod
    def _safe_visible_destination(
        cls,
        eye: EyeGaze,
        source: np.ndarray,
        destination: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool]:
        """Clamp a flow target before it can enter a lid-covered region.

        Returns ``(endpoint, limited_by_lid, safe)``.  Vertical motion is also
        capped to one quarter of the measured visible opening per frame.  This
        is a physical aperture guard, not a generic model-angle reduction.
        """

        source = np.asarray(source, dtype=np.float32)
        candidate = np.asarray(destination, dtype=np.float32).copy()
        if not cls._source_is_visible(eye, source):
            return source.copy(), False, False

        direction = candidate - source
        vertical_motion = float(np.dot(direction, eye.vertical_axis))
        vertical_limit = max(0.75, eye.height * 0.25)
        limited = False
        if abs(vertical_motion) > vertical_limit:
            candidate -= eye.vertical_axis * (
                vertical_motion - float(np.clip(vertical_motion, -vertical_limit, vertical_limit))
            )
            limited = True

        if cls._destination_is_visible(eye, candidate):
            return candidate.astype(np.float32), limited, True

        # Keep only the safe prefix of the source-to-target path.  This is
        # important when an otherwise useful horizontal correction also has a
        # small vertical component from head roll.
        direction = candidate - source
        low, high = 0.0, 1.0
        for _ in range(12):
            middle = (low + high) * 0.5
            trial = source + direction * middle
            if cls._destination_is_visible(eye, trial):
                low = middle
            else:
                high = middle
        if low <= 0.06:
            return source.copy(), True, False
        return (source + direction * low).astype(np.float32), True, True

    @staticmethod
    def _valid_frame_shape(frame_bgr: np.ndarray) -> tuple[int, int] | None:
        if not isinstance(frame_bgr, np.ndarray):
            return None
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None
        height, width = frame_bgr.shape[:2]
        if height < 2 or width < 2:
            return None
        return int(height), int(width)

    @staticmethod
    def _validated_landmarks(landmarks: np.ndarray) -> np.ndarray | None:
        try:
            points = np.asarray(landmarks, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if points.ndim != 2 or points.shape[0] < 478 or points.shape[1] < 2:
            return None
        points = points[:, :2]
        if not np.isfinite(points).all():
            return None
        return points

    @staticmethod
    def _canonicalize_points(points: np.ndarray, width: int, mirrored: bool) -> np.ndarray:
        result = np.asarray(points, dtype=np.float32).copy()
        if mirrored:
            result[:, 0] = float(width - 1) - result[:, 0]
        return result

    @staticmethod
    def _canonicalize_point(point: np.ndarray, width: int, mirrored: bool) -> np.ndarray:
        result = np.asarray(point, dtype=np.float32).copy()
        if mirrored:
            result[0] = float(width - 1) - result[0]
        return result

    @staticmethod
    def _decanonicalize_point(point: np.ndarray, width: int, mirrored: bool) -> np.ndarray:
        return NeuralEyeWarper._canonicalize_point(point, width, mirrored)

    @staticmethod
    def _eye_length(points: np.ndarray, spec: EyeModelSpec) -> float:
        corner_a, corner_b = points[list(spec.corner_indices), :2]
        # The exported model's crop was trained with the horizontal corner
        # span, not a diagonal Euclidean length.  Preserve that convention
        # under a small head roll so its fixed 64x48 input geometry remains
        # compatible with the feature maps below.
        return abs(float(corner_b[0] - corner_a[0]))

    @staticmethod
    def _is_profile_or_degenerate(points: np.ndarray) -> bool:
        right_length = NeuralEyeWarper._eye_length(points, EYE_MODEL_SPECS["R"])
        left_length = NeuralEyeWarper._eye_length(points, EYE_MODEL_SPECS["L"])
        if min(right_length, left_length) < 10.0:
            return True
        ratio = right_length / max(left_length, 1e-6)
        return not 0.42 <= ratio <= 2.38

    @staticmethod
    def _assign_model_sides(
        eyes: Sequence[EyeGaze], points: np.ndarray
    ) -> dict[int, str]:
        """Match current eye geometry to semantic model sides by image distance."""

        if not eyes:
            return {}
        semantic_centres = {
            side: np.mean(points[list(spec.corner_indices), :2], axis=0)
            for side, spec in EYE_MODEL_SPECS.items()
        }
        if len(eyes) == 1:
            distances = {
                side: float(np.linalg.norm(eyes[0].center - center))
                for side, center in semantic_centres.items()
            }
            return {0: min(distances, key=distances.get)}
        # Only two eye models exist.  For the usual two-eye input, evaluate
        # both assignments together so a mirrored preview cannot allocate the
        # same semantic model twice by a greedy nearest-neighbour tie.
        pairings = (("R", "L"), ("L", "R"))
        best = min(
            pairings,
            key=lambda sides: sum(
                float(np.linalg.norm(eyes[index].center - semantic_centres[side]))
                for index, side in enumerate(sides)
            ),
        )
        return {index: side for index, side in enumerate(best[: len(eyes)])}

    def _angle_from_canonical_delta(
        self,
        canonical_delta: np.ndarray,
        eye_length: float,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Map desired image motion through the measured model response.

        The old MVP treated iris motion as a spherical ``asin`` angle.  That
        is not the contract of this ONNX model and made a one- or two-pixel
        calibration request look like a large whole-eye deformation.  Each
        model now receives only a control angle obtained by inverting its own
        measured response curve.  The returned endpoint is evaluated forward
        through that same curve, so the preview marker always describes the
        request that will actually reach the renderer.
        """

        response = self._response_curves.get(side)
        if response is None:
            raise NeuralEyeModelError(f"No response curve is loaded for {side}-eye model")
        delta = np.asarray(canonical_delta, dtype=np.float32)
        if delta.shape != (2,) or not np.isfinite(delta).all():
            return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), True

        # Convert native camera pixels to the ONNX crop's 64x48 grid.  The
        # crop dimensions come from the documented model preprocessing, not
        # from the caller's local eye axes.
        x_scale = MODEL_WIDTH / max(1e-4, CROP_WIDTH_PER_EYE_LENGTH * float(eye_length))
        y_scale = MODEL_HEIGHT / max(1e-4, CROP_HEIGHT_PER_EYE_LENGTH * float(eye_length))
        desired_model_x = float(delta[0]) * x_scale * RESPONSE_TARGET_GAIN
        desired_model_y = float(delta[1]) * y_scale * RESPONSE_TARGET_GAIN
        active_x = abs(desired_model_x) >= MIN_AXIS_MODEL_PIXELS
        active_y = abs(desired_model_y) >= MIN_AXIS_MODEL_PIXELS
        if not active_x:
            desired_model_x = 0.0
        if not active_y:
            desired_model_y = 0.0
        if float(np.hypot(desired_model_x, desired_model_y)) < MIN_RENDER_MODEL_PIXELS:
            return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), False

        def inverse(axis: _ResponseAxis, target: float) -> tuple[float, bool]:
            lower = float(min(axis.pixels[0], axis.pixels[-1]))
            upper = float(max(axis.pixels[0], axis.pixels[-1]))
            clipped_target = float(np.clip(target, lower, upper))
            return axis.angle_for_pixels(clipped_target), not np.isclose(
                clipped_target, target, atol=1e-5
            )

        # ``input_ang`` is [vertical, horizontal].  Each response axis is
        # already expressed in image coordinates, including the vertical sign.
        requested_vertical, vertical_curve_limited = inverse(
            response.vertical, desired_model_y
        )
        requested_horizontal, horizontal_curve_limited = inverse(
            response.horizontal, desired_model_x
        )
        requested_angle = np.asarray(
            (requested_vertical, requested_horizontal), dtype=np.float32
        ) * self.angle_sign
        applied_angle = np.asarray(
            (
                np.clip(requested_angle[0], -SAFE_VERTICAL_DEGREES, SAFE_VERTICAL_DEGREES),
                np.clip(
                    requested_angle[1], -SAFE_HORIZONTAL_DEGREES, SAFE_HORIZONTAL_DEGREES
                ),
            ),
            dtype=np.float32,
        )
        inversion_angle = applied_angle / self.angle_sign
        applied_model_x = (
            response.horizontal.pixels_for_angle(float(inversion_angle[1]))
            if active_x
            else 0.0
        )
        applied_model_y = (
            response.vertical.pixels_for_angle(float(inversion_angle[0]))
            if active_y
            else 0.0
        )
        applied_delta = np.asarray(
            (applied_model_x / x_scale, applied_model_y / y_scale), dtype=np.float32
        )
        limited = (
            vertical_curve_limited
            or horizontal_curve_limited
            or not np.allclose(applied_angle, requested_angle, atol=1e-4)
        )
        return applied_angle, applied_delta, limited

    @staticmethod
    def _skipped_plan(
        index: int,
        eye: EyeGaze,
        reason: str,
        requested: np.ndarray | None = None,
        *,
        side: str | None = None,
        crop: NativeEyeCrop | None = None,
    ) -> NeuralEyeWarpPlan:
        source = np.asarray(eye.iris_center, dtype=np.float32).copy()
        target = (
            np.asarray(requested, dtype=np.float32).copy()
            if requested is not None
            else source.copy()
        )
        return NeuralEyeWarpPlan(
            source=source,
            requested_target=target,
            destination=source.copy(),
            delta=np.zeros(2, dtype=np.float32),
            applied=False,
            reason=reason,
            model_side=side,
            crop=crop,
            eye_index=index,
        )

    def _render_crop(
        self,
        source_frame: np.ndarray,
        canonical_landmarks: np.ndarray,
        crop: NativeEyeCrop,
        side: str,
        angle: np.ndarray | None,
    ) -> np.ndarray:
        if angle is None or np.asarray(angle).shape != (2,):
            raise NeuralEyeModelError("Missing neural gaze angle in render plan")
        session = self._sessions.get(side)
        tensor_names = self._tensor_names.get(side)
        if session is None or tensor_names is None:
            raise NeuralEyeModelError(f"Neural {side}-eye model is not loaded")
        source = source_frame[crop.top : crop.bottom, crop.left : crop.right]
        if source.shape[:2] != (crop.height, crop.width):
            raise NeuralEyeModelError("Eye crop no longer fits the current frame")
        model_image = cv2.resize(
            source, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32) / 255.0
        anchor_map = self.make_anchor_map(canonical_landmarks, crop)
        feeds = {
            tensor_names.image_input: model_image[None, ...],
            tensor_names.feature_input: anchor_map[None, ...],
            tensor_names.angle_input: np.asarray(angle, dtype=np.float32).reshape(1, 2),
        }
        try:
            flow_raw, lighting = session.run(
                [tensor_names.flow_output, tensor_names.lighting_output], feeds
            )
        except Exception as exc:  # pragma: no cover - backend error text varies.
            raise NeuralEyeModelError(f"ONNX inference failed for {side} eye: {exc}") from exc
        flow = self._as_model_hwc(flow_raw, 2, "flow_raw:0")
        lcm = self._as_model_hwc(lighting, 2, "lcm_map:0")
        return self.apply_flow_and_lighting(source, flow, lcm)

    @staticmethod
    def apply_flow_and_lighting(
        source_bgr: np.ndarray,
        flow_raw: np.ndarray,
        lcm_map: np.ndarray,
        *,
        preserve_sclera: bool = True,
    ) -> np.ndarray:
        """Apply the learned fields at the live crop's native resolution.

        ``flow_raw`` is passed through ``tanh`` as in the exported model.  The
        graph's 64 x 48 coordinate system is converted to native crop pixels
        with the same endpoint convention used by its TensorFlow export
        before ``cv2.remap``.
        """

        if source_bgr.ndim != 3 or source_bgr.shape[2] != 3:
            raise ValueError("source_bgr must be an H x W x 3 image")
        height, width = source_bgr.shape[:2]
        if height < 2 or width < 2:
            raise ValueError("source_bgr is too small for neural eye rendering")
        flow = NeuralEyeWarper._as_model_hwc(flow_raw, 2, "flow_raw:0")
        lcm = NeuralEyeWarper._as_model_hwc(lcm_map, 2, "lcm_map:0")
        native_flow = cv2.resize(
            np.tanh(flow).astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
        )
        native_lcm = cv2.resize(
            lcm.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
        )

        # Match the TensorFlow grid de-normalisation used when these ONNX
        # models were exported.  It is intentionally ``64 / 63`` and
        # ``48 / 47`` rather than a generic native-pixel identity grid: the
        # learned raw flow includes the corresponding small compensation.
        # Replacing it with a mathematically tidy identity grid makes real
        # model output subtly drift and can turn a correct iris move into a
        # blurry, offset eye.
        map_x = (
            np.arange(width, dtype=np.float32)[None, :]
            * (MODEL_WIDTH / float(MODEL_WIDTH - 1))
            + 0.5 * native_flow[..., 0] * width
        )
        map_y = (
            np.arange(height, dtype=np.float32)[:, None]
            * (MODEL_HEIGHT / float(MODEL_HEIGHT - 1))
            + 0.5 * native_flow[..., 1] * height
        )
        source_float = source_bgr.astype(np.float32) / 255.0
        warped = cv2.remap(
            source_float,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # The learned light-colour modulation has a scalar multiplicative and
        # additive field shared by the BGR channels: relit = warped * l0 + l1.
        gain = native_lcm[..., 0:1]
        bias = native_lcm[..., 1:2]
        relit = np.clip(warped * gain + bias, 0.0, 1.0)
        if not preserve_sclera:
            return np.rint(relit * 255.0).astype(np.uint8)

        # The learned lighting map is useful around the iris, but can make
        # bright low-saturation sclera turn gray.  Preserve the sharp,
        # full-resolution warped sclera while retaining the learned lighting
        # near coloured iris/pupil detail.  This remains entirely inside the
        # eye-aperture alpha used by ``apply``.
        maximum = np.max(warped, axis=2)
        minimum = np.min(warped, axis=2)
        saturation = (maximum - minimum) / (maximum + 1e-6)
        bright = _smoothstep(maximum, 0.42, 0.72)
        low_saturation = 1.0 - _smoothstep(saturation, 0.10, 0.26)
        sclera = cv2.GaussianBlur(
            (bright * low_saturation).astype(np.float32), (0, 0), 1.0
        )
        preserve = 0.96 * sclera[..., None]
        output = np.clip(relit * (1.0 - preserve) + warped * preserve, 0.0, 1.0)
        return np.rint(output * 255.0).astype(np.uint8)

    @staticmethod
    def _as_model_hwc(value: np.ndarray, channels: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.shape != (MODEL_HEIGHT, MODEL_WIDTH, channels):
            raise NeuralEyeModelError(
                f"{name} must have shape [1, {MODEL_HEIGHT}, {MODEL_WIDTH}, {channels}], "
                f"received {tuple(np.asarray(value).shape)}"
            )
        if not np.isfinite(array).all():
            raise NeuralEyeModelError(f"{name} contained non-finite values")
        return array

    @staticmethod
    def _validate_session(session: Any, side: str, path: Path) -> _ModelTensors:
        """Reject a same-named but incompatible ONNX file before webcam use."""

        try:
            inputs = {item.name: item for item in session.get_inputs()}
            outputs = {item.name: item for item in session.get_outputs()}
        except Exception as exc:
            raise NeuralEyeModelError(
                f"Neural {side}-eye model does not expose ONNX Runtime metadata: {path}"
            ) from exc
        required_inputs = {
            IMAGE_INPUT_NAME: (MODEL_HEIGHT, MODEL_WIDTH, 3),
            FEATURE_INPUT_NAME: (MODEL_HEIGHT, MODEL_WIDTH, FEATURE_CHANNELS),
            ANGLE_INPUT_NAME: (2,),
        }
        required_outputs = {
            MODEL_OUTPUT_NAME: (MODEL_HEIGHT, MODEL_WIDTH, 3),
            FLOW_OUTPUT_NAME: (MODEL_HEIGHT, MODEL_WIDTH, 2),
            LIGHTING_OUTPUT_NAME: (MODEL_HEIGHT, MODEL_WIDTH, 2),
        }
        absent = [name for name in (*required_inputs, *required_outputs) if name not in inputs and name not in outputs]
        if absent:
            raise NeuralEyeModelError(
                f"Neural {side}-eye model has incompatible named tensors: missing "
                + ", ".join(absent)
            )
        for name, trailing in required_inputs.items():
            NeuralEyeWarper._validate_tensor_shape(inputs[name], trailing, name, side)
        for name, trailing in required_outputs.items():
            NeuralEyeWarper._validate_tensor_shape(outputs[name], trailing, name, side)
        return _ModelTensors()

    @staticmethod
    def _validate_tensor_shape(item: Any, trailing: tuple[int, ...], name: str, side: str) -> None:
        shape = getattr(item, "shape", None)
        if shape is None or len(shape) != len(trailing) + 1:
            raise NeuralEyeModelError(
                f"Neural {side}-eye tensor {name} has an unexpected rank: {shape}"
            )
        for actual, expected in zip(shape[-len(trailing) :], trailing):
            # ONNX may expose a symbolic dynamic dimension.  A fixed incorrect
            # dimension is rejected, while dynamic batch dimensions are fine.
            if isinstance(actual, (int, np.integer)) and actual != expected:
                raise NeuralEyeModelError(
                    f"Neural {side}-eye tensor {name} expected trailing shape "
                    f"{trailing}, received {shape}"
                )
