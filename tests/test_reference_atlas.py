"""Geometry-only checks for the real personal eye-reference renderer."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from gaze_estimator import EyeGaze
from reference_atlas import (
    ATLAS_FORMAT,
    AtlasRecord,
    ReferenceAtlasRenderer,
    affine_from_geometries,
    aperture_alpha,
    canonical_eye_geometries,
    eye_geometry,
)


R_RING = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
L_RING = (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)
R_IRIS = (468, 469, 470, 471, 472)
L_IRIS = (473, 474, 475, 476, 477)


def _ring(center_x: float, center_y: float) -> np.ndarray:
    return np.asarray(
        (
            (-16, 0), (-13, -4), (-9, -6), (-4, -7), (1, -7), (7, -6),
            (12, -4), (15, -2), (16, 0), (15, 2), (11, 4), (6, 6),
            (0, 7), (-6, 6), (-11, 4), (-15, 2),
        ),
        dtype=np.float32,
    ) + np.asarray((center_x, center_y), dtype=np.float32)


def make_landmarks(*, r_center: tuple[float, float] = (82, 52), l_center: tuple[float, float] = (158, 52)) -> np.ndarray:
    points = np.zeros((478, 2), dtype=np.float32)
    for ring_ids, iris_ids, center in ((R_RING, R_IRIS, r_center), (L_RING, L_IRIS, l_center)):
        ring = _ring(*center)
        points[list(ring_ids)] = ring
        iris_center = np.asarray(center, dtype=np.float32)
        points[list(iris_ids)] = np.asarray(
            (
                iris_center,
                iris_center + (3, 0),
                iris_center + (0, 2),
                iris_center + (-3, 0),
                iris_center + (0, -2),
            ),
            dtype=np.float32,
        )
    return points


def eye_from_geometry(geometry: object) -> EyeGaze:
    return EyeGaze(
        contour=geometry.contour,
        iris_center=geometry.iris_center,
        ratio=geometry.iris_ratio,
        center=geometry.center,
        horizontal_axis=geometry.horizontal_axis,
        vertical_axis=geometry.vertical_axis,
        eye_width=geometry.width,
        eye_height=geometry.height,
        iris_radius=3.0,
    )


def make_record(
    geometry: object,
    *,
    side: str,
    crop_origin: tuple[float, float] | None = None,
    iris_offset: tuple[float, float] = (0, 0),
) -> AtlasRecord:
    if crop_origin is None:
        crop_origin = (float(geometry.center[0] - 46), float(geometry.center[1] - 28))
    origin = np.asarray(crop_origin, dtype=np.float32)
    image = np.full((58, 92, 3), 145, dtype=np.uint8)
    contour = geometry.contour - origin
    iris = geometry.iris_center - origin + np.asarray(iris_offset, dtype=np.float32)
    center = geometry.center - origin
    cv2.circle(image, tuple(iris.round().astype(int)), 4, (35, 35, 35), -1)
    mask = aperture_alpha(image.shape[:2], contour, geometry.height)
    assert mask is not None
    return AtlasRecord(
        record_id=f"record-{side}",
        pose_key="center",
        side=side,
        frame_index=0,
        pose=np.zeros(3, dtype=np.float32),
        image=image,
        contour=contour,
        iris_center=iris,
        center=center,
        horizontal_axis=geometry.horizontal_axis,
        vertical_axis=geometry.vertical_axis,
        width=geometry.width,
        height=geometry.height,
        openness=geometry.openness,
        sharpness=40.0,
        brightness=145.0,
        aperture_mask=mask,
    )


def serializable(record: AtlasRecord, image_file: str) -> dict[str, object]:
    return {
        "id": record.record_id,
        "pose_key": record.pose_key,
        "side": record.side,
        "frame_index": record.frame_index,
        "pose": record.pose.tolist(),
        "image_file": image_file,
        "contour": record.contour.tolist(),
        "iris_center": record.iris_center.tolist(),
        "center": record.center.tolist(),
        "horizontal_axis": record.horizontal_axis.tolist(),
        "vertical_axis": record.vertical_axis.tolist(),
        "eye_width": record.width,
        "eye_height": record.height,
        "openness": record.openness,
        "sharpness": record.sharpness,
        "brightness": record.brightness,
    }


class ReferenceAtlasTests(unittest.TestCase):
    def test_eye_geometry_and_aperture_mask_protects_outer_crop(self) -> None:
        geometry = eye_geometry(make_landmarks(), "R")
        assert geometry is not None
        self.assertGreater(geometry.width, 24.0)
        self.assertGreater(geometry.openness, 0.10)
        local = geometry.contour - np.asarray((36, 24), dtype=np.float32)
        alpha = aperture_alpha((58, 92), local, geometry.height)
        assert alpha is not None
        self.assertEqual(float(alpha[0, 0]), 0.0)
        self.assertGreater(float(alpha.max()), 0.65)
        self.assertLess(float(np.mean(alpha > 0.0)), 0.20)

    def test_canonical_cells_are_stable_when_landmark_groups_are_mirrored(self) -> None:
        points = make_landmarks()
        normal = canonical_eye_geometries(points)
        mirrored = points.copy()
        mirrored[:, 0] = 239 - mirrored[:, 0]
        # The named source landmark groups now occupy the opposite image
        # sides. The atlas cell labels must remain ordered by canonical X.
        mirrored_cells = canonical_eye_geometries(mirrored)
        self.assertLess(normal["R"].center[0], normal["L"].center[0])
        self.assertLess(mirrored_cells["R"].center[0], mirrored_cells["L"].center[0])

    def test_affine_tracks_eye_shape_not_donor_iris(self) -> None:
        reference_geometry = eye_geometry(make_landmarks(), "R")
        live_geometry = eye_geometry(make_landmarks(r_center=(106, 62)), "R")
        assert reference_geometry is not None and live_geometry is not None
        record = make_record(reference_geometry, side="R")
        result = affine_from_geometries(record, live_geometry)
        assert result is not None
        matrix, predicted_iris = result
        expected = cv2.transform(record.iris_center.reshape(1, 1, 2), matrix).reshape(2)
        self.assertTrue(np.allclose(predicted_iris, expected, atol=1e-4))
        self.assertTrue(np.allclose(predicted_iris, live_geometry.iris_center, atol=1e-4))

    def test_renderer_applies_only_real_apertures_and_mirror_path(self) -> None:
        points = make_landmarks()
        r_geometry = eye_geometry(points, "R")
        l_geometry = eye_geometry(points, "L")
        assert r_geometry is not None and l_geometry is not None
        records = (
            make_record(r_geometry, side="R", iris_offset=(4, 0)),
            make_record(l_geometry, side="L", iris_offset=(-4, 0)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_records: list[dict[str, object]] = []
            for record in records:
                filename = f"{record.side}.png"
                self.assertTrue(cv2.imwrite(str(root / filename), record.image))
                raw_records.append(serializable(record, filename))
            atlas_path = root / "atlas.json"
            atlas_path.write_text(json.dumps({"format": ATLAS_FORMAT, "records": raw_records}), encoding="utf-8")
            renderer = ReferenceAtlasRenderer(atlas_path)
            frame = np.full((120, 240, 3), 132, dtype=np.uint8)
            eyes = (eye_from_geometry(r_geometry), eye_from_geometry(l_geometry))
            # The requested old-gaze target equals the current iris here.
            # The atlas must still apply: its real donor iris is four pixels
            # away, so this protects against the former 0.9px target skip.
            plans = renderer.plan(
                frame,
                points,
                eyes,
                np.asarray([eye.ratio for eye in eyes], dtype=np.float32),
                1.0,
                np.zeros(2, dtype=np.float32),
            )
            self.assertTrue(all(plan.applied for plan in plans))
            corrected = renderer.apply(frame, points, plans)
            self.assertFalse(np.array_equal(corrected, frame))
            self.assertTrue(np.array_equal(corrected[0, 0], frame[0, 0]))

            mirrored_points = points.copy()
            mirrored_points[:, 0] = 239 - mirrored_points[:, 0]
            mirrored_eyes = tuple(
                EyeGaze(
                    contour=np.column_stack((239 - eye.contour[:, 0], eye.contour[:, 1])),
                    iris_center=np.asarray((239 - eye.iris_center[0], eye.iris_center[1]), dtype=np.float32),
                    ratio=eye.ratio,
                    center=np.asarray((239 - eye.center[0], eye.center[1]), dtype=np.float32),
                    horizontal_axis=np.asarray((-eye.horizontal_axis[0], eye.horizontal_axis[1]), dtype=np.float32),
                    vertical_axis=np.asarray((-eye.vertical_axis[0], eye.vertical_axis[1]), dtype=np.float32),
                    eye_width=eye.width,
                    eye_height=eye.height,
                    iris_radius=eye.iris_radius,
                )
                for eye in eyes
            )
            mirrored_frame = cv2.flip(frame, 1)
            mirrored_plans = renderer.plan(
                mirrored_frame,
                mirrored_points,
                mirrored_eyes,
                np.asarray([eye.ratio for eye in mirrored_eyes], dtype=np.float32),
                1.0,
                np.zeros(2, dtype=np.float32),
                mirrored=True,
            )
            self.assertTrue(all(plan.applied for plan in mirrored_plans))
            mirrored_corrected = renderer.apply(
                mirrored_frame, mirrored_points, mirrored_plans, mirrored=True
            )
            self.assertFalse(np.array_equal(mirrored_corrected, mirrored_frame))
            self.assertTrue(np.array_equal(mirrored_corrected, cv2.flip(corrected, 1)))


if __name__ == "__main__":
    unittest.main()
