from __future__ import annotations

import unittest

import numpy as np

from smoothing import ArrayEMA, ScalarEMA


class SmoothingTests(unittest.TestCase):
    def test_array_ema_blends_without_mutating_input(self) -> None:
        smoother = ArrayEMA(0.5)
        first = np.asarray([0.0, 10.0], dtype=np.float32)
        np.testing.assert_allclose(smoother.update("eye", first), first)
        first[0] = 99.0
        blended = smoother.update("eye", np.asarray([10.0, 20.0], dtype=np.float32))
        np.testing.assert_allclose(blended, [5.0, 15.0])

    def test_shape_change_resets_array_state(self) -> None:
        smoother = ArrayEMA(0.5)
        smoother.update("eye", np.zeros((2,), dtype=np.float32))
        value = smoother.update("eye", np.ones((3,), dtype=np.float32))
        np.testing.assert_allclose(value, np.ones((3,), dtype=np.float32))

    def test_scalar_ema(self) -> None:
        smoother = ScalarEMA(0.25)
        self.assertEqual(smoother.update(20.0), 20.0)
        self.assertAlmostEqual(smoother.update(4.0), 16.0)


if __name__ == "__main__":
    unittest.main()
