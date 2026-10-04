"""Small, dependency-free exponential moving average helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np


class ArrayEMA:
    """Smooth arrays independently by key and reset safely on shape changes."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._values: Dict[str, np.ndarray] = {}

    def update(self, key: str, value: np.ndarray) -> np.ndarray:
        incoming = np.asarray(value, dtype=np.float32)
        previous = self._values.get(key)
        if previous is None or previous.shape != incoming.shape:
            smoothed = incoming.copy()
        else:
            smoothed = self.alpha * incoming + (1.0 - self.alpha) * previous
        self._values[key] = smoothed
        return smoothed.copy()

    def clear(self) -> None:
        self._values.clear()


class ScalarEMA:
    """EMA for a single changing scalar, used for the FPS readout."""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._value: float | None = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = float(value)
        else:
            self._value = self.alpha * float(value) + (1.0 - self.alpha) * self._value
        return self._value

    def clear(self) -> None:
        self._value = None
