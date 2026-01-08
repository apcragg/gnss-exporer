"""Andrew Cragg 2024."""

import enum
import logging

import numpy as np

N_WINDOW_DEFAULT = 4


class AgcMode(enum.Enum):
    """Automatic gain control modes."""

    MOVING_AVERAGE = enum.auto()
    SECOND_ORDER = enum.auto()


class AutomaticGainControl:
    """Basic moving average gain control."""

    gain: float

    def __init__(
        self,
        alpha: float,
        *,
        amplitude: float = 1.0,
        agc_mode: AgcMode = AgcMode.MOVING_AVERAGE,
        n_window: int = N_WINDOW_DEFAULT,
    ) -> None:
        """Create gain control loop."""
        self.alpha = alpha
        self.amplitude = amplitude
        self.agc_mode = agc_mode
        self.gain = 1.0
        self.window = np.zeros(n_window, dtype=np.complex64)

    def update(self, x: np.complex64) -> None:
        """Update gain control with a new sample."""
        self.window = np.roll(self.window, 1)
        self.window[0] = x

        gk = self.amplitude / np.sqrt(np.mean(np.power(abs(self.window), 2)))

        if self.agc_mode == AgcMode.MOVING_AVERAGE:
            self.gain = self.gain * (1 - self.alpha) + (gk * self.alpha)
        else:
            logging.error(f"Unsupported AGC Mode {self.agc_mode}")
            raise ValueError

    def reset(self) -> None:
        """Reset gain control loop to defaults."""
        self.gain = 1.0
