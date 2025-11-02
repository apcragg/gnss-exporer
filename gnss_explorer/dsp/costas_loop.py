"""A. Cragg 2024."""

import enum

import numpy as np


class LoopState(enum.Enum):
    """Defines the state of the carrier tracking loop."""

    FLL = enum.auto()
    PLL = enum.auto()


class CarrierTrackingLoop:
    """Carrier tracking loop."""

    freq_estimate: float
    start_pll: bool

    def __init__(self, *, fll_bw: float, pll_bw: float, sample_rate: float) -> None:
        """Create a Costas carrier tracking loop.

        Args:
            fll_bw: Frequency Locked Loop bandwidth [Hz]
            pll_bw: FPhase Locked Loop bandwidth [Hz]
            sample_rate: Samples per second

        """
        self.state = LoopState.FLL
        self.fll_bw = fll_bw
        self.pll_bw = pll_bw
        self.sample_rate = sample_rate
        self.T = 1.0 / sample_rate
        self.freq_estimate = 0.0
        self.error = 0.0
        # FLL
        self.fll_w0 = 4 * fll_bw
        # PLL
        self.pll_w0 = pll_bw / 0.53
        self.pll_a2 = 1.414 * self.pll_w0

        self.n = 0

    def _fold_mod_pi(self, theta: float) -> float:
        """Fold angle to [-pi/2, +pi/2] by modulo-pi mapping.

        This cancels +/-pi jumps caused by data-bit flips.
        """
        return ((theta + 0.5 * np.pi) % np.pi) - 0.5 * np.pi

    def update(self, input_signal: np.complex64) -> None:
        """Update the tracking loop with a new sample.

        Args:
            input_signal: Complex sample

        Returns:
            in_phase: Real valued tracking output from the loop

        """
        if self.prev_sym is None:
            self.prev_sym = input_signal
            return

        # PLL
        self.error = np.atan(input_signal.imag / input_signal.real)
        pll_error = self.error / (2 * np.pi)

        # FLL
        d_phi = np.angle(input_signal * np.conj(self.prev_sym))  # [rad/s / T]
        fll_error = self._fold_mod_pi(d_phi) / (2 * np.pi)  # [Hz / T]

        integrator_old = self.integrator
        self.integrator += (self.fll_w0 * fll_error * self.T) + (
            (self.pll_w0**2) * pll_error * self.T
        )

        self.freq_estimate = ((self.integrator + integrator_old) * 0.5) + (
            self.pll_a2 * self.pll_w0 * pll_error
        )

        self.prev_sym = input_signal
        self.n += 1

        if self.start_pll:
            self.start_pll = False

    def reset(self) -> None:
        """Reset the tracking loop."""
        self.freq_estimate_pll = 0.0
        self.freq_estimate_fll = 0.0
        self.error = 0.0
        self.prev_sym = None
        self.integrator = 0
        self.sum_e_pll: float = 0
        self.sum_e: float = 0
        self.start_pll = False
