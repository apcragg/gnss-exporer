"""A. Cragg 2024."""

import enum

import numpy as np


class LoopState(enum.Enum):
    """Defines the state of the carrier tracking loop."""

    FLL = enum.auto()
    PLL = enum.auto()


class CarrierTrackingLoop:
    """Carrier tracking loop."""

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
        self.phase_estimate = 0.0
        self.freq_estimate = 0.0
        self.error = 0.0
        # FLL
        zeta = 0.707
        wn = fll_bw * 8 * zeta / (4 * zeta**2 + 1)
        self.fll_k1 = 2 * zeta * wn * self.T
        self.fll_k2 = (wn * self.T) ** 2
        # PLL
        zeta = 0.707
        wn = pll_bw * 8 * zeta / (4 * zeta**2 + 1)
        self.pll_k1 = 2 * zeta * wn
        self.pll_k2 = (wn) ** 2

        self.n = 0

    def _fold_mod_pi(self, theta: float) -> complex:
        """Fold angle to [-pi/2, +pi/2] by modulo-π mapping.

        This cancels ±π jumps caused by data-bit flips.
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

        if self.state == LoopState.PLL:
            # Calculate error signal
            self.error = np.atan(input_signal.imag / input_signal.real)

            self.sum_e += self.pll_k2 * self.error * self.T

            self.freq_estimate = self.pll_k1 * self.error + self.sum_e
        else:
            d_phi = np.angle(input_signal * np.conj(self.prev_sym))
            omega_mes = self._fold_mod_pi(d_phi)

            self.error = omega_mes * self.sample_rate  # [rad/s]

            self.sum_e += self.error
            self.freq_estimate += self.fll_k2 * self.sum_e + self.fll_k1 * self.error

        self.phase_estimate = self.phase_estimate % (2 * np.pi)

        self.prev_sym = input_signal
        self.n += 1
        if self.n == 2000:
            self.state = LoopState.PLL
            self.sum_e = self.freq_estimate

    def reset(self) -> None:
        """Reset the tracking loop."""
        self.phase_estimate = 0.0
        self.freq_estimate = 0.0
        self.error = 0.0
        self.prev_sym = None
        self.f_ll_integrator = 0
        self.sum_e = 0
