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
        zeta = 0.707
        wn = fll_bw * 8 * zeta / (4 * zeta**2 + 1)
        self.fll_k1 = 2 * zeta * wn
        self.fll_k2 = (wn) ** 2
        # PLL
        zeta = 0.707
        wn = pll_bw * 8 * zeta / (4 * zeta**2 + 1)
        self.pll_k1 = 2 * zeta * wn
        self.pll_k2 = (wn) ** 2

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
        self.sum_e_pll += self.pll_k2 * self.error * self.T
        self.freq_estimate_pll = self.pll_k1 * self.error + self.sum_e_pll

        # FLL
        d_phi = np.angle(input_signal * np.conj(self.prev_sym))  # [T * rad/s]
        omega_mes = self._fold_mod_pi(d_phi)  # [T * rad/s]

        self.sum_e += omega_mes
        self.freq_estimate_fll = self.fll_k1 * omega_mes + (self.sum_e * self.fll_k2) * self.T * 0.5

        self.freq_estimate = self.freq_estimate_pll * 1 + self.freq_estimate_fll * 1

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
        self.f_ll_integrator = 0
        self.sum_e_pll: float = 0
        self.sum_e: float = 0
        self.start_pll = False
