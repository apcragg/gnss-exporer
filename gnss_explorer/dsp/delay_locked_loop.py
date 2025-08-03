"""A. Cragg 2024."""


class DelayLockedLoop:
    """GPS Delay Locked Loop (DLL) for code tracking with carrier aiding."""

    def __init__(self, bandwidth: float, sample_rate: float) -> None:
        """Initialize the Delay Locked Loop.

        Args:
            bandwidth: Desired loop bandwidth in Hz.
            sample_rate: The sampling rate of the system in Hz.

        """
        self.sample_rate = sample_rate
        self.period = 1.0 / sample_rate

        # Loop filter design
        zeta = 0.707
        wn = 8 * bandwidth * zeta / (4 * zeta**2 + 1)

        self.k1 = 2 * zeta * wn * self.period
        self.k2 = (wn * self.period) ** 2

        self.offset = 0.0  # Current offset in samples
        self.freq_est_hz = 0.0

    def update(self, timing_error: float) -> None:
        """Update the delay offset based on the timing error from the discriminator.

        Args:
            timing_error: The timing error input from the timing discriminator.

        Returns:
            offset: The updated offset in fractional samples.

        """
        # Compute the control signal using the PI controller
        self.freq_est_hz += self.k2 * timing_error
        # Update the offset
        self.offset += self.k1 * timing_error

    def step(self, carrier_freq_est: float = 0.0) -> None:
        """Step the loop using an optional aiding frequency."""
        # The relationship for L1 is f_code_doppler = f_carrier_doppler / 1540
        # T_CODE is 1e-3, so we must scale correctly.
        # The carrier_freq_est is in Hz. The DLL freq_est is in samples/ms.
        # This scaling may need tuning based on your units.
        aiding_term = carrier_freq_est / 1540.0  # Divided by N_P_SYMBOLS
        self.offset += self.freq_est_hz * self.period  # + (aiding_term * self.period)

    def reset(self) -> None:
        """Reset the integrator and offset to zero."""
        self.freq_est_hz = 0.0
        self.offset = 0.0
