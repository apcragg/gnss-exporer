"""Andrew Cragg 2024."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from gnss_explorer.dsp import ca_code
from gnss_explorer.nav import nav

DEFAULT_F_DOPPLER_MAX_HZ = 25e3
DEFAULT_F_DOPPLER_STEP_HZ = 250

DEFAULT_DETECTION_THRESHOLD = 50


@dataclasses.dataclass
class CodeDetection:
    """Code detection data for the receiver."""

    n_offset: int
    f_doppler: float
    p_code_power: float


class L1CADetector:
    """Detector for the GPS L1 C/A code."""

    def __init__(
        self,
        f_s: float,
        *,
        f_max_doppler: float = DEFAULT_F_DOPPLER_MAX_HZ,
        f_step_doppler: float = DEFAULT_F_DOPPLER_STEP_HZ,
        p_ratio_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    ) -> None:
        """Create detector."""
        self.f_s = f_s
        self.est_frame = f_s * nav.T_CODE
        self.n_frame = int(self.est_frame)
        self.f_oversample = nav.FS_CHIP / f_s

        self.f_max_doppler = f_max_doppler
        self.f_step_doppler = f_step_doppler
        self.p_ratio_threshold = p_ratio_threshold

    def process(self, frame: np.ndarray, p_prn: int) -> CodeDetection | None:
        """Process a frame of data and add to queue if code is detected.

        Arguments:
            frame: Array of complex samples.
            p_prn: PRN number to search for.

        Returns:
            A CodeDetection object if detection exceeds threshold, or None.

        """
        n_frame = self.n_frame
        if len(frame) != n_frame:
            logging.error(f"Frame size {len(frame)} does not match expected frame size {n_frame}")
            msg = "Frame size mismatch"
            raise ValueError(msg)

        x_code = ca_code.ca_code_dll(self.f_oversample, 0.0, p_prn, n_frame)
        fft_code = np.conj(np.fft.fft(x_code, n_frame))

        sample_idx = np.arange(n_frame) / self.f_s

        doppler_range = np.arange(
            -self.f_max_doppler, self.f_max_doppler, self.f_step_doppler, dtype=float
        )

        best_corr = 0.0
        best_doppler = 0.0
        idx_max = 0

        corrs = []

        for doppler in doppler_range:
            t = sample_idx * doppler
            shift = np.exp(-2j * np.pi * t)

            x_shifted = frame * shift

            fft_x = np.fft.fft(x_shifted)
            corr = (abs(np.fft.ifft(fft_code * fft_x)) / nav.GPS_L1_CA_CHIPS) ** 2

            corrs.extend(corr)

            current_max = corr.max()
            if current_max > best_corr:
                best_corr = current_max
                idx_max = int(corr.argmax())
                best_doppler = doppler

        corrs.sort()
        ratio = corrs[-1] / corrs[-(4 * 5)]

        logging.debug(f"Best Correlation: {best_corr:.3f}, PRN {p_prn}, Ratio: {ratio:.3f}")

        if ratio > self.p_ratio_threshold:
            detection = CodeDetection(
                n_offset=idx_max, f_doppler=best_doppler, p_code_power=best_corr
            )
            logging.info(
                f"Detected PRN Code {p_prn} at offset {idx_max} with power {best_corr:.3f}"
                f" and Doppler shift {best_doppler:.1f} Hz"
            )
            return detection

        return None
