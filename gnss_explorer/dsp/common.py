"""A. Cragg 2024."""

from __future__ import annotations

import numba
import numpy as np
from numpy import typing as npt


@numba.njit()
def freq_shift(x: np.ndarray, fs: float, f_shift: float, phase: np.complex64) -> np.complex64:
    """Apply a frequency shift to a complex signal in place.

    This function multiplies each sample in the input array by an incrementing phase factor,
    shifting the signal in frequency. The operation is performed in place, and the final phase after
    processing is returned for potential use in continuous processing.

    Args:
        x (np.ndarray): Complex input signal (modified in place).
        fs (float): Sample rate of the signal.
        f_shift (float): Frequency shift in Hz.
        phase (complex): Initial phase offset.

    Returns:
        complex: The final phase offset after applying the frequency shift.

    """
    t_sample = 1.0 / fs
    phase_increment = np.exp(-2j * np.pi * f_shift * t_sample)

    # Acummulate phase as the shift is applied to avoid repeated calculations.
    current_phase = phase
    for idx in range(len(x)):
        x[idx] *= current_phase
        current_phase *= phase_increment

    return current_phase


def rms(x: npt.NDArray[np.complex64] | npt.NDArray[np.float32]) -> float:
    """Calculate root mean square of a signal.

    Args:
        x (np.ndarray[np.complex64] | np.ndarray[float]): Time series signal.

    Returns:
        Root-mean square value of time series signal.

    """
    return np.sqrt(np.mean(np.power(np.abs(x), 2)))


def quantize(
    x: npt.NDArray[np.complex64], n_bits: int, sigma_scale: float = 2
) -> npt.NDArray[np.complex64]:
    """Quantize a complex signal to a specified number of bits.

    This function quantizes the input complex signal by processing its real and imaginary
    components independently. First, the signal is scaled by the larger standard deviation
    of its real or imaginary part (multiplied by sigma_scale). Then it is scaled to the
    quantization range defined by n_bits, quantized by flooring, adjusted to be zero-mean,
    and finally rescaled back to the original range.

    Args:
        x (np.ndarray): Input complex signal (dtype np.complex64).
        n_bits (int): Number of bits for quantization.
        sigma_scale (float, optional): Scaling factor for determining the quantization threshold
            (default is 2).

    Returns:
        np.ndarray: Quantized complex signal (dtype np.complex64).

    """
    # Scale the signal before quantizing.
    # Use the larger standard deviation from the real or imaginary part.
    x_threshold = np.maximum(np.std(np.real(x)), np.std(np.imag(x))) * sigma_scale
    x /= x_threshold
    x *= 2 ** (n_bits - 1)

    # Quantize the real and imaginary signal components independently.
    x = np.floor(np.real(x)) + 1j * np.floor(np.imag(x))
    x_real = np.clip(np.real(x), -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
    x_imag = np.clip(np.imag(x), -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
    x = x_real + 1j * x_imag

    # Adjust the signal to be zero-mean (effective when quantization levels are even).
    x += 0.5 + 0.5j
    x /= 2 ** (n_bits - 1)

    # Rescale the signal to its original level.
    x *= x_threshold
    return x
