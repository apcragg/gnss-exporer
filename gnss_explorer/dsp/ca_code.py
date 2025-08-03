"""GPS C/A PRN Code Generator in Python.

This script generates the GPS C/A PRN codes using two linear feedback shift registers (LFSRs).
The LFSRs are combined to produce a Gold Code which is used by GPS satellites for signal spreading.
"""

from __future__ import annotations

import numba
import numpy as np
from numpy import typing as npt


# Function to generate the G1 and G2 sequences using Linear Feedback Shift Registers (LFSRs)
def _generate_g1_g2() -> tuple[list[int], list[int]]:
    # Initialize G1 and G2 registers (10 bits each, starting with all ones)
    g1 = [1] * 10
    g2 = [1] * 10

    # Lists to store the output sequences of G1 and G2
    g1_sequence = []
    g2_sequence = []

    # Generate G1 and G2 sequences for 1023 bits
    for _ in range(1023):
        # Append the output of G1 (first register) to the sequence
        g1_sequence.append(g1[-1])

        # Append the output of G2 (second register) to the sequence
        g2_sequence.append(g2[-1])

        # Feedback for G1: XOR of bits 3 and 10 (1-based index)
        g1_feedback = g1[2] ^ g1[9]

        # Feedback for G2: XOR of bits 2, 3, 6, 8, 9, and 10 (1-based index)
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        # Shift G1 register
        g1 = [g1_feedback, *g1[:-1]]

        # Shift G2 register
        g2 = [g2_feedback, *g2[:-1]]

    return g1_sequence, g2_sequence


# Function to generate the PRN code for a given satellite PRN number
def ca_code(prn_number: int) -> npt.NDArray[np.int_]:
    """Get CA code for specified PRN."""
    # Get the G1 and G2 sequences
    g1_sequence, g2_sequence = _generate_g1_g2()

    # Satellite-specific tap selection (used for G2 sequence)
    # This determines which two bits from G2 are XORed together to produce the PRN code.
    # The mapping is based on the GPS ICD (Interface Control Document).
    g2_tap_map = {
        1: (2, 6),
        2: (3, 7),
        3: (4, 8),
        4: (5, 9),
        5: (1, 9),
        6: (2, 10),
        7: (1, 8),
        8: (2, 9),
        9: (3, 10),
        10: (2, 3),
        11: (3, 4),
        12: (5, 6),
        13: (6, 7),
        14: (7, 8),
        15: (8, 9),
        16: (9, 10),
        17: (1, 4),
        18: (2, 5),
        19: (3, 6),
        20: (4, 7),
        21: (5, 8),
        22: (6, 9),
        23: (1, 3),
        24: (4, 6),
        25: (5, 7),
        26: (6, 8),
        27: (7, 9),
        28: (8, 10),
        29: (1, 6),
        30: (2, 7),
        31: (3, 8),
        32: (4, 9),
    }

    # Get the specific taps for the given PRN number
    tap1, tap2 = g2_tap_map[prn_number]

    # Generate the PRN sequence by XORing G1 with the specified G2 taps
    prn_sequence = []
    for i in range(1023):
        # G2 output is XOR of the selected taps (note: indices are zero-based here)
        g2_output = g2_sequence[i - (tap1 - 1)] ^ g2_sequence[i - (tap2 - 1)]

        # PRN output is G1 XOR G2
        prn_output = g1_sequence[i - 9] ^ g2_output
        prn_sequence.append(prn_output)

    return np.array(prn_sequence)


codes = np.array([ca_code(code + 1) for code in range(32)]) * 2.0 - 1.0


def ca_code_dll(fs: float, phase: float, prn_number: int, n_code: int) -> list[float]:
    """Sample the CA code for use in digital locked loop."""
    code = codes[prn_number - 1]
    x_code = []

    for idx in range(n_code):
        idx_code = int(np.round(idx * fs + phase)) % len(code)
        x_code.append(code[idx_code])

    return x_code


@numba.njit()
def ca_correlate(
    x: npt.NDArray[np.complex64],
    fs: float,
    phase: float,
    prn_number: int,
    norm: bool = True,  # noqa: FBT001, FBT002, Bug in numba.
) -> np.complex64:
    """Correlate signal with PRN code at a specified phase."""
    code = codes[prn_number - 1]
    code_len = len(code)
    x_corr: np.complex64 = np.complex64(0.0)

    idx_phase = phase

    for idx_signal in range(len(x)):
        idx_phase += fs
        idx_code = int(idx_phase) % code_len
        x_code = code[idx_code]
        x_corr += x[idx_signal] * x_code

    if norm:
        x_corr /= len(x)
    return x_corr
