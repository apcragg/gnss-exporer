"""Andrew Cragg 2024."""

from __future__ import annotations

import copy
import enum
import logging
import pathlib
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt

from gnss_explorer.detection import l1ca_detector
from gnss_explorer.dsp import common
from gnss_explorer.nav import nav, pvt
from gnss_explorer.receivers import l1ca

logging.getLogger().setLevel(logging.INFO)

FC_SIGNAL = 1575.420e6
FS_SIGNAL = 4.0e6
ACTIVE_PRNS = list(range(32))

N_RX = 100_000
T_LOAD_S = 120
N_LOAD = int(FS_SIGNAL * T_LOAD_S)

N_YIELD_DEFAULT = int(1e5)
N_FRAMES_DETECT = 10


class CaptureBinaryFormat(enum.Enum):
    """Supported on-disk encodings for complex I/Q sample captures.

    The enumeration distinguishes between two common interleaved
    integer formats and a native NumPy complex-64 format:

    Attributes:
        FILE_I16: Interleaved 16-bit signed integers
            (I0, Q0, I1, Q1 ...).
        FILE_I8:  Interleaved 8-bit signed integers
            (I0, Q0, I1, Q1 ...).
        FILE_NUMPY_C64: NumPy array of dtype ``complex64`` written
            directly to disk (real and imaginary 32-bit floats packed
            by NumPy's ``tofile`` / ``save``).

    """

    FILE_I16 = enum.auto()
    FILE_I8 = enum.auto()
    FILE_NUMPY_C64 = enum.auto()


def file_source_complex_mmap(
    file_path: pathlib.Path,
    capture_format: CaptureBinaryFormat,
    n_yield: int = N_YIELD_DEFAULT,
    *,
    f_shift: float = 0.0,
    n_load: float | None = None,
) -> Iterator[npt.NDArray[np.complex64]]:
    """Stream complex samples from a capture file via NumPy mem-mapping.

    The function memory-maps the input file, converts interleaved I/Q samples
    to complex64, applies an optional complex baseband frequency shift, and
    yields them in fixed-size blocks.

    Args:
        file_path: Path to the binary capture file.
        capture_format: Encoding of the samples on disk. Supported values
            are `CaptureBinaryFormat.FILE_I8`, `FILE_I16`, or
            `FILE_NUMPY_C64`.
        n_yield: Number of **complex** samples to emit per iteration
            (defaults to ``N_YIELD_DEFAULT``). Internally we load
            ``2 * n_yield`` scalar values because the file is
            interleaved I/Q.
        f_shift: Frequency shift in hertz to apply to the block before
            yielding. Use ``0.0`` to disable.
        n_load: If given, limits the total number of **complex** samples
            loaded from the file; useful for quick tests on long captures.
            ``None`` (default) loads the entire file.

    Yields:
        A 1-D `numpy.ndarray` of dtype ``complex64`` and length ``n_yield``
        containing a contiguous chunk of frequency-shifted I/Q samples.

    Raises:
        ValueError: If `capture_format` is not one of the supported
            ``CaptureBinaryFormat`` members.
        RuntimeError: If the file has an odd number of scalar samples,
            indicating a malformed interleaved I/Q file.

    """
    x: np.memmap
    if capture_format == CaptureBinaryFormat.FILE_I8:
        np_dtype = np.int8
        x = np.memmap(filename=file_path, dtype=np_dtype, mode="r")
    elif capture_format == CaptureBinaryFormat.FILE_I16:
        np_dtype = np.int16
        x = np.memmap(filename=file_path, dtype=np_dtype, mode="r")
    elif capture_format == CaptureBinaryFormat.FILE_NUMPY_C64:
        np_dtype = np.float32
        x = np.memmap(filename=file_path, dtype=np_dtype, mode="r")
    else:
        msg = f"Binary format {capture_format} is unsupported."
        raise ValueError(msg)

    if len(x) % 2 == 1:
        logging.error("Complex binary file should have even number of samples.")
        raise RuntimeError
    idx = int(4e6)
    n_samples = len(x)
    if n_load:
        n_samples = np.minimum(n_load * 2, n_samples)
    x_il = np.zeros(n_samples, dtype=np.complex64)
    x_c = np.zeros(n_samples // 2, dtype=np.complex64)
    end_phase = np.complex64(1.0)
    while True:
        n_remaining = n_samples - idx
        n_to_load = min(n_remaining, n_yield * 2)

        x_il = x[idx : idx + n_to_load].astype(np.complex64)
        x_c = x_il[0::2] + 1j * x_il[1::2]
        end_phase = common.freq_shift(x_c, fs=FS_SIGNAL, f_shift=f_shift, phase=end_phase)

        yield x_c

        idx += n_to_load

        if idx == n_samples:
            return


def main() -> None:
    """Run the GNSS Receiver."""
    n_load = int(T_LOAD_S * FS_SIGNAL)

    x_stream = file_source_complex_mmap(
        pathlib.Path("/home/apcragg/Documents/gps.bin"),
        capture_format=CaptureBinaryFormat.FILE_I16,
        n_yield=N_RX,
        n_load=n_load,
        f_shift=0,
    )

    receivers: dict[int, l1ca.L1CAReceiver] = {}
    detectors: dict[int, l1ca_detector.L1CADetector] = {}

    solver = pvt.PvtSolver(config=pvt.PvtSolverConfig(solution_period_hz=1.0))
    solver.start()

    config = l1ca.L1CAFixedConfig(
        f_s=FS_SIGNAL,
        f_c=FC_SIGNAL,
        p_prn=1,
        p_agc_alpha=0.1,
        f_fll_bw=4,
        f_pll_bw=18,
        f_dll_bw=5,
        n_subframe_lock=2,
        n_flywheel_allowed=10,
        b_sync_pattern=np.array([1, 0, 0, 0, 1, 0, 1, 1]),
    )

    for prn in ACTIVE_PRNS:
        prn_config = copy.deepcopy(config)
        prn_config.p_prn = prn
        receivers[prn] = l1ca.L1CAReceiver(config=prn_config, solver=solver)
        detectors[prn] = l1ca_detector.L1CADetector(
            f_s=FS_SIGNAL, f_max_doppler=10e3, f_step_doppler=250, p_ratio_threshold=1.6
        )

    t_start = time.time()

    n_pos = 0
    for x in x_stream:
        prn_to_detect = ACTIVE_PRNS[n_pos % len(ACTIVE_PRNS)]
        if prn_to_detect == n_pos:
            if receivers[prn_to_detect].state is l1ca.L1CAReceiverState.IDLE:
                detection = detectors[prn_to_detect].process(
                    frame=x[: detectors[prn_to_detect].n_frame * N_FRAMES_DETECT],
                    p_prn=prn_to_detect,
                )
                if detection:
                    receivers[prn_to_detect].update_detection(detection=detection)

        for prn in ACTIVE_PRNS:
            if receivers[prn].state is not l1ca.L1CAReceiverState.IDLE:
                receivers[prn].update(x, n_pos=n_pos * N_RX)

        n_pos += 1

    logging.info(f"Run time {time.time() - t_start:.3f} seconds")

    logging.info("|========== Final State ==========|")
    for prn in ACTIVE_PRNS:
        if receivers[prn].state is not l1ca.L1CAReceiverState.IDLE:
            logging.info(
                f"PRN: {prn:2}, State: {receivers[prn].state.name:10}, "
                f"Global Pos: {receivers[prn].n_global_pos}"
            )

    active_eph = []
    t_subframe = 0
    for eph in solver.ephemerides.values():
        if eph is None:
            continue
        active_eph.append(eph)
        t_subframe = eph.tow - 6

    for prn in ACTIVE_PRNS:
        if len(receivers[prn].b_symbols) > 0:
            data_real = np.array([x.symbol.real for x in receivers[prn].b_symbols])
            data_imag = np.array([x.symbol.imag for x in receivers[prn].b_symbols])
            data_c_n0_mm = np.array([x.c_n0_est_mm for x in receivers[prn].b_symbols])
            data_c_n0_nwpr = np.array([x.c_n0_est_nwpr for x in receivers[prn].b_symbols])
            b_code_phase = np.array(receivers[prn].b_code_phase) / FS_SIGNAL
            t_symbol_loop = np.arange(0, len(data_real), 1) / 50
            t_pseudo_symbol_loop = np.arange(0, len(receivers[prn].b_code_phase), 1) * nav.T_CODE

            data_c_n0_mm = np.convolve(data_c_n0_mm, [0.1] * 10, mode="full")[: -(10 - 1)]
            data_c_n0_nwpr = np.convolve(data_c_n0_nwpr, [0.1] * 10, mode="full")[: -(10 - 1)]

            plt.figure()
            ax1 = plt.subplot(2, 4, 1)
            plt.title(f"Symbols - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_symbol_loop, data_real, "ob")
            plt.plot(t_symbol_loop, data_imag, ".r")
            plt.axis((0.0, T_LOAD_S, -1.25, 1.25))

            ax2 = plt.subplot(2, 4, 2, sharex=ax1)
            plt.title(f"AGC Loop Gain - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_pseudo_symbol_loop, receivers[prn].b_gain)

            ax3 = plt.subplot(2, 4, 3, sharex=ax2)
            plt.title(f"C/N_0 - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_symbol_loop, data_c_n0_mm, "bo-")
            plt.plot(t_symbol_loop, data_c_n0_nwpr, "ro-")

            ax4 = plt.subplot(2, 4, 4, sharex=ax3)
            plt.title("Psudeo Symbols")
            plt.grid()
            plt.plot(t_pseudo_symbol_loop[:], np.real(receivers[prn].b_pseudo_symbols), "bo-")
            plt.plot(t_pseudo_symbol_loop[:], np.imag(receivers[prn].b_pseudo_symbols), "ro")
            plt.ylabel("Amplitude")
            plt.xlabel("Time (s)")

            t_calc_offset = 5
            n_off = int(t_calc_offset / nav.T_CODE)
            t_calc_offset = n_off * nav.T_CODE

            ax5 = plt.subplot(2, 4, 5, sharex=ax4)
            b_code_phase_trunc = b_code_phase[n_off:]
            if len(b_code_phase_trunc) > 3:
                resid = np.polynomial.polynomial.Polynomial.fit(
                    range(len(b_code_phase_trunc)), b_code_phase_trunc, deg=2
                )

                t_resid = np.arange(-n_off, -n_off + len(b_code_phase))
                b_code_resid = (b_code_phase - resid(t_resid)) * 1e9 * FS_SIGNAL

                plt.plot(t_pseudo_symbol_loop, b_code_resid)
                plt.vlines(
                    t_calc_offset, min(b_code_resid), max(b_code_resid), color="k", linestyles="dashed"
                )
                plt.title("Code Phase Residual")
                plt.xlabel("Time (s)")
                plt.ylabel("Nanoseconds")
                plt.grid()
            else:
                logging.warning(f"PRN {prn}: Not enough data for polynomial fit (need > 5s).")

            ax6 = plt.subplot(2, 4, 6, sharex=ax5)
            plt.title(f"Carrier Estimate - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_pseudo_symbol_loop, receivers[prn].b_carrier_est)

            ax7 = plt.subplot(2, 4, 7, sharex=ax6)
            plt.title(f"Code Error - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_pseudo_symbol_loop, receivers[prn].b_code_error)

            plt.subplot(2, 4, 8, sharex=ax7)
            plt.title(f"Code Phase Uncorrected - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_pseudo_symbol_loop, receivers[prn].b_code_phase_uncorr)
    plt.show()


if __name__ == "__main__":
    main()
