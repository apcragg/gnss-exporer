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
from gnss_explorer.nav import nav, orbit_plot, pvt
from gnss_explorer.receivers import l1ca

logging.getLogger().setLevel(logging.INFO)

FC_SIGNAL = 1575.420e6
FS_SIGNAL = 4.0e6 * (1.0000)  # add some PPMs
ACTIVE_PRNS = list(range(32))

N_RX = 100_000
T_LOAD_S = 120
N_LOAD = int(FS_SIGNAL * T_LOAD_S)

N_YEILD_DEAULT = int(1e5)

# Site location: Berkeley, Alameda County, California, United States
# 37.858429 lat, -122.273754 lon, 37.00 m alt
ecef_ant = (-2692314.79174219165, -4263143.23363866005, 3893072.19487998961)


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
    n_yield: int = N_YEILD_DEAULT,
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
            (defaults to ``N_YEILD_DEAULT``). Internally we load
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
    idx = 0
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
    """TODO."""
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
        f_fll_bw=10,
        f_pll_bw=5,
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
            f_s=FS_SIGNAL, f_max_doppler=10e3, f_step_doppler=250, p_ratio_threshold=1.7
        )

    t_start = time.time()

    n_pos = 0
    for x in x_stream:
        prn_to_detect = ACTIVE_PRNS[n_pos % len(ACTIVE_PRNS)]
        if prn_to_detect == n_pos:
            if receivers[prn_to_detect].state is l1ca.L1CAReceiverState.IDLE:
                detection = detectors[prn_to_detect].process(
                    frame=x[: detectors[prn_to_detect].n_frame],
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
    orbit_plot.plot_orbits_2d(active_eph, t_start=t_subframe)

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
            ax1 = plt.subplot(2, 3, 1)
            plt.title(f"Symbols - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_symbol_loop, data_real, "ob")
            plt.plot(t_symbol_loop, data_imag, ".r")
            plt.axis((0.0, T_LOAD_S, -1.25, 1.25))

            ax2 = plt.subplot(2, 3, 2, sharex=ax1)
            plt.title(f"(XXX carrier phase) AGC Loop Gain - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_pseudo_symbol_loop, receivers[prn].b_carrier_phase)

            ax2 = plt.subplot(2, 3, 3, sharex=ax1)
            plt.title(f"C/N_0 - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_symbol_loop, data_c_n0_mm, "bo-")
            plt.plot(t_symbol_loop, data_c_n0_nwpr, "ro-")

            ax3 = plt.subplot(2, 3, 4, sharex=ax2)
            plt.title("Psudeo Symbols")
            plt.grid()
            plt.plot(t_pseudo_symbol_loop[:], np.real(receivers[prn].b_pseudo_symbols), "bo-")
            plt.plot(t_pseudo_symbol_loop[:], np.imag(receivers[prn].b_pseudo_symbols), "ro")
            plt.ylabel("Amplitude")
            plt.xlabel("Time (s)")

            t_calc_offset = 2
            n_off = int(t_calc_offset / nav.T_CODE)
            t_calc_offset = n_off * nav.T_CODE

            b_code_phase_trunc = b_code_phase[n_off:]
            resid = np.polynomial.polynomial.Polynomial.fit(
                range(len(b_code_phase_trunc)), b_code_phase_trunc, deg=2
            )

            t_resid = np.arange(-n_off, -n_off + len(b_code_phase))
            b_code_resid = (b_code_phase - resid(t_resid)) * 1e9 * FS_SIGNAL

            ax4 = plt.subplot(2, 3, 5, sharex=ax3)
            plt.plot(t_pseudo_symbol_loop, b_code_resid)
            plt.vlines(
                t_calc_offset, min(b_code_resid), max(b_code_resid), color="k", linestyles="dashed"
            )
            plt.title("Code Phase Residual")
            plt.xlabel("Time (s)")
            plt.ylabel("Nanoseconds")
            plt.grid()

            plt.subplot(2, 3, 6, sharex=ax4)
            plt.title(f"Carrier Estimate - PRN {prn}")
            plt.grid()
            plt.xlabel("Time (s)")
            plt.plot(t_pseudo_symbol_loop, receivers[prn].b_carrier_est)

    plt.show()


if __name__ == "__main__":
    main()
