"""Andrew Cragg 2024."""

from __future__ import annotations

import dataclasses
import enum
import logging
import typing

import numpy as np
from numpy import typing as npt

from gnss_explorer.nav import nav_decoder

if typing.TYPE_CHECKING:
    from gnss_explorer.detection import l1ca_detector

from gnss_explorer.dsp import agc, ca_code, common, costas_loop, delay_locked_loop, symbol_sync
from gnss_explorer.nav import nav, pvt

# Parameters of the L1 C/A Code
T_CODE = 1e-3
F_CODE = 1 / T_CODE
GPS_L1_CA_CHIPS = 1023
FS_CHIP = GPS_L1_CA_CHIPS / T_CODE
N_P_SYMBOLS = 20
HALF_SAMPLE = 0.5

CODE_OFFSET_NARROW = 0.05
CODE_OFFSET_WIDE = 0.5

N_INT_CODE = nav.N_P_SYMBOLS


@dataclasses.dataclass()
class L1CAFixedConfig:
    """Fixed configuration for L1CA Receiver."""

    # Signal parameters
    f_s: float
    f_c: float
    p_prn: int

    # Analog gain control
    p_agc_alpha: float

    # Timing Loops
    f_fll_bw: float
    f_pll_bw: float
    f_dll_bw: float

    # Frame Sync Parameters
    n_subframe_lock: int
    n_flywheel_allowed: int
    b_sync_pattern: npt.NDArray[np.int_]

    # Computed parameters
    f_oversample: float = dataclasses.field(init=False)
    n_frame: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Initialize the computed parameters."""
        self.f_oversample = FS_CHIP / self.f_s
        self.n_frame = int(T_CODE * self.f_s)


class L1CAReceiverState(enum.Enum):
    """TODO."""

    IDLE = enum.auto()
    START = enum.auto()
    SYNCRONIZE = enum.auto()
    LOCKED = enum.auto()
    FLYWHEEL = enum.auto()


class L1CAReceiver:
    """Receiver for the GPS L1 C/A Signal."""

    config: L1CAFixedConfig
    state: L1CAReceiverState
    solver: pvt.PvtSolver

    agc_loop: agc.AutomaticGainControl
    carrier_tracking_loop: costas_loop.CarrierTrackingLoop
    delay_locked_loop: delay_locked_loop.DelayLockedLoop
    symbol_sync_loop: symbol_sync.SymbolSync
    frame_sync: nav_decoder.FrameSync

    n_code_detection_offset: int
    n_codes_processed: int
    n_global_pos: int
    b_samples: npt.NDArray[np.complex64]
    b_symbols: list[symbol_sync.L1CASymbol]

    n_pseudo_symbol_count: int | None
    b_next_symbol_start_of_word: bool
    b_narrow_correlator: bool

    n_dll_count_accumulate: int
    p_dll_err_accumulate: float

    def __init__(self, config: L1CAFixedConfig, solver: pvt.PvtSolver) -> None:
        """TODO."""
        self.config = config
        self.state = L1CAReceiverState.IDLE
        self.solver = solver

        self.agc_loop = agc.AutomaticGainControl(alpha=config.p_agc_alpha)

        self.carrier_tracking_loop = costas_loop.CarrierTrackingLoop(
            fll_bw=config.f_fll_bw,
            pll_bw=config.f_pll_bw,
            sample_rate=nav.F_CODE,
        )
        self.delay_locked_loop = delay_locked_loop.DelayLockedLoop(
            bandwidth=config.f_dll_bw, sample_rate=nav.F_CODE / N_INT_CODE
        )

        self.symbol_sync_loop = symbol_sync.SymbolSync(p_prn=self.config.p_prn)

        self.frame_sync = nav_decoder.FrameSync(
            f_sample_rate=self.config.f_s,
            b_sync_pattern=self.config.b_sync_pattern,
            p_prn=self.config.p_prn,
            n_subframe_lock=self.config.n_subframe_lock,
            n_flywheel_allowed=self.config.n_subframe_lock,
        )

        self.reset()

    def update_detection(self, detection: l1ca_detector.CodeDetection) -> None:
        """TODO."""
        if self.state == L1CAReceiverState.IDLE:
            self.reset()
            self.n_code_detection_offset = detection.n_offset
            self.f_doppler = detection.f_doppler
            self.state = L1CAReceiverState.START

            f_doppler_to_code = -(self.f_doppler / (self.config.f_c)) / T_CODE
            self.delay_locked_loop.freq_est_hz = f_doppler_to_code
        else:
            msg = "Should not receive new detection when not IDLE."
            raise RuntimeError(msg)

        logging.debug(
            f"Current offset: {self.n_code_detection_offset}, New Offset {detection.n_offset}"
        )

    def reset(self) -> None:
        """Reset receiver loops and offsets."""
        self.state = L1CAReceiverState.IDLE
        self.p_last_nco_phase: np.complex64 = np.complex64(1 + 0j)
        self.p_last_nco_phase_carrier: np.complex64 = np.complex64(1 + 0j)
        self.p_code_discim_offset = CODE_OFFSET_WIDE

        self.n_b_samples_pos = 0
        self.n_global_pos = 0
        self.n_codes_processed = 0
        self.n_code_detection_offset = 0
        self.n_code_track_offset = 0
        self.n_total_offset = 0
        self.n_dll_count_accumulate = 0
        self.p_dll_err_accumulate = 0

        # Reset loops to initial state
        self.agc_loop.reset()
        self.carrier_tracking_loop.reset()
        self.delay_locked_loop.reset()
        self.symbol_sync_loop.reset()
        self.frame_sync.reset()

        # Clear buffers
        self.b_samples = np.zeros(self.config.n_frame, dtype=np.complex64)
        self.b_gain = []
        self.b_code_phase = []
        self.b_code_error = []
        self.b_carrier_est = []
        self.b_code_phase_uncorr = []
        self.b_pseudo_symbols = []
        self.b_symbols = []

        # Clear counters
        self.n_pseudo_symbol_count = 0
        self.b_next_symbol_start_of_word = False
        self.b_narrow_correlator = False

    def _run_dll_discrim(self, samples: npt.NDArray[np.complex64]) -> np.complex64:
        p_dll_phase = self.delay_locked_loop.offset

        x_corr_early = np.abs(
            ca_code.ca_correlate(
                samples,
                self.config.f_oversample,
                p_dll_phase - self.p_code_discim_offset,
                self.config.p_prn,
            )
        )
        x_corr_prompt = ca_code.ca_correlate(
            samples, self.config.f_oversample, p_dll_phase, self.config.p_prn
        )
        x_corr_late = np.abs(
            ca_code.ca_correlate(
                samples,
                self.config.f_oversample,
                p_dll_phase + self.p_code_discim_offset,
                self.config.p_prn,
            )
        )

        if (x_corr_early + x_corr_late) == 0:
            discrim = 0.0
        else:
            discrim = 0.5 * (x_corr_late - x_corr_early) / (x_corr_early + x_corr_late)

        if self.b_narrow_correlator and self.n_dll_count_accumulate == 0:
            self.b_narrow_correlator = False
            self.p_code_discim_offset = CODE_OFFSET_NARROW

        discrim *= 1.0 - self.p_code_discim_offset
        self.p_dll_err_accumulate += discrim

        if self.n_dll_count_accumulate == (N_INT_CODE - 1):
            self.delay_locked_loop.update(self.p_dll_err_accumulate / N_INT_CODE)
            self.p_dll_err_accumulate = 0

        self.n_dll_count_accumulate = (self.n_dll_count_accumulate + 1) % N_INT_CODE

        return x_corr_prompt

    def _pseudo_symbol_tracking(
        self, samples: npt.NDArray[np.complex64], n_frame_global_pos: int
    ) -> symbol_sync.L1CAPseudoSymbol:
        f_total_carrier_freq = self.f_doppler + (
            self.carrier_tracking_loop.freq_estimate / (2 * np.pi)
        )

        self.p_last_nco_phase = common.freq_shift(
            samples,
            fs=self.config.f_s,
            f_shift=f_total_carrier_freq,
            phase=self.p_last_nco_phase,
        )

        x_corr_prompt = self._run_dll_discrim(samples=samples)
        pseudo_symbol = x_corr_prompt * self.agc_loop.gain
        self.carrier_tracking_loop.update(pseudo_symbol)

        self.delay_locked_loop.step(
            f_total_carrier_freq, start_aiding=self.carrier_tracking_loop.start_pll
        )

        pseudo_symbol = x_corr_prompt * self.agc_loop.gain
        self.carrier_tracking_loop.update(pseudo_symbol)

        # T/2 correction
        # Carrier estimate is for the middle of the integration window
        # self.p_last_nco_phase_carrier *= np.exp(
        #     -1j * self.carrier_tracking_loop.freq_estimate * self.carrier_tracking_loop.T / 2
        # )

        self.agc_loop.update(x_corr_prompt)

        if self.delay_locked_loop.offset > HALF_SAMPLE:
            self.delay_locked_loop.offset -= self.config.f_oversample * 1
            self.n_code_track_offset = -1
        elif self.delay_locked_loop.offset < -HALF_SAMPLE:
            self.delay_locked_loop.offset += self.config.f_oversample * 1
            self.n_code_track_offset = 1
        else:
            self.n_code_track_offset = 0
        self.n_total_offset += self.n_code_track_offset

        t_receiver = (
            n_frame_global_pos
            + self.n_code_track_offset
            - (self.delay_locked_loop.offset / self.config.f_oversample)
        ) / self.config.f_s

        self.b_gain.append(self.carrier_tracking_loop.error)
        self.b_pseudo_symbols.append(pseudo_symbol)
        self.b_code_phase.append(t_receiver)
        self.b_code_error.append(self.delay_locked_loop.timing_error)
        self.b_code_phase_uncorr.append(self.delay_locked_loop.offset)
        self.b_carrier_est.append(self.carrier_tracking_loop.integrator)

        return symbol_sync.L1CAPseudoSymbol(
            p_prn=self.config.p_prn,
            pseudo_symbol=pseudo_symbol,
            n_count=self.n_pseudo_symbol_count,
            t_receiver=t_receiver,
        )

    def _process_recv_frame(self, n_frame_global_pos: int) -> None:
        """Process code-frame of data."""
        # No-op when IDLE
        if self.state == L1CAReceiverState.IDLE:
            return

        # Transition from start mode which is used to set up loops and buffers
        # after receiving a detection while in IDLE mode.
        if self.state == L1CAReceiverState.START:
            # Begin sync loop
            self.state = L1CAReceiverState.SYNCRONIZE
            self.symbol_sync_loop.start()

        # Process frames for all non-IDLE states
        pseudo_symbol = self._pseudo_symbol_tracking(
            samples=self.b_samples, n_frame_global_pos=n_frame_global_pos
        )

        symbol = self.symbol_sync_loop.update(
            pseudo_symbol=pseudo_symbol, start_of_word=self.b_next_symbol_start_of_word
        )
        self.b_next_symbol_start_of_word = False

        # Increment pseudosymbol count modulo the number of them in a full Symbol
        if symbol and self.n_pseudo_symbol_count is None:
            self.n_pseudo_symbol_count = 0

        if self.n_pseudo_symbol_count is not None:
            self.n_pseudo_symbol_count = (self.n_pseudo_symbol_count + 1) % nav.N_P_SYMBOLS

        self.solver.add_measurement(pseudo_symbol=pseudo_symbol, symbol=symbol)

        if self.symbol_sync_loop.state == symbol_sync.SymbolSyncState.UNLOCKED:
            # Symbol sync should not lose lock. If it does we need to reset because the
            # receiver timing will be irrecoverably altered.
            logging.error(
                f"Symbol Sync no longer locked. Resetting receiever for PRN {self.config.p_prn}."
            )
            self.state = L1CAReceiverState.IDLE
            return

        if symbol:
            self.b_symbols.append(symbol)
            self._process_nav_symbol(symbol=symbol)
        else:
            # Symbol not produced for every pseudosymbol
            pass

    def _process_nav_symbol(self, symbol: symbol_sync.L1CASymbol) -> None:
        """Process L1CASymbol."""
        subframe = self.frame_sync.process(symbol=symbol)

        # Don't expect a subframe for every processed symbol
        if not subframe:
            return

        self.b_next_symbol_start_of_word = True
        self._process_subframe(subframe=subframe)

    def _process_subframe(self, subframe: nav.NavSubframe) -> None:
        if self.state == L1CAReceiverState.SYNCRONIZE:
            if self.frame_sync.state == nav_decoder.FrameSyncState.LOCKED:
                self.state = L1CAReceiverState.LOCKED
                self.b_narrow_correlator = True
            else:
                msg = f"Frame produced but FrameSync is {self.frame_sync.state}, not LOCKED"
                raise RuntimeError(msg)

        if self.state == L1CAReceiverState.LOCKED:
            if subframe.parity:
                self.solver.add_subframe(subframe)
            else:
                logging.debug(f"Bad frame on PRN {self.config.p_prn}")

        self.n_codes_processed += 1

    def update(self, x: npt.NDArray[np.complex64], n_pos: int) -> None:
        """TODO.

        Args:
            x: Buffer of new complex samples
            n_pos: Global sample posiiton of the first sample in `x`

        """
        if self.n_global_pos != n_pos:
            if self.state != L1CAReceiverState.START:
                logging.error(
                    f"Unexpected jump in global sample position. Expected {self.n_global_pos}, "
                    f"Got {n_pos}. Resetting receiver."
                )
                self.reset()
            self.n_b_samples_pos = 0
            self.n_global_pos = n_pos

        # Use the code detection offset once before restting it
        n_input_buf_pos = self.n_code_detection_offset
        self.n_code_detection_offset = 0

        n_total_len = len(x)
        # This used to be `n_total_len - 1``and that caused the 4001 global pos diff weirdness
        # XXX: Why?
        while n_input_buf_pos < n_total_len:
            n_samples_needed = self.config.n_frame - self.n_b_samples_pos
            n_samples_remaining = n_total_len - n_input_buf_pos

            # If out of samples, store them and wait for more
            if n_samples_remaining < n_samples_needed:
                self.b_samples[
                    self.n_b_samples_pos : self.n_b_samples_pos + n_samples_remaining
                ] = x[n_input_buf_pos : n_input_buf_pos + n_samples_remaining]
                self.n_b_samples_pos += n_samples_remaining
                n_input_buf_pos += n_samples_remaining
            # There are enough samples to process a full frame
            else:
                self.b_samples[self.n_b_samples_pos :] = x[
                    n_input_buf_pos : n_input_buf_pos + n_samples_needed
                ]
                n_input_buf_pos += n_samples_needed
                # After processing a frame, account for the code tracking loop offset
                n_input_buf_pos += self.n_code_track_offset

                n_frame_global_pos = n_pos + n_input_buf_pos - self.config.n_frame
                self._process_recv_frame(n_frame_global_pos=n_frame_global_pos)
                self.n_b_samples_pos = 0

        self.n_global_pos += n_total_len
