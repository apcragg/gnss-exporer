"""A. Cragg 2024."""

from __future__ import annotations

import dataclasses
import enum
import logging

import numpy as np
from numpy import typing as npt

from gnss_explorer.dsp import symbol_sync
from gnss_explorer.nav import nav


class FrameSyncState(enum.Enum):
    """Frame Synchronization State."""

    SEARCHING = enum.auto()
    FLYWHEEL = enum.auto()
    LOCKED = enum.auto()


@dataclasses.dataclass
class SyncTheory:
    """Hypothesis for frame synchronization."""

    start_offset: int
    phase: bool
    count: int


class FrameSync:
    """Frame Synchronization.

    Finds the start of the 300-bit subframe within the stream of navigation symbols.
    """

    state: FrameSyncState
    b_sync_pattern: npt.NDArray[np.int_]
    b_subframe_bits: list[int]
    b_subframe_symbols: list[symbol_sync.L1CASymbol]
    p_prn: int
    locked: bool
    n_locked_offset: int
    n_flywheel_allowed: int
    n_subframe_lock: int
    n_offset: int
    phase: bool
    theories: list[SyncTheory]

    def __init__(
        self,
        f_sample_rate: float,
        b_sync_pattern: npt.NDArray[np.int_],
        p_prn: int,
        n_subframe_lock: int = 3,
        n_flywheel_allowed: int = 10,
    ) -> None:
        """Create Frame Syncrhonization class."""
        self.state = FrameSyncState.SEARCHING
        self.f_sample_rate = f_sample_rate
        self.b_sync_pattern = b_sync_pattern
        self.p_prn = p_prn
        self.n_subframe_lock = n_subframe_lock
        self.n_flywheel_allowed = n_flywheel_allowed
        self.reset()

    def _sync_correlate(self) -> int:
        sync_corr = 0
        for bit in range(len(self.b_sync_pattern)):
            sync_corr += not (
                self.b_subframe_bits[nav.N_BITS_SUBFRAME - bit - 1] ^ self.b_sync_pattern[bit]
            )
        return sync_corr

    def _update_theories(self, *, phase: bool) -> None:
        theory_updated = False
        theory_offset = self.n_offset
        for theory in self.theories:
            if (theory.phase == phase) and (
                (theory_offset - theory.start_offset) % nav.N_BITS_SUBFRAME == 0
            ):
                theory.count += 1
                theory_updated = True
        if not theory_updated:
            self.theories.append(SyncTheory(start_offset=theory_offset, phase=phase, count=1))
            logging.debug(f"Considering new theory - {self.theories[-1]}")

    def _process_searching(self) -> None:
        sync_corr = self._sync_correlate()

        if sync_corr == len(self.b_sync_pattern):
            self._update_theories(phase=False)
        elif sync_corr == 0:
            self._update_theories(phase=True)
        else:
            return

        for theory in self.theories:
            if theory.count == self.n_subframe_lock:
                self.state = FrameSyncState.LOCKED
                self.n_locked_offset = theory.start_offset
                self.phase = theory.phase
                self.theories = []
                logging.info(
                    f"Frame Sync Locked to PRN {self.p_prn} starting at offset "
                    f"{self.n_locked_offset}, {self.n_offset}"
                )
                break

    def process(self, symbol: symbol_sync.L1CASymbol) -> nav.NavSubframe | None:
        """Process a new symbol and potentially produce a subframe."""
        bit = 1 if symbol.symbol.real > 0.0 else 0
        self.b_subframe_bits = [bit, *self.b_subframe_bits[:-1]]
        self.b_subframe_symbols = [symbol, *self.b_subframe_symbols[:-1]]

        subframe = None
        if self.state is FrameSyncState.SEARCHING:
            self._process_searching()
        if self.state is FrameSyncState.FLYWHEEL:
            pass
        if self.state is FrameSyncState.LOCKED:
            if (self.n_offset - self.n_locked_offset) % nav.N_BITS_SUBFRAME == 0:
                subframe = self.publish_subframe()

        self.n_offset += 1
        return subframe

    def _compute_parity_bits(self, d_bits: list[int], d_29_prev: int, d_30_prev: int) -> list[int]:
        """Compute patiy bits of the subframe word.

        Partiy bits computed per 20.3.5.2 User Parity Algorithm.

        Args:
            d_bits: D1 - D30 of current word
            d_29_prev: D29 from previous word
            d_30_prev: D30 from previous word

        Returns:
            party: Parity bits D25 - D30 of the current word

        """
        parity = [0] * nav.N_BITS_PARITY

        d25_list = [1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23]
        d26_list = [2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24]
        d27_list = [1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22]
        d28_list = [2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23]
        d29_list = [1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24]
        d30_list = [3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24]

        parity[0] = (d_29_prev + sum([d_bits[idx - 1] for idx in d25_list])) % 2
        parity[1] = (d_30_prev + sum([d_bits[idx - 1] for idx in d26_list])) % 2
        parity[2] = (d_29_prev + sum([d_bits[idx - 1] for idx in d27_list])) % 2
        parity[3] = (d_30_prev + sum([d_bits[idx - 1] for idx in d28_list])) % 2
        parity[4] = (d_30_prev + sum([d_bits[idx - 1] for idx in d29_list])) % 2
        parity[5] = (d_29_prev + sum([d_bits[idx - 1] for idx in d30_list])) % 2

        return parity

    def publish_subframe(self, n_subframe: int = 0) -> nav.NavSubframe | None:
        """Publish the current buffer as a subframe."""
        offset = n_subframe * nav.N_BITS_SUBFRAME
        logging.debug(f"Publishing subframe at offset {offset}, Invert: {self.phase}")

        subframe_bitstream = (
            np.logical_xor(np.flip(self.b_subframe_bits), self.phase).astype(int).tolist()
        )

        # 20.3.5.2 User Parity Algorithm
        # If D30*, then invert data bits d1-d24 of the following frame
        for idx_word in range(nav.N_WORDS_SUBFRAME):
            if idx_word == 0:
                d30_prev = self.d30_prev
            else:
                d30_prev = subframe_bitstream[idx_word * nav.N_BITS_WORD - 1]
            if d30_prev:
                idx_st = idx_word * nav.N_BITS_WORD
                idx_end = idx_word * nav.N_BITS_WORD + 24
                subframe_bitstream[idx_st:idx_end] = [
                    bit ^ 1 for bit in subframe_bitstream[idx_st:idx_end]
                ]

        subframe_parity_check = True
        for idx_word in range(nav.N_WORDS_SUBFRAME):
            calc_parity = self._compute_parity_bits(
                subframe_bitstream[idx_word * nav.N_BITS_WORD : idx_word * nav.N_BITS_WORD + 24],
                self.d29_prev,
                self.d30_prev,
            )

            self.d29_prev = subframe_bitstream[(idx_word + 1) * nav.N_BITS_WORD - 2]
            self.d30_prev = subframe_bitstream[(idx_word + 1) * nav.N_BITS_WORD - 1]

            recv_parity = subframe_bitstream[
                idx_word * nav.N_BITS_WORD + 24 : idx_word * nav.N_BITS_WORD + 30
            ]

            parity_check = np.array_equal(calc_parity, recv_parity)
            subframe_parity_check = subframe_parity_check and parity_check
            logging.debug(f"Word {idx_word} Partiy Check: {[parity_check]}")
        logging.debug(f"Frame Partiy Check: {[subframe_parity_check]} {self.p_prn}")

        return nav.NavSubframe.from_bits(
            bitstream=subframe_bitstream,
            t_offset_s=self.b_subframe_symbols[nav.N_BITS_SUBFRAME].t_receiver,
            p_prn=self.p_prn,
            parity=subframe_parity_check,
        )

    def reset(self) -> None:
        """Reset the tracking loop."""
        self.n_locked_offset = 0
        self.n_offset = 0
        self.phase = True

        self.d29_prev = 0
        self.d30_prev = 0

        # Data structures
        self.b_subframe_bits = [0] * nav.N_BITS_SUBFRAME * self.n_subframe_lock
        self.b_subframe_symbols = (
            [
                symbol_sync.L1CASymbol(
                    p_prn=self.p_prn,
                    symbol=np.complex64(0.0),
                    t_receiver=0.0,
                    n_count=0,
                    c_n0_est_mm=0.0,
                    c_n0_est_nwpr=0.0,
                )
            ]
            * nav.N_BITS_SUBFRAME
            * self.n_subframe_lock
        )
        self.theories = []
