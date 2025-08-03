"""Andrew Cragg 2024."""

from __future__ import annotations

import dataclasses
import enum

import numpy as np
from numpy import typing as npt

from gnss_explorer.nav import nav

DEFAULT_N_CHECK_FOR_LOCK = 200
DEFAULT_N_FLYWHEEL_MAX = 10


@dataclasses.dataclass()
class L1CASymbol:
    """TODO."""

    p_prn: int
    symbol: np.complex64
    n_count: (
        int | None
    )  # Count between [0..300] indicating which position the symbol is in the SubFrame
    t_receiver: float  # Time relative to the receiver epoch [secs]
    # Signal quality metrics
    c_n0_est_mm: float
    c_n0_est_nwpr: float


@dataclasses.dataclass()
class L1CAPseudoSymbol:
    """TODO."""

    p_prn: int
    pseudo_symbol: np.complex64
    n_count: int | None  # Count between [0..19] indicating which position the pseduo symbol is
    t_receiver: float  # Time relative to the receiver epoch [secs]


class SymbolSyncState(enum.Enum):
    """TODO."""

    UNLOCKED = enum.auto()
    CHECK = enum.auto()
    LOCKED = enum.auto()
    FLYWHEEL = enum.auto()


class SymbolSync:
    """TODO."""

    state: SymbolSyncState
    b_pseudo_symbols: list[L1CAPseudoSymbol]
    s_delta_k: npt.NDArray[np.float32]
    n_delta: np.intp
    n_count: int | None  # Position of previous symbol in the word. `None` until full Word received
    b_start_of_word: bool

    def __init__(
        self,
        p_prn: int,
        n_check_for_lock: int = DEFAULT_N_CHECK_FOR_LOCK,
        n_flywheel_max: int = DEFAULT_N_FLYWHEEL_MAX,
    ) -> None:
        """TODO."""
        self.p_prn = p_prn
        self.n_check_for_lock = n_check_for_lock
        self.n_flywheel_max = n_flywheel_max
        self.s_delta_k = np.zeros((nav.N_P_SYMBOLS, n_check_for_lock), dtype=np.float32)
        self.reset()

    def _calc_moments_db(self, n_delta: np.intp) -> float:
        """Calculate Moment Method C/N0 Estmation."""
        data: npt.NDArray[np.complex64] = np.array(
            [
                sym.pseudo_symbol
                for sym in self.b_pseudo_symbols[n_delta : n_delta + nav.N_P_SYMBOLS]
            ]
        )
        m2 = np.mean(np.power(np.abs(data), 2))
        m4 = np.mean(np.power(np.abs(data), 4))
        pd_sqr = 2 * m2**2 - m4
        if pd_sqr < 0.0:
            return float("nan")
        pd = np.sqrt(pd_sqr)
        pn = m2 - pd

        # C/N0 estimate in dB
        return 10 * np.log10((1 / nav.T_CODE) * pd / pn)

    def _calc_nwpr_db(self, n_delta: np.intp) -> float:
        """Calculate Narrowband Wideband Power Ratio C/N0 Estmation."""
        data: npt.NDArray[np.complex64] = np.array(
            [
                sym.pseudo_symbol
                for sym in self.b_pseudo_symbols[n_delta : n_delta + nav.N_P_SYMBOLS]
            ]
        )

        wbp_k = np.sum(np.abs(data) ** 2)
        nbp_k = np.sum(np.real(data)) ** 2 + np.sum(np.imag(data)) ** 2

        r = nbp_k / wbp_k
        gamm = (1 / (nav.T_CODE)) * (r - 1) / (nav.N_P_SYMBOLS - r)

        return 10 * np.log10(gamm) + 2 if gamm > 0 else float("nan")

    def _calc_s_delta(self, n_delta: np.intp) -> np.complex64:
        """Calculate the objective function for r(t)."""
        symbol = np.complex64(0)
        for n_idx in range(nav.N_P_SYMBOLS):
            symbol += self.b_pseudo_symbols[n_delta + n_idx].pseudo_symbol
        return symbol / nav.N_P_SYMBOLS

    def _max_s_delta(self) -> np.intp:
        s_delta = np.sum(self.s_delta_k, axis=1)
        return np.argmax(s_delta)

    def _update_objective(self) -> None:
        for n_delta in np.arange(nav.N_P_SYMBOLS):
            s_delta_k = abs(self._calc_s_delta(n_delta=n_delta))
            # Shift in newest calculation
            self.s_delta_k[n_delta] = np.roll(self.s_delta_k[n_delta], shift=1)
            self.s_delta_k[n_delta][0] = s_delta_k

    def _update_n_count(self, *, start_of_word: bool = False) -> None:
        """Update the Symbol count tracking."""
        if self.n_count is not None:
            self.n_count = (self.n_count + 1) % nav.N_BITS_SUBFRAME

        if start_of_word:
            if self.n_count is not None:
                if self.n_count != 0:
                    print("unexpected n count", self.n_count, self.p_prn)
            self.n_count = 0
            self.b_start_of_word = False

    def update(
        self, pseudo_symbol: L1CAPseudoSymbol, *, start_of_word: bool = False
    ) -> L1CASymbol | None:
        """Shift in the next pseudosymbol and (maybe) produce an L1CASymbol."""
        # 1) shift the buffer
        self.b_pseudo_symbols[1:] = self.b_pseudo_symbols[:-1]
        self.b_pseudo_symbols[0] = pseudo_symbol
        self.n_offset += 1

        if start_of_word:
            self.b_start_of_word = start_of_word

        # 2) only every nav.N_P_SYMBOLS samples do we advance the bit counter
        if (self.n_offset - 1) % nav.N_P_SYMBOLS == 0:
            self.n_bits_received += 1
            self._update_objective()
        else:
            return None

        # 3) handle the “lock-acquisition” state
        if self.state is SymbolSyncState.CHECK:
            if self.n_bits_received == self.n_check_for_lock:
                self.n_delta = self._max_s_delta()
                self.state = SymbolSyncState.LOCKED
            return None

        # 4) handle transitions out of LOCKED
        if self.state is SymbolSyncState.LOCKED:
            if self._max_s_delta() != self.n_delta:
                self.state = SymbolSyncState.FLYWHEEL
                self.n_flywheel_count = 1
                self.n_check = 0
            # fall through to output phase

        # 5) handle FLYWHEEL behavior
        elif self.state is SymbolSyncState.FLYWHEEL:
            if self._max_s_delta() != self.n_delta:
                self.n_flywheel_count += 1
                self.n_check = 0
            else:
                self.n_check += 1

            if self.n_check >= self.n_flywheel_max:
                self.state = SymbolSyncState.LOCKED
                self.n_flywheel_count = 0
            elif self.n_flywheel_count >= self.n_flywheel_max:
                self.state = SymbolSyncState.UNLOCKED

            # fall through if still in FLYWHEEL

        else:
            msg = f"Unknown state: {self.state!r}"
            raise RuntimeError(msg)

        # 6) at this point we're guaranteed to be in LOCKED or (still) in FLYWHEEL,
        #    so symbol, c_mm and c_nwpr can be safely computed:
        symbol = self._calc_s_delta(self.n_delta)
        c_mm = self._calc_moments_db(self.n_delta)
        c_nwpr = self._calc_nwpr_db(self.n_delta)

        self._update_n_count(start_of_word=self.b_start_of_word)

        return L1CASymbol(
            p_prn=self.p_prn,
            symbol=symbol,
            t_receiver=self.b_pseudo_symbols[self.n_delta].t_receiver,
            n_count=self.n_count,
            c_n0_est_mm=c_mm,
            c_n0_est_nwpr=c_nwpr,
        )

    def start(self) -> None:
        """Start the tracking loop."""
        self.state = SymbolSyncState.CHECK

    def reset(self) -> None:
        """TODO."""
        self.state = SymbolSyncState.UNLOCKED
        self.b_pseudo_symbols = (
            [
                L1CAPseudoSymbol(
                    p_prn=self.p_prn, pseudo_symbol=np.complex64(0), t_receiver=0, n_count=0
                )
            ]
            * 2
            * nav.N_P_SYMBOLS
        )
        self.n_offset = 0
        self.n_delta = np.intp(0)
        self.n_check = 0
        self.n_flywheel_count = 0
        self.n_bits_received = 0
        self.n_count = None
        self.b_start_of_word = False
