"""GPS L1 C/A navigation-message utilities.

This module collects constants and helper classes used to decode
300-bit navigation sub-frames broadcast by GPS satellites.  It exposes:

Constants
    * T_CODE / F_CODE      - one C/A-code period and its chipping rate
    * GPS_L1_CA_CHIPS      - chips per C/A code (1023)
    * FS_CHIP              - chip-rate-scaled sample clock
    * C_GPS                - defined speed of light for GPS (m/s)
    * Bit-length helpers   - word, sub-frame, frame sizes, etc.
    * NAV_PREAMBLE         - 8-bit preamble pattern
    * ACTIVE_PRNS          - default list of healthy PRN indices
    * ALL_SUBFRAMES        - {1, 2, 3, 4, 5}

Classes
    TelemetryWord
        Extracts the single integrity bit from the first NAV word.

    HandoverWord
        Parses the hand-over word (HOW) containing time-of-week and
        status flags.

    NavSubframe[1-5]
        Typed sub-frame parsers registered by sub-frame ID.  Sub-frames 1-3
        decode ICD-GPS-200 clock and ephemeris parameters; sub-frames 4-5
        are placeholders for system and almanac data still to be mapped.

    NavFrame
        Simple container that groups one instance of each of the five
        sub-frames into a complete 1500-bit navigation frame.

All bit offsets, scales and two-s-complement rules follow ICD-GPS-200H.
Times are expressed in GPS seconds; angles are in semicircles unless
otherwise noted.
"""

from __future__ import annotations

import logging
import textwrap
from typing import Any, ClassVar

from gnss_explorer.dsp import bits

# -----------------------------------------------------------------------------
# GPS / NAV constants
# -----------------------------------------------------------------------------
T_CODE = 1e-3
F_CODE = 1 / T_CODE  # 1 kHz
GPS_L1_CA_CHIPS = 1023
FS_CHIP = GPS_L1_CA_CHIPS / T_CODE

C_GPS = 299_792_458  # [m/s]

N_BITS_WORD = 30
N_BITS_PARITY = 6
N_BITS_DATA = N_BITS_WORD - N_BITS_PARITY  # 24 bits
N_WORDS_SUBFRAME = 10
N_BITS_SUBFRAME = N_BITS_WORD * N_WORDS_SUBFRAME  # 300 bits
N_SUBFRAMES = 5
N_BITS_FRAME = N_SUBFRAMES * N_BITS_SUBFRAME  # 1500 bits
N_FRAMES = 25
N_P_SYMBOLS = 20
T_BIT = N_P_SYMBOLS * T_CODE  # 20 msec
T_SUBFRAME = T_BIT * N_BITS_SUBFRAME  # 6 seconds
T_FRAME = T_SUBFRAME * N_SUBFRAMES  # 30 seconds

NAV_PREAMBLE = [1, 0, 0, 0, 1, 0, 1, 1]
ACTIVE_PRNS = list(range(32))
ALL_SUBFRAMES = set(range(1, N_SUBFRAMES + 1))

N_LEAP_S = 18  # As of September 2025. Last leap second was in 2021.


class TelemetryWord(bits.BitStream):
    """NAV telemetry (TLM) word containing the integrity flag.

    The telemetry word is the first 30-bit word of every GPS L1 C/A
    subframe. This helper extracts the single-bit integrity field so that
    downstream processing can quickly determine whether the previous
    subframe was considered valid by the satellite.

    Attributes:
        integrity: Integer 0 or 1. A value of 1 indicates no detected
            parity errors in the preceding subframe.

    """

    integrity: int

    def __init__(self, bitstream: list[int]) -> None:
        """Create a `TelemetryWord` from a raw bit list.

        Args:
            bitstream: A list of 0/1 integers at least `N_BITS_DATA` long,
                ordered most-significant bit first.

        Raises:
            ValueError: If `bitstream` is shorter than `N_BITS_DATA`.

        """
        if len(bitstream) < N_BITS_DATA:
            msg = (
                "Bitstream too short for TelemetryWord: "
                f"expected at least {N_BITS_DATA}, got {len(bitstream)}"
            )
            raise ValueError(msg)
        super().__init__(bitstream=bitstream)
        self.integrity = self.bitstream[23]

    def __repr__(self) -> str:
        return f"TelemetryWord(integrity={self.integrity})"


class HandoverWord(bits.BitStream):
    """NAV hand-over word (HOW) with time and status information.

    The hand-over word is the second 30-bit word of every GPS L1 C/A
    subframe. It carries the GPS time of week for the next subframe and
    several status flags.

    Attributes:
        time_of_week: GPS time of week (seconds) of the first bit of the
            next subframe.
        alert:     Alert flag (1 signals satellite or system anomaly).
        anti_spoof: Anti-spoof flag (1 when P-code is encrypted).
        subframe_id: Integer 1-5 identifying the current subframe.

    """

    time_of_week: int
    alert: int
    anti_spoof: int
    subframe_id: int

    def __init__(self, bitstream: list[int]) -> None:
        """Parse a `HandoverWord` from raw bits.

        Args:
            bitstream: A list of 0/1 integers at least two words long, MSB first.

        Raises:
            ValueError: If `bitstream` is shorter than two words.

        """
        if len(bitstream) < N_BITS_WORD * 2:
            msg = (
                f"Bitstream too short for HandoverWord: expected at least {N_BITS_WORD * 2}, "
                f"got {len(bitstream)}"
            )
            raise ValueError(msg)
        super().__init__(bitstream=bitstream)
        self.time_of_week = self._get_field_int(31, 17, scale=6)
        self.alert = self._get_field_int(48, 1)
        self.anti_spoof = self._get_field_int(49, 1)
        self.subframe_id = self._get_field_int(50, 3)

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""\
            |-------- Handover Word --------|
            subframe_id: {self.subframe_id}
            time_of_week: {self.time_of_week} seconds
            alert: {self.alert}
            anti_spoof: {self.anti_spoof}
            |-------------------------------|
            """
        ).rstrip()


class NavSubframe(bits.BitStream):
    """Generic Navigation Subframe."""

    frame_id: ClassVar[int]
    _registry: ClassVar[dict[int, type[NavSubframe]]] = {}

    def __init_subclass__(cls, *, frame_id: int, **kwargs: dict[str, Any]) -> None:
        super().__init_subclass__(**kwargs)
        cls.frame_id = frame_id
        NavSubframe._registry[frame_id] = cls

    @classmethod
    def from_bits(
        cls,
        bitstream: list[int],
        t_offset_s: float,
        p_prn: int,
        *,
        parity: bool,
    ) -> NavSubframe | None:
        """Parse common fields and dispatch to the right subclass."""
        if len(bitstream) < N_BITS_SUBFRAME:
            msg = (
                "Bitstream too short for NavSubframe: "
                f"expected >={N_BITS_SUBFRAME}, got {len(bitstream)}"
            )
            raise ValueError(msg)
        # Peak at the embedded subframe ID
        handover = HandoverWord(bitstream)
        fid = handover.subframe_id
        subcls = cls._registry.get(fid)
        if subcls is None:
            msg = f"Unknown subframe ID: {fid}"
            logging.debug(msg)
            return None
        return subcls(bitstream, t_offset_s, p_prn, parity=parity)

    def __init__(
        self,
        bitstream: list[int],
        t_offset_s: float,
        p_prn: int,
        *,
        parity: bool,
    ) -> None:
        super().__init__(bitstream=bitstream)
        self.t_offset_s = t_offset_s
        self.p_prn = p_prn
        self.parity = parity

        self.telemetry_word = TelemetryWord(bitstream)
        self.handover_word = HandoverWord(bitstream)

    @property
    def subframe_id(self) -> int:
        """Subframe ID from the handover word."""
        return self.handover_word.subframe_id

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""\
            |================ Subframe ================|
             Offset: {self.t_offset_s:.9f} s
             PRN: {self.p_prn}
             Parity: {self.parity}
            {self.handover_word}
            |================++++++++++================|
            """
        ).rstrip()

    def __repr__(self) -> str:
        return (
            f"NavSubframe(t_offset_s={self.t_offset_s}, p_prn={self.p_prn}, parity={self.parity})"
        )


class NavSubframe1(NavSubframe, frame_id=1):
    """Navigation Subframe 1 with clock and health parameters."""

    gps_week: int
    code_on_l2: int
    ura_index: int
    sv_health: int
    iodc: int
    l2_p_data_flag: int
    t_gd: float
    toc: int
    af2: float
    af1: float
    af0: float

    def __init__(
        self,
        bitstream: list[int],
        t_offset_s: float,
        p_prn: int,
        *,
        parity: bool,
    ) -> None:
        super().__init__(bitstream, t_offset_s, p_prn, parity=parity)
        # Word 3
        self.gps_week = self._get_field_int(61, 10)
        self.code_on_l2 = self._get_field_int(71, 2)
        self.ura_index = self._get_field_int(73, 4)
        self.sv_health = self._get_field_int(77, 6)
        self.iodc = self._get_field_int_spanning([(83, 2), (211, 8)])
        # Word 4
        self.l2_p_data_flag = self._get_field_int(91, 1)
        # Word 7
        self.t_gd = self._get_field_real(197, 8, scale=2**-31, twos_complement=True)
        # Word 8
        self.toc = self._get_field_int(219, 16, scale=2**4, twos_complement=False)
        # Word 9
        self.af2 = self._get_field_real(241, 8, scale=2**-55, twos_complement=True)
        self.af1 = self._get_field_real(249, 16, scale=2**-43, twos_complement=True)
        # Word 10
        self.af0 = self._get_field_real(271, 22, scale=2**-31, twos_complement=True)

    def __repr__(self) -> str:
        return f"NavSubframe1(gps_week={self.gps_week}, iodc={self.iodc}, toc={self.toc})"


class NavSubframe2(NavSubframe, frame_id=2):
    """Navigation Subframe 2 with ephemeris data (part 1)."""

    iode: int
    c_rs: float
    delta_n: float
    m_0: float
    c_uc: float
    e: float
    c_us: float
    root_a: float
    t_oe: float
    fit_interval_flag: int
    aodo: int

    def __init__(
        self,
        bitstream: list[int],
        t_offset_s: float,
        p_prn: int,
        *,
        parity: bool,
    ) -> None:
        super().__init__(bitstream, t_offset_s, p_prn, parity=parity)
        # Word 3
        self.iode = self._get_field_int(61, 8)
        self.c_rs = self._get_field_real(69, 16, scale=2**-5, twos_complement=True)
        # Word 4
        self.delta_n = self._get_field_real(91, 16, scale=2**-43, twos_complement=True)
        # Words 4 & 5
        self.m_0 = self._get_field_real_spanning(
            fields=[(107, 8), (121, 24)], scale=2**-31, twos_complement=True
        )
        # Word 6
        self.c_uc = self._get_field_real(151, 16, scale=2**-29, twos_complement=True)
        # Words 6 & 7
        self.e = self._get_field_real_spanning(
            fields=[(167, 8), (181, 24)], scale=2**-33, twos_complement=False
        )
        # Word 8
        self.c_us = self._get_field_real(211, 16, scale=2**-29, twos_complement=True)
        # Words 8 & 9
        self.root_a = self._get_field_real_spanning(
            fields=[(227, 8), (241, 24)], scale=2**-19, twos_complement=False
        )
        # Word 10
        self.t_oe = self._get_field_real(271, 16, scale=2**4, twos_complement=False)
        self.fit_interval_flag = self._get_field_int(287, 1)
        self.aodo = self._get_field_int(288, 5)

    def __repr__(self) -> str:
        return f"NavSubframe2(iode={self.iode}, t_oe={self.t_oe})"


class NavSubframe3(NavSubframe, frame_id=3):
    """Navigation Subframe 3 with ephemeris data (part 2)."""

    c_ic: float
    omega_0: float
    c_is: float
    i_0: float
    c_rc: float
    omega: float
    omega_dot: float
    iode: int
    i_dot: float

    def __init__(
        self,
        bitstream: list[int],
        t_offset_s: float,
        p_prn: int,
        *,
        parity: bool,
    ) -> None:
        super().__init__(bitstream, t_offset_s, p_prn, parity=parity)
        # Word 3
        self.c_ic = self._get_field_real(61, 16, scale=2**-29, twos_complement=True)
        # Words 3 & 4
        self.omega_0 = self._get_field_real_spanning(
            [(77, 8), (91, 24)], scale=2**-31, twos_complement=True
        )
        # Word 5
        self.c_is = self._get_field_real(121, 16, scale=2**-29, twos_complement=True)
        # Words 5 & 6
        self.i_0 = self._get_field_real_spanning(
            [(137, 8), (151, 24)], scale=2**-31, twos_complement=True
        )
        # Word 7
        self.c_rc = self._get_field_real(181, 16, scale=2**-5, twos_complement=True)
        # Words 7 & 8
        self.omega = self._get_field_real_spanning(
            [(197, 8), (211, 24)], scale=2**-31, twos_complement=True
        )
        # Word 9
        self.omega_dot = self._get_field_real(241, 24, scale=2**-43, twos_complement=True)
        # Word 10
        self.iode = self._get_field_int(271, 8)
        self.i_dot = self._get_field_real(279, 14, scale=2**-43, twos_complement=True)

    def __repr__(self) -> str:
        return f"NavSubframe3(omega_0={self.omega_0}, i_dot={self.i_dot})"


class NavSubframe4(NavSubframe, frame_id=4):
    """Navigation Subframe 4 (data fields TBD)."""

    def __init__(
        self,
        bitstream: list[int],
        t_offset_s: float,
        p_prn: int,
        *,
        parity: bool,
    ) -> None:
        super().__init__(bitstream, t_offset_s, p_prn, parity=parity)

    def __repr__(self) -> str:
        return f"NavSubframe4(t_offset_s={self.t_offset_s}, p_prn={self.p_prn})"


class NavSubframe5(NavSubframe, frame_id=5):
    """Navigation Subframe 5 (data fields TBD)."""

    def __init__(
        self,
        bitstream: list[int],
        t_offset_s: float,
        p_prn: int,
        *,
        parity: bool,
    ) -> None:
        super().__init__(bitstream, t_offset_s, p_prn, parity=parity)

    def __repr__(self) -> str:
        return f"NavSubframe5(t_offset_s={self.t_offset_s}, p_prn={self.p_prn})"


class NavFrame:
    """Navigation Frame composed of 5 subframes."""

    subframe1: NavSubframe1
    subframe2: NavSubframe2
    subframe3: NavSubframe3
    subframe4: NavSubframe4
    subframe5: NavSubframe5

    def __init__(
        self,
        subframe1: NavSubframe1,
        subframe2: NavSubframe2,
        subframe3: NavSubframe3,
        subframe4: NavSubframe4,
        subframe5: NavSubframe5,
    ) -> None:
        self.subframe1 = subframe1
        self.subframe2 = subframe2
        self.subframe3 = subframe3
        self.subframe4 = subframe4
        self.subframe5 = subframe5

    def __repr__(self) -> str:
        return (
            f"NavFrame(subframe1={self.subframe1}, "
            f"subframe2={self.subframe2}, "
            f"subframe3={self.subframe3}, "
            f"subframe4={self.subframe4}, "
            f"subframe5={self.subframe5})"
        )
