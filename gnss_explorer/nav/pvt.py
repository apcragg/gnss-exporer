"""PVT (Position, Velocity, Time) Solver."""

import dataclasses
import enum
import math
import threading
import time
from typing import NamedTuple

import numpy as np
from numpy import typing as npt

from gnss_explorer.dsp import symbol_sync
from gnss_explorer.nav import ephemeris, nav

N_MIN_SV_FOR_SOLUTION = 4
N_MAX_ITR = 10


@dataclasses.dataclass(frozen=True)
class Vec3d:
    """3d Vector."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3d") -> "Vec3d":
        return Vec3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3d") -> "Vec3d":
        return Vec3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vec3d":
        return Vec3d(-self.x, -self.y, -self.z)

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_ndarray(self) -> npt.NDArray[np.float64]:
        return np.array(self.to_tuple(), dtype=float)

    # allow numpy to convert it automatically
    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[np.float64]:
        return np.array(self.to_tuple(), dtype=dtype if dtype is not None else float)

    def __repr__(self) -> str:
        return f"Vec3d(x={self.x}, y={self.y}, z={self.z})"


@dataclasses.dataclass()
class PvtSolverConfig:
    """Configuration class for the Position-Velocity-Time solver."""

    solution_period_hz: float


class PvtSolverState(enum.Enum):
    """State of the Position-Velocity-Time solver."""

    SEARCHING = enum.auto()
    UNCONVERGED = enum.auto()
    CONVERGED = enum.auto()


class PseudorangeMeasurement(NamedTuple):
    """Wrapper for Pseudorange Measurements and correct Space Vehicle time."""

    prn: int
    pseudorange: float  # Calculated pseudorange to space vehicle from receiver [m]
    t_rx: float  # Estimated receiver time [sec]
    t_tx: float  # (20.3.3.3.3. Eq. 1) Corrected space vehicle tranmsission time [sec]
    ephemerides: ephemeris.GpsEphemeris


class PvtSolver:
    """Position-Velocity-Time (PVT) Solver.

    Solves for the receiver position and clock bias using pseudorange measurements
    and satellite ephemerides.
    """
    config: PvtSolverConfig
    candidate_ephemerides: dict[int, ephemeris.GpsEphemeris | None]
    ephemerides: dict[int, ephemeris.GpsEphemeris | None]
    pseudo_symbols: dict[int, list[symbol_sync.L1CAPseudoSymbol]]
    nav_symbols: dict[int, list[symbol_sync.L1CASymbol]]
    current_subframe: dict[int, int]
    # Mapped by PRN -> Start of Frame Time -> Subframe ID
    subframe_buffers: dict[int, dict[int, dict[int, nav.NavSubframe]]]

    # Solution
    p_ecef: Vec3d | None  # Solution coordinates in ECEF [m, m,m]
    t_clock_bias: float | None  # Solution clock bias [sec]

    def __init__(self, config: PvtSolverConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._solution_thread: threading.Thread | None = None
        self._running = False
        self.reset()
        self._last_solution_time = time.time()

    def start(self) -> None:
        """Begin periodic PVT solving in a background thread."""
        if self._running:
            return
        self._running = True
        self._solution_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._solution_thread.start()

    def stop(self) -> None:
        """Stop background PVT solving."""
        self._running = False
        if self._solution_thread:
            self._solution_thread.join()

    def _run_loop(self) -> None:
        interval = 1.0 / self.config.solution_period_hz
        while self._running:
            time.sleep(interval)
            self.solve_pvt()

    def add_measurement(
        self,
        pseudo_symbol: symbol_sync.L1CAPseudoSymbol,
        symbol: symbol_sync.L1CASymbol | None = None,
    ) -> None:
        """Buffer a pseudo-symbol (with timestamp) and optional symbol sync output."""
        prn = pseudo_symbol.p_prn
        with self._lock:
            self.pseudo_symbols[prn].append(pseudo_symbol)
            if symbol:
                self.nav_symbols[prn].append(symbol)

    def add_subframe(self, subframe: nav.NavSubframe) -> None:
        """Buffer a NAV subframe, grouping subframes by their broadcast frame start time.

        Only assemble into a NavFrame when all subframes (1-5) from the same frame epoch
        have been received for a PRN.
        """
        p_prn = subframe.p_prn
        subframe_id = subframe.handover_word.subframe_id

        # Z-count gives subframe start time in GPS seconds (multiple of 6s)
        t_tow = subframe.handover_word.time_of_week
        # compute frame start epoch (GPS seconds)
        t_frame_start = t_tow - (subframe_id - 1) * 6

        with self._lock:
            self.current_subframe[p_prn] = subframe_id % nav.N_SUBFRAMES
            # nested buffer: prn -> frame_start -> {sf_id: subframe}
            prn_bufs = self.subframe_buffers.setdefault(p_prn, {})
            frame_buf = prn_bufs.setdefault(t_frame_start, {})
            frame_buf[subframe_id] = subframe

            # Once we have all N_SUBFRAMES for this frame_start, assemble ephemeris
            if set(frame_buf.keys()) == nav.ALL_SUBFRAMES:
                ordered_subframes = {
                    f"subframe{i}": frame_buf[i] for i in range(1, nav.N_SUBFRAMES + 1)
                }
                frame = nav.NavFrame(**ordered_subframes)  # type: ignore[reportArgumentType]
                eph = ephemeris.GpsEphemeris(p_prn, frame)
                self.ephemerides[p_prn] = eph
                # clear just this frame's buffer
                del prn_bufs[t_frame_start]

    def _get_measurements(self) -> list[PseudorangeMeasurement] | None:
        with self._lock:
            prns = self._get_prns_by_c_n0()

            measurements: list[PseudorangeMeasurement] = []
            for prn in prns:
                symbols = self.nav_symbols.get(prn, [])
                if not len(symbols) > 0:
                    continue
                latest_symbol = symbols[-1]

                eph = self.ephemerides[prn]
                if eph is None:
                    # Should not happen since we filter by PRNs with valid Es/N0 measurements.
                    continue

                if latest_symbol.n_count is None:
                    # Symbols are produced before their position in the Subframe is known. Don't use
                    # these symbols for the PVT solution.
                    continue

                frame_n_count = (
                    latest_symbol.n_count + nav.N_BITS_SUBFRAME * (self.current_subframe[prn])
                )
                updated_tow = eph.tow + nav.T_FRAME + frame_n_count * nav.T_BIT
                updated_toc = eph.toc + nav.T_FRAME + frame_n_count * nav.T_BIT
                t_sv = updated_tow

                # Apply User Algorithm for SV Clock Corrections (20.3.3.3.3.1)
                t_rel = ephemeris.F * eph.e * math.sqrt(eph.A) * math.sin(eph.calculate_e_k(t=t_sv))

                # (20.3.3.3.3. Eq 1.) uses transmit time `t` but we can approximate by `t_sv`.
                t_sv_delta = (
                    eph.af0
                    + (t_sv - updated_toc) * eph.af1
                    + ((t_sv - updated_toc) ** 2) * eph.af2
                    - eph.t_gd
                    + t_rel
                )
                t_tx = t_sv - t_sv_delta
                pseudorange = (latest_symbol.t_receiver - t_tx) * nav.C_GPS
                measurements.append(
                    PseudorangeMeasurement(
                        prn=prn,
                        pseudorange=pseudorange,
                        t_rx=latest_symbol.t_receiver,
                        t_tx=t_tx,
                        ephemerides=eph,
                    )
                )

            filtered_measurements: list[PseudorangeMeasurement] = []
            if len(measurements) >= N_MIN_SV_FOR_SOLUTION:
                latest_time = max([m.t_rx for m in measurements])
                filtered_measurements.extend(m for m in measurements if m.t_rx > latest_time - 1)

            if len(filtered_measurements) >= N_MIN_SV_FOR_SOLUTION:
                self.state = PvtSolverState.UNCONVERGED
                return filtered_measurements
            return None

    def _build_h(self, p_svs: list[Vec3d], p_hat_user: Vec3d) -> npt.NDArray[np.float64]:
        n_sv = len(p_svs)

        h = np.zeros(shape=(n_sv, 4), dtype=np.float64)

        for sv_j, p_sv_j in enumerate(p_svs):
            r_hat_j = np.linalg.norm(p_sv_j - p_hat_user)
            h[sv_j, 0] = -(p_sv_j.x - p_hat_user.x) / r_hat_j
            h[sv_j, 1] = -(p_sv_j.y - p_hat_user.y) / r_hat_j
            h[sv_j, 2] = -(p_sv_j.z - p_hat_user.z) / r_hat_j
            h[sv_j, 3] = 1

        return h

    def solve_pvt(self) -> None:
        """Solve the Position-Velocity-Time solution."""
        measurements = self._get_measurements()
        if measurements is None:
            return

        # Initial guess at the center of the earth
        p_hat_user = Vec3d()
        t_receiver_bias_s = (self.t_clock_bias) if self.t_clock_bias else 0.0  # [m]

        # Least squares iterations
        n_itr = 0
        while n_itr < N_MAX_ITR:
            # Spacecraft positions at corrected transmission time
            p_svs = [Vec3d(*m.ephemerides.as_ecef(t=m.t_tx)) for m in measurements]

            rho_hat = np.array(
                [
                    np.linalg.norm(p_sv - p_hat_user) + t_receiver_bias_s * nav.C_GPS
                    for p_sv in p_svs
                ]
            )

            rho = np.array([m.pseudorange for m in measurements])
            rho_delta = rho - rho_hat

            h = self._build_h(p_svs=p_svs, p_hat_user=p_hat_user)

            (p_hat_user_update, *_) = np.linalg.lstsq(h, rho_delta, rcond=None)

            p_hat_user = p_hat_user + Vec3d(
                p_hat_user_update[0], p_hat_user_update[1], p_hat_user_update[2]
            )

            v = rho_delta - h @ p_hat_user_update

            t_receiver_bias_s += p_hat_user_update[3] / nav.C_GPS
            n_itr += 1
        print(
            f"ECEF: {p_hat_user}",
            f"Clock bias: {t_receiver_bias_s:15.9f} sec",
        )

        print(
            f"rss={float(v @ v):.3f} "
            "residuals_per_sv="
            + str({m.prn: float(r) for m, r in zip(measurements, v, strict=False)})
        )

        self.p_ecef = p_hat_user
        self.t_clock_bias = t_receiver_bias_s

    def _get_prns_by_c_n0(self) -> list[int]:
        """Return list of PRNs sorted by descending average C/N0."""
        cn0s: dict[int, float] = {}
        for prn, syms in self.nav_symbols.items():
            if prn not in self.ephemerides or not syms:
                continue
            vals = [s.c_n0_est_mm for s in syms]
            cn0s[prn] = sum(vals) / len(vals)
        return sorted(cn0s, key=lambda p: cn0s[p], reverse=True)

    def reset(self) -> None:
        """Reset the solver state to the initial condition."""
        self.state = PvtSolverState.SEARCHING
        self.candidate_ephemerides = dict.fromkeys(nav.ACTIVE_PRNS)
        self.ephemerides = dict.fromkeys(nav.ACTIVE_PRNS)
        self.pseudo_symbols = {prn: [] for prn in nav.ACTIVE_PRNS}
        self.nav_symbols = {prn: [] for prn in nav.ACTIVE_PRNS}
        self.subframe_buffers = {}
        self.current_subframe = {}

        self.p_ecef = None
        self.t_clock_bias = None
