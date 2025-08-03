"""Andrew Cragg 2025."""

import math

import pyproj

from gnss_explorer.nav import nav

C_GPS = 299792458.0  # GPS standard Speed of light [m/s]


MU = 3.986005e14  # Earth's gravitational constant [m^3/s^2]
OMEGA_E_DOT = 7.2921151467e-5  # Earth's rotation rate [rad/s]
R_EARTH = 6378137.0  # Earth's radius [m]
F = -4.442807633e-10  # Constant used in relativistic correction [s/(m^1/2)]

DEFAULT_N_MAX_ITR_KEPLER = 100
DEFAULT_TOL_KEPLER = 1e-10

WGS84_LLA_PROJ = "epsg:4326"
WGS84_ECEF_PROJ = "epsg:4978"


def solve_keplers_equation(
    m_t: float, e: float, tol: float = DEFAULT_TOL_KEPLER, max_itr: int = DEFAULT_N_MAX_ITR_KEPLER
) -> float:
    """Solve Kepler's equation iteratively."""
    e_n = m_t  # initial guess
    for _ in range(max_itr):
        f = e_n - e * math.sin(e_n) - m_t
        f_prime = 1 - e * math.cos(e_n)
        delta = f / f_prime
        e_n -= delta
        if abs(delta) < tol:
            break
    return e_n


class GpsEphemeris:
    """Epehemeris data for a GPS space vehicle."""

    tow: int  # latest time [seconds]
    gps_week: int  # GPS week number
    toc: int  # time of clock [sec]
    t_gd: float  # group delay correct [sec]
    sv_prn: int  # space vehcile prn
    A: float  # semi-major axis [m]
    e: float  # eccentricity
    m_0: float  # mean anomaly at reference epoch [rad]
    delta_n: float  # corrected mean motion [rad/s]
    t_oe: float  # time of ephemeris [s]
    omega: float  # argument of perigee [rad]
    omega_0: float  # longitude of ascending node at reference epoch [rad]
    omega_dot: float  # rate of change of longitude of ascending node [semi-circles/sec]
    i_0: float  # inclination at reference epoch [rad]
    i_dot: float  # rate of change of inclination [rad/s]
    c_us: float  # harmonic correction term for argument of latitude [rad]
    c_uc: float  # harmonic correction term for argument of latitude [rad]
    c_rs: float  # harmonic correction term for radius [m]
    c_rc: float  # harmonic correction term for radius [m]
    c_is: float  # harmonic correction term for inclination [rad]
    c_ic: float  # harmonic correction term for inclination [rad]

    af0: float  # clock correction term 0 [sec]
    af1: float  # clock correction term 1 [sec/sec]
    af2: float  # clock correction term 2 [sec/sec^2]

    def __init__(self, sv: int, frame: nav.NavFrame) -> None:
        """Create GPS Ephemeris class from NavFrame."""
        self.tow = frame.subframe1.handover_word.time_of_week
        self.gps_week = frame.subframe1.gps_week
        self.toc = frame.subframe1.toc  # [secs]
        self.t_gd = frame.subframe1.t_gd

        self.sv_prn = sv
        self.A = frame.subframe2.root_a**2  # [sqrt(m)] to [m]
        self.e = frame.subframe2.e
        self.m_0 = frame.subframe2.m_0 * math.pi  # [semicircles/s] to [rad/s]
        self.delta_n = frame.subframe2.delta_n * math.pi  # [semicircles/s] to [rad/s]
        self.t_oe = frame.subframe2.t_oe

        self.omega = frame.subframe3.omega * math.pi  # [semicircles] to [rad]
        self.omega_dot = frame.subframe3.omega_dot * math.pi  # [semicircles/s] to [rad/s]
        self.i_0 = frame.subframe3.i_0 * math.pi  # [semicircles] to [rad]
        self.i_dot = frame.subframe3.i_dot * math.pi  # [semicircles/s] to [rad/s]
        self.omega_0 = frame.subframe3.omega_0 * math.pi  # [semicircles] to [rad]

        self.c_us = frame.subframe2.c_us
        self.c_uc = frame.subframe2.c_uc
        self.c_rs = frame.subframe2.c_rs
        self.c_rc = frame.subframe3.c_rc
        self.c_is = frame.subframe3.c_is
        self.c_ic = frame.subframe3.c_ic

        self.af0 = frame.subframe1.af0
        self.af1 = frame.subframe1.af1
        self.af2 = frame.subframe1.af2

        self.transformer = pyproj.Transformer.from_crs(WGS84_ECEF_PROJ, WGS84_LLA_PROJ)

    def calculate_e_k(self, t: float) -> float:
        """Calculate `e_k` at a given time."""
        t_k = t - self.t_oe
        n = math.sqrt(MU / self.A**3) + self.delta_n
        m_k = self.m_0 + n * t_k
        return solve_keplers_equation(m_k, self.e)

    def is_valid(self) -> bool:
        """Ephemerides received recently and with valid issue time."""
        # XXX: Implement ephemeris validity check.
        return True

    def _calculate_position(self, t: float, *, is_eci: bool) -> tuple[float, float, float]:
        """Calculate position for either ECEF or ECI frame."""
        t_k = t - self.t_oe
        e_k = self.calculate_e_k(t=t)
        v_k = math.atan2(math.sqrt(1 - self.e**2) * math.sin(e_k), math.cos(e_k) - self.e)
        phi_k = v_k + self.omega
        delta_u_k = self.c_us * math.sin(2 * phi_k) + self.c_uc * math.cos(2 * phi_k)
        u_k = phi_k + delta_u_k
        delta_r_k = self.c_rs * math.sin(2 * phi_k) + self.c_rc * math.cos(2 * phi_k)
        r_k = self.A * (1 - self.e * math.cos(e_k)) + delta_r_k
        delta_i_k = self.c_is * math.sin(2 * phi_k) + self.c_ic * math.cos(2 * phi_k)
        i_k = self.i_0 + self.i_dot * t_k + delta_i_k

        if is_eci:
            # For ECI, we don't account for Earth's rotation
            omega_k = self.omega_0 + self.omega_dot * t_k
        else:
            # For ECEF, we account for Earth's rotation
            omega_k = self.omega_0 + (self.omega_dot - OMEGA_E_DOT) * t_k - OMEGA_E_DOT * self.t_oe

        x_prime_k = r_k * math.cos(u_k)
        y_prime_k = r_k * math.sin(u_k)
        x = x_prime_k * math.cos(omega_k) - y_prime_k * math.cos(i_k) * math.sin(omega_k)
        y = x_prime_k * math.sin(omega_k) + y_prime_k * math.cos(i_k) * math.cos(omega_k)
        z = y_prime_k * math.sin(i_k)
        return x, y, z

    def as_ecef(self, t: float) -> tuple[float, float, float]:
        """Calculate satellite position in ECEF coordinates at time t."""
        return self._calculate_position(t, is_eci=False)

    def as_eci(self, t: float) -> tuple[float, float, float]:
        """Calculate satellite position in ECI coordinates at time t."""
        return self._calculate_position(t, is_eci=True)

    def as_lla(self, t: float) -> tuple[float, float, float]:
        """Return the ephemeris data in Lat, Lon, Alt (LLA) coordinates.

        Args:
            t: Reference time (s)

        Returns:
            (lat, long, alt): Satellite position in LLA coordinates [deg, deg, m]

        """
        (x, y, z) = self.as_ecef(t)
        (lat, lon, alt) = self.transformer.transform(x, y, z)
        return (lat, lon, alt)

    def __str__(self) -> str:
        """Return nice representation of the ephemeris data."""
        return (
            "Ephemeris Data:\n"
            f"  A       : {self.A:.3f} m    (semi-major axis)\n"
            f"  e       : {self.e:.3f}      (eccentricity)\n"
            f"  m_0     : {self.m_0:.3f} rad  (mean anomaly at reference epoch)\n"
            f"  delta_n : {self.delta_n:.3f} rad/s (corrected mean motion)\n"
            f"  t_oe    : {self.t_oe:.3f} s   (time of ephemeris)\n"
            f"  omega   : {self.omega:.3f} rad  (argument of perigee)\n"
            f"  i_0     : {self.i_0:.3f} rad  (inclination at reference epoch)\n"
            f"  i_dot   : {self.i_dot:.3f} rad/s (rate of change of inclination)\n"
            f"  c_us    : {self.c_us:.3f} rad  (harmonic correction term for argument of latitude)"
            "\n"
            f"  c_uc    : {self.c_uc:.3f} rad  (harmonic correction term for argument of latitude)"
            "\n"
            f"  c_rs    : {self.c_rs:.3f} m    (harmonic correction term for radius)\n"
            f"  c_rc    : {self.c_rc:.3f} m    (harmonic correction term for radius)\n"
            f"  c_is    : {self.c_is:.3f} rad  (harmonic correction term for inclination)\n"
            f"  c_ic    : {self.c_ic:.3f} rad  (harmonic correction term for inclination)"
        )
