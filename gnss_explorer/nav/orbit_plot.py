"""Plots satellite orbits around the Earth in 3D."""

from typing import TYPE_CHECKING, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

if TYPE_CHECKING:
    from cartopy.mpl import geoaxes

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gnss_explorer.nav import ephemeris


def plot_earth(ax: Axes3D) -> None:
    """Plot a 3D sphere representing the Earth."""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = ephemeris.R_EARTH * np.outer(np.cos(u), np.sin(v))
    y = ephemeris.R_EARTH * np.outer(np.sin(u), np.sin(v))
    z = ephemeris.R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(
        x, y, z, color="b", alpha=0.3, rstride=5, cstride=5, linewidth=0, antialiased=False
    )


def plot_orbits(
    ephs: list[ephemeris.GpsEphemeris], t_start: float, coord_frame: str = "ECEF"
) -> None:
    """Generate and display the orbit plot for one or more satellites."""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    plot_earth(ax)

    # Define a color cycle for the orbits
    colors = ["gold", "cyan", "magenta", "green", "orange", "purple"]
    all_orbit_points = []

    for i, eph in enumerate(ephs):
        # Select the correct function based on the desired coordinate frame
        calc_func = eph.as_eci if coord_frame.upper() == "ECI" else eph.as_ecef

        # Calculate the orbital path
        orbital_period_s = 2 * 3600
        t_range = np.linspace(t_start - orbital_period_s, t_start, num=1000)
        orbit_points = np.array([calc_func(t) for t in t_range])
        all_orbit_points.append(orbit_points)

        # Calculate satellite position at t_oe
        sat_pos_toe = calc_func(t_start)

        # Plot the orbital path
        color = colors[i % len(colors)]
        ax.plot(
            orbit_points[:, 0],
            orbit_points[:, 1],
            orbit_points[:, 2],
            label=f"SV {eph.sv_prn} Orbit Path",
            color=color,
        )

        # Plot the satellite's position at t_oe + tow
        ax.scatter(
            xs=[sat_pos_toe[0]],
            ys=[sat_pos_toe[1]],
            zs=[sat_pos_toe[2]],  # type: ignore[reportArgumentType]
            color="red",
            s=100,
            label=f"SV {eph.sv_prn} at t_oe + tow",
            depthshade=True,
            zorder=10,
        )

    # Customize the plot
    ax.set_title(f"GPS Satellite Orbits in 3D ({coord_frame.upper()})", fontsize=16)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.legend()

    # Set aspect ratio to be equal for a spherical view
    if all_orbit_points:
        full_orbit_data = np.concatenate(all_orbit_points)
        max_range = full_orbit_data.max()
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    ax.set_aspect("equal", adjustable="box")

    # Improve viewing angle
    ax.view_init(elev=30.0, azim=45)
    plt.show()


def plot_orbits_2d(
    ephs: list[ephemeris.GpsEphemeris],
    t_start: float,
    *,
    duration_s: int = 2 * 3600,
    projection: str = "robinson",
) -> None:
    """Plot satellite ground tracks on a world map (Mercator by default).

    Parameters
    ----------
    ephs
        List of satellite ephemerides.
    t_start
        GPS time-of-week, seconds.
    coord_frame
        ``"ECEF"`` (default) or ``"ECI"``.
    duration_s
        Length of history to show leading up to *t_start* [s].
    projection
        Map projection name - one of ``"mercator"``, ``"platecarree"``, or
        ``"robinson"``.  Case-insensitive.

    """
    proj_map = {
        "mercator": ccrs.Mercator(),
        "platecarree": ccrs.PlateCarree(),
        "robinson": ccrs.Robinson(),
    }
    proj = proj_map.get(projection.lower(), ccrs.Mercator())

    plt.figure(figsize=(14, 7))
    ax: geoaxes.GeoAxes = cast("geoaxes.GeoAxes", plt.axes(projection=proj))
    ax.set_global()

    # Basemap styling
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.coastlines(resolution="110m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)

    colors = ["gold", "cyan", "magenta", "green", "orange", "purple"]

    for i, eph in enumerate(ephs):
        calc = eph.as_lla
        ts = np.linspace(t_start - duration_s, t_start, 720)
        lla_track = np.asarray([calc(t) for t in ts])
        lats = lla_track[:, 0]
        lons = lla_track[:, 1]

        # Unwrap longitudes to avoid 180° jumps when plotting
        lons = np.degrees(np.unwrap(np.radians(lons)))

        color = colors[i % len(colors)]
        ax.plot(
            lons,
            lats,
            color=color,
            linewidth=1.2,
            transform=ccrs.PlateCarree(),
            label=f"SV {eph.sv_prn} track",
        )

        # Mark the "now" sub-satellite point
        lat_now = lats[-1]
        lon_now = lons[-1]
        ax.plot(
            lon_now, lat_now, marker="o", markersize=6, color="red", transform=ccrs.PlateCarree()
        )

    ax.set_title(f"GPS Satellite Ground Tracks ({projection.title()} projection)")
    ax.legend(loc="lower left")

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False

    plt.tight_layout()
    plt.show()
