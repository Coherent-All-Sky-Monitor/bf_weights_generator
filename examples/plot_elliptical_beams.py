#!/usr/bin/env python3
"""
Plot 64 elliptical beams on an Alt/Az sky map.

Generates a beam grid using the current antenna layout (from CSV) and
plots each beam as an ellipse whose size reflects the E-W and N-S FWHM
computed from the array baselines.

Usage:
    python examples/plot_elliptical_beams.py
    python examples/plot_elliptical_beams.py --layout casm_antenna_layout_current.csv
    python examples/plot_elliptical_beams.py --n-beams 32 --output beams_32.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.colors as mcolors
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib",
          file=sys.stderr)
    sys.exit(1)

from bf_weights_generator import (
    Array64Config,
    compute_beam_fwhm,
    generate_beam_grid,
)


def plot_elliptical_beams(
    pointings,
    fwhm_ew_deg,
    fwhm_ns_deg,
    title="Elliptical Beam Grid",
    output_path=None,
    figsize=(12, 12),
):
    """
    Plot beams as ellipses on a polar Alt/Az sky map.

    Parameters
    ----------
    pointings : list of StationaryPointing
        Beam pointing directions.
    fwhm_ew_deg : float
        E-W beam FWHM in degrees (azimuth direction).
    fwhm_ns_deg : float
        N-S beam FWHM in degrees (altitude direction).
    title : str
        Plot title.
    output_path : str or Path, optional
        Save figure to this path.
    figsize : tuple
        Figure size in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw sky as a circle (zenith angle 0-90)
    theta = np.linspace(0, 2 * np.pi, 256)

    # Horizon circle
    ax.plot(90 * np.cos(theta), 90 * np.sin(theta), 'k-', linewidth=1.5)

    # Altitude circles (zenith angle = 90 - alt)
    for alt in [15, 30, 45, 60, 75]:
        za = 90 - alt
        ax.plot(za * np.cos(theta), za * np.sin(theta),
                color='gray', linewidth=0.5, linestyle='--', alpha=0.4)
        ax.text(0, -za - 1.5, f'{alt}°', ha='center', va='top',
                fontsize=8, color='gray', alpha=0.6)

    # Azimuth lines
    for az in np.arange(0, 360, 45):
        az_rad = np.deg2rad(az)
        ax.plot([0, 90 * np.sin(az_rad)], [0, 90 * np.cos(az_rad)],
                color='gray', linewidth=0.5, linestyle='--', alpha=0.4)

    # Cardinal labels
    offset = 95
    ax.text(0, offset, 'N', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(offset, 0, 'E', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0, -offset, 'S', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(-offset, 0, 'W', ha='center', va='center', fontsize=14, fontweight='bold')

    # Colormap for beams by altitude
    alts = [p.alt_deg for p in pointings]
    norm = mcolors.Normalize(vmin=min(alts), vmax=max(alts))
    cmap = plt.cm.viridis

    # Plot each beam as an ellipse
    for i, p in enumerate(pointings):
        za = 90.0 - p.alt_deg  # zenith angle
        az_rad = np.deg2rad(p.az_deg)

        # Position in Cartesian projection (x=East, y=North)
        x = za * np.sin(az_rad)
        y = za * np.cos(az_rad)

        # Ellipse dimensions
        # N-S FWHM maps to altitude direction (radial in this projection)
        # E-W FWHM maps to azimuth direction, but scales with 1/cos(alt) on sky
        # In the flat zenith-angle projection, the radial extent = fwhm_ns
        # and the tangential extent = fwhm_ew (already in degrees on sky)
        height = fwhm_ns_deg  # radial (altitude) direction
        width = fwhm_ew_deg   # tangential (azimuth) direction

        # Ellipse orientation: the radial direction points from center to beam
        # angle = azimuth measured from +y axis (North), so rotation is -az
        angle = -p.az_deg

        color = cmap(norm(p.alt_deg))
        ellipse = Ellipse(
            (x, y), width=width, height=height, angle=angle,
            facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.55,
        )
        ax.add_patch(ellipse)

        # Small dot at beam center
        ax.plot(x, y, 'k.', markersize=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label('Altitude (°)', fontsize=12)

    # Zenith marker
    ax.plot(0, 0, 'r+', markersize=12, markeredgewidth=2, zorder=10)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_aspect('equal')
    ax.set_xlabel('← West     East →', fontsize=11)
    ax.set_ylabel('← South     North →', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Info box
    info = (
        f"{len(pointings)} beams\n"
        f"FWHM: {fwhm_ew_deg:.1f}° (E-W) × {fwhm_ns_deg:.1f}° (N-S)\n"
        f"Alt range: {min(alts):.0f}° – {max(alts):.0f}°"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props, family='monospace')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    return fig, ax


def plot_elliptical_beams_rectangular(
    pointings,
    fwhm_ew_deg,
    fwhm_ns_deg,
    title="Elliptical Beam Grid",
    output_path=None,
    figsize=(16, 8),
):
    """
    Plot beams as ellipses on a rectangular Alt/Az sky map.

    Parameters
    ----------
    pointings : list of StationaryPointing
        Beam pointing directions.
    fwhm_ew_deg : float
        E-W beam FWHM in degrees (azimuth direction).
    fwhm_ns_deg : float
        N-S beam FWHM in degrees (altitude direction).
    title : str
        Plot title.
    output_path : str or Path, optional
        Save figure to this path.
    figsize : tuple
        Figure size in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Colormap for beams by altitude
    alts = [p.alt_deg for p in pointings]
    norm = mcolors.Normalize(vmin=min(alts), vmax=max(alts))
    cmap = plt.cm.viridis

    # Plot each beam as an ellipse
    for i, p in enumerate(pointings):
        az = p.az_deg
        alt = p.alt_deg

        # E-W FWHM is the azimuth extent on sky; correct for cos(alt)
        # convergence so the ellipse width in azimuth degrees is larger
        # at higher altitudes
        cos_alt = np.cos(np.deg2rad(alt))
        if cos_alt > 0.01:
            width_az = fwhm_ew_deg / cos_alt
        else:
            width_az = 360.0
        height_alt = fwhm_ns_deg

        color = cmap(norm(alt))
        ellipse = Ellipse(
            (az, alt), width=width_az, height=height_alt, angle=0,
            facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.55,
        )
        ax.add_patch(ellipse)

        # Small dot at beam center
        ax.plot(az, alt, 'k.', markersize=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Altitude (°)', fontsize=12)

    ax.set_xlim(-5, 365)
    ax.set_ylim(min(alts) - fwhm_ns_deg, 95)
    ax.set_xlabel('Azimuth (°)', fontsize=12)
    ax.set_ylabel('Altitude (°)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Cardinal direction labels on azimuth axis
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.set_xticklabels(['N\n0', 'NE\n45', 'E\n90', 'SE\n135',
                         'S\n180', 'SW\n225', 'W\n270', 'NW\n315', 'N\n360'])

    ax.grid(True, linestyle='--', alpha=0.3)

    # Info box
    info = (
        f"{len(pointings)} beams\n"
        f"FWHM: {fwhm_ew_deg:.1f}° (E-W) × {fwhm_ns_deg:.1f}° (N-S)\n"
        f"Alt range: {min(alts):.0f}° – {max(alts):.0f}°"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props, family='monospace')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot elliptical beams on Alt/Az sky map.",
    )
    parser.add_argument(
        "--layout", "-l",
        default="casm_antenna_layout_current.csv",
        help="Antenna layout CSV file (default: casm_antenna_layout_current.csv)",
    )
    parser.add_argument(
        "--n-beams", "-n", type=int, default=64,
        help="Number of beams (default: 64)",
    )
    parser.add_argument(
        "--alt-min", type=float, default=30.0,
        help="Minimum altitude in degrees (default: 30)",
    )
    parser.add_argument(
        "--freq-mhz", type=float, default=None,
        help="Reference frequency in MHz (default: center of band)",
    )
    parser.add_argument(
        "--style", "-s",
        choices=["polar", "rectangular", "both"],
        default="polar",
        help="Plot style: polar (zenithal), rectangular (Alt vs Az), or both (default: polar)",
    )
    parser.add_argument(
        "--exclude-outtrigger", action="store_true",
        help="Exclude out-trigger antenna (pos_id starting with 'O') from FWHM and grid computation",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save plot to file (e.g., beams.png)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        default=True,
        help="Don't display plot interactively (default: True, savefig only)",
    )
    args = parser.parse_args()

    # Load array
    csv_path = Path(args.layout)
    if not csv_path.exists():
        # Try relative to project root
        csv_path = Path(__file__).parent.parent / args.layout
    if not csv_path.exists():
        print(f"Error: CSV file not found: {args.layout}", file=sys.stderr)
        sys.exit(1)

    array = Array64Config.from_csv(str(csv_path))
    print(f"Loaded array: {array.n_active} active antennas from {csv_path.name}")

    # Get positions, optionally excluding out-trigger
    positions = array.active_positions
    subtitle = csv_path.stem
    if args.exclude_outtrigger:
        # Exclude antennas whose pos_id starts with 'O' (out-trigger)
        keep = []
        for idx, ai in enumerate(array.active_indices):
            pid = array.pos_ids[ai]
            if pid.upper().startswith('O'):
                print(f"  Excluding out-trigger: ant64={ai} pos_id={pid}")
            else:
                keep.append(idx)
        positions = positions[keep]
        subtitle += " (no out-trigger)"
        print(f"  Using {len(positions)} antennas after excluding out-trigger")

    freq_hz = args.freq_mhz * 1e6 if args.freq_mhz else None
    fwhm_ew, fwhm_ns = compute_beam_fwhm(positions, freq_hz=freq_hz)
    print(f"Beam FWHM: {fwhm_ew:.1f}° (E-W) × {fwhm_ns:.1f}° (N-S)")

    # Generate beam grid
    beams = generate_beam_grid(
        n_beams=args.n_beams,
        positions_enu=positions,
        freq_hz=freq_hz,
        alt_min_deg=args.alt_min,
    )
    print(f"Generated {len(beams)} beams (target: {args.n_beams})")

    title = f"{len(beams)} Elliptical Beams — {subtitle}"

    # Determine output paths for each style
    output_base = Path(args.output) if args.output else None

    if args.style in ("polar", "both"):
        if output_base and args.style == "both":
            polar_out = str(output_base.with_stem(output_base.stem + "_polar"))
        else:
            polar_out = str(output_base) if output_base else None
        plot_elliptical_beams(
            beams, fwhm_ew, fwhm_ns,
            title=title,
            output_path=polar_out,
        )

    if args.style in ("rectangular", "both"):
        if output_base and args.style == "both":
            rect_out = str(output_base.with_stem(output_base.stem + "_rectangular"))
        else:
            rect_out = str(output_base) if output_base else None
        plot_elliptical_beams_rectangular(
            beams, fwhm_ew, fwhm_ns,
            title=title,
            output_path=rect_out,
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
