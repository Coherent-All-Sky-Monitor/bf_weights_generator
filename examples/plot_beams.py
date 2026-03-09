#!/usr/bin/env python3
"""
Plot beam pointings on an Alt/Az sky map.

Creates a polar plot showing beam positions, with zenith at center
and horizon at the edge. Useful for visualizing beam coverage.

Usage:
    python plot_beams.py                    # Plot default transit survey beams
    python plot_beams.py --beams zenith     # Single zenith beam
    python plot_beams.py --beams "90:0,70:45,60:90"  # Custom beams
    python plot_beams.py --output beams.png # Save to file
    python plot_beams.py --weights snap_weights.h5  # Load from HDF5 file

Examples:
    # Plot transit survey beams and save to file
    python plot_beams.py --output transit_beams.png

    # Plot beams from an existing weights file
    python plot_beams.py --weights /tmp/layout1_weights.h5 --output beams.png

    # Custom beams with labels
    python plot_beams.py --beams "90:0,70:0,70:90,70:180,70:270"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from bf_weights_generator import (
    TRANSIT_SURVEY_BEAMS,
    StationaryPointing,
    parse_beams_arg,
    compute_beam_fwhm,
)
from bf_weights_generator.io import load_int8_weights_hdf5


def _ellipse_patch_polar(az_deg, alt_deg, fwhm_ew_deg, fwhm_ns_deg, n_pts=64):
    """Compute ellipse vertices in polar (az_rad, zenith_angle) coordinates.

    Generates points for an ellipse centered at (az_deg, alt_deg) with
    semi-axes fwhm_ew/2 (azimuth) and fwhm_ns/2 (altitude), correctly
    accounting for the cos(alt) convergence in azimuth.

    Returns arrays of (theta, r) in polar plot coordinates.
    """
    za_center = 90.0 - alt_deg
    az_center_rad = np.deg2rad(az_deg)

    t = np.linspace(0, 2 * np.pi, n_pts)
    # Half-widths in degrees on the sky
    half_ns = fwhm_ns_deg / 2.0   # altitude (radial) direction
    half_ew = fwhm_ew_deg / 2.0   # azimuth (tangential) direction

    # Offsets in zenithal equidistant projection (degrees)
    # d_alt along radial, d_az along tangential
    d_alt = half_ns * np.sin(t)   # radial offset
    cos_alt = np.cos(np.deg2rad(alt_deg))
    if cos_alt > 0.01:
        d_az = half_ew * np.cos(t) / cos_alt  # azimuth offset (degrees)
    else:
        d_az = half_ew * np.cos(t) * 100  # near zenith, large azimuth

    az_pts = az_center_rad + np.deg2rad(d_az)
    za_pts = za_center + d_alt

    return az_pts, za_pts


def plot_beams_polar(
    pointings,
    title="Beam Pointings (Alt/Az)",
    output_path=None,
    show_grid=True,
    fwhm_ew_deg=None,
    fwhm_ns_deg=None,
    beam_radius_deg=4.0,
    figsize=(10, 10),
):
    """
    Create a polar plot of beam pointings with elliptical beam footprints.

    Parameters
    ----------
    pointings : list of StationaryPointing
        Beam pointing directions.
    title : str
        Plot title.
    output_path : str or Path, optional
        If provided, save figure to this path.
    show_grid : bool
        Show altitude/azimuth grid lines.
    fwhm_ew_deg : float, optional
        E-W beam FWHM in degrees. If provided (with fwhm_ns_deg), draws
        elliptical beam footprints instead of circles.
    fwhm_ns_deg : float, optional
        N-S beam FWHM in degrees.
    beam_radius_deg : float
        Fallback beam radius if FWHM not provided (default: 4 deg).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    elliptical = fwhm_ew_deg is not None and fwhm_ns_deg is not None
    if not elliptical:
        fwhm_ew_deg = beam_radius_deg * 2
        fwhm_ns_deg = beam_radius_deg * 2

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    # Configure polar plot
    # Azimuth: 0=North at top, increasing clockwise (E=90, S=180, W=270)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # Clockwise

    # Radial axis: zenith angle (0 at center = 90 alt, 90 at edge = 0 alt)
    ax.set_rlim(0, 90)
    ax.set_rticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(['90', '75', '60', '45', '30', '15', '0'])

    # Grid
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.5)

    # Azimuth labels
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    # Colormap by altitude
    alts = [p.alt_deg for p in pointings]
    norm = mcolors.Normalize(vmin=min(alts), vmax=max(alts))
    cmap = plt.cm.viridis

    # Plot each beam as an ellipse polygon
    for i, p in enumerate(pointings):
        az_rad = np.deg2rad(p.az_deg)
        zenith_angle = 90.0 - p.alt_deg

        color = cmap(norm(p.alt_deg))

        # Draw elliptical footprint
        az_pts, za_pts = _ellipse_patch_polar(
            p.az_deg, p.alt_deg, fwhm_ew_deg, fwhm_ns_deg)
        verts = np.column_stack([az_pts, za_pts])
        poly = Polygon(verts, closed=True, facecolor=color,
                       edgecolor='black', linewidth=0.8, alpha=0.55,
                       transform=ax.transProjectionAffine + ax.transAxes)
        # Use transData for correct polar mapping
        poly.set_transform(ax.transData)
        ax.add_patch(poly)

        # Beam center dot
        ax.plot(az_rad, zenith_angle, 'k.', markersize=1)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label='Altitude (°)')

    # Info box
    info = f"{len(pointings)} beams"
    if elliptical:
        info += f"\nFWHM: {fwhm_ew_deg:.1f}° (E-W) × {fwhm_ns_deg:.1f}° (N-S)"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, info, fontsize=9, verticalalignment='bottom',
             bbox=props, family='monospace')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    return fig, ax


def plot_beams_cartesian(
    pointings,
    title="Beam Pointings (Cartesian)",
    output_path=None,
    beam_radius_deg=4.0,
    figsize=(12, 8),
):
    """
    Create a Cartesian (flat) plot of beam pointings in l,m coordinates.

    Parameters
    ----------
    pointings : list of StationaryPointing
        Beam pointing directions.
    title : str
        Plot title.
    output_path : str or Path, optional
        If provided, save figure to this path.
    beam_radius_deg : float
        Approximate beam radius for visualization.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot horizon circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.5, label='Horizon')

    # Plot altitude circles
    for alt in [30, 60]:
        r = np.cos(np.deg2rad(alt))
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'k:', linewidth=0.5, alpha=0.3)
        ax.annotate(f'{alt} alt', (r*0.7, r*0.7), fontsize=8, alpha=0.5)

    # Color map
    n_beams = len(pointings)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_beams, 10)))

    # Beam radius in l,m coordinates
    beam_radius_lm = np.sin(np.deg2rad(beam_radius_deg))

    # Plot each beam
    for i, p in enumerate(pointings):
        l, m = p.l, p.m

        # Plot beam as circle
        circle = plt.Circle((l, m), beam_radius_lm,
                            fill=True, facecolor=colors[i % 10],
                            edgecolor='black', linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)

        # Label
        label = p.name if p.name else f"Beam {i}"
        ax.annotate(label, (l, m), fontsize=9, ha='center', va='center',
                    fontweight='bold', color='white' if p.alt_deg > 45 else 'black')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('l (East)', fontsize=12)
    ax.set_ylabel('m (North)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add cardinal directions
    ax.annotate('N', (0, 1.05), ha='center', fontsize=12, fontweight='bold')
    ax.annotate('S', (0, -1.05), ha='center', fontsize=12, fontweight='bold')
    ax.annotate('E', (1.05, 0), va='center', fontsize=12, fontweight='bold')
    ax.annotate('W', (-1.05, 0), va='center', fontsize=12, fontweight='bold')

    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)

    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot beam pointings on Alt/Az sky map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--beams", "-b",
        default="transit",
        help="Beam specification: 'transit', 'zenith', or 'alt:az,alt:az,...' (default: transit)",
    )
    group.add_argument(
        "--weights", "-w",
        help="Load beams from an int8 weights HDF5 file",
    )

    parser.add_argument(
        "--output", "-o",
        help="Save plot to file (e.g., beams.png)",
    )
    parser.add_argument(
        "--style",
        choices=["polar", "cartesian", "both"],
        default="polar",
        help="Plot style: polar (Alt/Az), cartesian (l,m), or both (default: polar)",
    )
    parser.add_argument(
        "--beam-radius",
        type=float,
        default=None,
        help="Fallback beam radius in degrees when FWHM unavailable (default: 4.0)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        default=True,
        help="Don't display plot interactively (default: True, savefig only)",
    )

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting.", file=sys.stderr)
        print("Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    # Load beams
    fwhm_ew = fwhm_ns = None
    if args.weights:
        print(f"Loading beams from: {args.weights}")
        weights = load_int8_weights_hdf5(args.weights)
        pointings = weights.pointings
        title = f"Beams from {Path(args.weights).name}"
        # Compute FWHM from array config if available
        if hasattr(weights, 'array_config') and weights.array_config is not None:
            positions = weights.array_config.active_positions
            fwhm_ew, fwhm_ns = compute_beam_fwhm(positions)
            print(f"Beam FWHM: {fwhm_ew:.1f}° (E-W) × {fwhm_ns:.1f}° (N-S)")
    else:
        try:
            pointings = parse_beams_arg(args.beams)
        except ValueError as e:
            print(f"Error parsing beams: {e}", file=sys.stderr)
            sys.exit(1)
        title = f"Beam Pointings ({len(pointings)} beams)"

    beam_radius = args.beam_radius if args.beam_radius is not None else 4.0

    print(f"Plotting {len(pointings)} beams:")
    for i, p in enumerate(pointings):
        print(f"  Beam {i}: {p.name or 'unnamed'} at Alt={p.alt_deg:.1f}, Az={p.az_deg:.1f}")

    # Determine output paths
    if args.output:
        output_base = Path(args.output)
        if args.style == "both":
            polar_output = output_base.with_stem(output_base.stem + "_polar")
            cart_output = output_base.with_stem(output_base.stem + "_cartesian")
        else:
            polar_output = cart_output = output_base
    else:
        polar_output = cart_output = None

    # Create plots
    if args.style in ["polar", "both"]:
        fig1, ax1 = plot_beams_polar(
            pointings,
            title=title,
            output_path=polar_output if args.style != "both" else str(polar_output),
            fwhm_ew_deg=fwhm_ew,
            fwhm_ns_deg=fwhm_ns,
            beam_radius_deg=beam_radius,
        )

    if args.style in ["cartesian", "both"]:
        fig2, ax2 = plot_beams_cartesian(
            pointings,
            title=title,
            output_path=cart_output if args.style != "both" else str(cart_output),
            beam_radius_deg=beam_radius,
        )

    if not args.no_show and not args.output:
        plt.show()
    elif not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
