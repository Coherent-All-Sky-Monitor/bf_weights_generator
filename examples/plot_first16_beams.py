#!/usr/bin/env python3
"""
Plot the first 16 beams from a weights file on a zenithal projection
with elliptical beam footprints.

Usage:
    python examples/plot_first16_beams.py
    python examples/plot_first16_beams.py --weights /path/to/weights.h5 --output beams.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

from bf_weights_generator.io import load_int8_weights_hdf5
from bf_weights_generator import compute_beam_fwhm


DEFAULT_WEIGHTS = (
    "/home/casm/software/vishnu/beamforming_weights/geo_fallback/"
    "int8_weights_512beam_feb19_svd.h5"
)


def plot_zenithal_beams(pointings, fwhm_ew, fwhm_ns, output_path=None):
    """Plot beams as ellipses on a zenithal equidistant projection."""
    alts = [p.alt_deg for p in pointings]
    norm = mcolors.Normalize(vmin=min(alts), vmax=max(alts))
    cmap = plt.cm.viridis
    theta = np.linspace(0, 2 * np.pi, 256)

    # Auto-scale to fit data
    xs, ys = [], []
    for p in pointings:
        za = 90.0 - p.alt_deg
        xs.append(za * np.sin(np.deg2rad(p.az_deg)))
        ys.append(za * np.cos(np.deg2rad(p.az_deg)))
    margin = max(fwhm_ew, fwhm_ns) * 1.2
    lim = max(max(abs(v) for v in xs), max(abs(v) for v in ys)) + margin

    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait

    # Altitude circles
    for alt in [15, 30, 45, 60, 75, 80, 85]:
        za = 90 - alt
        if za <= lim * 1.2:
            ax.plot(za * np.cos(theta), za * np.sin(theta),
                    color='gray', linewidth=0.5, linestyle='--', alpha=0.4)
            ax.text(0.4, -za - 0.3, f'{alt}°', ha='left', va='top',
                    fontsize=8, color='gray', alpha=0.6)

    # Azimuth lines
    for az in np.arange(0, 360, 45):
        az_rad = np.deg2rad(az)
        ax.plot([0, lim * np.sin(az_rad)], [0, lim * np.cos(az_rad)],
                color='gray', linewidth=0.5, linestyle='--', alpha=0.4)

    # Cardinal labels (S and W offset slightly to avoid axis labels)
    off = lim + 1.1
    ax.text(0, off, 'N', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(off, 0, 'E', ha='left', va='center', fontsize=14, fontweight='bold')
    ax.text(1.0, -off, 'S', ha='left', va='top', fontsize=14, fontweight='bold')
    ax.text(-off, 0.8, 'W', ha='right', va='bottom', fontsize=14, fontweight='bold')

    # Beams as ellipses with beam number labels
    for i, p in enumerate(pointings):
        za = 90.0 - p.alt_deg
        az_rad = np.deg2rad(p.az_deg)
        x = za * np.sin(az_rad)
        y = za * np.cos(az_rad)

        if za < fwhm_ns:
            diameter = max(fwhm_ew, fwhm_ns)
            width = height = diameter
            angle = 0
        else:
            height = fwhm_ns
            width = fwhm_ew
            angle = -p.az_deg

        color = cmap(norm(p.alt_deg))
        ellipse = Ellipse((x, y), width=width, height=height, angle=angle,
                          facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.55)
        ax.add_patch(ellipse)
        ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center',
                    fontweight='bold', color='k')

    ax.plot(0, 0, 'r+', markersize=12, markeredgewidth=2, zorder=10)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('← West     East →', fontsize=12)
    ax.set_ylabel('← South     North →', fontsize=11)

    info = (f"{len(pointings)} beams | "
            f"FWHM: {fwhm_ew:.1f}° (E-W) × {fwhm_ns:.1f}° (N-S)")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props, family='monospace')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.03, label='Altitude (°)')

    plt.subplots_adjust(left=0.08, right=0.92, top=0.97, bottom=0.05)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot first 16 beams from a weights file.",
    )
    parser.add_argument(
        "--weights", "-w", default=DEFAULT_WEIGHTS,
        help="Int8 weights HDF5 file",
    )
    parser.add_argument(
        "--n-beams", "-n", type=int, default=16,
        help="Number of beams to plot (default: 16)",
    )
    parser.add_argument(
        "--output", "-o", default="first_16_beams_zenithal.png",
        help="Output file (default: first_16_beams_zenithal.png)",
    )
    args = parser.parse_args()

    weights = load_int8_weights_hdf5(args.weights)
    pointings = weights.pointings[:args.n_beams]
    positions = weights.array_config.active_positions
    fwhm_ew, fwhm_ns = compute_beam_fwhm(positions)

    print(f"FWHM: {fwhm_ew:.1f}° (E-W) × {fwhm_ns:.1f}° (N-S)")
    print(f"Plotting {len(pointings)} beams from {Path(args.weights).name}")
    for i, p in enumerate(pointings):
        print(f"  Beam {i:2d}: Alt={p.alt_deg:.1f}°, Az={p.az_deg:.1f}°")

    plot_zenithal_beams(pointings, fwhm_ew, fwhm_ns, output_path=args.output)


if __name__ == "__main__":
    main()
