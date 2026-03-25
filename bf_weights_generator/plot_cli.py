"""
CLI tool to plot beam positions on a zenithal projection.

Usage:
    casm-bf-plotter weights.h5                        # Plot all beams, save to PNG
    casm-bf-plotter weights.h5 -n 16                  # Plot first 16 beams
    casm-bf-plotter weights.h5 -o beams.png            # Custom output path
    casm-bf-plotter weights.h5 --freq 400e6            # FWHM at specific frequency
"""

import argparse
import os
import sys

try:
    import h5py
except ImportError:
    h5py = None

try:
    import numpy as np
except ImportError:
    np = None


def _read_beam_data(filepath):
    """Read pointings and antenna positions from HDF5 weight file.

    Returns (alt_deg, az_deg, positions_enu, freq_hz, format_type).
    """
    with h5py.File(filepath, "r") as f:
        fmt = str(f.attrs.get("format_type", ""))

        # Read pointings
        if "pointings" in f:
            alt_deg = f["pointings"]["alt_deg"][:]
            az_deg = f["pointings"]["az_deg"][:]
        else:
            raise ValueError("No pointings found in file")

        # Read antenna positions - try different config group names
        positions_enu = None
        for grp_name in ("array_config", "compute_array_config"):
            if grp_name in f:
                grp = f[grp_name]
                positions_enu = grp["positions_enu"][:]
                # Filter to active antennas if mask exists
                if "active_mask" in grp:
                    mask = grp["active_mask"][:]
                    positions_enu = positions_enu[mask]
                elif "antenna_flags" in grp:
                    flags = grp["antenna_flags"][:]
                    positions_enu = positions_enu[flags.astype(bool)]
                break

        if positions_enu is None:
            raise ValueError("No array_config found in file")

        # Read frequencies for center freq
        freq_hz = None
        if "frequencies_hz" in f:
            freqs = f["frequencies_hz"][:]
            freq_hz = float(np.mean(freqs))

        return alt_deg, az_deg, positions_enu, freq_hz, fmt


def _compute_fwhm(positions_enu, freq_hz=None):
    """Compute beam FWHM in E-W and N-S directions (degrees)."""
    if freq_hz is None:
        freq_hz = 437.5e6

    wavelength = 299792458.0 / freq_hz

    d_ew = positions_enu[:, 0].max() - positions_enu[:, 0].min()
    d_ns = positions_enu[:, 1].max() - positions_enu[:, 1].min()

    fwhm_ew = np.rad2deg(wavelength / d_ew) if d_ew > 0 else 180.0
    fwhm_ns = np.rad2deg(wavelength / d_ns) if d_ns > 0 else 180.0

    return fwhm_ew, fwhm_ns


def plot_zenithal_beams(alt_deg, az_deg, fwhm_ew, fwhm_ns, output_path, title=None):
    """Plot beams as ellipses on a zenithal equidistant projection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.colors as mcolors

    n_beams = len(alt_deg)
    norm = mcolors.Normalize(vmin=alt_deg.min(), vmax=alt_deg.max())
    cmap = plt.cm.viridis
    theta = np.linspace(0, 2 * np.pi, 256)

    # Zenith angle and projected coordinates
    za = 90.0 - alt_deg
    az_rad = np.deg2rad(az_deg)
    xs = za * np.sin(az_rad)
    ys = za * np.cos(az_rad)

    margin = max(fwhm_ew, fwhm_ns) * 1.2
    lim = max(np.max(np.abs(xs)), np.max(np.abs(ys))) + margin

    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait

    # Altitude circles
    for alt in [15, 30, 45, 60, 75, 80, 85]:
        r = 90 - alt
        if r <= lim * 1.2:
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
            ax.text(0.4, -r - 0.3, f"{alt}\u00b0", ha="left", va="top",
                    fontsize=8, color="gray", alpha=0.6)

    # Azimuth lines
    for az in np.arange(0, 360, 45):
        az_r = np.deg2rad(az)
        ax.plot([0, lim * np.sin(az_r)], [0, lim * np.cos(az_r)],
                color="gray", linewidth=0.5, linestyle="--", alpha=0.4)

    # Cardinal labels
    off = lim + 1.1
    ax.text(0, off, "N", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.text(off, 0, "E", ha="left", va="center", fontsize=14, fontweight="bold")
    ax.text(1.0, -off, "S", ha="left", va="top", fontsize=14, fontweight="bold")
    ax.text(-off, 0.8, "W", ha="right", va="bottom", fontsize=14, fontweight="bold")

    # Beams as ellipses
    for i in range(n_beams):
        x, y = xs[i], ys[i]

        if za[i] < fwhm_ns:
            diameter = max(fwhm_ew, fwhm_ns)
            w = h = diameter
            angle = 0
        else:
            w = fwhm_ew
            h = fwhm_ns
            angle = -az_deg[i]

        color = cmap(norm(alt_deg[i]))
        ellipse = Ellipse((x, y), width=w, height=h, angle=angle,
                          facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.55)
        ax.add_patch(ellipse)
        ax.annotate(str(i), (x, y), fontsize=max(4, min(8, 200 // n_beams)),
                    ha="center", va="center", fontweight="bold", color="k")

    ax.plot(0, 0, "r+", markersize=12, markeredgewidth=2, zorder=10)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("\u2190 West     East \u2192", fontsize=12)
    ax.set_ylabel("\u2190 South     North \u2192", fontsize=11)

    info = (f"{n_beams} beams | "
            f"FWHM: {fwhm_ew:.1f}\u00b0 (E-W) \u00d7 {fwhm_ns:.1f}\u00b0 (N-S)")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.85)
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", bbox=props, family="monospace")

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.03, label="Altitude (\u00b0)")

    plt.subplots_adjust(left=0.08, right=0.92, top=0.97, bottom=0.05)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_main():
    """Entry point for casm-bf-plotter CLI."""
    parser = argparse.ArgumentParser(
        prog="casm-bf-plotter",
        description="Plot beam positions on a zenithal projection from a weight file.",
    )
    parser.add_argument("file", help="HDF5 weight file to read")
    parser.add_argument(
        "-n", "--n-beams", type=int, default=None,
        help="Number of beams to plot (default: all)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image path (default: <input_name>_beams.png)",
    )
    parser.add_argument(
        "--freq", type=float, default=None,
        help="Reference frequency in Hz for FWHM calculation (default: center of band)",
    )
    parser.add_argument(
        "--title", default=None,
        help="Plot title (default: filename)",
    )

    args = parser.parse_args()

    if h5py is None:
        print("Error: h5py is required. Install with: pip install h5py", file=sys.stderr)
        sys.exit(1)
    if np is None:
        print("Error: numpy is required. Install with: pip install numpy", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        alt_deg, az_deg, positions_enu, file_freq_hz, fmt = _read_beam_data(args.file)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine frequency for FWHM
    freq_hz = args.freq if args.freq is not None else file_freq_hz

    # Subset beams
    n_beams = len(alt_deg)
    if args.n_beams is not None:
        if args.n_beams > n_beams:
            print(f"Warning: requested {args.n_beams} beams but file has {n_beams}", file=sys.stderr)
        else:
            n_beams = args.n_beams
            alt_deg = alt_deg[:n_beams]
            az_deg = az_deg[:n_beams]

    fwhm_ew, fwhm_ns = _compute_fwhm(positions_enu, freq_hz)

    print(f"File: {os.path.basename(args.file)}")
    print(f"Beams: {n_beams}")
    print(f"FWHM: {fwhm_ew:.1f}\u00b0 (E-W) \u00d7 {fwhm_ns:.1f}\u00b0 (N-S)")
    if freq_hz:
        print(f"Ref freq: {freq_hz / 1e6:.2f} MHz")

    # Output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.file))[0]
        output_path = f"{base}_beams.png"

    title = args.title or os.path.basename(args.file)

    try:
        plot_zenithal_beams(alt_deg, az_deg, fwhm_ew, fwhm_ns, output_path, title=title)
    except ImportError:
        print("Error: matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    plot_main()
