#!/usr/bin/env python3
"""
Visualize how astronomical sources transit through the beam grid on a given day.

Produces a two-panel figure:
  Left  — Zenithal projection of the beam grid with source tracks overlaid
  Right — Altitude vs local time showing transit windows

Usage:
    # Sun and Cyg-A, today 6 AM - 3 PM PST
    python examples/plot_source_transit.py \\
        weights_512beam/int8_weights_512beam_feb19_svd.h5 \\
        --sources Sun Cyg-A \\
        --date 2026-03-09 \\
        --time-tz America/Los_Angeles \\
        --time-start 06:00 --time-end 15:00 \\
        -o /tmp/source_transit_2026-03-09.png

    # All default sources (Sun, Cas-A, Cyg-A), full day UTC
    python examples/plot_source_transit.py weights.h5
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_sun
    from astropy.time import Time
    import astropy.units as u
except ImportError:
    print("Error: astropy is required. Install with: pip install astropy",
          file=sys.stderr)
    sys.exit(1)

from bf_weights_generator.config import OVRO_LAT_DEG, OVRO_LON_DEG, OVRO_ALT_M

# Reuse from check_source_visibility
from examples.check_source_visibility import (
    KNOWN_SOURCES,
    OVRO,
    compute_source_track,
    load_beams_from_hdf5,
    check_beam_hits,
)

# Distinct colors for sources
SOURCE_COLORS = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
]


def plot_transit(
    sources_data, beam_alt, beam_az, fwhm_ew, fwhm_ns,
    output_path, title, tz, date_str,
):
    """
    Two-panel figure: zenithal beam map + altitude vs time.

    Parameters
    ----------
    sources_data : list of dict
        Each dict has keys: name, times, alt, az, hits, color
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates

    fig, (ax_zen, ax_alt) = plt.subplots(1, 2, figsize=(22, 10))

    # ── Left panel: Zenithal projection ──────────────────────────────────
    n_beams = len(beam_alt)
    norm = mcolors.Normalize(vmin=beam_alt.min(), vmax=beam_alt.max())
    cmap = plt.cm.viridis
    theta = np.linspace(0, 2 * np.pi, 256)

    # Zenith angle and projected coordinates
    za = 90.0 - beam_alt
    az_rad = np.deg2rad(beam_az)
    xs = za * np.sin(az_rad)
    ys = za * np.cos(az_rad)

    margin = max(fwhm_ew, fwhm_ns) * 1.2
    lim = max(np.max(np.abs(xs)), np.max(np.abs(ys))) + margin

    # Altitude circles
    for alt in [15, 30, 45, 60, 75, 80, 85]:
        r = 90 - alt
        if r <= lim * 1.2:
            ax_zen.plot(r * np.cos(theta), r * np.sin(theta),
                        color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
            ax_zen.text(0.4, -r - 0.3, f"{alt}\u00b0", ha="left", va="top",
                        fontsize=8, color="gray", alpha=0.6)

    # Azimuth lines
    for az in np.arange(0, 360, 45):
        az_r = np.deg2rad(az)
        ax_zen.plot([0, lim * np.sin(az_r)], [0, lim * np.cos(az_r)],
                    color="gray", linewidth=0.5, linestyle="--", alpha=0.4)

    # Cardinal labels
    off = lim + 1.1
    ax_zen.text(0, off, "N", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax_zen.text(off, 0, "E", ha="left", va="center", fontsize=14, fontweight="bold")
    ax_zen.text(1.0, -off, "S", ha="left", va="top", fontsize=14, fontweight="bold")
    ax_zen.text(-off, 0.8, "W", ha="right", va="bottom", fontsize=14, fontweight="bold")

    # Beam ellipses
    for i in range(n_beams):
        x, y = xs[i], ys[i]
        if za[i] < fwhm_ns:
            diameter = max(fwhm_ew, fwhm_ns)
            w = h = diameter
            angle = 0
        else:
            w = fwhm_ew
            h = fwhm_ns
            angle = -beam_az[i]

        color = cmap(norm(beam_alt[i]))
        ell = Ellipse((x, y), width=w, height=h, angle=angle,
                      facecolor=color, edgecolor="gray", linewidth=0.5, alpha=0.35)
        ax_zen.add_patch(ell)

    # Collect all hit beam indices across all sources for highlighting
    all_hit_beams = {}
    for sd in sources_data:
        for h in sd["hits"]:
            bi = h["beam_idx"]
            if bi not in all_hit_beams:
                all_hit_beams[bi] = sd["color"]

    # Highlight hit beams
    for bi, col in all_hit_beams.items():
        x, y = xs[bi], ys[bi]
        if za[bi] < fwhm_ns:
            diameter = max(fwhm_ew, fwhm_ns)
            w = h = diameter
            angle = 0
        else:
            w = fwhm_ew
            h = fwhm_ns
            angle = -beam_az[bi]
        ell = Ellipse((x, y), width=w, height=h, angle=angle,
                      facecolor=col, edgecolor=col, linewidth=1.5, alpha=0.5)
        ax_zen.add_patch(ell)
        ax_zen.annotate(str(bi), (x, y), fontsize=5, ha="center", va="center",
                        fontweight="bold", color="k", zorder=9)

    # Source tracks on zenithal projection
    for sd in sources_data:
        above = sd["alt"] > 0
        if not np.any(above):
            continue
        src_za = 90.0 - sd["alt"]
        src_az_rad = np.deg2rad(sd["az"])
        src_xs = src_za * np.sin(src_az_rad)
        src_ys = src_za * np.cos(src_az_rad)

        above_idx = np.where(above)[0]
        # Split at az wraps
        d_az = np.abs(np.diff(sd["az"][above_idx]))
        breaks = np.where(d_az > 90)[0] + 1
        segments = np.split(above_idx, breaks)

        for seg in segments:
            if len(seg) < 2:
                continue
            ax_zen.plot(src_xs[seg], src_ys[seg], color=sd["color"],
                        linewidth=2.0, alpha=0.8, label=sd["name"]
                        if seg is segments[0] else None)

        # Mark beam crossing peaks
        for h in sd["hits"]:
            peak_za = 90.0 - h["peak_alt"]
            peak_az_rad = np.deg2rad(h["peak_az"])
            ax_zen.plot(peak_za * np.sin(peak_az_rad),
                        peak_za * np.cos(peak_az_rad),
                        '*', color=sd["color"], markersize=12,
                        markeredgecolor='k', markeredgewidth=0.5, zorder=10)

    ax_zen.plot(0, 0, "r+", markersize=12, markeredgewidth=2, zorder=10)
    ax_zen.set_xlim(-lim, lim)
    ax_zen.set_ylim(-lim, lim)
    ax_zen.set_aspect("equal")
    ax_zen.set_xlabel("\u2190 West     East \u2192", fontsize=12)
    ax_zen.set_ylabel("\u2190 South     North \u2192", fontsize=11)
    ax_zen.set_title("Beam Grid — Zenithal Projection", fontsize=13, fontweight="bold")

    # Deduplicated legend
    handles, labels = ax_zen.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax_zen.legend(by_label.values(), by_label.keys(),
                      loc="upper right", fontsize=10)

    info = (f"{n_beams} beams | "
            f"FWHM: {fwhm_ew:.1f}\u00b0 (E-W) \u00d7 {fwhm_ns:.1f}\u00b0 (N-S)")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.85)
    ax_zen.text(0.02, 0.02, info, transform=ax_zen.transAxes, fontsize=10,
                verticalalignment="bottom", bbox=props, family="monospace")

    # ── Right panel: Altitude vs Time ────────────────────────────────────
    # Shade beam altitude band
    if len(beam_alt) > 0:
        ax_alt.axhspan(beam_alt.min() - fwhm_ns / 2,
                       beam_alt.max() + fwhm_ns / 2,
                       color="green", alpha=0.1, label="Beam coverage")

    tz_name = str(tz)
    for sd in sources_data:
        above = sd["alt"] > 0
        if not np.any(above):
            continue

        # Convert astropy times to matplotlib dates in the chosen timezone
        dt_list = [t.to_datetime(timezone=tz) for t in sd["times"]]
        above_idx = np.where(above)[0]

        # Split at time gaps > 2 hours (e.g. source set and rose again)
        if len(above_idx) > 1:
            dt_seconds = np.array([(sd["times"][above_idx[i+1]] -
                                    sd["times"][above_idx[i]]).sec
                                   for i in range(len(above_idx) - 1)])
            breaks = np.where(dt_seconds > 7200)[0] + 1
            segments = np.split(above_idx, breaks)
        else:
            segments = [above_idx]

        for j, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            seg_times = [dt_list[i] for i in seg]
            ax_alt.plot(seg_times, sd["alt"][seg], color=sd["color"],
                        linewidth=2.0, alpha=0.8,
                        label=sd["name"] if j == 0 else None)

    ax_alt.set_ylim(0, 95)
    ax_alt.set_xlabel(f"Time ({tz_name})", fontsize=12)
    ax_alt.set_ylabel("Altitude (\u00b0)", fontsize=12)
    ax_alt.set_title("Source Altitude vs Time", fontsize=13, fontweight="bold")
    ax_alt.grid(True, linestyle="--", alpha=0.3)
    ax_alt.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax_alt.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate(rotation=45)

    handles, labels = ax_alt.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax_alt.legend(by_label.values(), by_label.keys(),
                      loc="upper right", fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize source transits through the beam grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "weights", help="HDF5 weights file path",
    )
    parser.add_argument(
        "--date", default=None,
        help="Date to plot (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--time-tz", default="UTC",
        help="IANA timezone for display and --date interpretation "
             "(e.g. America/Los_Angeles)",
    )
    parser.add_argument(
        "--sources", nargs="+", default=["Cas-A", "Cyg-A", "Sun"],
        help="Source names (from: Sun, Tau-A, Cas-A, Cyg-A, Vir-A)",
    )
    parser.add_argument(
        "--dt", type=float, default=2.0,
        help="Time step in minutes",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: source_transit_YYYY-MM-DD.png)",
    )
    parser.add_argument(
        "--time-start", default="00:00",
        help="Start time (HH:MM in --time-tz)",
    )
    parser.add_argument(
        "--time-end", default="23:59",
        help="End time (HH:MM in --time-tz)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Text-only output (skip plot generation)",
    )
    args = parser.parse_args()

    # ── Timezone ─────────────────────────────────────────────────────────
    try:
        tz = ZoneInfo(args.time_tz)
    except KeyError:
        print(f"Error: unknown timezone {args.time_tz!r}", file=sys.stderr)
        sys.exit(1)

    # ── Date range ───────────────────────────────────────────────────────
    if args.date:
        date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        date = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        date = date.replace(tzinfo=None)  # naive for combining below

    h_start, m_start = map(int, args.time_start.split(":"))
    h_end, m_end = map(int, args.time_end.split(":"))

    start_local = date.replace(hour=h_start, minute=m_start,
                               second=0, microsecond=0)
    start_local = start_local.replace(tzinfo=tz)
    end_local = date.replace(hour=h_end, minute=m_end,
                             second=0, microsecond=0)
    end_local = end_local.replace(tzinfo=tz)

    # Convert to UTC for astropy
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    date_str = date.strftime("%Y-%m-%d")
    print(f"Date: {date_str}")
    print(f"Time range: {args.time_start} - {args.time_end} {args.time_tz}")
    print(f"  UTC: {start_utc.strftime('%Y-%m-%d %H:%M')} - "
          f"{end_utc.strftime('%Y-%m-%d %H:%M')}")

    # ── Load beams ───────────────────────────────────────────────────────
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading beams from: {weights_path}")
    beam_alt, beam_az, beam_names, fwhm_ew, fwhm_ns = \
        load_beams_from_hdf5(str(weights_path))
    print(f"  {len(beam_alt)} beams, FWHM: {fwhm_ew:.2f}\u00b0 \u00d7 "
          f"{fwhm_ns:.2f}\u00b0")
    print(f"  Beam alt range: {beam_alt.min():.1f}\u00b0 \u2013 "
          f"{beam_alt.max():.1f}\u00b0")

    # ── Compute source tracks ────────────────────────────────────────────
    sources_data = []
    for i, source_name in enumerate(args.sources):
        color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        print(f"\n{'='*60}")
        print(f"Source: {source_name}")
        print(f"{'='*60}")

        times, src_alt, src_az = compute_source_track(
            source_name, start_utc, end_utc, dt_minutes=args.dt,
        )

        above = src_alt > 0
        if np.any(above):
            print(f"  Visible alt range: {src_alt[above].min():.1f}\u00b0 \u2013 "
                  f"{src_alt[above].max():.1f}\u00b0")
        else:
            print(f"  {source_name} never rises above horizon in this period.")

        hits = check_beam_hits(
            src_alt, src_az, beam_alt, beam_az, fwhm_ew, fwhm_ns, times,
        )

        if hits:
            print(f"\n  {len(hits)} beam crossing(s) found:")
            print(f"  {'Beam':<10s} {'Entry':<22s} {'Exit':<22s} "
                  f"{'Duration':>10s} {'Peak Alt':>9s} {'Peak Az':>9s}")
            print(f"  {'-'*10} {'-'*22} {'-'*22} {'-'*10} {'-'*9} {'-'*9}")
            for h in sorted(hits, key=lambda x: x["entry_time"].datetime):
                entry = h["entry_time"].to_datetime(timezone=tz)
                exit_ = h["exit_time"].to_datetime(timezone=tz)
                dur = f"{h['duration_min']:.0f} min"
                print(f"  beam_{h['beam_idx']:03d}   "
                      f"{entry.strftime('%Y-%m-%d %H:%M'):<22s} "
                      f"{exit_.strftime('%Y-%m-%d %H:%M'):<22s} "
                      f"{dur:>10s} {h['peak_alt']:>8.1f}\u00b0 "
                      f"{h['peak_az']:>8.1f}\u00b0")

            unique_beams = sorted(set(h["beam_idx"] for h in hits))
            print(f"\n  Unique beams hit: {len(unique_beams)} / {len(beam_alt)}")
            print(f"  Beam indices: {unique_beams}")
        else:
            print(f"\n  No beam crossings found.")

        sources_data.append({
            "name": source_name,
            "times": times,
            "alt": src_alt,
            "az": src_az,
            "hits": hits,
            "color": color,
        })

    # ── Plot ─────────────────────────────────────────────────────────────
    if not args.no_plot:
        if args.output:
            output_path = args.output
        else:
            output_path = f"source_transit_{date_str}.png"

        title = (f"Source Transit at OVRO \u2014 {date_str} "
                 f"({args.time_start}\u2013{args.time_end} {args.time_tz})")

        plot_transit(
            sources_data, beam_alt, beam_az, fwhm_ew, fwhm_ns,
            output_path, title, tz, date_str,
        )


if __name__ == "__main__":
    main()
