"""Public ``plot_source_transit`` API.

Visualize source tracks through a beam grid stored in a SNAP int8
weights HDF5. Promoted from ``examples/plot_source_transit.py`` (and
its helpers in ``examples/check_source_visibility.py``) so notebooks
can import directly without manipulating ``sys.path``.

Usage in a notebook
-------------------

>>> from bf_weights_generator import plot_source_transit
>>> figs = plot_source_transit(
...     "weights.h5",
...     sources=["sun", "cas-a", "cyg-a"],
...     date="2026-05-07",
...     time_tz="America/Los_Angeles",
...     time_start="06:00", time_end="15:00",
... )

Usage from the CLI
------------------

::

    casm-bf-source-transit weights.h5 \\
        --sources sun cas-a cyg-a \\
        --date 2026-05-07 \\
        --time-tz America/Los_Angeles \\
        --time-start 06:00 --time-end 15:00 \\
        -o /tmp/transit.png
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u

from bf_weights_generator.config import OVRO_LAT_DEG, OVRO_LON_DEG, OVRO_ALT_M


# Source catalog (kept local so notebooks don't need to know about
# casm_vis_analysis.sources). Mirrors the entries used by the CASM stack;
# 'sun' uses astropy.get_sun, others are J2000 SkyCoords.
KNOWN_SOURCES = {
    "sun":      None,
    "tau-a":    SkyCoord(ra="05h34m31.94s", dec="+22d00m52.2s"),
    "crab":     SkyCoord(ra="05h34m31.94s", dec="+22d00m52.2s"),
    "cas-a":    SkyCoord(ra="23h23m24.00s", dec="+58d48m54.0s"),
    "cyg-a":    SkyCoord(ra="19h59m28.36s", dec="+40d44m02.1s"),
    "vir-a":    SkyCoord(ra="12h30m49.42s", dec="+12d23m28.0s"),
    "b0329+54": SkyCoord(ra="03h32m59.4096s", dec="+54d34m43.329s"),
}

OVRO_LOCATION = EarthLocation(
    lat=OVRO_LAT_DEG * u.deg,
    lon=OVRO_LON_DEG * u.deg,
    height=OVRO_ALT_M * u.m,
)

SOURCE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


def _compute_source_track(source_name, start_utc, end_utc, dt_minutes=2.0):
    key = source_name.lower().replace(" ", "-")
    is_sun = key == "sun"
    if not is_sun and key not in KNOWN_SOURCES:
        raise ValueError(
            f"Unknown source: {source_name!r}. "
            f"Known: sun, {', '.join(k for k in KNOWN_SOURCES if k not in ('sun', 'crab'))}"
        )
    t_start = Time(start_utc)
    t_end = Time(end_utc)
    n_steps = max(1, int((t_end - t_start).sec / (dt_minutes * 60)))
    times = t_start + np.linspace(0, (t_end - t_start).sec, n_steps) * u.s
    altaz_frame = AltAz(obstime=times, location=OVRO_LOCATION)
    coords = (get_sun(times) if is_sun else KNOWN_SOURCES[key]).transform_to(altaz_frame)
    return times, coords.alt.deg, coords.az.deg


def _load_beams_from_hdf5(filepath):
    """Read beam alt/az + array geometry from a SNAP int8 weights HDF5."""
    import h5py
    with h5py.File(filepath, "r") as f:
        alt = f["pointings/alt_deg"][:]
        az = f["pointings/az_deg"][:]
        names = json.loads(f["pointings"].attrs["names"])
        positions = f["array_config/positions_enu"][:]
        active = f["array_config/active_mask"][:]

    from bf_weights_generator import compute_beam_fwhm
    active_pos = positions[active]
    fwhm_ew, fwhm_ns = compute_beam_fwhm(active_pos)
    return alt, az, names, fwhm_ew, fwhm_ns


def _check_beam_hits(source_alt, source_az, beam_alt, beam_az,
                     fwhm_ew, fwhm_ns, times):
    """Find every (beam, contiguous-time-interval) the source crosses through."""
    half_ew = fwhm_ew / 2.0
    half_ns = fwhm_ns / 2.0
    hits = []
    for bi in range(len(beam_alt)):
        b_alt = beam_alt[bi]; b_az = beam_az[bi]
        d_alt = source_alt - b_alt
        d_az = ((source_az - b_az + 180) % 360) - 180
        cos_alt = np.cos(np.deg2rad(b_alt))
        d_az_sky = d_az * cos_alt
        r_ew = d_az_sky / half_ew
        r_ns = d_alt / half_ns
        r = np.sqrt(r_ew**2 + r_ns**2)
        in_beam = r < 1.0
        if not np.any(in_beam):
            continue
        idxs = np.where(in_beam)[0]
        splits = np.where(np.diff(idxs) > 1)[0] + 1
        for grp in np.split(idxs, splits):
            i_entry, i_exit = grp[0], grp[-1]
            i_peak = grp[np.argmin(r[grp])]
            hits.append({
                "beam_idx": bi,
                "entry_time": times[i_entry],
                "exit_time": times[i_exit],
                "peak_alt": float(source_alt[i_peak]),
                "peak_az": float(source_az[i_peak]),
                "min_dist_deg": float(r[i_peak] * max(half_ew, half_ns)),
                "duration_min": float((times[i_exit] - times[i_entry]).sec / 60),
            })
    return hits


def _plot_two_panel(sources_data, beam_alt, beam_az, fwhm_ew, fwhm_ns,
                    title, tz):
    """Render the two-panel figure (zenith projection + alt vs time)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates

    fig, (ax_zen, ax_alt) = plt.subplots(1, 2, figsize=(20, 9))

    # ── Left panel: zenithal projection ─────────────────────────────────
    norm = mcolors.Normalize(vmin=beam_alt.min(), vmax=beam_alt.max())
    cmap = plt.cm.viridis
    theta = np.linspace(0, 2 * np.pi, 256)

    za = 90.0 - beam_alt
    az_rad = np.deg2rad(beam_az)
    xs = za * np.sin(az_rad)
    ys = za * np.cos(az_rad)

    margin = max(fwhm_ew, fwhm_ns) * 1.2
    lim = max(np.max(np.abs(xs)), np.max(np.abs(ys))) + margin

    for alt in [15, 30, 45, 60, 75, 80, 85]:
        r = 90 - alt
        if r <= lim * 1.2:
            ax_zen.plot(r * np.cos(theta), r * np.sin(theta),
                        color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
            ax_zen.text(0.4, -r - 0.3, f"{alt}°", ha="left", va="top",
                        fontsize=8, color="gray", alpha=0.6)
    for az in np.arange(0, 360, 45):
        az_r = np.deg2rad(az)
        ax_zen.plot([0, lim * np.sin(az_r)], [0, lim * np.cos(az_r)],
                    color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    off = lim + 1.1
    ax_zen.text(0, off, "N", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax_zen.text(off, 0, "E", ha="left",   va="center", fontsize=14, fontweight="bold")
    ax_zen.text(1.0, -off, "S", ha="left", va="top",   fontsize=14, fontweight="bold")
    ax_zen.text(-off, 0.8, "W", ha="right", va="bottom", fontsize=14, fontweight="bold")

    for i in range(len(beam_alt)):
        x, y = xs[i], ys[i]
        if za[i] < fwhm_ns:
            w = h = max(fwhm_ew, fwhm_ns); angle = 0
        else:
            w = fwhm_ew; h = fwhm_ns; angle = -beam_az[i]
        color = cmap(norm(beam_alt[i]))
        ax_zen.add_patch(Ellipse((x, y), width=w, height=h, angle=angle,
                                 facecolor=color, edgecolor="gray",
                                 linewidth=0.5, alpha=0.35))

    all_hit_beams = {}
    for sd in sources_data:
        for h in sd["hits"]:
            all_hit_beams.setdefault(h["beam_idx"], sd["color"])
    for bi, col in all_hit_beams.items():
        x, y = xs[bi], ys[bi]
        if za[bi] < fwhm_ns:
            w = h = max(fwhm_ew, fwhm_ns); angle = 0
        else:
            w = fwhm_ew; h = fwhm_ns; angle = -beam_az[bi]
        ax_zen.add_patch(Ellipse((x, y), width=w, height=h, angle=angle,
                                 facecolor=col, edgecolor=col, linewidth=1.5,
                                 alpha=0.5))
        ax_zen.annotate(str(bi), (x, y), fontsize=5, ha="center", va="center",
                        fontweight="bold", color="k", zorder=9)

    for sd in sources_data:
        above = sd["alt"] > 0
        if not np.any(above):
            continue
        src_za = 90.0 - sd["alt"]
        src_az_rad = np.deg2rad(sd["az"])
        sx = src_za * np.sin(src_az_rad); sy = src_za * np.cos(src_az_rad)
        idxs = np.where(above)[0]
        d_az = np.abs(np.diff(sd["az"][idxs]))
        breaks = np.where(d_az > 90)[0] + 1
        for j, seg in enumerate(np.split(idxs, breaks)):
            if len(seg) < 2:
                continue
            ax_zen.plot(sx[seg], sy[seg], color=sd["color"], linewidth=2.0,
                        alpha=0.85, label=sd["name"] if j == 0 else None)
        for h in sd["hits"]:
            pza = 90.0 - h["peak_alt"]; par = np.deg2rad(h["peak_az"])
            ax_zen.plot(pza * np.sin(par), pza * np.cos(par),
                        '*', color=sd["color"], markersize=12,
                        markeredgecolor='k', markeredgewidth=0.5, zorder=10)

    ax_zen.plot(0, 0, "r+", markersize=12, markeredgewidth=2, zorder=10)
    ax_zen.set_xlim(-lim, lim); ax_zen.set_ylim(-lim, lim)
    ax_zen.set_aspect("equal")
    ax_zen.set_xlabel("← West     East →"); ax_zen.set_ylabel("← South     North →")
    ax_zen.set_title("Beam grid — Zenithal projection", fontweight="bold")
    handles, labels = ax_zen.get_legend_handles_labels()
    if labels:
        ax_zen.legend(dict(zip(labels, handles)).values(),
                      dict(zip(labels, handles)).keys(),
                      loc="upper right", fontsize=10)

    # ── Right panel: altitude vs time ───────────────────────────────────
    if len(beam_alt) > 0:
        ax_alt.axhspan(beam_alt.min() - fwhm_ns / 2, beam_alt.max() + fwhm_ns / 2,
                       color="green", alpha=0.1, label="Beam coverage")
    for sd in sources_data:
        above = sd["alt"] > 0
        if not np.any(above):
            continue
        dt_list = [t.to_datetime(timezone=tz) for t in sd["times"]]
        idxs = np.where(above)[0]
        if len(idxs) > 1:
            gaps = np.array([(sd["times"][idxs[i+1]] - sd["times"][idxs[i]]).sec
                             for i in range(len(idxs)-1)])
            breaks = np.where(gaps > 7200)[0] + 1
            segs = np.split(idxs, breaks)
        else:
            segs = [idxs]
        for j, seg in enumerate(segs):
            if len(seg) < 2:
                continue
            ax_alt.plot([dt_list[i] for i in seg], sd["alt"][seg],
                        color=sd["color"], linewidth=2.0, alpha=0.85,
                        label=sd["name"] if j == 0 else None)

    ax_alt.set_ylim(0, 95)
    ax_alt.set_xlabel(f"Time ({tz})"); ax_alt.set_ylabel("Altitude (°)")
    ax_alt.set_title("Source altitude vs time", fontweight="bold")
    ax_alt.grid(True, linestyle="--", alpha=0.3)
    ax_alt.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax_alt.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate(rotation=45)
    handles, labels = ax_alt.get_legend_handles_labels()
    if labels:
        ax_alt.legend(dict(zip(labels, handles)).values(),
                      dict(zip(labels, handles)).keys(),
                      loc="upper right", fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_source_transit(weights, sources, *, date=None, time_tz="UTC",
                        time_start="00:00", time_end="23:59", dt_minutes=2.0,
                        output_path=None):
    """Visualize source tracks across the beam grid stored in ``weights``.

    Parameters
    ----------
    weights : str or Path
        Path to an int8 SNAP weights HDF5 with a ``pointings`` group and
        ``array_config`` group (the format ``save_int8_weights_hdf5``
        / ``save_combined_weights_hdf5`` write).
    sources : list of str
        Source names: ``'sun'``, ``'cas-a'``, ``'cyg-a'``, ``'tau-a'`` /
        ``'crab'``, ``'vir-a'``, ``'b0329+54'``.
    date : str, optional
        ``YYYY-MM-DD``. Default: today (in ``time_tz``).
    time_tz : str
        IANA tz for ``date`` and ``time_start`` / ``time_end``. E.g.
        ``'America/Los_Angeles'``.
    time_start, time_end : str
        ``HH:MM`` window in ``time_tz``.
    dt_minutes : float
        Time-step granularity.
    output_path : str or Path, optional
        Save PNG instead of returning inline. ``None`` returns the Figure.

    Returns
    -------
    Figure or None
    """
    weights = Path(weights)
    beam_alt, beam_az, _, fwhm_ew, fwhm_ns = _load_beams_from_hdf5(weights)

    try:
        tz = ZoneInfo(time_tz)
    except KeyError as e:
        raise ValueError(f"unknown timezone {time_tz!r}") from e
    if date is None:
        d = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
        d = d.replace(tzinfo=None)
    else:
        d = datetime.strptime(date, "%Y-%m-%d")
    h0, m0 = map(int, time_start.split(":"))
    h1, m1 = map(int, time_end.split(":"))
    start_local = d.replace(hour=h0, minute=m0).replace(tzinfo=tz)
    end_local   = d.replace(hour=h1, minute=m1).replace(tzinfo=tz)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc   = end_local.astimezone(timezone.utc)

    sources_data = []
    for i, name in enumerate(sources):
        color = SOURCE_COLORS[i % len(SOURCE_COLORS)]
        times, alt, az = _compute_source_track(name, start_utc, end_utc,
                                               dt_minutes=dt_minutes)
        hits = _check_beam_hits(alt, az, beam_alt, beam_az,
                                fwhm_ew, fwhm_ns, times)
        sources_data.append({"name": name, "times": times, "alt": alt,
                             "az": az, "hits": hits, "color": color})

    title = (f"Source transit at OVRO — {d:%Y-%m-%d}  "
             f"({time_start}–{time_end} {time_tz})")
    fig = _plot_two_panel(sources_data, beam_alt, beam_az, fwhm_ew, fwhm_ns,
                          title, tz)

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return None
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize source transits across a SNAP weights beam grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("weights", help="SNAP int8 weights HDF5")
    parser.add_argument("--sources", nargs="+",
                        default=["sun", "cas-a", "cyg-a"])
    parser.add_argument("--date", default=None,
                        help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--time-tz", default="UTC",
                        help="IANA timezone (e.g. America/Los_Angeles)")
    parser.add_argument("--time-start", default="00:00")
    parser.add_argument("--time-end",   default="23:59")
    parser.add_argument("--dt", type=float, default=2.0,
                        help="Time step (minutes)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG path")
    args = parser.parse_args(argv)
    if args.output is None:
        date_label = args.date or datetime.now().strftime("%Y-%m-%d")
        args.output = f"source_transit_{date_label}.png"
    plot_source_transit(
        args.weights, sources=args.sources,
        date=args.date, time_tz=args.time_tz,
        time_start=args.time_start, time_end=args.time_end,
        dt_minutes=args.dt, output_path=args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
