#!/usr/bin/env python3
"""
Generate int8 beamformer weights for CASM SNAP hardware.

Produces int8 weights combined with 3 different SVD calibration weight sets:
  - Feb 14 correlator SVD (pre_feb16 layout → current layout remapping)
  - Feb 19 correlator SVD (current layout)
  - Voltage self-cal SVD (current layout)

Beams are spaced at FWHM (default) or Nyquist (FWHM/2) intervals. When the
full grid exceeds the target count, beams are trimmed from mid-altitudes first
to preserve both low-altitude and zenith coverage.

Modes:
  --mode both      Generate weights AND plot (default)
  --mode weights   Generate weights only (no plot)
  --mode plot      Generate beam plot only (no weights)

Usage:
    python examples/generate_128beam_weights.py
    python examples/generate_128beam_weights.py -n 256 --spacing nyquist
    python examples/generate_128beam_weights.py --mode plot -n 128
    python examples/generate_128beam_weights.py --mode weights -d /tmp/weights
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bf_weights_generator import (
    Array64Config,
    compute_beam_fwhm,
    load_calibration_weights,
    save_int8_weights_hdf5,
)
from bf_weights_generator.weights import generate_beam_grid_altaz
from bf_weights_generator.snap_weights import SnapWeightsGenerator


# ── Default calibration weight paths ──────────────────────────────────────────
_CASM_VOLTAGE = Path("/home/casm/software/vishnu/casm-voltage-analysis")

DEFAULT_CAL_FEB14 = str(
    _CASM_VOLTAGE
    / "correlator_analysis/results/svd_beamform_no_rfi"
    / "svd_weights_sun_2026-02-14_phase-only_thr2.0_norfi.npz"
)
DEFAULT_CAL_FEB19 = str(
    _CASM_VOLTAGE
    / "correlator_analysis/results/multi_day_svd"
    / "svd_weights_feb19.npz"
)
DEFAULT_CAL_VOLTAGE = str(
    _CASM_VOLTAGE
    / "voltage_beamforming/results/selfcal_svd"
    / "svd_weights_voltage_selfcal.npz"
)


# ── Calibration configuration ────────────────────────────────────────────────
CAL_CONFIGS = [
    {
        "name": "feb14_svd",
        "cal_path_arg": "cal_feb14",
        "compute_layout_arg": "compute_layout_pre_feb16",
        "output_layout_arg": "output_layout",
        "output_filename": "int8_weights_{n}beam_feb14_svd.h5",
        "description": "Feb 14 correlator SVD",
    },
    {
        "name": "feb19_svd",
        "cal_path_arg": "cal_feb19",
        "compute_layout_arg": "output_layout",  # current layout
        "output_layout_arg": "output_layout",
        "output_filename": "int8_weights_{n}beam_feb19_svd.h5",
        "description": "Feb 19 correlator SVD",
    },
    {
        "name": "voltage_selfcal",
        "cal_path_arg": "cal_voltage",
        "compute_layout_arg": "output_layout",  # current layout
        "output_layout_arg": "output_layout",
        "output_filename": "int8_weights_{n}beam_voltage_selfcal.h5",
        "description": "Voltage self-cal SVD",
    },
]


def generate_beam_grid(
    positions_enu: np.ndarray,
    n_beams: int,
    spacing_mode: str = "fwhm",
    alt_min_deg: float = 30.0,
    alt_max_deg: float = 90.0,
    freq_hz: float = None,
) -> tuple:
    """
    Generate a beam grid trimmed to n_beams.

    Parameters
    ----------
    positions_enu : np.ndarray
        Antenna positions (n_ant, 3) in ENU coordinates.
    n_beams : int
        Target number of beams.
    spacing_mode : str
        "fwhm" for FWHM spacing, "nyquist" for FWHM/2 spacing.
    alt_min_deg, alt_max_deg : float
        Altitude range in degrees.
    freq_hz : float, optional
        Reference frequency for FWHM computation.

    Returns
    -------
    beams : list of StationaryPointing
    fwhm_ew, fwhm_ns : float
        Beam FWHM in degrees.
    spacing_ew, spacing_ns : float
        Actual grid spacing in degrees.
    """
    fwhm_ew, fwhm_ns = compute_beam_fwhm(positions_enu, freq_hz=freq_hz)

    if spacing_mode == "nyquist":
        spacing_ew = fwhm_ew / 2.0
        spacing_ns = fwhm_ns / 2.0
    else:
        spacing_ew = fwhm_ew
        spacing_ns = fwhm_ns

    label = "Nyquist (FWHM/2)" if spacing_mode == "nyquist" else "FWHM"

    # Generate full grid
    all_beams = generate_beam_grid_altaz(
        alt_min_deg=alt_min_deg,
        alt_max_deg=alt_max_deg,
        spacing_ew_deg=spacing_ew,
        spacing_ns_deg=spacing_ns,
    )

    print(f"  Full {label} grid: {len(all_beams)} beams "
          f"(alt {alt_min_deg:.0f}°–{alt_max_deg:.0f}°)")

    # Trim to n_beams if needed, keeping both low and high altitude beams
    if len(all_beams) > n_beams:
        n_drop = len(all_beams) - n_beams
        # Group beams by altitude level
        all_beams.sort(key=lambda p: (-p.alt_deg, p.az_deg))
        alt_levels = {}
        for b in all_beams:
            key = round(b.alt_deg, 2)
            alt_levels.setdefault(key, []).append(b)
        sorted_alts = sorted(alt_levels.keys())

        # Drop beams from middle altitude levels first
        # Always protect the lowest and highest levels (index 0 and -1)
        n_levels = len(sorted_alts)
        drop_priority = []
        for i, alt in enumerate(sorted_alts):
            if i == 0 or i == n_levels - 1:
                continue  # never drop lowest or highest altitude level
            dist_from_edge = min(i, n_levels - 1 - i)
            drop_priority.append((dist_from_edge, alt))
        # Sort: highest distance-from-edge first (= most middle)
        drop_priority.sort(key=lambda x: -x[0])

        dropped = 0
        drop_alts = set()
        for _, alt in drop_priority:
            if dropped >= n_drop:
                break
            level_count = len(alt_levels[alt])
            if dropped + level_count <= n_drop:
                drop_alts.add(alt)
                dropped += level_count

        beams = [b for b in all_beams if round(b.alt_deg, 2) not in drop_alts]

        # If we still have too many (couldn't drop whole levels evenly),
        # trim from the middle of what remains
        if len(beams) > n_beams:
            beams.sort(key=lambda p: (-p.alt_deg, p.az_deg))
            # Remove from the middle
            mid = len(beams) // 2
            excess = len(beams) - n_beams
            beams = beams[:mid - excess//2] + beams[mid - excess//2 + excess:]

        beams.sort(key=lambda p: (-p.alt_deg, p.az_deg))
        print(f"  Trimmed {n_drop} beams from mid-altitudes "
              f"(dropped {len(drop_alts)} altitude level(s))")
    else:
        beams = all_beams

    # Rename beams sequentially
    for i, b in enumerate(beams):
        b.name = f"beam_{i:03d}"

    alts = [b.alt_deg for b in beams]
    print(f"  Selected {len(beams)} beams "
          f"(alt {min(alts):.1f}°–{max(alts):.1f}°)")

    return beams, fwhm_ew, fwhm_ns, spacing_ew, spacing_ns


def generate_beam_plots(
    beams, fwhm_ew, fwhm_ns, spacing_ew, spacing_ns, spacing_mode, output_dir,
):
    """Generate both polar (zenith) and rectangular (Alt vs Az) beam plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not available, skipping plots", file=sys.stderr)
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from plot_elliptical_beams import (
        plot_elliptical_beams,
        plot_elliptical_beams_rectangular,
    )

    label = "Nyquist (FWHM/2)" if spacing_mode == "nyquist" else "FWHM"
    spacing_tag = "nyquist" if spacing_mode == "nyquist" else "fwhm"
    title = (
        f"{len(beams)} {label}-spaced beams\n"
        f"Spacing: {spacing_ew:.2f}° (E-W) × {spacing_ns:.2f}° (N-S)"
    )

    # Polar (zenith) plot
    polar_path = output_dir / f"beam_positions_{len(beams)}_{spacing_tag}_polar.png"
    plot_elliptical_beams(
        beams,
        fwhm_ew_deg=fwhm_ew,
        fwhm_ns_deg=fwhm_ns,
        title=title,
        output_path=str(polar_path),
    )

    # Rectangular (Alt vs Az) plot
    rect_path = output_dir / f"beam_positions_{len(beams)}_{spacing_tag}_altaz.png"
    plot_elliptical_beams_rectangular(
        beams,
        fwhm_ew_deg=fwhm_ew,
        fwhm_ns_deg=fwhm_ns,
        title=title,
        output_path=str(rect_path),
    )


def resolve_csv_path(path_str: str) -> str:
    """Resolve CSV path, trying CWD then project root."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    alt = Path(__file__).parent.parent / path_str
    if alt.exists():
        return str(alt)
    raise FileNotFoundError(f"CSV not found: {path_str}")


def do_weights(args, beams, layout_paths, cal_paths, out_dir):
    """Generate int8 weights for all calibration sets."""
    print(f"\n{'='*60}")
    print(f"Generating int8 weights for {len(CAL_CONFIGS)} calibration sets")
    print(f"{'='*60}")

    summary_rows = []

    for cfg in CAL_CONFIGS:
        print(f"\n--- {cfg['description']} ---")

        # Load cal weights
        cal_path = cal_paths[cfg["cal_path_arg"]]
        print(f"  Cal weights: {Path(cal_path).name}")
        cal = load_calibration_weights(cal_path)
        n_good = int(np.sum(cal.flags))
        print(f"  Good channels: {n_good}/{len(cal.flags)}")

        # Load compute and output layouts
        compute_csv = layout_paths[cfg["compute_layout_arg"]]
        output_csv = layout_paths[cfg["output_layout_arg"]]
        compute_array = Array64Config.from_csv(compute_csv)
        output_array_cfg = Array64Config.from_csv(output_csv)
        print(f"  Compute layout: {Path(compute_csv).name} "
              f"({compute_array.n_active} ants)")
        print(f"  Output layout:  {Path(output_csv).name} "
              f"({output_array_cfg.n_active} ants)")

        # Create generator and compute int8 weights
        gen = SnapWeightsGenerator(
            compute_array, output_array_config=output_array_cfg,
        )
        weights = gen.compute_int8_weights(
            beams, cal_weights=cal, geo_fallback=args.geo_fallback,
        )
        print(f"  Int8 shape: {weights.shape}")
        if args.geo_fallback:
            print(f"  Geo fallback: ON (flagged channels use geo-only weights)")

        # Post-processing: flip to descending freq if requested
        # compute_int8_weights outputs ascending (SNAP order: low→high)
        if args.freq_order == "descending":
            weights.weights_int8 = weights.weights_int8[:, ::-1, :, :, :].copy()
            weights.frequencies_hz = weights.frequencies_hz[::-1].copy()
            print(f"  Freq order: descending "
                  f"({weights.frequencies_hz[0]/1e6:.2f}→"
                  f"{weights.frequencies_hz[-1]/1e6:.2f} MHz)")
        else:
            print(f"  Freq order: ascending "
                  f"({weights.frequencies_hz[0]/1e6:.2f}→"
                  f"{weights.frequencies_hz[-1]/1e6:.2f} MHz)")

        # Save
        out_file = out_dir / cfg["output_filename"].format(n=len(beams))
        save_int8_weights_hdf5(weights, str(out_file), overwrite=args.overwrite)
        file_size_mb = out_file.stat().st_size / 1e6
        print(f"  Saved: {out_file} ({file_size_mb:.1f} MB)")

        summary_rows.append({
            "name": cfg["description"],
            "file": out_file.name,
            "good_ch": n_good,
            "size_mb": file_size_mb,
        })

    return summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate int8 beamformer weights for CASM SNAP hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", "-m", choices=["both", "weights", "plot"], default="both",
        help="Operation mode: 'both' (weights + plot), 'weights' (no plot), "
             "'plot' (no weights)",
    )
    parser.add_argument(
        "--output-dir", "-d", default="./weights_512beam",
        help="Output directory",
    )
    parser.add_argument(
        "--n-beams", "-n", type=int, default=512,
        help="Target beam count",
    )
    parser.add_argument(
        "--spacing", "-s", choices=["fwhm", "nyquist"], default="fwhm",
        help="Beam spacing: 'fwhm' (1× FWHM) or 'nyquist' (FWHM/2)",
    )
    parser.add_argument(
        "--alt-min", type=float, default=30.0,
        help="Minimum altitude (deg)",
    )
    parser.add_argument(
        "--alt-max", type=float, default=90.0,
        help="Maximum altitude (deg)",
    )
    parser.add_argument(
        "--freq-order", choices=["descending", "ascending"], default="descending",
        help="Frequency order in output files",
    )
    parser.add_argument(
        "--cal-feb14", default=DEFAULT_CAL_FEB14,
        help="Feb 14 SVD calibration weights (.npz)",
    )
    parser.add_argument(
        "--cal-feb19", default=DEFAULT_CAL_FEB19,
        help="Feb 19 SVD calibration weights (.npz)",
    )
    parser.add_argument(
        "--cal-voltage", default=DEFAULT_CAL_VOLTAGE,
        help="Voltage self-cal SVD weights (.npz)",
    )
    parser.add_argument(
        "--compute-layout-pre-feb16",
        default="casm_antenna_layout_pre_feb16.csv",
        help="Pre-Feb16 antenna layout CSV (for Feb 14 cal geo computation)",
    )
    parser.add_argument(
        "--output-layout",
        default="casm_antenna_layout_current.csv",
        help="Current antenna layout CSV (SNAP ordering + FWHM computation)",
    )
    parser.add_argument(
        "--overwrite", "-f", action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--geo-fallback", action="store_true",
        help="Use geo-only weights on flagged channels instead of zeroing them",
    )
    args = parser.parse_args()

    # ── Resolve layout paths ──────────────────────────────────────────────
    output_layout_path = resolve_csv_path(args.output_layout)
    pre_feb16_layout_path = resolve_csv_path(args.compute_layout_pre_feb16)

    layout_paths = {
        "output_layout": output_layout_path,
        "compute_layout_pre_feb16": pre_feb16_layout_path,
    }

    cal_paths = {
        "cal_feb14": args.cal_feb14,
        "cal_feb19": args.cal_feb19,
        "cal_voltage": args.cal_voltage,
    }

    # ── Load output layout and compute FWHM ───────────────────────────────
    output_array = Array64Config.from_csv(output_layout_path)
    print(f"Output layout: {output_array.n_active} active antennas "
          f"({Path(output_layout_path).name})")

    positions = output_array.active_positions
    spacing_label = "Nyquist (FWHM/2)" if args.spacing == "nyquist" else "FWHM"

    # ── Generate beam grid ────────────────────────────────────────────────
    print(f"\nGenerating {args.n_beams} {spacing_label}-spaced beams...")
    beams, fwhm_ew, fwhm_ns, spacing_ew, spacing_ns = generate_beam_grid(
        positions_enu=positions,
        n_beams=args.n_beams,
        spacing_mode=args.spacing,
        alt_min_deg=args.alt_min,
        alt_max_deg=args.alt_max,
    )
    print(f"  FWHM:    {fwhm_ew:.2f}° (E-W) × {fwhm_ns:.2f}° (N-S)")
    print(f"  Spacing: {spacing_ew:.2f}° (E-W) × {spacing_ns:.2f}° (N-S) "
          f"[{spacing_label}]")

    # ── Create output directory ───────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate weights ──────────────────────────────────────────────────
    summary_rows = []
    if args.mode in ("both", "weights"):
        summary_rows = do_weights(args, beams, layout_paths, cal_paths, out_dir)

    # ── Generate plots ────────────────────────────────────────────────────
    if args.mode in ("both", "plot"):
        print(f"\nGenerating beam plots...")
        generate_beam_plots(
            beams, fwhm_ew, fwhm_ns, spacing_ew, spacing_ns,
            args.spacing, out_dir,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    alts = [b.alt_deg for b in beams]
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Mode:            {args.mode}")
    print(f"  Beams:           {len(beams)}")
    print(f"  FWHM:            {fwhm_ew:.2f}° (E-W) × {fwhm_ns:.2f}° (N-S)")
    print(f"  Spacing:         {spacing_ew:.2f}° (E-W) × {spacing_ns:.2f}° (N-S) "
          f"[{spacing_label}]")
    print(f"  Alt range:       {min(alts):.1f}° – {max(alts):.1f}°")
    if args.mode in ("both", "weights"):
        print(f"  Freq order:      {args.freq_order}")
        print(f"  Geo fallback:    {'ON' if args.geo_fallback else 'OFF'}")
    print(f"  Output dir:      {out_dir.resolve()}")

    if summary_rows:
        print()
        print(f"  {'Cal Set':<25s} {'Good Ch':>8s} {'Size':>8s}  File")
        print(f"  {'-'*25} {'-'*8} {'-'*8}  {'-'*40}")
        for row in summary_rows:
            print(f"  {row['name']:<25s} {row['good_ch']:>4d}/3072 "
                  f"{row['size_mb']:>6.1f}MB  {row['file']}")


if __name__ == "__main__":
    main()
