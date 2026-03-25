#!/usr/bin/env python3
"""
gen_fwhm_beams.py  [TESTING ONLY]
==================================
Generate int8 SNAP beamformer weights with FWHM-spaced beams.

Two beam-selection modes (--mode):

  coverage  [default]
      Full FWHM grid, then DROP middle altitude levels to reach --n-beams.
      Sky coverage is maintained — every part of the sky above alt-min has
      at least one nearby beam. Beam count may be slightly under --n-beams
      if whole-level drops overshoot.

  exact
      Scale beam spacing UP from FWHM until the grid has exactly --n-beams.
      Beams above alt-min are sorted by altitude and the densest rings are
      pruned. WARNING: gaps in sky coverage are introduced — some sky
      directions will be further than one FWHM from any beam.

Example commands
----------------
# Mode: coverage — 512 beams, full sky coverage, with cal
python examples/gen_fwhm_beams.py \\
    --layout casm_antenna_layout_current.csv \\
    --output weights_coverage_512.h5 \\
    --n-beams 512 --mode coverage \\
    --cal svd_weights_mar10.npz --overwrite

# Mode: coverage — full FWHM grid (no trimming), geo-only
python examples/gen_fwhm_beams.py \\
    --layout casm_antenna_layout_current.csv \\
    --output weights_coverage_full.h5 \\
    --mode coverage --overwrite

# Mode: exact — exactly 512 beams (coverage gaps), with cal
python examples/gen_fwhm_beams.py \\
    --layout casm_antenna_layout_current.csv \\
    --output weights_exact_512.h5 \\
    --n-beams 512 --mode exact \\
    --cal svd_weights_mar10.npz --overwrite

# Mode: exact — exactly 128 beams, geo-only
python examples/gen_fwhm_beams.py \\
    --layout casm_antenna_layout_current.csv \\
    --output weights_exact_128.h5 \\
    --n-beams 128 --mode exact --overwrite
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bf_weights_generator import (
    Array64Config,
    FrequencyConfig,
    SnapWeightsGenerator,
    StationaryPointing,
    compute_beam_fwhm,
    generate_beam_grid_altaz,
    load_calibration_weights,
    save_int8_weights_hdf5,
)


# ---------------------------------------------------------------------------
# Beam grid builders
# ---------------------------------------------------------------------------

def _label_beams(beams):
    """Rename beams sequentially in-place."""
    for i, b in enumerate(beams):
        b.name = f"beam_{i:03d}"


def build_coverage_grid(positions_enu, n_beams=None, alt_min_deg=30.0):
    """FWHM grid trimmed to n_beams by dropping middle altitude levels.

    Generates the full one-FWHM-spaced grid first, then drops whole
    altitude rings starting from the most central ones until the count
    is at or below n_beams. Preserves sky coverage — every direction
    above alt_min has at least one nearby beam.

    Parameters
    ----------
    positions_enu : ndarray (n_ant, 3)
    n_beams : int or None
        Target count. None = return full FWHM grid.
    alt_min_deg : float

    Returns
    -------
    beams : list[StationaryPointing]
    fwhm_ew, fwhm_ns : float  (degrees)
    """
    fwhm_ew, fwhm_ns = compute_beam_fwhm(positions_enu)
    print(f"  FWHM: EW={fwhm_ew:.3f} deg, NS={fwhm_ns:.3f} deg")

    beams = generate_beam_grid_altaz(
        alt_min_deg=alt_min_deg,
        spacing_ew_deg=fwhm_ew,
        spacing_ns_deg=fwhm_ns,
    )
    print(f"  Full FWHM grid: {len(beams)} beams")

    if n_beams is not None and len(beams) > n_beams:
        n_drop = len(beams) - n_beams

        beams.sort(key=lambda p: (-p.alt_deg, p.az_deg))
        alt_levels = {}
        for b in beams:
            alt_levels.setdefault(round(b.alt_deg, 2), []).append(b)
        sorted_alts = sorted(alt_levels.keys())
        n_levels = len(sorted_alts)

        # Drop whole levels from the middle out; always keep lowest and highest.
        drop_priority = sorted(
            [(min(i, n_levels - 1 - i), alt)
             for i, alt in enumerate(sorted_alts)
             if i not in (0, n_levels - 1)],
            key=lambda x: -x[0],
        )
        dropped, drop_alts = 0, set()
        for _, alt in drop_priority:
            if dropped >= n_drop:
                break
            lc = len(alt_levels[alt])
            if dropped + lc <= n_drop:
                drop_alts.add(alt)
                dropped += lc

        beams = [b for b in beams if round(b.alt_deg, 2) not in drop_alts]

        # Fallback: if whole-level drops left us still over, trim from centre.
        if len(beams) > n_beams:
            beams.sort(key=lambda p: (-p.alt_deg, p.az_deg))
            mid = len(beams) // 2
            excess = len(beams) - n_beams
            beams = beams[: mid - excess // 2] + beams[mid - excess // 2 + excess :]

        beams.sort(key=lambda p: (-p.alt_deg, p.az_deg))
        print(f"  Dropped {n_drop} beams from middle levels → {len(beams)} beams remaining")

    _label_beams(beams)
    alts = [b.alt_deg for b in beams]
    print(f"  Final: {len(beams)} beams, alt {min(alts):.1f}–{max(alts):.1f} deg")
    return beams, fwhm_ew, fwhm_ns


def build_exact_grid(positions_enu, n_beams, alt_min_deg=30.0):
    """Exactly n_beams by scaling spacing above FWHM.

    Generates the full FWHM grid, then sorts by altitude (highest first)
    and keeps only the top n_beams. This effectively scales the spacing
    up to fit exactly n_beams in the sky above alt_min, but introduces
    gaps in coverage.

    WARNING: some sky directions will be more than one FWHM from the
    nearest beam. Use 'coverage' mode if full-sky sampling matters.

    Parameters
    ----------
    positions_enu : ndarray (n_ant, 3)
    n_beams : int
    alt_min_deg : float

    Returns
    -------
    beams : list[StationaryPointing]
    fwhm_ew, fwhm_ns : float  (degrees)
    """
    fwhm_ew, fwhm_ns = compute_beam_fwhm(positions_enu)
    print(f"  FWHM: EW={fwhm_ew:.3f} deg, NS={fwhm_ns:.3f} deg")

    all_beams = generate_beam_grid_altaz(
        alt_min_deg=alt_min_deg,
        spacing_ew_deg=fwhm_ew,
        spacing_ns_deg=fwhm_ns,
    )
    print(f"  Full FWHM grid: {len(all_beams)} beams")

    if len(all_beams) <= n_beams:
        warnings.warn(
            f"Full FWHM grid has only {len(all_beams)} beams — fewer than "
            f"requested {n_beams}. Returning all beams.",
            stacklevel=2,
        )
        beams = all_beams
    else:
        # Sort highest-altitude first (zenith-biased selection), then trim.
        all_beams.sort(key=lambda p: (-p.alt_deg, p.az_deg))
        beams = all_beams[:n_beams]
        beams.sort(key=lambda p: (-p.alt_deg, p.az_deg))
        warnings.warn(
            f"exact mode: selected {n_beams} of {len(all_beams)} FWHM beams. "
            f"Sky directions below alt={beams[-1].alt_deg:.1f} deg and some "
            f"azimuth ranges are not covered.",
            stacklevel=2,
        )
        print(f"  WARNING: kept highest-altitude {n_beams} beams — "
              f"{len(all_beams) - n_beams} low-altitude beams dropped (coverage gaps)")

    _label_beams(beams)
    alts = [b.alt_deg for b in beams]
    print(f"  Final: {len(beams)} beams, alt {min(alts):.1f}–{max(alts):.1f} deg")
    return beams, fwhm_ew, fwhm_ns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="[TEST] Generate int8 SNAP beamformer weights (FWHM spacing).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--layout", "-l", required=True,
                        help="Antenna layout CSV (Array64Config format)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output HDF5 path")
    parser.add_argument("--n-beams", "-n", type=int, default=None,
                        help="Target beam count. "
                             "coverage mode: trims to this count. "
                             "exact mode: required. "
                             "Omit to get full FWHM grid (coverage mode only).")
    parser.add_argument("--mode", default="coverage",
                        choices=["coverage", "exact"],
                        help="coverage: full FWHM grid, drop middle levels to reach n-beams "
                             "(default). "
                             "exact: keep highest-altitude n-beams only; warns about gaps.")
    parser.add_argument("--alt-min", type=float, default=30.0,
                        help="Minimum altitude in degrees (default: 30)")
    parser.add_argument("--cal", default=None,
                        help="SVD calibration weights NPZ (optional)")
    parser.add_argument("--overwrite", "-f", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    if args.mode == "exact" and args.n_beams is None:
        parser.error("--mode exact requires --n-beams")

    # Load array layout
    print(f"Layout: {args.layout}")
    array_config = Array64Config.from_csv(args.layout)
    print(f"  {array_config.n_active}/64 active antennas")

    # Build beam grid
    print(f"\nBuilding beam grid  mode={args.mode}  alt_min={args.alt_min} deg")
    if args.mode == "coverage":
        beams, fwhm_ew, fwhm_ns = build_coverage_grid(
            array_config.active_positions,
            n_beams=args.n_beams,
            alt_min_deg=args.alt_min,
        )
    else:
        beams, fwhm_ew, fwhm_ns = build_exact_grid(
            array_config.active_positions,
            n_beams=args.n_beams,
            alt_min_deg=args.alt_min,
        )

    # Load cal weights
    cal = None
    if args.cal:
        print(f"\nCal weights: {args.cal}")
        cal = load_calibration_weights(args.cal)
        print(f"  shape={cal.weights.shape}  good={cal.flags.sum()}/{len(cal.flags)}"
              f"  source={cal.source}  ref_ant={cal.ref_ant_id}")

    # Compute int8 weights
    print("\nComputing int8 weights...")
    gen = SnapWeightsGenerator(array_config, freq_config=FrequencyConfig())
    weights = gen.compute_int8_weights(pointings=beams, cal_weights=cal)
    print(f"  shape: {weights.weights_int8.shape}  dtype: {weights.weights_int8.dtype}")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_int8_weights_hdf5(weights, out, overwrite=args.overwrite)
    print(f"\nSaved: {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
