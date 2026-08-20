#!/usr/bin/env python
"""Canonical CASM calibration + beamforming-weights recipe, in one driver.

This module ORCHESTRATES existing library functions; it contains no new
signal-processing algorithm. Every numerical code path is copied verbatim
from the runs that produced the deployed products:

  * cal solve + solar-track grid + int8 save + self-checks
    /mnt/nvme5/solar0819/newcal_build/make_cal0819_weights.py
  * static build (``build_manual_static``) + static-subtracted solve
    /mnt/nvme5/solar0819/cal_static_night/run_static_night_aug20.py
    (itself a verbatim copy of /home/casm/scratch/solar_weights_20260815/
    run_drill_aug15.py, the historical canonical recipe)
  * all-sky grid variant + nearest-beam checks
    /mnt/nvme5/solar0819/newcal_build/make_aug20_grid2090_weights.py
  * rank-1-vs-frequency plot template
    /mnt/nvme5/solar0819/svd_census/make_plots_aug19.py
  * per-antenna phase/delay diff vs a previous cal
    /mnt/nvme5/solar0819/cal_diff/run_cal_diff.py
  * pointing verification by cal division
    /mnt/nvme5/solar0819/pointing_verify/verify_pointing.py

Bit-exact reproduction of all three of those products through this driver is
checked in docs/canonical-recipe.md.

FIXED policy (deliberately not knobs):
  FrequencyConfig.layout_64ant(); full-band per-channel solve with
  SVDConfig(threshold=1.0, PHASE_ONLY, block_size=1, masked_band_strategy=
  "zero"); fringe_stop sign=-1; scale_factor=127.0; ``[..., :64]`` slice;
  freq_order="descending"; cal saved through casm_calibrator.save_calibration.

TUNABLE: the calibrator source, the time windows, the antenna set, the
reference antenna, the layout, the grid bounds and spacing search, and the
output naming.

Nothing here ever uploads. The deploy command is printed for the operator.

Invocation (no console entry point: pyproject.toml carries unrelated
uncommitted edits, so it was left untouched):

    /home/casm/software/dev/casm_venvs/casm_offline_env/bin/python \\
      -m bf_weights_generator.make_cal_and_weights --help

Usage from Python:

    from bf_weights_generator.make_cal_and_weights import RecipeParams, run
    run(RecipeParams(tag="20260820", out_dir="/mnt/nvme5/...", ...))
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import sys
from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# FIXED policy constants (see module docstring)
# ---------------------------------------------------------------------------

FMT_NAME = "layout_64ant"
FRINGE_SIGN = -1
SVD_THRESHOLD = 1.0
SVD_BLOCK_SIZE = 1
SVD_MASKED_BAND_STRATEGY = "zero"
SCALE_FACTOR = 127.0
FREQ_ORDER = "descending"
N_CB_SLOTS = 64
FWHM_FREQ_HZ = 450e6

# Diagnostic band conventions (svd_census/make_plots_aug19.py, cal_diff).
SUNBAND = (435.0, 460.0)
RFI_LINES = [(462.0, 464.0), (450.0, 452.0), (436.5, 438.5), (400.0, 402.0)]
FITBAND = (398.0, 480.0)
TAUS_NS = np.arange(-300.0, 300.0001, 0.02)

DEFAULT_LAYOUT = ("/home/casm/software/dev/antenna_layouts/"
                  "casm_antenna_layout_2026-08-07.csv")
VENV_PY = "/home/casm/software/dev/casm_venvs/casm_offline_env/bin/python"
DEPLOY_PY = ("/home/casm/software/dev/bf_weights_generator/"
             "bf_weights_generator/deploy_bf_weights.py")
TRANSIT_BIN = ("/home/casm/software/dev/casm_venvs/casm_offline_env/bin/"
               "casm-bf-source-transit")
VERIFY_POINTING = "/mnt/nvme5/solar0819/pointing_verify/verify_pointing.py"


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class RecipeParams:
    """Every knob of the recipe. Anything not here is fixed policy."""

    # ---- outputs -----------------------------------------------------------
    out_dir: str = "."
    tag: str = "run"

    # ---- cal solve ---------------------------------------------------------
    cal_source: str = "sun"
    """Source name understood by casm_vis_analysis.sources: sun, cyg-a,
    cas-a, tau-a, vir-a, b0329+54 (dashes/plus/space are normalised to
    underscores by the catalog). For anything not in the catalog set
    ``cal_ra``/``cal_dec`` as well and this name becomes the label."""

    cal_ra: Optional[str] = None
    """ICRS RA for a source not in the catalog, e.g. "19h59m28.36s" or
    "299.868deg". Requires ``cal_dec``. Registered into the catalog under
    ``cal_source`` for the duration of the run (fringe_stop only accepts
    catalog names)."""

    cal_dec: Optional[str] = None

    source_window: Optional[Tuple[str, str]] = None
    """(UTC start, UTC end) of the on-source data to solve on."""

    static_window: Optional[Tuple[str, str]] = None
    """(UTC start, UTC end) of an off-source window to average into a static
    visibility and subtract before solving. None = no static subtraction."""

    static_path: Optional[str] = None
    """Where the static npz lives. If it exists it is loaded, else built from
    ``static_window`` (build_manual_static semantics). Default:
    <out_dir>/static_<tag>.npz."""

    static_notes: str = ""

    antennas: Optional[Sequence[int]] = None
    """Antenna IDs to solve/beamform with. None = every antenna active in the
    layout CSV."""

    ref_ant: int = 9
    layout_csv: str = DEFAULT_LAYOUT
    min_alt_deg: float = 10.0

    cal_path: Optional[str] = None
    """Reuse an existing cal h5 instead of solving. Skips the whole read/solve
    step, so grid-only rebuilds cost seconds."""

    # ---- grid --------------------------------------------------------------
    make_weights: bool = True
    grid_mode: str = "bounds"          # "bounds" | "track"
    alt_min_deg: float = 20.0
    alt_max_deg: float = 90.0
    az_min_deg: float = 0.0
    az_max_deg: float = 360.0
    n_beams: int = 512
    mult_min: float = 0.05
    mult_max: float = 0.60
    mult_n: int = 56
    """Spacing multiplier search: spacing = mult * FWHM per axis, over
    np.linspace(mult_min, mult_max, mult_n). The grid kept is the one with the
    FEWEST beams that still reaches n_beams."""

    mult_break_below_target: bool = False
    """run_drill_aug15.py stopped the search at the first multiplier that
    undershot n_beams. The Aug-19/Aug-20 builds did not. Only affects results
    if the beam count is non-monotonic in mult."""

    trim: str = "lowest"               # "lowest" | "highest" altitudes kept
    track_sources: Sequence[str] = field(default_factory=list)
    track_date: Optional[str] = None
    track_min_alt_deg: float = 0.0
    track_pad_deg: float = 0.0
    """grid_mode="track": alt/az bounds are the bounding box of the named
    source tracks on track_date (sampled each minute, above track_min_alt_deg),
    padded by track_pad_deg. Beams are then laid on that box by the same
    bounds-mode search. Placing beams ALONG a track (rather than on its
    bounding box) has no helper in bf_weights_generator today: TODO."""

    # ---- verification / diagnostics ---------------------------------------
    verify_beams: Sequence[int] = (0, 256, 511)
    """Beams put through the cal-division pointing fit. Out-of-range entries
    are clipped to the grid."""

    nearest_targets: Sequence[Tuple[str, float, float]] = ()
    """(name, alt_deg, az_deg) rows for the nearest-beam table."""

    nearest_sources: Sequence[str] = ()
    """Named sources; their transit alt/az on ``nearest_date`` are appended to
    the nearest-beam table."""

    nearest_date: Optional[str] = None
    prev_cal_path: Optional[str] = None
    """Previous cal h5 for the per-antenna phase/delay diff figure."""

    diagnostics: bool = True
    notebook: bool = True
    execute_notebook: bool = True
    fringe_plot_max_baselines: int = 20

    def resolved(self) -> "RecipeParams":
        """Fill in derived defaults."""
        if self.static_path is None and self.static_window is not None:
            self.static_path = os.path.join(self.out_dir,
                                            f"static_{self.tag}.npz")
        return self


# ---------------------------------------------------------------------------
# Source handling
# ---------------------------------------------------------------------------

def resolve_source(params: RecipeParams) -> str:
    """Return the source key to hand to fringe_stop.

    fringe_stop resolves names through casm_vis_analysis.sources.CATALOG
    (plus the special-cased "sun"), normalising '-', '+' and ' ' to '_'.
    An explicit RA/Dec is registered into that catalog under the requested
    name so the rest of the pipeline is unchanged.
    """
    from casm_vis_analysis import sources as _src

    name = params.cal_source
    if params.cal_ra is None and params.cal_dec is None:
        if name.lower() != "sun":
            key = name.lower().replace("-", "_").replace(" ", "_").replace("+", "_")
            if key not in _src.CATALOG:
                raise ValueError(
                    f"Unknown source {name!r}. Catalog: sun, "
                    f"{', '.join(sorted(_src.CATALOG))}. For anything else "
                    f"pass cal_ra and cal_dec.")
        return name
    if params.cal_ra is None or params.cal_dec is None:
        raise ValueError("cal_ra and cal_dec must be given together.")
    from astropy.coordinates import SkyCoord
    key = name.lower().replace("-", "_").replace(" ", "_").replace("+", "_")
    _src.CATALOG[key] = SkyCoord(params.cal_ra, params.cal_dec, frame="icrs")
    print(f"  registered {name} -> {key} at RA {params.cal_ra} "
          f"Dec {params.cal_dec}", flush=True)
    return name


def source_transit_altaz(name: str, date_utc: str,
                         step_minutes: float = 1.0) -> Tuple[float, float]:
    """Alt/az of a source at its highest point on a UTC date."""
    from astropy.time import Time
    from casm_vis_analysis.sources import source_altaz
    t0 = Time(f"{date_utc} 00:00:00", scale="utc").unix
    grid = t0 + np.arange(0.0, 86400.0, step_minutes * 60.0)
    alt, az = source_altaz(name, grid)
    k = int(np.argmax(alt))
    return float(alt[k]), float(az[k])


# ---------------------------------------------------------------------------
# Step 1: calibration
# ---------------------------------------------------------------------------

def build_manual_static(path, t0, t1, note, fmt):
    """Verbatim copy of run_drill_aug15.py::build_manual_static, UTC times."""
    from casm_io.correlator import read_visibilities
    from casm_vis_analysis.offsource import (average_visibility,
                                             save_static_visibility,
                                             load_static_visibility)
    if os.path.exists(path):
        print(f"  reusing existing static {path}", flush=True)
        return load_static_visibility(path)
    dn = read_visibilities(t0, t1, time_tz="UTC", data_root="/mnt", fmt=fmt,
                           verbose=False)
    t = np.asarray(dn["time_unix"])
    sv = average_visibility(dn, time_range_unix=(float(t[0]), float(t[-1])),
                            apply_freq_mask=True).astype(np.complex64)
    save_static_visibility(path, sv, freq_mhz=np.asarray(dn["freq_mhz"]),
                           window_unix=(float(t[0]), float(t[-1])),
                           altitudes=None, notes=note)
    print(f"  built {path} ({(t[-1]-t[0])/60:.0f} min, {len(t)} integrations)",
          flush=True)
    del dn
    gc.collect()
    return load_static_visibility(path)


def solve_calibration(params: RecipeParams, mapping, ant, source: str,
                      static_vis=None, label="cal"):
    """One solve: read -> (optional static subtract) -> fringe_stop -> SVD.

    Code path identical to make_cal0819_weights.py step 1 and to
    run_static_night_aug20.py::run_case.
    """
    from casm_io.correlator import read_visibilities, load_format
    from casm_vis_analysis.fringe_stop import fringe_stop
    from casm_vis_analysis.offsource import subtract_static_visibility
    from casm_calibrator import svd_calibrate, SVDConfig
    from casm_calibrator.svd import SVDMode

    cfg = SVDConfig(threshold=SVD_THRESHOLD, svd_mode=SVDMode.PHASE_ONLY,
                    block_size=SVD_BLOCK_SIZE,
                    masked_band_strategy=SVD_MASKED_BAND_STRATEGY)
    t0, t1 = params.source_window
    print(f"[cal:{label}] solving {t0} -> {t1} UTC, "
          f"static={'yes' if static_vis is not None else 'no'}", flush=True)
    d = read_visibilities(t0, t1, time_tz="UTC", data_root="/mnt",
                          fmt=load_format(FMT_NAME), verbose=False)
    nt = np.asarray(d["vis"]).shape[0]
    dd = d if static_vis is None else subtract_static_visibility(d, static_vis)
    fs = fringe_stop(dd, ant, ref_ant=params.ref_ant, source=source,
                     sign=FRINGE_SIGN, min_alt_deg=params.min_alt_deg)
    cal = svd_calibrate(fs, ant, data=dd, config=cfg)
    r1 = np.asarray(cal["rank1_ratios"])
    print(f"  nt={nt} solved {int(np.asarray(cal['flags']).sum())}/"
          f"{len(np.asarray(cal['freqs_mhz']))} rank1 median "
          f"{np.nanmedian(r1):.2f}", flush=True)
    return cal, nt, fs


def band_mask(freq, band=SUNBAND, lines=RFI_LINES):
    """In-band, RFI lines removed (svd_census/run_static_night convention)."""
    b = (freq > band[0]) & (freq < band[1])
    for lo, hi in lines:
        b &= ~((freq > lo) & (freq < hi))
    return b


# ---------------------------------------------------------------------------
# Step 2: beam grid + weights
# ---------------------------------------------------------------------------

def track_bounds(params: RecipeParams):
    """Alt/az bounding box of the named source tracks on ``track_date``."""
    from astropy.time import Time
    from casm_vis_analysis.sources import source_altaz
    if not params.track_sources or params.track_date is None:
        raise ValueError("grid_mode='track' needs track_sources and track_date")
    t0 = Time(f"{params.track_date} 00:00:00", scale="utc").unix
    grid = t0 + np.arange(0.0, 86400.0, 60.0)
    alts, azs = [], []
    for name in params.track_sources:
        a, z = source_altaz(name, grid)
        keep = a >= params.track_min_alt_deg
        if not keep.any():
            print(f"  WARNING: {name} never rises above "
                  f"{params.track_min_alt_deg} deg on {params.track_date}",
                  flush=True)
            continue
        alts.append(np.asarray(a)[keep])
        azs.append(np.asarray(z)[keep])
    if not alts:
        raise ValueError("no track samples above track_min_alt_deg")
    alt = np.concatenate(alts)
    az = np.concatenate(azs)
    # Unwrap azimuth about the circular mean so a track crossing 0 deg does
    # not produce a 0-360 box.
    ctr = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(az)).mean())) % 360.0
    azu = ctr + (((az - ctr) + 180.0) % 360.0 - 180.0)
    p = params.track_pad_deg
    lo, hi = float(alt.min()) - p, float(alt.max()) + p
    az_lo, az_hi = float(azu.min()) - p, float(azu.max()) + p
    if az_hi - az_lo > 350.0:
        # A box that wide is the whole sky; use the full-azimuth branch of
        # generate_beam_grid_altaz rather than a 359-deg linspace.
        az_lo, az_hi = 0.0, 360.0
    bounds = (max(0.0, lo), min(90.0, hi), az_lo, az_hi)
    print(f"  track box from {list(params.track_sources)} on "
          f"{params.track_date}: alt {bounds[0]:.2f}-{bounds[1]:.2f}, "
          f"az {bounds[2]:.2f}-{bounds[3]:.2f}", flush=True)
    return bounds


def build_grid(params: RecipeParams, arr):
    """Spacing search + trim to exactly n_beams.

    Verbatim from make_cal0819_weights.py step 2 / make_aug20_grid2090_
    weights.py: densest grid (fewest beams) that still reaches n_beams, then
    np.argsort on altitude and keep the first n_beams indices in file order.
    """
    from bf_weights_generator import generate_beam_grid_altaz
    from bf_weights_generator.config import compute_beam_fwhm

    if params.grid_mode == "track":
        alt_min, alt_max, az_min, az_max = track_bounds(params)
    elif params.grid_mode == "bounds":
        alt_min, alt_max = params.alt_min_deg, params.alt_max_deg
        az_min, az_max = params.az_min_deg, params.az_max_deg
    else:
        raise ValueError(f"grid_mode must be 'bounds' or 'track', "
                         f"got {params.grid_mode!r}")

    fw_ew, fw_ns = compute_beam_fwhm(arr.active_positions, freq_hz=FWHM_FREQ_HZ)
    print(f"  {arr.n_active}-ant FWHM @450 MHz: {fw_ew:.2f} x {fw_ns:.2f} deg",
          flush=True)
    best = None
    for mult in np.linspace(params.mult_min, params.mult_max, params.mult_n):
        g = generate_beam_grid_altaz(alt_min_deg=alt_min, alt_max_deg=alt_max,
                                     az_min_deg=az_min, az_max_deg=az_max,
                                     spacing_ew_deg=mult * fw_ew,
                                     spacing_ns_deg=mult * fw_ns)
        if len(g) >= params.n_beams and (best is None or len(g) < len(best[1])):
            best = (mult, g)
        if params.mult_break_below_target and len(g) < params.n_beams:
            break
    if best is None:
        raise SystemExit(
            f"spacing search found no grid with >= {params.n_beams} beams over "
            f"mult {params.mult_min}-{params.mult_max}; widen the search")
    mult, grid = best
    n_pre = len(grid)
    alts_pre = [b.alt_deg for b in grid]
    if params.trim == "lowest":
        order = np.argsort(alts_pre)
    elif params.trim == "highest":
        order = np.argsort(alts_pre)[::-1]
    else:
        raise ValueError(f"trim must be 'lowest' or 'highest', "
                         f"got {params.trim!r}")
    grid = [grid[i] for i in sorted(order[:params.n_beams])]
    alts = np.array([b.alt_deg for b in grid])
    azs = np.array([b.az_deg for b in grid])
    print(f"  mult={mult:.3f} ({n_pre} beams pre-trim) -> {len(grid)} beams, "
          f"alt {alts.min():.2f}-{alts.max():.2f}, "
          f"az {azs.min():.1f}-{azs.max():.1f}, "
          f"spacing {mult*fw_ew:.2f} x {mult*fw_ns:.2f} deg", flush=True)
    meta = dict(mult=float(mult), n_pre_trim=int(n_pre),
                fwhm_ew_deg=float(fw_ew), fwhm_ns_deg=float(fw_ns),
                spacing_ew_deg=float(mult * fw_ew),
                spacing_ns_deg=float(mult * fw_ns),
                alt_min=float(alts.min()), alt_max=float(alts.max()),
                az_min=float(azs.min()), az_max=float(azs.max()),
                bounds=[float(alt_min), float(alt_max),
                        float(az_min), float(az_max)])
    return grid, meta


def build_weights(params: RecipeParams, arr, grid, cal_file, out_path):
    """Combine + int8 + save. Verbatim from make_cal0819_weights.py step 2."""
    from bf_weights_generator import (generate_combined_weights,
                                      load_calibration_weights,
                                      save_int8_weights_hdf5)
    from bf_weights_generator.config import FrequencyConfig
    freq_cfg = FrequencyConfig.layout_64ant()
    comb = generate_combined_weights(grid, arr,
                                     cal_weights=load_calibration_weights(cal_file),
                                     freq_config=freq_cfg,
                                     freq_order=FREQ_ORDER)
    w8 = comb.to_int8(scale_factor=SCALE_FACTOR)
    w8.weights_int8 = w8.weights_int8[..., :N_CB_SLOTS]
    if os.path.exists(out_path):
        os.remove(out_path)
    save_int8_weights_hdf5(w8, out_path)
    print(f"  wrote {out_path}", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Step 3: verification
# ---------------------------------------------------------------------------

def sep_deg(alt, az, alt0, az0):
    """Angular separation, make_aug20_grid2090_weights.py."""
    a, z = np.deg2rad(alt), np.deg2rad(az)
    a0, z0 = np.deg2rad(alt0), np.deg2rad(az0)
    c = np.sin(a) * np.sin(a0) + np.cos(a) * np.cos(a0) * np.cos(z - z0)
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def _load_verify_pointing():
    """Import verify_pointing.py from its canonical location."""
    import importlib.util
    if not os.path.exists(VERIFY_POINTING):
        return None
    spec = importlib.util.spec_from_file_location("_verify_pointing",
                                                  VERIFY_POINTING)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # pragma: no cover
        print(f"  verify_pointing import failed ({type(exc).__name__}: {exc})",
              flush=True)
        return None
    return mod


def verify(params: RecipeParams, weights_file, cal_file, grid_meta, report):
    """Pointings stats, int8 sanity, cal-division coherence, nearest beams."""
    import h5py
    from bf_weights_generator.snap_weights import load_calibration_weights

    print("\n[verify]", flush=True)
    with h5py.File(weights_file) as f:
        W = f["weights_int8"]
        shape = tuple(int(s) for s in W.shape)
        alt = f["pointings/alt_deg"][...]
        az = f["pointings/az_deg"][...]
        sl = W[:, 1500].astype(np.int32)
        attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                 for k, v in f.attrs.items()}
    print(f"  weights_int8 shape {shape}", flush=True)
    print(f"  pointings: n={len(alt)} alt {alt.min():.2f}-{alt.max():.2f} "
          f"(median {np.median(alt):.2f}) az {az.min():.2f}-{az.max():.2f}",
          flush=True)
    print(f"  int8 @ch1500: min={int(sl.min())} max={int(sl.max())} "
          f"mean|v|={np.abs(sl).mean():.2f} (must be NONZERO)", flush=True)
    if np.abs(sl).max() == 0:
        raise SystemExit("FAIL: all-zero int8 payload at ch1500")
    report["pointings"] = dict(n=int(len(alt)), alt_min=float(alt.min()),
                               alt_max=float(alt.max()),
                               alt_median=float(np.median(alt)),
                               az_min=float(az.min()), az_max=float(az.max()))
    report["int8"] = dict(shape=list(shape), ch1500_min=int(sl.min()),
                          ch1500_max=int(sl.max()),
                          ch1500_mean_abs=float(np.abs(sl).mean()))
    report["weights_attrs"] = {k: str(v) for k, v in attrs.items()}

    # ---- cal-division pointing coherence -------------------------------
    vp = _load_verify_pointing()
    rows = []
    if vp is None:
        print("  pointing fit SKIPPED: verify_pointing.py not readable",
              flush=True)
    else:
        meta = vp.load_file(weights_file)
        calw = load_calibration_weights(cal_file)
        beams = sorted({int(np.clip(b, 0, len(alt) - 1))
                        for b in params.verify_beams})
        print(f"  cal-division pointing fit on beams {beams}", flush=True)
        print(f"  {'beam':>5} {'stated_alt':>10} {'stated_az':>10} "
              f"{'fit_alt':>9} {'fit_az':>9} {'coh_fit':>8} {'coh_stated':>10} "
              f"{'dalt':>7} {'daz':>8}", flush=True)
        for b in beams:
            r = vp.fit_beam(weights_file, b, meta, calw)
            daz = (r["fit_az"] - r["stated_az"] + 180) % 360 - 180
            r["dalt"] = r["fit_alt"] - r["stated_alt"]
            r["daz"] = daz
            rows.append(r)
            print(f"  {b:5d} {r['stated_alt']:10.3f} {r['stated_az']:10.3f} "
                  f"{r['fit_alt']:9.3f} {r['fit_az']:9.3f} {r['coh_fit']:8.4f} "
                  f"{r['coh_stated']:10.4f} {r['dalt']:7.3f} {daz:8.3f}",
                  flush=True)
    report["pointing_fit"] = [{k: (float(v) if isinstance(v, (int, float,
                                                              np.floating))
                                   else v)
                               for k, v in r.items()} for r in rows]

    # ---- nearest-beam table --------------------------------------------
    targets = [(str(n), float(a), float(z)) for n, a, z in params.nearest_targets]
    for name in params.nearest_sources:
        if params.nearest_date is None:
            raise ValueError("nearest_sources needs nearest_date")
        a, z = source_transit_altaz(name, params.nearest_date)
        targets.append((f"{name} transit {params.nearest_date}", a, z))
    near = []
    if targets:
        print("  nearest beam to each target:", flush=True)
    for name, a0, z0 in targets:
        dd = sep_deg(alt, az, a0, z0)
        k = int(np.argmin(dd))
        near.append(dict(target=name, target_alt=a0, target_az=z0, beam=k,
                         beam_alt=float(alt[k]), beam_az=float(az[k]),
                         sep_deg=float(dd[k])))
        print(f"    {name}: alt={a0:.2f} az={z0:.2f} -> beam {k} "
              f"alt={alt[k]:.2f} az={az[k]:.2f} sep={dd[k]:.2f} deg",
              flush=True)
    report["nearest_beams"] = near
    report["grid"] = grid_meta
    return report


def print_operator_commands(params: RecipeParams, weights_file):
    """Dry-run / deploy / transit commands. Nothing is executed here."""
    dry = os.path.join(params.out_dir, f"dryrun_{params.tag}")
    srcs = " ".join(params.nearest_sources) or params.cal_source
    print(f"""
NEXT STEPS (operator; this driver never uploads)
  1) dry-run inspect (no network side effects):
     {VENV_PY} \\
       {DEPLOY_PY} \\
       {weights_file} -o {dry}
  2) deploy (uploads; a human action, per casm-wiki weights-and-deploy.md):
     {VENV_PY} \\
       {DEPLOY_PY} \\
       {weights_file} --upload --save-defaults -o {dry}
     Omit --save-defaults only if you want the next restart to revert.
  3) beam schedule for the day:
     {TRANSIT_BIN} \\
       {weights_file} --sources {srcs} \\
       --date {params.nearest_date or params.track_date or 'YYYY-MM-DD'} \\
       --time-tz America/Los_Angeles
""", flush=True)


# ---------------------------------------------------------------------------
# Step 4: diagnostic figures
# ---------------------------------------------------------------------------

def _roll(r):
    import pandas as pd
    return pd.Series(r).rolling(32, center=True, min_periods=8).median()


def plot_rank1(cases, out_png, title):
    """Rank-1 ratio vs frequency, svd_census/make_plots_aug19.py template.

    cases: list of (freq, rank1, label, colour); the first is primary.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fp, rp = np.asarray(cases[0][0]), np.asarray(cases[0][1])
    med_p = float(np.nanmedian(rp))
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.plot(fp, rp, lw=0.4, alpha=0.55, color=cases[0][3],
            label="primary per-channel")
    meds = {}
    for f, r, lab, col in cases:
        f, r = np.asarray(f), np.asarray(r)
        m = float(np.nanmedian(r))
        sb = float(np.nanmedian(r[band_mask(f)]))
        meds[lab] = (m, sb)
        ax.plot(f, _roll(r), lw=1.8 if lab == cases[0][2] else 1.3, color=col,
                label=f"{lab} rolling median ({m:.2f}, in-band {sb:.2f})")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.axhline(med_p, color="tab:blue", ls=":", lw=1.2,
               label=f"primary median {med_p:.2f}")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("rank-1 ratio")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    return meds


def plot_fringe_stopped(fs, mapping, out_dir, max_baselines):
    """Fringe-stopped visibility waterfalls via the casm_vis_analysis helper."""
    from casm_vis_analysis.plotting.fringe_diag import plot_fringe_diagnostic
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target_aids = list(fs["target_aids"])
    target_labels = list(fs["target_labels"])
    target_snaps = [mapping.snap_adc(a)[0] for a in target_aids]
    ref_snap = mapping.snap_adc(int(fs["ref_ant"]))[0]
    panels = [("Raw phase", np.asarray(fs["vis"])),
              ("Geometric", np.asarray(fs["geometric_phase"])),
              ("Fringe-stopped", np.asarray(fs["vis_stopped"]))]
    figs = plot_fringe_diagnostic(panels, np.asarray(fs["time_unix"]),
                                  np.asarray(fs["freq_mhz"]), target_labels,
                                  target_snaps, ref_snap, output_dir=out_dir,
                                  split_max=max_baselines,
                                  freq_mask=fs.get("freq_mask"))
    for fig in figs:
        plt.close(fig)
    pngs = sorted(os.path.join(out_dir, p) for p in os.listdir(out_dir)
                  if p.endswith(".png"))
    print(f"  fringe-stopped waterfalls: {len(pngs)} figure(s) in {out_dir}",
          flush=True)
    return pngs


def _fitmask(freq, *goods):
    m = (freq > FITBAND[0]) & (freq < FITBAND[1])
    for lo, hi in RFI_LINES:
        m &= ~((freq > lo) & (freq < hi))
    for gd in goods:
        m &= gd
    return m


def _delay_fit(dphi, freq_mhz, m):
    """run_cal_diff.py::delay_fit, verbatim."""
    f = freq_mhz[m] * 1e6
    z = np.exp(1j * dphi[m])
    ph = np.exp(-2j * np.pi * np.outer(TAUS_NS * 1e-9, f))
    amp = np.abs((z[None, :] * ph).mean(axis=1))
    k = int(np.argmax(amp))
    tau = TAUS_NS[k]
    res = np.angle(z * np.exp(-2j * np.pi * tau * 1e-9 * f))
    res = res - np.angle(np.exp(1j * res).mean())
    res = np.angle(np.exp(1j * res))
    return tau, float(np.sqrt(np.mean(res ** 2))), float(amp[k])


def plot_cal_diff(cal_file, prev_cal_file, ref_ant, out_png, out_csv):
    """Per-antenna phase & delay diff vs a previous cal (run_cal_diff.py)."""
    import h5py
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _load(path):
        with h5py.File(path, "r") as f:
            g = f["gains"][:]
            ants = [int(a) for a in f["ant_ids"][:]]
            flags = f["flags"][:].astype(bool)
            freq = f["freqs_mhz"][:]
        ph = np.angle(g * np.conj(g[ants.index(ref_ant)])[None, :])
        return dict(ants=ants, freq=freq, phase=ph,
                    good=flags & (np.abs(g).min(axis=0) > 0))

    A, B = _load(prev_cal_file), _load(cal_file)
    if not np.allclose(A["freq"], B["freq"]):
        print("  cal diff SKIPPED: frequency axes differ", flush=True)
        return None, None
    common = [a for a in B["ants"] if a in A["ants"]]
    freq = B["freq"]
    rows, dphi_all = [], np.full((len(common), len(freq)), np.nan)
    for i, a in enumerate(common):
        d = np.angle(np.exp(1j * (B["phase"][B["ants"].index(a)]
                                  - A["phase"][A["ants"].index(a)])))
        m = _fitmask(freq, A["good"], B["good"])
        dphi_all[i] = np.where(m, d, np.nan)
        if m.sum() < 50:
            rows.append(dict(ant=a, nchan=int(m.sum())))
            continue
        tau, resid, coh = _delay_fit(d, freq, m)
        rows.append(dict(ant=a, nchan=int(m.sum()),
                         mean_abs_dphi_rad=float(np.mean(np.abs(d[m]))),
                         circ_coherence=float(np.abs(np.exp(1j * d[m]).mean())),
                         delay_ns=float(tau), resid_rms_rad=resid,
                         coh_after_delay=coh))
    tab = pd.DataFrame(rows)
    tab.to_csv(out_csv, index=False)
    m = _fitmask(freq, A["good"], B["good"])
    loss = np.full(len(freq), np.nan)
    loss[m] = np.abs(np.nanmean(np.exp(1j * dphi_all)[:, m], axis=0)) ** 2

    ncol = int(np.ceil(np.sqrt(len(common))))
    nrow = int(np.ceil(len(common) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for k, a in enumerate(common):
        ax = axes.flat[k]
        ax.plot(freq, dphi_all[k], ".", ms=1.2)
        r = tab[tab.ant == a]
        t = f"ant {a}"
        if len(r) and "delay_ns" in r and not r.delay_ns.isna().all():
            t += (f"\n{r.delay_ns.values[0]:+.1f} ns, "
                  f"res {r.resid_rms_rad.values[0]:.2f} rad")
        ax.set_title(t, fontsize=8)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(alpha=.3)
    for k in range(len(common), nrow * ncol):
        axes.flat[k].axis("off")
    fig.suptitle(f"Delta cal phase vs freq, new minus previous (ref ant "
                 f"{ref_ant});\nband-avg predicted coherent beam power "
                 f"{np.nanmean(loss):.3f}", fontsize=11)
    fig.supxlabel("frequency (MHz)")
    fig.supylabel("delta phase (rad)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  cal diff vs {prev_cal_file}: band-avg predicted coherence "
          f"{np.nanmean(loss):.3f}", flush=True)
    return out_png, float(np.nanmean(loss))


def plot_grid_map(weights_file, out_png, near, title):
    """Beam-grid alt/az scatter (no all-sky image)."""
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with h5py.File(weights_file) as f:
        alt = f["pointings/alt_deg"][...]
        az = f["pointings/az_deg"][...]
    fig, ax = plt.subplots(figsize=(9, 5))
    s = ax.scatter(az, alt, c=np.arange(len(alt)), s=10, cmap="viridis")
    fig.colorbar(s, ax=ax, label="beam index")
    for row in near:
        ax.plot(row["target_az"], row["target_alt"], "r*", ms=13)
        ax.annotate(row["target"], (row["target_az"], row["target_alt"]),
                    fontsize=7, color="r",
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("altitude (deg)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Step 5: diagnostics notebook
# ---------------------------------------------------------------------------

def write_notebook(params: RecipeParams, nb_path, sections, provenance):
    """Build cal_<tag>_diagnostics.ipynb and execute it so it reads as a report.

    Cells reference the PNGs by path relative to the notebook, and execution
    embeds the images as base64 outputs, so the file survives being copied
    away from out_dir.
    """
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(
        f"# CASM cal + weights diagnostics: `{params.tag}`\n\n"
        f"Generated by `bf_weights_generator/make_cal_and_weights.py` "
        f"(canonical recipe, see `docs/canonical-recipe.md`).\n\n"
        f"## Provenance\n\n```json\n"
        f"{json.dumps(provenance, indent=2, default=str)}\n```\n\n"
        f"## Parameters\n\n```json\n"
        f"{json.dumps(asdict(params), indent=2, default=str)}\n```\n")]
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "from IPython.display import Image, display\n"
        f"os.chdir({os.path.dirname(os.path.abspath(nb_path))!r})\n"
        "print(os.getcwd())"))
    for title, text, pngs in sections:
        md = f"## {title}\n"
        if text:
            md += f"\n{text}\n"
        cells.append(nbf.v4.new_markdown_cell(md))
        if pngs:
            rel = [os.path.relpath(p, os.path.dirname(os.path.abspath(nb_path)))
                   for p in pngs]
            cells.append(nbf.v4.new_code_cell(
                "for p in %s:\n    display(Image(filename=p))" % repr(rel)))
    nb["cells"] = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3",
                                 "language": "python", "name": "python3"}
    if params.execute_notebook:
        try:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=600,
                           kernel_name="python3",
                           resources={"metadata": {
                               "path": os.path.dirname(os.path.abspath(nb_path))}}
                           ).execute()
        except Exception as exc:                   # pragma: no cover
            print(f"  notebook execution failed "
                  f"({type(exc).__name__}: {exc}); saving unexecuted",
                  flush=True)
    with open(nb_path, "w") as fh:
        nbf.write(nb, fh)
    print(f"  wrote {nb_path}", flush=True)
    return nb_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(params: RecipeParams):
    """Execute the recipe. Returns a dict of output paths + the report."""
    from casm_io.correlator import AntennaMapping, load_format
    from bf_weights_generator import Array64Config

    params = params.resolved()
    os.makedirs(params.out_dir, exist_ok=True)
    fmt = load_format(FMT_NAME)
    mapping = AntennaMapping.load(params.layout_csv)
    all_active = set(mapping.active_antennas())
    ants = sorted(all_active if params.antennas is None
                  else set(int(a) for a in params.antennas))
    missing = sorted(set(ants) - all_active)
    if missing:
        raise ValueError(f"antennas not active in {params.layout_csv}: {missing}")
    ant = mapping.with_inactive(sorted(all_active - set(ants)))
    assert sorted(ant.active_antennas()) == ants
    print(f"[setup] layout {params.layout_csv}\n"
          f"  {len(ants)} antennas: {ants}\n  ref_ant {params.ref_ant}",
          flush=True)

    out = dict(out_dir=params.out_dir, tag=params.tag)
    report = dict(params=asdict(params), antennas=ants)
    sections = []
    fs_primary = None
    cal_cases = []

    # ---------------- 1. calibration -----------------------------------
    if params.cal_path:
        cal_file = params.cal_path
        print(f"[cal] reusing {cal_file} (no solve)", flush=True)
        report["cal_reused"] = True
    else:
        if params.source_window is None:
            raise ValueError("source_window is required unless cal_path is set")
        source = resolve_source(params)
        static = None
        if params.static_window is not None:
            print("[static] building/loading off-source static", flush=True)
            static = build_manual_static(
                params.static_path, params.static_window[0],
                params.static_window[1],
                params.static_notes or
                f"{params.static_window[0]}-{params.static_window[1]} UTC "
                f"static for cal tag {params.tag}.", fmt)
        cal, nt, fs_primary = solve_calibration(
            params, mapping, ant, source,
            static_vis=None if static is None else static["static_vis"],
            label="primary")
        cal_cases.append((np.asarray(cal["freqs_mhz"]),
                          np.asarray(cal["rank1_ratios"]),
                          f"{params.cal_source}, "
                          f"{'static-subtracted' if static is not None else 'no static'}",
                          "tab:green"))
        if static is not None and params.diagnostics:
            # A/B: the same window without the static, for the standard figure.
            cal0, _, _ = solve_calibration(params, mapping, ant, source,
                                           static_vis=None, label="nostatic-AB")
            cal_cases.append((np.asarray(cal0["freqs_mhz"]),
                              np.asarray(cal0["rank1_ratios"]),
                              f"{params.cal_source}, no static", "tab:red"))
            r1s = np.asarray(cal["rank1_ratios"])
            r10 = np.asarray(cal0["rank1_ratios"])
            f = np.asarray(cal["freqs_mhz"])
            report["static_ab"] = dict(
                with_static_full=float(np.nanmedian(r1s)),
                with_static_band=float(np.nanmedian(r1s[band_mask(f)])),
                no_static_full=float(np.nanmedian(r10)),
                no_static_band=float(np.nanmedian(r10[band_mask(f)])))
            print(f"  static A/B: with {report['static_ab']['with_static_band']:.3f} "
                  f"vs without {report['static_ab']['no_static_band']:.3f} "
                  f"(in-band rank-1 median)", flush=True)

        from casm_calibrator import save_calibration
        cal_file = os.path.join(params.out_dir, f"cal_{params.tag}.h5")
        save_calibration(cal, cal_file, n_time_averaged=nt, overwrite=True)
        print(f"  wrote {cal_file}", flush=True)
        report["cal"] = dict(n_time_averaged=int(nt),
                             n_solved=int(np.asarray(cal["flags"]).sum()),
                             rank1_median=float(np.nanmedian(
                                 np.asarray(cal["rank1_ratios"]))))
        np.savez(os.path.join(params.out_dir, f"rank1_vs_freq_{params.tag}.npz"),
                 freq_mhz=np.asarray(cal["freqs_mhz"]),
                 rank1=np.asarray(cal["rank1_ratios"]), antennas=np.array(ants))
    out["cal_file"] = cal_file

    # ---------------- 2. weights ---------------------------------------
    weights_file = None
    grid_meta = {}
    if params.make_weights:
        print("\n[weights]", flush=True)
        arr = Array64Config.from_antenna_mapping(ant)
        grid, grid_meta = build_grid(params, arr)
        weights_file = os.path.join(params.out_dir,
                                    f"weights_{params.tag}_"
                                    f"{arr.n_active}ant_{len(grid)}_int8.h5")
        build_weights(params, arr, grid, cal_file, weights_file)
        out["weights_file"] = weights_file
        verify(params, weights_file, cal_file, grid_meta, report)
        print_operator_commands(params, weights_file)

    # ---------------- 3. diagnostics ------------------------------------
    if params.diagnostics:
        print("\n[diagnostics]", flush=True)
        if cal_cases:
            png = os.path.join(params.out_dir, f"rank1_vs_freq_{params.tag}.png")
            meds = plot_rank1(cal_cases, png,
                              f"Fringe-stopped {params.cal_source} rank-1 vs "
                              f"frequency, {len(ants)} ants, "
                              f"{params.source_window[0]} - "
                              f"{params.source_window[1]} UTC")
            report["rank1_medians"] = {k: list(v) for k, v in meds.items()}
            sections.append((
                "Rank-1 ratio vs frequency",
                "Per-channel rank-1 ratio of the fringe-stopped visibility "
                "matrix (phase-only SVD). Values well above 1 mean the source "
                "dominates and the solve is trustworthy in that channel. "
                "Dashed line at 1.0, dotted line at the primary case median.",
                [png]))
        if fs_primary is not None:
            fdir = os.path.join(params.out_dir, f"fringe_{params.tag}")
            os.makedirs(fdir, exist_ok=True)
            fpngs = plot_fringe_stopped(fs_primary, mapping, fdir,
                                        params.fringe_plot_max_baselines)
            sections.append((
                "Fringe-stopped visibilities",
                "Waterfalls per SNAP pair: raw phase, the geometric model, and "
                "the fringe-stopped residual. Flat residual phase across the "
                "window means the source direction and antenna positions are "
                "consistent.", fpngs))
        if params.prev_cal_path:
            png = os.path.join(params.out_dir, f"cal_diff_{params.tag}.png")
            csv = os.path.join(params.out_dir, f"cal_diff_{params.tag}.csv")
            p, loss = plot_cal_diff(cal_file, params.prev_cal_path,
                                    params.ref_ant, png, csv)
            if p:
                report["cal_diff_band_avg_coherence"] = loss
                sections.append((
                    "Per-antenna phase and delay diff vs the previous cal",
                    f"New cal minus `{params.prev_cal_path}`, referenced to "
                    f"antenna {params.ref_ant}. Band-averaged predicted "
                    f"coherent beam power if the OLD cal were kept: "
                    f"{loss:.3f}. Table: `{os.path.basename(csv)}`.", [png]))
        if "static_ab" in report:
            ab = report["static_ab"]
            sections.append((
                "Static A/B",
                f"In-band rank-1 median with the static subtracted: "
                f"{ab['with_static_band']:.3f}; without: "
                f"{ab['no_static_band']:.3f}. Full band: "
                f"{ab['with_static_full']:.3f} vs {ab['no_static_full']:.3f}.",
                []))
        if weights_file:
            png = os.path.join(params.out_dir, f"beam_grid_{params.tag}.png")
            plot_grid_map(weights_file, png, report.get("nearest_beams", []),
                          f"Beam grid {params.tag}: {report['pointings']['n']} "
                          f"pointings, alt "
                          f"{report['pointings']['alt_min']:.1f}-"
                          f"{report['pointings']['alt_max']:.1f} deg")
            sections.append((
                "Beam grid",
                "Alt/az of every beam, coloured by beam index. Red stars are "
                "the nearest-beam targets.", [png]))

    # ---------------- 4. report + notebook -------------------------------
    rep_path = os.path.join(params.out_dir, f"report_{params.tag}.json")
    with open(rep_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    out["report_json"] = rep_path
    out["report"] = report
    print(f"\n[report] {rep_path}", flush=True)

    if params.notebook and sections:
        provenance = dict(driver=os.path.abspath(__file__),
                          cal_file=cal_file, weights_file=weights_file,
                          layout=params.layout_csv, antennas=ants,
                          grid=grid_meta,
                          report_json=rep_path,
                          fixed_policy=dict(
                              freq_config="FrequencyConfig.layout_64ant()",
                              svd=dict(threshold=SVD_THRESHOLD,
                                       mode="PHASE_ONLY",
                                       block_size=SVD_BLOCK_SIZE,
                                       masked_band_strategy=SVD_MASKED_BAND_STRATEGY),
                              fringe_stop_sign=FRINGE_SIGN,
                              scale_factor=SCALE_FACTOR,
                              freq_order=FREQ_ORDER,
                              cb_slots=N_CB_SLOTS))
        nb_path = os.path.join(params.out_dir,
                               f"cal_{params.tag}_diagnostics.ipynb")
        write_notebook(params, nb_path, sections, provenance)
        out["notebook"] = nb_path

    print("MAKE_CAL_AND_WEIGHTS_DONE", flush=True)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _coerce(name, value):
    """Parse a --param VALUE string into the dataclass field's type."""
    ftype = {f.name: f.type for f in fields(RecipeParams)}[name]
    text = str(ftype)
    if "bool" in text and "Optional" not in text:
        return str(value).lower() in ("1", "true", "yes", "on")
    if value == "None":
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Canonical CASM calibration + beamforming-weights recipe. "
                    "Orchestrates casm_io / casm_vis_analysis / casm_calibrator "
                    "/ bf_weights_generator. Never uploads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Knobs (--param NAME VALUE, VALUE parsed as a Python literal):\n"
               + "\n".join(f"  {f.name:26s} {f.type}"
                           for f in fields(RecipeParams)))
    ap.add_argument("--config", help="JSON file of parameter overrides")
    ap.add_argument("--param", nargs=2, action="append", metavar=("NAME", "VALUE"),
                    default=[], help="Override one parameter; repeatable")
    ap.add_argument("--print-params", action="store_true",
                    help="Print the resolved parameters and exit")
    args = ap.parse_args(argv)

    valid = {f.name for f in fields(RecipeParams)}
    kwargs = {}
    if args.config:
        with open(args.config) as fh:
            for k, v in json.load(fh).items():
                if k not in valid:
                    raise SystemExit(f"unknown parameter in config: {k}")
                kwargs[k] = v
    for name, value in args.param:
        if name not in valid:
            raise SystemExit(f"unknown parameter: {name}. Known: "
                             f"{', '.join(sorted(valid))}")
        kwargs[name] = _coerce(name, value)
    params = RecipeParams(**kwargs)
    if args.print_params:
        print(json.dumps(asdict(params), indent=2, default=str))
        return 0
    run(params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
