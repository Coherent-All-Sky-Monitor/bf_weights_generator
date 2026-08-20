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
  * pointing verification by cal division
    /mnt/nvme5/solar0819/pointing_verify/verify_pointing.py, now vendored
    into ``bf_weights_generator/recipe_verify.py``

Every diagnostic figure lives in ``bf_weights_generator/recipe_diagnostics.py``
(which documents where each plot was ported from). Diagnostics are read-only:
no figure code can change a product byte.

Bit-exact reproduction of the three reference products through this driver is
checked by ``tests/validate_recipe.py`` (see docs/canonical-recipe.md).

FIXED policy (deliberately not knobs):
  FrequencyConfig.layout_64ant(); full-band per-channel solve with
  SVDConfig(threshold=1.0, PHASE_ONLY, block_size=1, masked_band_strategy=
  "zero"); fringe_stop sign=-1; scale_factor=127.0; ``[..., :64]`` slice;
  freq_order="descending"; cal saved through casm_calibrator.save_calibration.

TUNABLE: the calibrator source, the time windows, the antenna set, the
reference antenna, the layout, the grid bounds and spacing search, the
verification thresholds, and the output naming.

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
import contextlib
import gc
import json
import os
import sys
import traceback
from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Sequence, Tuple

import numpy as np

from bf_weights_generator import recipe_diagnostics as rd
from bf_weights_generator import recipe_verify as rv

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

# Diagnostic band conventions live with the diagnostics (svd_census/cal_diff).
SUNBAND = rd.SUNBAND
RFI_LINES = rd.RFI_LINES
FITBAND = rd.FITBAND
TAUS_NS = rd.TAUS_NS
band_mask = rd.band_mask
LOCAL_TZ = rd.LOCAL_TZ

DEFAULT_LAYOUT = ("/home/casm/software/dev/antenna_layouts/"
                  "casm_antenna_layout_2026-08-07.csv")
VENV_PY = "/home/casm/software/dev/casm_venvs/casm_offline_env/bin/python"
DEPLOY_PY = ("/home/casm/software/dev/bf_weights_generator/"
             "bf_weights_generator/deploy_bf_weights.py")
TRANSIT_BIN = ("/home/casm/software/dev/casm_venvs/casm_offline_env/bin/"
               "casm-bf-source-transit")


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
    """ICRS RA for a source NOT already in the catalog, e.g. "19h59m28.36s"
    or "299.868deg". Requires ``cal_dec``. Registered into the catalog under
    ``cal_source`` for the duration of the solve and removed afterwards
    (fringe_stop only accepts catalog names). Rejected for ``cal_source="sun"``
    (the sun is special-cased ahead of the catalog, so coordinates would be
    silently ignored) and for any name already in the catalog."""

    cal_dec: Optional[str] = None

    source_window: Optional[Tuple[str, str]] = None
    """(UTC start, UTC end) of the on-source data to solve on."""

    static_window: Optional[Tuple[str, str]] = None
    """(UTC start, UTC end) of an off-source window to average into a static
    visibility and subtract before solving."""

    static_path: Optional[str] = None
    """Where the static npz lives. If it exists it is loaded (and its recorded
    window checked against ``static_window`` when that is set), else built from
    ``static_window``. Default when only ``static_window`` is given:
    <out_dir>/static_<tag>.npz. Setting either one turns static subtraction
    ON."""

    static_window_tol_s: float = 300.0
    """How far the window recorded inside an existing static npz may sit from
    ``static_window`` before the run aborts. Not zero because the realised
    window is the first/last integration inside the request, which observation
    boundaries can push minutes away from it."""

    static_notes: str = ""

    antennas: Optional[Sequence[int]] = None
    """Antenna IDs to solve/beamform with. None = every antenna active in the
    layout CSV."""

    ref_ant: int = 9
    layout_csv: str = DEFAULT_LAYOUT
    min_alt_deg: float = 10.0

    min_time_on_source_frac: float = 0.5
    """Minimum fraction of the solve window during which the source must be
    above ``min_alt_deg``. fringe_stop silently falls back to an all-ones time
    mask when the source never rises, which would otherwise solve on noise."""

    cal_path: Optional[str] = None
    """Reuse an existing cal h5 instead of solving. Skips the whole read/solve
    step, so grid-only rebuilds cost seconds. Mutually exclusive with
    source_window / static_window / static_path / cal_ra."""

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

    # ---- verification ------------------------------------------------------
    verify_pointing: bool = True
    """Run the cal-division pointing fit. A failure aborts the run: a weights
    file that does not point where its table says is not deployable."""

    verify_beams: Sequence[int] = (0, 256, 511)
    """Beams put through the cal-division pointing fit. Out-of-range entries
    are clipped to the grid."""

    min_subband_good_frac: float = 0.9
    """Per-subband (512 channels) fraction of solved, non-zero cal channels
    below which the run aborts. A dead subband means the F-engine delivered no
    data there and the beamformer would run on a fraction of the band."""

    nearest_targets: Sequence[Tuple[str, float, float]] = ()
    """(name, alt_deg, az_deg) rows for the nearest-beam table."""

    nearest_sources: Sequence[str] = ()
    """Named sources; their transit alt/az on ``nearest_date`` are appended to
    the nearest-beam table."""

    nearest_date: Optional[str] = None

    # ---- diagnostics -------------------------------------------------------
    prev_cal_path: Optional[str] = None
    """Previous cal h5 for the per-antenna phase/delay diff figure and as the
    comparison case in the beamformed-source check."""

    diagnostics: bool = True
    notebook: bool = True
    execute_notebook: bool = True
    fringe_plot_max_baselines: int = 20
    phase_baselines_per_class: int = 3
    """How many short / medium / long reference-antenna baselines go into the
    phase-vs-frequency figures."""

    beam_check: bool = True
    beam_check_source: str = "cyg-a"
    beam_check_fallback: str = "cas-a"
    """Tried when ``beam_check_source`` is not up, or its data is not on disk."""

    beam_check_date: Optional[str] = None
    beam_check_window: Optional[Tuple[str, str]] = None
    """Explicit (UTC start, UTC end). Default: centred on the source's transit
    on ``beam_check_date`` (or nearest_date / track_date / the solve date)."""

    beam_check_minutes: float = 60.0
    beam_check_off_alt_deg: float = -25.0
    """Altitude offset of the off-source null beam."""

    transit_sources: Sequence[str] = ("sun", "cyg-a", "cas-a", "b0329+54")
    transit_date: Optional[str] = None

    def resolved(self) -> "RecipeParams":
        """Fill in derived defaults."""
        if self.static_path is None and self.static_window is not None:
            self.static_path = os.path.join(self.out_dir,
                                            f"static_{self.tag}.npz")
        return self


def validate_params(params: RecipeParams, active_antennas) -> None:
    """Reject parameter sets that state two intents at once."""
    if params.cal_path:
        clashes = [n for n in ("source_window", "static_window", "static_path",
                               "cal_ra", "cal_dec")
                   if getattr(params, n) is not None]
        if clashes:
            raise ValueError(
                f"cal_path reuses an existing cal, so {', '.join(clashes)} "
                f"would be ignored. Drop cal_path to solve, or drop "
                f"{', '.join(clashes)} to reuse.")
    elif params.source_window is None:
        raise ValueError("source_window is required unless cal_path is set")
    if params.ref_ant not in active_antennas:
        raise ValueError(
            f"ref_ant {params.ref_ant} is not in the antenna set "
            f"{list(active_antennas)}")
    if params.trim not in ("lowest", "highest"):
        raise ValueError(f"trim must be 'lowest' or 'highest', "
                         f"got {params.trim!r}")
    if params.grid_mode not in ("bounds", "track"):
        raise ValueError(f"grid_mode must be 'bounds' or 'track', "
                         f"got {params.grid_mode!r}")


# ---------------------------------------------------------------------------
# Source handling
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def registered_source(params: RecipeParams):
    """Yield the source key for fringe_stop, registering ad-hoc coordinates.

    fringe_stop resolves names through casm_vis_analysis.sources.CATALOG
    (plus the special-cased "sun"), normalising '-', '+' and ' ' to '_'.
    An explicit RA/Dec is registered into that catalog under the requested
    name FOR THE DURATION OF THE SOLVE ONLY: the key is removed again on the
    way out, success or failure, so nothing leaks into the next run in the
    same process.
    """
    from casm_vis_analysis import sources as _src

    name = params.cal_source
    key = name.lower().replace("-", "_").replace(" ", "_").replace("+", "_")

    if params.cal_ra is None and params.cal_dec is None:
        if key != "sun" and key not in _src.CATALOG:
            raise ValueError(
                f"Unknown source {name!r}. Catalog: sun, "
                f"{', '.join(sorted(_src.CATALOG))}. For anything else "
                f"pass cal_ra and cal_dec.")
        yield name
        return

    if params.cal_ra is None or params.cal_dec is None:
        raise ValueError("cal_ra and cal_dec must be given together.")
    if key == "sun":
        raise ValueError(
            "cal_source='sun' ignores cal_ra/cal_dec: the sun is computed "
            "from ephemeris ahead of the catalog lookup. Use a different "
            "cal_source name for a fixed-coordinate source.")
    if key in _src.CATALOG:
        raise ValueError(
            f"{name!r} is already in the source catalog; passing cal_ra/"
            f"cal_dec would overwrite the built-in J2000 position for every "
            f"consumer in this process. Drop cal_ra/cal_dec to use the "
            f"catalog position, or pick a new name.")

    from astropy.coordinates import SkyCoord
    _src.CATALOG[key] = SkyCoord(params.cal_ra, params.cal_dec, frame="icrs")
    print(f"  registered {name} -> {key} at RA {params.cal_ra} "
          f"Dec {params.cal_dec}", flush=True)
    try:
        yield name
    finally:
        _src.CATALOG.pop(key, None)


def source_transit_altaz(name: str, date_utc: str,
                         step_minutes: float = 1.0) -> Tuple[float, float]:
    """Alt/az of a source at its highest point on a UTC date."""
    alt, az, _ = source_transit(name, date_utc, step_minutes)
    return alt, az


def source_transit(name: str, date_utc: str,
                   step_minutes: float = 1.0) -> Tuple[float, float, float]:
    """(alt_deg, az_deg, unix_time) of a source's highest point on a UTC date."""
    from astropy.time import Time
    from casm_vis_analysis.sources import source_altaz
    t0 = Time(f"{date_utc} 00:00:00", scale="utc").unix
    grid = t0 + np.arange(0.0, 86400.0, step_minutes * 60.0)
    alt, az = source_altaz(name, grid)
    k = int(np.argmax(alt))
    return float(alt[k]), float(az[k]), float(grid[k])


def _utc_string(unix_time: float) -> str:
    from astropy.time import Time
    return Time(float(unix_time), format="unix", scale="utc").iso[:19]


# ---------------------------------------------------------------------------
# Step 1: calibration
# ---------------------------------------------------------------------------

def _read_visibilities(t0, t1, fmt, what):
    """read_visibilities with the window in the error message.

    A truncated/stub .dat on disk raises a nameless "negative dimensions"
    ValueError from deep inside the reader; without this the operator has no
    idea which window to go and look at.
    """
    from casm_io.correlator import read_visibilities
    try:
        return read_visibilities(t0, t1, time_tz="UTC", data_root="/mnt",
                                 fmt=fmt, verbose=False)
    except Exception as exc:
        raise RuntimeError(
            f"reading {what} visibilities {t0} -> {t1} UTC failed "
            f"(data_root=/mnt, fmt={FMT_NAME}): {type(exc).__name__}: {exc}"
        ) from exc


def build_manual_static(path, window, note, fmt, tol_s=300.0):
    """Load or build the off-source static visibility.

    Numerics verbatim from run_drill_aug15.py::build_manual_static (UTC
    times, ``average_visibility(apply_freq_mask=True)``). An existing file is
    reused only if the window it records matches ``window``; otherwise the run
    stops rather than silently subtracting the wrong sky.
    """
    from casm_vis_analysis.offsource import (average_visibility,
                                             save_static_visibility,
                                             load_static_visibility)
    if os.path.exists(path):
        st = load_static_visibility(path)
        stored = st.get("window_unix")
        if window is not None and stored is not None:
            from astropy.time import Time
            want = [Time(w, scale="utc").unix for w in window]
            off = np.abs(np.asarray(stored) - np.asarray(want))
            if np.any(off > tol_s):
                raise ValueError(
                    f"existing static {path} covers "
                    f"{_utc_string(stored[0])} -> {_utc_string(stored[1])} UTC, "
                    f"but static_window asks for {window[0]} -> {window[1]} "
                    f"({off.max():.0f} s away, tolerance {tol_s:.0f} s). "
                    f"Delete the file to rebuild it, point static_path at the "
                    f"right one, or raise static_window_tol_s.")
            print(f"  reusing existing static {path} "
                  f"(window offset {off.max():.0f} s)", flush=True)
        else:
            print(f"  reusing existing static {path}", flush=True)
        return st
    if window is None:
        raise ValueError(
            f"static_path {path} does not exist and static_window is not set, "
            f"so there is nothing to build the static from.")
    dn = _read_visibilities(window[0], window[1], fmt, "static")
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


def solve_calibration(params: RecipeParams, mapping, ant, source,
                      static_vis=None, label="cal", collect_diag=False):
    """One solve: read -> (optional static subtract) -> fringe_stop -> SVD.

    Code path identical to make_cal0819_weights.py step 1 and to
    run_static_night_aug20.py::run_case. ``collect_diag`` additionally copies
    the per-antenna autocorrelations out of the cube before it is freed; that
    is a pure read and cannot affect the solution.
    """
    from casm_io.correlator import load_format
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
    d = _read_visibilities(t0, t1, load_format(FMT_NAME), "solve-window")
    nt = np.asarray(d["vis"]).shape[0]
    dd = d if static_vis is None else subtract_static_visibility(d, static_vis)
    fs = fringe_stop(dd, ant, ref_ant=params.ref_ant, source=source,
                     sign=FRINGE_SIGN, min_alt_deg=params.min_alt_deg)

    # fringe_stop returns an all-True time mask when the source never rises;
    # solving on that is solving on noise, so check it explicitly.
    tmask = np.asarray(fs["time_mask"], dtype=bool)
    frac = float(tmask.mean())
    print(f"  {source} above {params.min_alt_deg} deg for "
          f"{tmask.sum()}/{len(tmask)} integrations ({frac:.0%})", flush=True)
    if frac < params.min_time_on_source_frac:
        raise ValueError(
            f"{source} is above min_alt_deg={params.min_alt_deg} for only "
            f"{frac:.0%} of {t0} -> {t1} UTC (need "
            f"{params.min_time_on_source_frac:.0%}). Pick a window centred on "
            f"the transit, or lower min_time_on_source_frac deliberately.")

    diag = dict(time_mask_frac=frac)
    if collect_diag:
        # RAW autocorrelations: dead/railed feeds are what this panel is for,
        # and subtracting the static would hide a railed level.
        autos, labels = rd.collect_autocorrelations(d, ant)
        diag.update(autos=autos, auto_labels=labels,
                    freq_mhz=np.asarray(d["freq_mhz"]),
                    time_unix=np.asarray(d["time_unix"]))

    cal = svd_calibrate(fs, ant, data=dd, config=cfg)
    r1 = np.asarray(cal["rank1_ratios"])
    print(f"  nt={nt} solved {int(np.asarray(cal['flags']).sum())}/"
          f"{len(np.asarray(cal['freqs_mhz']))} rank1 median "
          f"{np.nanmedian(r1):.2f}", flush=True)
    del d, dd
    gc.collect()
    return cal, nt, fs, diag


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
    a stable argsort on altitude keeping the first n_beams indices in file
    order.
    """
    from bf_weights_generator import generate_beam_grid_altaz
    from bf_weights_generator.config import compute_beam_fwhm

    if params.grid_mode == "track":
        alt_min, alt_max, az_min, az_max = track_bounds(params)
    else:
        alt_min, alt_max = params.alt_min_deg, params.alt_max_deg
        az_min, az_max = params.az_min_deg, params.az_max_deg

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
        raise ValueError(
            f"spacing search found no grid with >= {params.n_beams} beams over "
            f"mult {params.mult_min}-{params.mult_max}; widen the search")
    mult, grid = best
    n_pre = len(grid)
    alts_pre = [b.alt_deg for b in grid]
    # kind="stable": ties in altitude must break by grid order, not by
    # whatever introsort does with them, or two runs of the same config could
    # keep different beams.
    order = np.argsort(alts_pre, kind="stable")
    if params.trim == "highest":
        order = order[::-1]
    grid = [grid[i] for i in sorted(order[:params.n_beams])]
    alts = np.array([b.alt_deg for b in grid])
    azs = np.array([b.az_deg for b in grid])
    print(f"  mult={mult:.3f} ({n_pre} beams pre-trim) -> {len(grid)} beams, "
          f"alt {alts.min():.2f}-{alts.max():.2f}, "
          f"az {azs.min():.1f}-{azs.max():.1f}, "
          f"spacing {mult*fw_ew:.2f} x {mult*fw_ns:.2f} deg", flush=True)
    # In track mode the requested box is an UNWRAPPED arc (it can run past
    # 360 deg); the per-beam azimuths come back wrapped into 0-360, so report
    # both rather than pretending the grid spans 0-360.
    meta = dict(mult=float(mult), n_pre_trim=int(n_pre),
                fwhm_ew_deg=float(fw_ew), fwhm_ns_deg=float(fw_ns),
                spacing_ew_deg=float(mult * fw_ew),
                spacing_ns_deg=float(mult * fw_ns),
                alt_min=float(alts.min()), alt_max=float(alts.max()),
                az_min=float(azs.min()), az_max=float(azs.max()),
                az_arc_deg=float(az_max - az_min),
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


def check_cal_subbands(params: RecipeParams, cal_file, report):
    """Per-subband solved-channel fractions; abort on a dead subband."""
    rows = rv.subband_flag_table(cal_file)
    print("  cal subbands (512 ch each):", flush=True)
    for r in rows:
        print(f"    sb{r['subband']} ch {r['chan_lo']:4d}-{r['chan_hi']:4d} "
              f"{r['freq_lo_mhz']:7.2f}-{r['freq_hi_mhz']:7.2f} MHz  "
              f"good {r['n_good']:4d}/{r['n_chan']:4d} "
              f"({r['good_frac']:.3f})", flush=True)
    report["cal_subbands"] = rows
    bad = [r for r in rows if r["good_frac"] < params.min_subband_good_frac]
    if bad:
        raise RuntimeError(
            "cal has dead subband(s): "
            + ", ".join(f"sb{r['subband']} {r['good_frac']:.3f}" for r in bad)
            + f" (need {params.min_subband_good_frac:.2f}). Check the "
              "F-engine / missing-subband diagnostic before deploying.")
    return rows


def verify(params: RecipeParams, weights_file, cal_file, grid_meta, report):
    """Pointing stats, int8 sanity, cal-division coherence, nearest beams."""
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
        os.remove(weights_file)       # never leave a deployable-looking dud
        raise RuntimeError(
            f"all-zero int8 payload at ch1500; removed {weights_file}")
    report["pointings"] = dict(n=int(len(alt)), alt_min=float(alt.min()),
                               alt_max=float(alt.max()),
                               alt_median=float(np.median(alt)),
                               az_min=float(az.min()), az_max=float(az.max()))
    report["int8"] = dict(shape=list(shape), ch1500_min=int(sl.min()),
                          ch1500_max=int(sl.max()),
                          ch1500_mean_abs=float(np.abs(sl).mean()))
    report["weights_attrs"] = {k: str(v) for k, v in attrs.items()}

    check_cal_subbands(params, cal_file, report)

    # ---- cal-division pointing coherence -------------------------------
    rows = []
    if not params.verify_pointing:
        print("  pointing fit SKIPPED (verify_pointing=False)", flush=True)
    else:
        meta = rv.load_file(weights_file)
        calw = load_calibration_weights(cal_file)
        beams = sorted({int(np.clip(b, 0, len(alt) - 1))
                        for b in params.verify_beams})
        print(f"  cal-division pointing fit on beams {beams}", flush=True)
        print(f"  {'beam':>5} {'stated_alt':>10} {'stated_az':>10} "
              f"{'fit_alt':>9} {'fit_az':>9} {'coh_fit':>8} {'coh_stated':>10} "
              f"{'dalt':>7} {'daz':>8}", flush=True)
        for b in beams:
            r = rv.fit_beam(weights_file, b, meta, calw)
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
# Step 4: diagnostic figures (all drawing lives in recipe_diagnostics)
# ---------------------------------------------------------------------------

def _beam_check_window(params: RecipeParams, source):
    """(start, end) UTC strings centred on the source's transit."""
    if params.beam_check_window is not None:
        return tuple(params.beam_check_window)
    date = (params.beam_check_date or params.nearest_date or params.track_date
            or params.transit_date)
    if date is None and params.source_window is not None:
        date = str(params.source_window[0])[:10]
    if date is None:
        return None
    _alt, _az, t_transit = source_transit(
        source.lower().replace("-", "_").replace("+", "_"), date)
    half = params.beam_check_minutes * 30.0     # minutes/2 -> seconds
    return (_utc_string(t_transit - half), _utc_string(t_transit + half))


def run_diagnostics(params, mapping, ant, ants, cal, cal_file, fs_primary,
                    solve_diag, cal_cases, weights_file, report, sections):
    """Every figure the notebook shows. Failures here never lose a product."""
    figs_dir = os.path.join(params.out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    tag = params.tag

    def fig(name):
        return os.path.join(figs_dir, f"{name}_{tag}.png")

    def section(title, text, pngs):
        sections.append((title, text, [p for p in pngs if p]))

    def step(name, fn):
        """Run one diagnostic; log and continue if it fails."""
        try:
            return fn()
        except Exception as exc:                       # pragma: no cover
            print(f"  DIAGNOSTIC '{name}' FAILED "
                  f"({type(exc).__name__}: {exc})", flush=True)
            traceback.print_exc()
            report.setdefault("diagnostic_failures", {})[name] = (
                f"{type(exc).__name__}: {exc}")
            sections.append((name, f"**SKIPPED** - this diagnostic raised "
                                   f"`{type(exc).__name__}: {exc}`.", []))
            return None

    # ---- 1. autocorrelation spectra ------------------------------------
    if solve_diag and "autos" in solve_diag:
        def _autos():
            png = fig("autocorr")
            rd.plot_autocorr_spectra(
                solve_diag["autos"], solve_diag["freq_mhz"],
                solve_diag["auto_labels"], solve_diag["time_unix"], png,
                f"Autocorrelation spectra, solve window "
                f"({rd.utc_local_title(solve_diag['time_unix'])})",
                freq_mask=fs_primary.get("freq_mask") if fs_primary else None)
            section("Autocorrelation spectra of the active antennas",
                    "Time-averaged autocorrelation power per antenna over the "
                    "same visibilities the cal was solved on.\n\n"
                    "*Healthy*: every panel shows the same smooth bandpass "
                    "shape at a similar level, with the known narrow RFI "
                    "lines. *Sick*: a flat/featureless panel (dead feed), one "
                    "sitting 12+ dB above the rest (railed ADC), or a whole "
                    "512-channel block at the floor (missing subband).",
                    [png])
        step("Autocorrelation spectra", _autos)

    # ---- 2-4. phase vs frequency through the stages --------------------
    if fs_primary is not None:
        def _phases():
            sel, labels, lengths = rd.select_ref_baselines(
                fs_primary, ant, params.phase_baselines_per_class)
            report["phase_baselines"] = [
                dict(index=int(i), label=l.replace("\n", " "),
                     length_m=float(x))
                for i, l, x in zip(sel, labels, lengths)]
            freq = np.asarray(fs_primary["freq_mhz"])
            tu = np.asarray(fs_primary["time_unix"])

            saw = fig("phase_raw_sawtooth")
            rd.plot_sawtooth(fs_primary, sel, labels, lengths, saw,
                             f"Raw cross-correlation phase vs frequency, "
                             f"ref ant {params.ref_ant} "
                             f"({rd.utc_local_title(tu)})")
            raw = fig("phase_stage1_raw")
            rd.plot_phase_stage(np.asarray(fs_primary["vis"])[:, :, sel], freq,
                                labels, raw, tu)
            section("Raw cross-correlation phase vs frequency (the delay "
                    "sawtooth)",
                    "Time-averaged raw phase for short, medium and long "
                    "reference-antenna baselines, ordered by baseline length. "
                    "First figure is wrapped phase (the sawtooth), second is "
                    "unwrapped.\n\n"
                    "*Healthy*: a clean linear ramp, wrapping many times "
                    "across the band - that slope is the geometric delay "
                    "plus the antenna's cable delay, so it grows with "
                    "baseline length only roughly. *Sick*: phase noise with "
                    "no ramp (no fringes on that baseline) or jumps at "
                    "subband edges.",
                    [saw, raw])

            stopped = fig("phase_stage2_fringe_stopped")
            rd.plot_phase_stage(
                np.asarray(fs_primary["vis_stopped"])[:, :, sel], freq,
                labels, stopped, tu)
            section("The same baselines after fringe-stopping",
                    "Geometric delay toward the calibrator removed "
                    "(`sign=-1`); only instrumental phase is left.\n\n"
                    "*Healthy*: the fast ramp is gone, leaving slow "
                    "structure - a residual cable delay of a few ns plus "
                    "bandpass ripple. *Sick*: the ramp survives (wrong source "
                    "or wrong antenna positions).",
                    [stopped])

            resid = fig("phase_stage3_calibrated")
            rd.plot_phase_stage(
                rd.calibrated_baselines(fs_primary, cal, sel), freq, labels,
                resid, tu, unwrap=False)
            section("After fringe-stopping AND the solved cal",
                    "The same baselines with the per-antenna solution divided "
                    "out: `V_ij conj(g_i) g_j`. Plotted wrapped, not "
                    "unwrapped.\n\n"
                    "*Healthy*: flat at zero across the band on every "
                    "baseline. *Sick*: any surviving slope (the delay the "
                    "solve failed to absorb) or a band where the residual "
                    "scatters over the full +/-pi.",
                    [resid])
        step("Phase vs frequency through the stages", _phases)

    # ---- 5. per-antenna gain phase + delay fit -------------------------
    def _gainfits():
        png = fig("gain_delay_fits")
        csv = os.path.join(params.out_dir, f"gain_delay_fits_{tag}.csv")
        _p, tab = rd.plot_gain_delay_fits(cal_file, params.ref_ant, png, csv,
                                          tag=tag)
        if "resid_rms_rad" in tab:
            report["gain_delay_fits"] = dict(
                median_resid_rms_rad=float(np.nanmedian(tab["resid_rms_rad"])),
                max_abs_delay_ns=float(np.nanmax(np.abs(tab["delay_ns"]))))
        section("Per-antenna solved gain phase with the fitted delay",
                f"Blue: the solved gain phase referenced to antenna "
                f"{params.ref_ant}. Red: the best-fit pure delay "
                f"(+/-300 ns brute-force search, `run_cal_diff.py`). Panel "
                f"titles carry the delay and the residual rms. Table: "
                f"`gain_delay_fits_{tag}.csv`.\n\n"
                "*Healthy*: the red line tracks the blue points across the "
                "whole band with residual rms well under ~0.5 rad. *Sick*: "
                "a large residual (structure a delay cannot describe) or a "
                "delay jumping by an ADC-sample multiple vs the last cal.",
                [png])
    step("Per-antenna gain phase with delay fit", _gainfits)

    # ---- 6. rank-1 ratio -----------------------------------------------
    if cal_cases:
        def _rank1():
            png = fig("rank1_vs_freq")
            meds = rd.plot_rank1(cal_cases, png,
                                 f"Fringe-stopped {params.cal_source} rank-1 "
                                 f"vs frequency, {len(ants)} ants, "
                                 f"{params.source_window[0]} - "
                                 f"{params.source_window[1]} UTC")
            report["rank1_medians"] = {k: list(v) for k, v in meds.items()}
            section("Rank-1 ratio vs frequency",
                    "Per-channel sigma_1/sigma_2 of the fringe-stopped "
                    "visibility matrix (phase-only SVD). Dashed line at 1.0, "
                    "dotted at the primary median.\n\n"
                    "*Healthy*: well above 1 across the band - the source "
                    "dominates and the channel is trustworthy. *Sick*: "
                    "hugging 1.0, or collapsing over a whole subband.",
                    [png])
        step("Rank-1 ratio vs frequency", _rank1)

    # ---- 7. singular values vs frequency -------------------------------
    if cal is not None and "singular_values" in cal:
        def _svd():
            png = fig("svd_vs_freq")
            _p, stats = rd.plot_svd_vs_freq(
                cal, png,
                f"SVD of the solve matrix, {len(ants)} ants, "
                f"{params.cal_source} {params.source_window[0]} - "
                f"{params.source_window[1]} UTC")
            report["svd_vs_freq"] = stats
            section("SVD singular values vs frequency",
                    "sigma_1..6 of the per-channel phase-only solve matrix "
                    "(left, log scale) and the rank-1 fraction "
                    "sigma_1/sum(sigma) (right), the "
                    "`svd_census/svd_vs_freq` view. The entries are "
                    "unit-modulus with a zeroed diagonal, so a perfectly "
                    "coherent point source gives sigma_1 = N-1 and a rank-1 "
                    "fraction of 0.5, not 1.\n\n"
                    "*Healthy*: sigma_1 pinned near N-1, far above "
                    "sigma_2..k, and the fraction near 0.5 in band. *Sick*: "
                    "the singular values bunch together and the fraction "
                    "falls toward the incoherent floor 1/N.",
                    [png])
        step("SVD singular values vs frequency", _svd)

    # ---- 8. fringe-stop waterfalls -------------------------------------
    if fs_primary is not None:
        def _waterfalls():
            fdir = os.path.join(figs_dir, f"fringe_{tag}")
            os.makedirs(fdir, exist_ok=True)
            fpngs = rd.plot_fringe_stopped(fs_primary, mapping, fdir,
                                           params.fringe_plot_max_baselines)
            section("Fringe-stopped visibility waterfalls",
                    "Per SNAP pair: raw phase, the geometric model, and the "
                    "fringe-stopped residual, against local time.\n\n"
                    "*Healthy*: raw and model show the same fringe pattern "
                    "and the residual is vertically banded (constant in "
                    "time). *Sick*: residual fringes still sloping in time "
                    "(bad positions/source) or a baseline that is pure "
                    "speckle (dead antenna).",
                    fpngs)
        step("Fringe-stopped visibility waterfalls", _waterfalls)

    # ---- 9. static A/B --------------------------------------------------
    if "static_ab" in report:
        ab = report["static_ab"]
        section("Static A/B",
                f"In-band rank-1 median with the static subtracted: "
                f"{ab['with_static_band']:.3f}; without: "
                f"{ab['no_static_band']:.3f}. Full band: "
                f"{ab['with_static_full']:.3f} vs {ab['no_static_full']:.3f}."
                f"\n\n*Healthy*: subtracting the static raises the in-band "
                f"rank-1 median. If it lowers it, the static window was not "
                f"quiet.", [])

    # ---- 10. cal diff vs the previous cal -------------------------------
    if params.prev_cal_path:
        def _caldiff():
            png = fig("cal_diff")
            csv = os.path.join(params.out_dir, f"cal_diff_{tag}.csv")
            p, loss = rd.plot_cal_diff(cal_file, params.prev_cal_path,
                                       params.ref_ant, png, csv)
            if p:
                report["cal_diff_band_avg_coherence"] = loss
                section("Per-antenna phase and delay diff vs the previous cal",
                        f"New cal minus `{params.prev_cal_path}`, referenced "
                        f"to antenna {params.ref_ant}. Band-averaged "
                        f"predicted coherent beam power if the OLD cal were "
                        f"kept: {loss:.3f}. Table: `cal_diff_{tag}.csv`.\n\n"
                        "*Healthy*: differences flat and small; predicted "
                        "coherence near 1 means nothing moved. *Sick*: clean "
                        "delay slopes per antenna (something re-cabled or an "
                        "ADC slipped) - deploying the new cal is then "
                        "necessary, not optional.", [png])
        step("Cal diff vs previous cal", _caldiff)

    # ---- 11. beamformed response on a bright source ---------------------
    if params.beam_check:
        def _beam():
            tried = []
            for src in [params.beam_check_source, params.beam_check_fallback]:
                if not src or src in tried:
                    continue
                tried.append(src)
                win = _beam_check_window(params, src)
                if win is None:
                    continue
                print(f"  beam check on {src}: {win[0]} -> {win[1]} UTC",
                      flush=True)
                res = rd.beamform_source_check(
                    cal_file, params.layout_csv, ants, src, win,
                    fig(f"beam_check_{src.replace('-', '')}"),
                    prev_cal_file=params.prev_cal_path,
                    off_alt_deg=params.beam_check_off_alt_deg,
                    min_alt_deg=params.min_alt_deg)
                if "skipped" in res:
                    print(f"    SKIP: {res['skipped']}", flush=True)
                    continue
                report["beam_check"] = res
                lines = "; ".join(f"{k} = {v['coh']:+.4f}"
                                  for k, v in res["summary"].items())
                print(f"    {lines}", flush=True)
                section(f"Beamformed response on {src}",
                        f"Coherent beam steered at {src} through the "
                        f"available span with the just-solved cal, in the "
                        f"visibility domain "
                        f"(`P = 2 Re sum_ij w_i w_j* V_ij`, normalised by "
                        f"`2 sum |V_ij|` so 1.0 is a perfectly phased point "
                        f"source). Window {win[0]} - {win[1]} UTC, "
                        f"{res['n_integrations']} integrations, altitude "
                        f"{res['alt_range'][0]:.1f}-{res['alt_range'][1]:.1f} "
                        f"deg. Band-averaged: {lines}.\n\n"
                        "*Healthy*: the new cal is clearly above zero and "
                        "beats both the off-source null and the previous "
                        "cal. *Sick*: on-source sitting at the null level - "
                        "the cal does not phase the array on an independent "
                        "source. Caveat: the null is a rough control, not a "
                        "zero - an altitude-offset beam can still land on "
                        "real sky structure, so read it as a scale, not as "
                        "a floor.",
                        [res["png"]])
                return
            section(f"Beamformed response on {params.beam_check_source}",
                    f"**SKIPPED** - no usable window for "
                    f"{' or '.join(tried) or 'the requested source'}: the "
                    f"visibilities are not on disk, or the source does not "
                    f"rise above {params.min_alt_deg} deg there. Set "
                    f"`beam_check_window` explicitly to force one.", [])
        step("Beamformed source check", _beam)

    # ---- 12. beam grid + transit coverage -------------------------------
    if weights_file:
        def _grid():
            png = fig("beam_grid")
            rd.plot_grid_map(weights_file, png, report.get("nearest_beams", []),
                             f"Beam grid {tag}: {report['pointings']['n']} "
                             f"pointings, alt "
                             f"{report['pointings']['alt_min']:.1f}-"
                             f"{report['pointings']['alt_max']:.1f} deg")
            section("Beam grid",
                    "Alt/az of every beam, coloured by beam index; red stars "
                    "are the nearest-beam targets.\n\n"
                    "*Healthy*: the targets sit inside the beam cloud, well "
                    "under a beam FWHM from the nearest beam. *Sick*: a "
                    "target outside the grid, or a visible hole in the "
                    "coverage.", [png])
        step("Beam grid", _grid)

        def _transit():
            date = (params.transit_date or params.nearest_date
                    or params.track_date
                    or (str(params.source_window[0])[:10]
                        if params.source_window else None))
            if date is None:
                raise ValueError("no date for the transit plot; set "
                                 "transit_date")
            png = fig("source_transit")
            rd.plot_transit_coverage(weights_file, params.transit_sources,
                                     date, png)
            report["transit_plot"] = dict(date=date,
                                          sources=list(params.transit_sources))
            section("Source-transit coverage of the new grid",
                    f"`bf_weights_generator.plot_source_transit` (the same "
                    f"machinery as `casm-bf-source-transit`) for "
                    f"{', '.join(params.transit_sources)} on {date}, local "
                    f"time.\n\n"
                    "*Healthy*: each source of interest passes through beams "
                    "for a usable stretch of the day. *Sick*: a source that "
                    "never enters a beam - the grid is pointed somewhere "
                    "else.", [png])
        step("Source-transit coverage", _transit)


# ---------------------------------------------------------------------------
# Step 5: diagnostics notebook
# ---------------------------------------------------------------------------

def write_notebook(params: RecipeParams, nb_path, sections, provenance):
    """Build cal_<tag>_diagnostics.ipynb and execute it so it reads as a report.

    Cells reference the PNGs by path relative to the notebook, and execution
    embeds the images as base64 outputs, so the file survives being copied
    away from out_dir.
    """
    try:
        import nbformat as nbf
    except ImportError as exc:
        print(f"  notebook SKIPPED: nbformat not importable ({exc})",
              flush=True)
        return None

    here = os.path.dirname(os.path.abspath(nb_path))
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(
        f"# CASM cal + weights diagnostics: `{params.tag}`\n\n"
        f"Generated by `bf_weights_generator/make_cal_and_weights.py` "
        f"(canonical recipe, see `docs/canonical-recipe.md`). Every figure is "
        f"described under its own heading, with what a healthy and a sick "
        f"version look like. All time axes are local time at OVRO.\n\n"
        f"## Provenance\n\n```json\n"
        f"{json.dumps(provenance, indent=2, default=str)}\n```\n\n"
        f"## Parameters\n\n```json\n"
        f"{json.dumps(asdict(params), indent=2, default=str)}\n```\n")]
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "from IPython.display import Image, display\n"
        f"os.chdir({here!r})\n"
        "print(os.getcwd())"))
    for title, text, pngs in sections:
        md = f"## {title}\n"
        if text:
            md += f"\n{text}\n"
        cells.append(nbf.v4.new_markdown_cell(md))
        if pngs:
            rel = [os.path.relpath(p, here) for p in pngs]
            cells.append(nbf.v4.new_code_cell(
                "for p in %s:\n"
                "    if os.path.exists(p):\n"
                "        display(Image(filename=p))\n"
                "    else:\n"
                "        print('MISSING FIGURE:', p)" % repr(rel)))
    nb["cells"] = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3",
                                 "language": "python", "name": "python3"}
    if params.execute_notebook:
        try:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=900, kernel_name="python3",
                           allow_errors=True,
                           resources={"metadata": {"path": here}}).execute()
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

def _write_report(report, path):
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return path


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
    validate_params(params, ants)
    ant = mapping.with_inactive(sorted(all_active - set(ants)))
    assert sorted(ant.active_antennas()) == ants
    print(f"[setup] layout {params.layout_csv}\n"
          f"  {len(ants)} antennas: {ants}\n  ref_ant {params.ref_ant}",
          flush=True)

    out = dict(out_dir=params.out_dir, tag=params.tag)
    report = dict(params=asdict(params), antennas=ants,
                  versions=dict(numpy=np.__version__,
                                python=sys.version.split()[0]))
    rep_path = os.path.join(params.out_dir, f"report_{params.tag}.json")
    sections = []
    fs_primary = None
    solve_diag = None
    cal = None
    cal_cases = []

    # ---------------- 1. calibration -----------------------------------
    if params.cal_path:
        cal_file = params.cal_path
        print(f"[cal] reusing {cal_file} (no solve)", flush=True)
        report["cal_reused"] = True
    else:
        static = None
        if params.static_path is not None:
            print("[static] building/loading off-source static", flush=True)
            static = build_manual_static(
                params.static_path, params.static_window,
                params.static_notes or
                f"static for cal tag {params.tag}.", fmt,
                tol_s=params.static_window_tol_s)
        with registered_source(params) as source:
            cal, nt, fs_primary, solve_diag = solve_calibration(
                params, mapping, ant, source,
                static_vis=None if static is None else static["static_vis"],
                label="primary", collect_diag=params.diagnostics)
            cal_cases.append((np.asarray(cal["freqs_mhz"]),
                              np.asarray(cal["rank1_ratios"]),
                              f"{params.cal_source}, "
                              f"{'static-subtracted' if static is not None else 'no static'}",
                              "tab:green"))
            if static is not None and params.diagnostics:
                # A/B: the same window without the static, for the figure.
                cal0, _, _, _ = solve_calibration(params, mapping, ant, source,
                                                  static_vis=None,
                                                  label="nostatic-AB")
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
                print(f"  static A/B: with "
                      f"{report['static_ab']['with_static_band']:.3f} "
                      f"vs without {report['static_ab']['no_static_band']:.3f} "
                      f"(in-band rank-1 median)", flush=True)

        from casm_calibrator import save_calibration
        cal_file = os.path.join(params.out_dir, f"cal_{params.tag}.h5")
        save_calibration(cal, cal_file, n_time_averaged=nt, overwrite=True)
        print(f"  wrote {cal_file}", flush=True)
        report["cal"] = dict(n_time_averaged=int(nt),
                             n_solved=int(np.asarray(cal["flags"]).sum()),
                             rank1_median=float(np.nanmedian(
                                 np.asarray(cal["rank1_ratios"]))),
                             time_on_source_frac=solve_diag["time_mask_frac"])
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
    else:
        # No weights to verify, but the cal itself still has to be sound.
        print("\n[verify]", flush=True)
        check_cal_subbands(params, cal_file, report)

    # The products exist and are verified at this point: write the report
    # before anything cosmetic can fail.
    _write_report(report, rep_path)
    out["report_json"] = rep_path
    print(f"\n[report] {rep_path}", flush=True)

    # ---------------- 3. diagnostics ------------------------------------
    if params.diagnostics:
        print("\n[diagnostics]", flush=True)
        try:
            run_diagnostics(params, mapping, ant, ants, cal, cal_file,
                            fs_primary, solve_diag, cal_cases, weights_file,
                            report, sections)
        except Exception as exc:                       # pragma: no cover
            print(f"  DIAGNOSTICS BLOCK FAILED ({type(exc).__name__}: {exc}); "
                  f"products are already written", flush=True)
            traceback.print_exc()
            report["diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    # ---------------- 4. notebook + final report -------------------------
    if params.notebook and sections:
        provenance = dict(driver=os.path.abspath(__file__),
                          cal_file=cal_file, weights_file=weights_file,
                          layout=params.layout_csv, antennas=ants,
                          grid=grid_meta,
                          report_json=rep_path,
                          numpy=np.__version__,
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
        if write_notebook(params, nb_path, sections, provenance):
            out["notebook"] = nb_path

    _write_report(report, rep_path)
    out["report"] = report
    print("MAKE_CAL_AND_WEIGHTS_DONE", flush=True)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_FIELD_TYPES = None


def _field_type(name):
    global _FIELD_TYPES
    if _FIELD_TYPES is None:
        _FIELD_TYPES = {f.name: str(f.type) for f in fields(RecipeParams)}
    if name not in _FIELD_TYPES:
        raise SystemExit(f"unknown parameter: {name}. Known: "
                         f"{', '.join(sorted(_FIELD_TYPES))}")
    return _FIELD_TYPES[name]


def _typed(name, value):
    """Coerce an already-parsed value to the declared field type.

    Applied to BOTH ``--param`` and ``--config`` values so a config file that
    says ``"mult_min": 0`` or ``"source_window": [...]`` produces exactly what
    a Python caller would have passed.
    """
    t = _field_type(name)
    if value is None:
        if "Optional" not in t:
            raise SystemExit(f"{name} is not optional (type {t})")
        return None
    if t == "bool":
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if t in ("Optional[Tuple[str, str]]", "Tuple[str, str]"):
        v = list(value)
        if len(v) != 2:
            raise SystemExit(f"{name} needs exactly two entries, got {v!r}")
        return (str(v[0]), str(v[1]))
    if t in ("Sequence[Tuple[str, float, float]]",):
        return tuple((str(a), float(b), float(c)) for a, b, c in value)
    if t in ("Optional[Sequence[int]]", "Sequence[int]"):
        return tuple(int(x) for x in value)
    if t in ("Sequence[str]",):
        return tuple(str(x) for x in value)
    if t == "float":
        return float(value)
    if t == "int":
        return int(value)
    if t in ("str", "Optional[str]"):
        return str(value)
    return value


def _coerce(name, value):
    """Parse a --param VALUE string into the dataclass field's type."""
    t = _field_type(name)
    if t == "bool":
        return str(value).lower() in ("1", "true", "yes", "on")
    if value == "None":
        return _typed(name, None)
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value
    return _typed(name, parsed)


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

    kwargs = {}
    if args.config:
        with open(args.config) as fh:
            for k, v in json.load(fh).items():
                kwargs[k] = _typed(k, v)
    for name, value in args.param:
        kwargs[name] = _coerce(name, value)
    params = RecipeParams(**kwargs)
    if args.print_params:
        print(json.dumps(asdict(params), indent=2, default=str))
        return 0
    run(params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
