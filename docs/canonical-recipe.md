# Canonical cal + weights recipe (`make_cal_and_weights.py`)

`bf_weights_generator/make_cal_and_weights.py` is THE entrypoint for building a
calibration and a 512-beam int8 weights file. It orchestrates existing library
functions (`casm_io`, `casm_vis_analysis`, `casm_calibrator`,
`bf_weights_generator`) and contains no signal-processing algorithm of its own.
It never uploads: the deploy command is printed for the operator.

Everything a run touches is parameterised in one `RecipeParams` dataclass.
Anything not in that dataclass is fixed policy.

## Invocation

There is no console entry point. `pyproject.toml` carried unrelated
uncommitted edits when this driver was added, so it was left untouched.

```bash
/home/casm/software/dev/casm_venvs/casm_offline_env/bin/python \
  -m bf_weights_generator.make_cal_and_weights --help
```

Parameters come from a JSON file, from repeatable `--param NAME VALUE`
overrides (VALUE is parsed as a Python literal), or both (`--param` wins):

```bash
python -m bf_weights_generator.make_cal_and_weights \
  --config params.json --param tag "'20260820b'" --param alt_min_deg 25.0
```

`--print-params` dumps the resolved parameter block and exits.

From Python:

```python
from bf_weights_generator.make_cal_and_weights import RecipeParams, run
out = run(RecipeParams(out_dir="/mnt/nvme5/...", tag="20260820", ...))
```

## Fixed policy (not knobs)

| item | value |
| --- | --- |
| frequency config | `FrequencyConfig.layout_64ant()` |
| solve | full band, per channel: `SVDConfig(threshold=1.0, PHASE_ONLY, block_size=1, masked_band_strategy="zero")` |
| fringe stop | `sign=-1` |
| cal save | `casm_calibrator.save_calibration` (h5) |
| int8 | `to_int8(scale_factor=127.0)`, `[..., :64]` slice |
| weights save | `save_int8_weights_hdf5`, `freq_order="descending"` |
| trim | stable `argsort` on altitude, first `n_beams` kept in file order |

## Knobs

Cal solve: `cal_source`, `cal_ra`/`cal_dec`, `source_window`, `static_window`,
`static_path`, `static_window_tol_s`, `static_notes`, `antennas`, `ref_ant`
(default 9), `layout_csv`, `min_alt_deg`, `min_time_on_source_frac`,
`cal_path` (reuse an existing cal h5 and skip the solve).

Grid: `n_beams` (512 in every build so far, but a knob), `grid_mode` (`bounds`
or `track`), `alt_min_deg`/`alt_max_deg`/`az_min_deg`/`az_max_deg`,
`mult_min`/`mult_max`/`mult_n`, `mult_break_below_target`, `trim`
(`lowest`/`highest`), `track_sources`/`track_date`/`track_min_alt_deg`/
`track_pad_deg`.

Verification: `verify_pointing`, `verify_beams`, `min_subband_good_frac`,
`nearest_targets`, `nearest_sources`/`nearest_date`.

Outputs and diagnostics: `out_dir`, `tag`, `make_weights`, `prev_cal_path`,
`diagnostics`, `notebook`, `execute_notebook`, `fringe_plot_max_baselines`,
`phase_baselines_per_class`, `beam_check`, `beam_check_source`,
`beam_check_fallback`, `beam_check_date`, `beam_check_window`,
`beam_check_minutes`, `beam_check_off_alt_deg`, `transit_sources`,
`transit_date`.

### What aborts a run

The driver refuses to produce a product it cannot vouch for:

* the calibrator source is above `min_alt_deg` for less than
  `min_time_on_source_frac` of the solve window (`fringe_stop` silently falls
  back to an all-ones time mask when the source never rises, so this is
  checked explicitly rather than trusted);
* an existing `static_path` records a window more than `static_window_tol_s`
  away from `static_window` (the tolerance is not zero: the realised window is
  the first/last integration inside the request, which observation boundaries
  can push a minute or two away from it);
* `cal_path` is combined with `source_window` / `static_window` /
  `static_path` / `cal_ra` / `cal_dec` (two intents in one config);
* `ref_ant` is not in the antenna set;
* any 512-channel subband of the cal has a solved, non-zero fraction below
  `min_subband_good_frac` (a dead subband means the F-engine delivered nothing
  there; see casm-wiki `missing-subbands-diagnostic`);
* the int8 payload at channel 1500 is all zero (the file is deleted first);
* the cal-division pointing fit fails (`verify_pointing=False` to opt out).

Diagnostics are the exception: every figure is wrapped, and a figure that
raises becomes a `**SKIPPED**` section in the notebook instead of losing a
product that is already written and verified.

### Ad-hoc calibrator coordinates

`cal_ra`/`cal_dec` register `cal_source` into
`casm_vis_analysis.sources.CATALOG` for the duration of the solve and remove
it again afterwards, success or failure. They are rejected for
`cal_source="sun"` (the sun is computed from ephemeris before the catalog is
consulted, so the coordinates would be silently ignored) and for any name
already in the catalog (overwriting a built-in J2000 position would leak into
every other consumer in the process).

### Sources

`cal_source` is passed to `casm_vis_analysis.sources`, which knows `sun`,
`cyg-a`, `cas-a`, `tau-a`, `vir-a`, `b0329+54` (dashes, plus signs and spaces
are normalised to underscores). For anything outside that catalog, set
`cal_ra` and `cal_dec` as well; the driver registers the coordinate under
`cal_source` for the run, because `fringe_stop` only accepts catalog names.

### Grid modes

`bounds` lays the grid on an explicit alt/az box. The spacing search is the
proven one: for each `mult` in `np.linspace(mult_min, mult_max, mult_n)` build
`generate_beam_grid_altaz(spacing_ew_deg=mult*fwhm_ew, spacing_ns_deg=mult*fwhm_ns)`,
keep the grid with the FEWEST beams that still reaches `n_beams`, then
`np.argsort(kind="stable")` on altitude and keep the first `n_beams` indices in
file order (`trim="lowest"` keeps the low beams, `"highest"` the high ones).
FWHM comes from `compute_beam_fwhm(arr.active_positions, freq_hz=450e6)`.

Reproducing `run_drill_aug15.py` (the Aug-15 build): its loop used
`spacing = m * fwhm / 2` over `np.linspace(0.5, 3.0, 51)`, stopped at the first
multiplier that undershot 512 beams, and kept the HIGHEST altitudes. In this
driver's parameterisation the multipliers are **halved**, i.e.
`mult_min=0.25, mult_max=1.5, mult_n=51, mult_break_below_target=true,
trim="highest"`.

`track` derives that box from named source tracks: sample each source every
minute over `track_date` (UTC), keep samples above `track_min_alt_deg`, take
the alt/az bounding box (azimuth unwrapped about the circular mean), pad by
`track_pad_deg`, then run the bounds-mode search on it. A box wider than 350
deg in azimuth is replaced by the full 0-360 range so the grid generator uses
its full-azimuth branch.

TODO: placing beams ALONG a track rather than on its bounding box. No helper
in `bf_weights_generator` does that today, and this driver adds no new
geometry, so it is not implemented. Use `bounds`/`track` and check the
nearest-beam table.

## What a run produces in `out_dir`

* `cal_<tag>.h5` (unless `cal_path` was reused) and `rank1_vs_freq_<tag>.npz`
* `static_<tag>.npz` when `static_window` is set and `static_path` is unset
* `weights_<tag>_<N>ant_<nbeams>_int8.h5`
* `report_<tag>.json`: parameters, package versions, grid metadata, pointing
  stats, int8 sanity, per-subband cal occupancy, pointing-fit rows,
  nearest-beam table, static A/B medians, delay-fit and beam-check summaries.
  Written as soon as the products are verified, then rewritten after the
  diagnostics, so a figure that raises cannot lose it.
* `cal_diff_<tag>.csv`, `gain_delay_fits_<tag>.csv`
* `figs/` (see the table below) and `figs/fringe_<tag>/fringe_diag_snap*.png`
* `cal_<tag>_diagnostics.ipynb`: markdown header with parameters, file paths
  and provenance, then one section per figure — each with a one-line "healthy
  vs sick" note — and executed cells displaying the PNGs. Images are embedded
  as base64 outputs, so the notebook reads as a report without being re-run
  and survives being copied out of `out_dir`.

### Diagnostic figures

All drawing lives in `bf_weights_generator/recipe_diagnostics.py`; nothing in
it can touch a product byte. Every time axis is local time at OVRO
(`America/Los_Angeles`, labelled "Local Time (OVRO)"), matching
`casm_vis_analysis.solar_waterfall`; UTC appears in titles only.

| figure in `figs/` | what it shows | ported from |
| --- | --- | --- |
| `autocorr_<tag>.png` | per-antenna autocorrelation spectra over the solve window (raw, before static subtraction) | `casm_vis_analysis.plotting.autocorr` |
| `phase_raw_sawtooth_<tag>.png` | raw wrapped cross-phase vs frequency, short/medium/long ref-ant baselines, with the net turns across the band | new panel over `fs['vis']` |
| `phase_stage1_raw_<tag>.png` | the same baselines, unwrapped | `casm_vis_analysis.plotting.phase_freq` |
| `phase_stage2_fringe_stopped_<tag>.png` | after fringe-stopping | same |
| `phase_stage3_calibrated_<tag>.png` | after fringe-stopping **and** dividing the solved cal out (`V_ij conj(g_i) g_j`) | same |
| `gain_delay_fits_<tag>.png` | solved gain phase per antenna with the fitted delay overlaid; delay + residual rms in each panel title | `cal_diff/run_cal_diff.py` |
| `rank1_vs_freq_<tag>.png` | rank-1 ratio vs frequency (all solve cases) | `svd_census/make_plots_aug19.py` |
| `svd_vs_freq_<tag>.png` | sigma_1..6 of the solve matrix + rank-1 fraction | `svd_census/run_census_aug19.py` |
| `fringe_<tag>/fringe_diag_snap*.png` | raw / geometric / fringe-stopped waterfalls per SNAP pair | `casm_vis_analysis.plotting.fringe_diag` |
| `cal_diff_<tag>.png` | per-antenna phase and delay diff vs `prev_cal_path` | `cal_diff/run_cal_diff.py` |
| `beam_check_<source>_<tag>.png` | coherent-beam response on a bright independent source (default Cyg A, falls back to Cas A) vs time and vs frequency, new cal / previous cal / off-source null | `cyga_cal_test/run_task1_cyga.py`, time panel from `visbeam_test/plot_coh_vs_time.py` |
| `beam_grid_<tag>.png` | alt/az of every beam with the nearest-beam targets | this driver |
| `source_transit_<tag>.png` | source tracks through the new grid for `transit_sources` | `bf_weights_generator.plot_source_transit` (the `casm-bf-source-transit` machinery) |

The beam check needs the visibilities for its window on disk. When they are
not there, or the source never rises, the notebook gets a `**SKIPPED**`
section stating why instead of failing the run. Sky images are deliberately
not produced: the question this notebook answers is "is this cal good", not
"what is on the sky".

Verification printed by every run: pointings table stats, int8 payload sanity
at channel 1500, per-subband cal occupancy, cal-division pointing coherence on
`verify_beams` (`bf_weights_generator/recipe_verify.py`, vendored from
`/mnt/nvme5/solar0819/pointing_verify/verify_pointing.py` so a swept scratch
directory cannot break a run), the nearest-beam table, and the dry-run /
deploy / `casm-bf-source-transit` commands.

---

## Example 1: today's solar-track build (Aug-19, 16 antennas)

Solve on the hour after solar transit + 0.74 h, no static, then a dense grid on
the sun's remaining track (alt 0-33, az 258-292), trimmed to the 512 lowest.
This reproduces `weights_aug19_16ant_solartrack_512_int8_CAL0819.h5`.

```json
{
  "out_dir": "/mnt/nvme5/solar0819/newcal_build",
  "tag": "aug19_solartrack",
  "cal_source": "sun",
  "source_window": ["2026-08-19 20:41:30", "2026-08-19 21:41:30"],
  "antennas": [9, 10, 15, 19, 22, 23, 24, 26, 30, 32, 36, 38, 40, 42, 44, 45],
  "ref_ant": 9,
  "grid_mode": "bounds",
  "alt_min_deg": 0.0, "alt_max_deg": 33.0,
  "az_min_deg": 258.0, "az_max_deg": 292.0,
  "mult_min": 0.05, "mult_max": 0.60, "mult_n": 56,
  "trim": "lowest",
  "verify_beams": [30, 83, 353],
  "nearest_sources": ["sun"], "nearest_date": "2026-08-19",
  "prev_cal_path": "/home/casm/scratch/solar_weights_20260815/cal_sun_2026-08-15_thr1_phaseonly_17ant_no33.h5"
}
```

Measured: 16-ant FWHM 18.79 x 4.02 deg at 450 MHz, `mult=0.170` (519 beams
pre-trim) -> 512 beams, alt 0.00-32.78, az 258.0-292.0, spacing 3.19 x 0.68 deg;
solve 3072/3072 channels, rank-1 median 4.16, 27 integrations.

## Example 2: the aug20 alt-20-90 build (night-static cal, all-sky grid)

Same sun window, minus the Aug-20 02:45-03:15 UT night static, then an all-sky
grid on alt 20-90 that serves both the sun and B0329+54 the next day. Because
`static_window` is set, the driver also solves the same window without the
static and puts both curves on the rank-1 figure.

```json
{
  "out_dir": "/mnt/nvme5/solar0819/recipe_demo_20260820",
  "tag": "20260820",
  "cal_source": "sun",
  "source_window": ["2026-08-19 20:41:30", "2026-08-19 21:41:30"],
  "static_window": ["2026-08-20 02:45", "2026-08-20 03:15"],
  "static_path": "/mnt/nvme5/solar0819/cal_static_night/static_aug20_0245_0315UT.npz",
  "antennas": [9, 10, 15, 19, 22, 23, 24, 26, 30, 32, 36, 38, 40, 42, 44, 45],
  "ref_ant": 9,
  "grid_mode": "bounds",
  "alt_min_deg": 20.0, "alt_max_deg": 90.0,
  "mult_min": 0.40, "mult_max": 0.90, "mult_n": 51,
  "trim": "lowest",
  "verify_beams": [0, 256, 511],
  "nearest_sources": ["sun", "b0329+54"], "nearest_date": "2026-08-20",
  "prev_cal_path": "/home/casm/scratch/solar_weights_20260815/cal_sun_2026-08-15_thr1_phaseonly_17ant_no33.h5"
}
```

Measured 2026-08-19 (`/mnt/nvme5/solar0819/recipe_demo_20260820/`): in-band
rank-1 median 9.51 with the static, 6.03 without; `mult=0.600` (514 pre-trim)
-> 512 beams, alt 20.00-85.07, spacing 11.27 x 2.41 deg; int8 at ch1500
min -127 max 127 mean|v| 19.92; pointing fit on beams 0/256/511 gives
`coh_stated` 1.0000/1.0000/1.0000 and |dalt| <= 0.06 deg; nearest beam to the
B0329 transit is beam 478 at 0.46 deg, to the sun transit beam 435 at 1.62 deg;
band-averaged predicted coherent beam power against the Aug-15 cal is 0.266
(that cal is stale, which is why this build exists).

Re-run 2026-08-20 with the full diagnostics set (same params, in
`params_20260820.json` next to the products): sun above 10 deg for 27/27
integrations; all six cal subbands 512/512 solved; median per-antenna delay-fit
residual 0.195 rad with delays out to -224 ns; solve-matrix sigma_1 median
14.79 of a possible 15 and rank-1 fraction 0.50 in band; Cyg A beam check over
05:29-06:29 UTC (86 deg altitude, 27 integrations, 2425 channels) gives
coherence +0.0703 +/- 0.0026 for the new cal, +0.0105 +/- 0.0032 for the stale
Aug-15 cal and +0.0188 for the 25-deg-off null.

## Example 3: a future Cyg A calibration

Cyg A instead of the sun. Same fixed policy; only the source, the window and
the grid change. Use `grid_mode="track"` to put the grid on Cyg A's own track,
and reuse the resulting cal for later grid-only rebuilds via `cal_path`.

```json
{
  "out_dir": "/mnt/nvme5/cyga_cal_20260901",
  "tag": "cyga_20260901",
  "cal_source": "cyg-a",
  "source_window": ["2026-09-01 06:00:00", "2026-09-01 08:00:00"],
  "static_window": ["2026-09-01 12:45:00", "2026-09-01 13:15:00"],
  "antennas": [9, 10, 15, 19, 22, 23, 24, 26, 30, 32, 36, 38, 40, 42, 44, 45],
  "ref_ant": 9,
  "min_alt_deg": 30.0,
  "grid_mode": "track",
  "track_sources": ["cyg-a"], "track_date": "2026-09-01",
  "track_min_alt_deg": 30.0, "track_pad_deg": 3.0,
  "mult_min": 0.10, "mult_max": 0.90, "mult_n": 81,
  "trim": "highest",
  "verify_beams": [0, 256, 511],
  "nearest_sources": ["cyg-a", "cas-a"], "nearest_date": "2026-09-01",
  "prev_cal_path": "/mnt/nvme5/solar0819/cal_static_night/cal_sun_2026-08-19_thr1_phaseonly_16ant_nightstaticB.h5"
}
```

A source that is not in the catalog (say a new calibrator field) needs its
coordinates:

```json
{"cal_source": "3c48", "cal_ra": "01h37m41.30s", "cal_dec": "+33d09m35.1s"}
```

Windows must be chosen so the source is above `min_alt_deg` for most of the
window. `fringe_stop` masks the samples below it — but if the source never
rises it masks nothing and quietly solves on noise, so the driver aborts when
the on-source fraction is below `min_time_on_source_frac` (default 0.5). The
fraction is printed and recorded in `report_<tag>.json`
(`cal.time_on_source_frac`).

Static windows need the same check on the OTHER bright sources. Cas A is
circumpolar at OVRO: its altitude never drops below about 6 deg, and that
minimum falls near 20:00 UTC in September when the sun is up at ~60 deg, so no
window is free of everything. On 2026-09-01 the least-bad night window is
about 12:45-13:15 UTC: Cyg A 9-13 deg, Cas A ~46 deg, sun below -10 deg. Run
`casm_vis_analysis.sources.source_altaz` over any candidate window before
trusting the static it produces.

Check the in-band rank-1 median in `report_<tag>.json` before deploying
anything built on a non-solar cal: no Cyg A cal has been solved through this
driver yet, so there is no reference number to compare against. The Cyg A
beam check in the diagnostics notebook is a useful independent referee — for
the Aug-20 demo build it read +0.070 for the new cal against +0.011 for the
stale Aug-15 cal.

---

## Byte-for-byte validation

Script: `tests/validate_recipe.py` (in this repo), last run 2026-08-20 after
the diagnostics and review-fix pass. Three existing products were rebuilt
through the driver into a
scratch directory and every HDF5 dataset compared with `np.array_equal` plus an
md5 of the dataset bytes.

| target | driver params | datasets compared | result |
| --- | --- | --- | --- |
| `newcal_build/cal_sun_2026-08-19_thr1_phaseonly_16ant.h5` | sun 20:41:30-21:41:30 UT, 16 ant, ref 9, no static | ant_ids, flags, freqs_hz, freqs_mhz, gains, rank1_ratios, weights | all 7 bit-exact |
| `cal_static_night/cal_sun_2026-08-19_thr1_phaseonly_16ant_nightstaticB.h5` | same window, minus static 2026-08-20 02:45-03:15 UT | same 7 | all 7 bit-exact |
| `newcal_build/weights_aug19_16ant_solartrack_512_int8_CAL0819.h5` | `cal_path` = the no-static cal, alt 0-33, az 258-292, mult 0.05-0.60/56, trim lowest | weights_int8, pointings/alt_deg, pointings/az_deg, array_config/{active_mask,antenna_ids,positions_enu}, frequencies_hz | all 7 bit-exact |

md5 of `weights_int8`: `6f560350a1718e9ea8a0051adfdf0522` (both files).
md5 of the no-static `gains`: `c891977215275cb5ba80f59e685d29e8`; of the
night-static `gains`: `7d3c5090db147e17e16c440dcce224d4`.

The T2 run pointed `static_path` at the existing `static_aug20_0245_0315UT.npz`,
which is what `build_manual_static` does on a rerun (it loads the file if it
exists, after checking the window it records). Rebuilding the static from raw
visibilities is the same `average_visibility(apply_freq_mask=True)` call over
the same window. That existing npz records 02:46:31-03:14:00 UTC, 91 s from
the requested start — which is why `static_window_tol_s` defaults to 300 and
not 60.

Run with numpy 2.4.2 / Python 3.13.5 (both recorded in
`report_<tag>.json["versions"]`). The altitude trim uses
`np.argsort(kind="stable")`; the three targets are bit-exact with it, so the
tie-breaking it pins was already what the reference builds got.

## Related

* `docs/recipes.md`: code-level API recipes for the individual steps.
* `docs/int8_weights.md`, `docs/ib_weights.md`, `docs/beam_grid.md`.
* casm-wiki `weights-and-deploy.md`: operational deploy procedure and the
  rules about `--save-defaults` and who runs `--upload`.
* IB weights are still a separate companion product
  (`gen_ib_from_cb.py`); this driver builds the CB file only.
