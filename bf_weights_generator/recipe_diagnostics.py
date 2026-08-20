"""Diagnostic figures for the canonical cal + weights recipe.

Everything ``make_cal_and_weights.py`` draws lives here, so the driver stays
an orchestrator and the plots can be reused from a notebook. No product byte
is written from this module: it only reads visibilities, cal h5 files and the
weights h5 file.

Where each figure comes from (ported, not re-invented):

===========================================  ==================================
figure                                       source
===========================================  ==================================
autocorrelation spectra                      ``casm_vis_analysis.plotting.autocorr``
                                             (old notebook sections 4 / "Auto-
                                             correlation spectra")
raw / fringe-stopped / calibrated phase      ``casm_vis_analysis.plotting.phase_freq``
                                             (old notebook "Fringe-stop toward Sun")
fringe-stop waterfalls                       ``casm_vis_analysis.plotting.fringe_diag``
per-antenna gain phase + delay fit           ``/mnt/nvme5/solar0819/cal_diff/run_cal_diff.py``
rank-1 ratio vs frequency                    ``/mnt/nvme5/solar0819/svd_census/make_plots_aug19.py``
singular values vs frequency                 ``/mnt/nvme5/solar0819/svd_census/run_census_aug19.py``
cal phase/delay diff vs a previous cal       ``/mnt/nvme5/solar0819/cal_diff/run_cal_diff.py``
beamformed response on a bright source       ``/mnt/nvme5/solar0819/cyga_cal_test/run_task1_cyga.py``
                                             (+ time-series panel from
                                             ``visbeam_test/plot_coh_vs_time.py``)
beam grid / source-transit coverage          ``bf_weights_generator.plot_transit``
===========================================  ==================================

Time-axis convention: every time axis is LOCAL time at OVRO
(``America/Los_Angeles``), labelled "Local Time (OVRO)", matching
``casm_vis_analysis.solar_waterfall``. UTC appears in titles only.
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

# ---------------------------------------------------------------------------
# Conventions shared with the driver
# ---------------------------------------------------------------------------

LOCAL_TZ = "America/Los_Angeles"
TIME_AXIS_LABEL = "Local Time (OVRO)"

#: Band the solar rank-1 medians are quoted over (svd_census convention).
SUNBAND = (435.0, 460.0)
#: Persistent narrow-band RFI, excluded from every fit and band average.
RFI_LINES = [(462.0, 464.0), (450.0, 452.0), (436.5, 438.5), (400.0, 402.0)]
#: Band the per-antenna delay fits use (cal_diff convention).
FITBAND = (398.0, 480.0)
#: Delay search grid, ns (cal_diff convention).
TAUS_NS = np.arange(-300.0, 300.0001, 0.02)

C_M_S = 299792458.0


def _plt():
    """matplotlib.pyplot with the Agg backend forced (headless runs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Masks and fits (verbatim numerics from svd_census / cal_diff)
# ---------------------------------------------------------------------------

def band_mask(freq, band=SUNBAND, lines=RFI_LINES):
    """In-band, RFI lines removed (svd_census/run_static_night convention)."""
    b = (freq > band[0]) & (freq < band[1])
    for lo, hi in lines:
        b &= ~((freq > lo) & (freq < hi))
    return b


def fit_mask(freq, *goods):
    """Full fit band minus RFI lines, ANDed with any extra good-channel masks."""
    m = (freq > FITBAND[0]) & (freq < FITBAND[1])
    for lo, hi in RFI_LINES:
        m &= ~((freq > lo) & (freq < hi))
    for gd in goods:
        m &= gd
    return m


def delay_fit(dphi, freq_mhz, m):
    """run_cal_diff.py::delay_fit, verbatim.

    Returns (tau_ns, residual_rms_rad, coherence_after_delay).
    """
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


def delay_fit_phase0(dphi, freq_mhz, m, tau_ns):
    """Constant phase offset of the fitted delay line (for overlaying it)."""
    f = freq_mhz[m] * 1e6
    z = np.exp(1j * dphi[m])
    return float(np.angle(np.mean(z * np.exp(-2j * np.pi * tau_ns * 1e-9 * f))))


def _roll(r, win=32, minp=8):
    import pandas as pd
    return pd.Series(r).rolling(win, center=True, min_periods=minp).median()


# ---------------------------------------------------------------------------
# Local-time helpers
# ---------------------------------------------------------------------------

def local_datetimes(time_unix, tz=LOCAL_TZ):
    """Unix seconds -> tz-aware datetimes in the OVRO local zone."""
    zone = ZoneInfo(tz)
    return [datetime.fromtimestamp(float(t), tz=zone) for t in
            np.asarray(time_unix, dtype=float)]


def format_local_axis(ax, tz=LOCAL_TZ, label=True, rotate=30):
    """Apply the casm-solar-waterfall local-time formatting to a date axis."""
    import matplotlib.dates as mdates
    zone = ZoneInfo(tz)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=zone))
    if label:
        ax.set_xlabel(TIME_AXIS_LABEL)
    for lab in ax.get_xticklabels():
        lab.set_rotation(rotate)
        lab.set_ha("right")


def localize_hours_axis(fig, t0_unix, tz=LOCAL_TZ):
    """Relabel an "hours since t0" x-axis as local clock time.

    ``casm_vis_analysis.plotting.fringe_diag`` plots hours-since-start; the
    operator convention is local wall-clock, so the tick labels (not the data)
    are rewritten here and the shared x-label replaced.
    """
    from matplotlib.ticker import FuncFormatter
    zone = ZoneInfo(tz)

    def _fmt(x, _pos):
        return datetime.fromtimestamp(float(t0_unix) + x * 3600.0,
                                      tz=zone).strftime("%H:%M:%S")

    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
        for lab in ax.get_xticklabels():
            lab.set_rotation(30)
            lab.set_ha("right")
    fig.supxlabel(TIME_AXIS_LABEL)
    return fig


def utc_local_title(time_unix, tz=LOCAL_TZ):
    """"HH:MM:SS-HH:MM:SS PDT (UTC ...)" one-liner for figure titles."""
    zone = ZoneInfo(tz)
    t0 = datetime.fromtimestamp(float(time_unix[0]), tz=zone)
    t1 = datetime.fromtimestamp(float(time_unix[-1]), tz=zone)
    u0 = datetime.fromtimestamp(float(time_unix[0]), tz=ZoneInfo("UTC"))
    u1 = datetime.fromtimestamp(float(time_unix[-1]), tz=ZoneInfo("UTC"))
    return (f"{t0:%Y-%m-%d %H:%M:%S}-{t1:%H:%M:%S} {t0:%Z} "
            f"(UTC {u0:%Y-%m-%d %H:%M:%S}-{u1:%H:%M:%S})")


# ---------------------------------------------------------------------------
# Data harvesting during the solve (read-only; never touches the product)
# ---------------------------------------------------------------------------

def collect_autocorrelations(data, ant):
    """Time series of the autocorrelations of every active antenna.

    Returns ``(autos, labels)`` where ``autos`` is (T, F, n_ant) taken
    straight out of the full upper-triangle visibility cube. Copies, so the
    caller can free the cube.
    """
    from casm_io.correlator.baselines import triu_flat_index
    vis = data["vis"] if hasattr(data, "__getitem__") else data.vis
    n_bl = vis.shape[-1]
    n_inputs = int((-1 + (1 + 8 * n_bl) ** 0.5) / 2)
    active = sorted(ant.active_antennas())
    idx = [triu_flat_index(n_inputs, ant.packet_index(a), ant.packet_index(a))
           for a in active]
    autos = np.asarray(vis[:, :, idx]).copy()
    labels = [f"Ant {a} | S{ant.snap_adc(a)[0]}A{ant.snap_adc(a)[1]}"
              for a in active]
    return autos, labels


# ---------------------------------------------------------------------------
# 1. Autocorrelation spectra
# ---------------------------------------------------------------------------

def plot_autocorr_spectra(autos, freq_mhz, labels, time_unix, out_png, title,
                          freq_mask=None, tz=LOCAL_TZ):
    """Per-antenna autocorrelation power spectrum (dB) on the solve window.

    Straight call into ``casm_vis_analysis.plotting.autocorr.plot_autocorr``,
    which is what the hand-vetted notebooks use.
    """
    from casm_vis_analysis.plotting.autocorr import plot_autocorr
    plt = _plt()
    flagged = None if freq_mask is None else ~np.asarray(freq_mask, bool)
    fig = plot_autocorr(np.asarray(autos), np.asarray(freq_mhz), labels,
                        time_avg=True, freq_mask=flagged, ncols=4,
                        time_unix=np.asarray(time_unix), snap_label=title,
                        time_tz=tz)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# 2-4. Phase vs frequency through the stages
# ---------------------------------------------------------------------------

def select_ref_baselines(fs, ant, n_per_class=3):
    """Short / medium / long reference-antenna baselines, sorted by length.

    ``fs['vis']`` holds exactly the ref<->target baselines, so the selection
    is an index into its last axis.
    """
    ref = int(fs["ref_ant"])
    targets = list(fs["target_aids"])
    df = ant.dataframe

    def pos(a):
        return df.loc[df["antenna_id"] == a, ["x_m", "y_m", "z_m"]].values[0]

    r0 = pos(ref)
    length = np.array([float(np.linalg.norm(pos(a) - r0)) for a in targets])
    order = np.argsort(length)
    n = len(order)
    k = max(1, min(n_per_class, n // 3 if n >= 3 else 1))
    mid = n // 2
    pick = sorted(set(list(order[:k])
                      + list(order[max(0, mid - k // 2):mid - k // 2 + k])
                      + list(order[-k:])),
                  key=lambda i: length[i])
    labels = [f"Ant {ref}x{targets[i]}\n|b|={length[i]:.1f} m" for i in pick]
    return list(pick), labels, length[pick]


def calibrated_baselines(fs, cal, sel):
    """Fringe-stopped ref<->target visibilities with the solved cal divided out.

    ``fs['vis'][..., k] = V_(ref, target_k)`` and the SVD model is
    ``V_ij = g_i conj(g_j)``, so the residual is ``V_ij conj(g_i) g_j`` and a
    good solution leaves it at zero phase across the band.
    """
    ants = [int(a) for a in np.asarray(cal["ant_ids"])]
    gains = np.asarray(cal["gains"])                      # (n_ant, n_chan)
    freq_cal = np.asarray(cal["freqs_mhz"])
    freq = np.asarray(fs["freq_mhz"])
    if not (freq_cal.shape == freq.shape and np.allclose(freq_cal, freq)):
        raise ValueError("cal and fringe-stop frequency axes differ")
    ref = int(fs["ref_ant"])
    targets = list(fs["target_aids"])
    g_ref = gains[ants.index(ref)]
    corr = np.stack([np.conj(g_ref) * gains[ants.index(targets[k])]
                     for k in sel], axis=-1)              # (n_chan, n_sel)
    out = np.asarray(fs["vis_stopped"])[:, :, sel] * corr[None, :, :]
    # Channels with no solution (gain zeroed) would plot as a phase of 0;
    # NaN them so they read as gaps instead of a fake flat residual.
    bad = np.abs(corr) == 0
    out = np.where(bad[None, :, :], np.nan + 1j * np.nan, out)
    return out


def plot_phase_stage(vis_sel, freq_mhz, labels, out_png, time_unix,
                     unwrap=True, tz=LOCAL_TZ):
    """One stage of the phase-vs-frequency diagnostic (one column per stage)."""
    from casm_vis_analysis.plotting.phase_freq import plot_phase_vs_freq
    plt = _plt()
    panels = [("phase vs frequency", np.asarray(vis_sel))]
    figs = plot_phase_vs_freq(panels, np.asarray(freq_mhz),
                              baseline_labels=labels, unwrap=unwrap,
                              output_path=None,
                              time_unix=np.asarray(time_unix),
                              split_max=len(labels), time_tz=tz)
    fig = figs[0]
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    for f in figs:
        plt.close(f)
    return out_png


def plot_sawtooth(fs, sel, labels, lengths, out_png, title, tz=LOCAL_TZ):
    """Raw (wrapped) cross-phase vs frequency, all selected baselines overlaid.

    The delay "sawtooth": each baseline's phase wraps through 2 pi at a rate
    set by its geometric delay, so long baselines produce many more wraps
    across the band than short ones.
    """
    plt = _plt()
    freq = np.asarray(fs["freq_mhz"])
    good = np.asarray(fs.get("freq_mask", np.ones(len(freq), bool)), bool)
    vis = np.asarray(fs["vis"])[:, :, sel]
    tmask = np.asarray(fs.get("time_mask", np.ones(vis.shape[0], bool)), bool)
    if not tmask.any():
        tmask = np.ones(vis.shape[0], bool)
    avg = np.mean(vis[tmask], axis=0)                    # (F, n_sel)
    n = len(sel)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.5 * n + 1.2), sharex=True,
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for k in range(n):
        ax = axes[k, 0]
        ph = np.where(good, np.angle(avg[:, k]), np.nan)
        ax.plot(freq, ph, ".", ms=1.0,
                color=cmap(k / max(1, n - 1)))
        unw = np.unwrap(np.angle(avg[good, k]))
        wraps = abs(unw[-1] - unw[0]) / (2 * np.pi)     # net turns, not noise
        ax.set_ylabel(f"{labels[k]}", fontsize=7)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(alpha=0.3)
        ax.text(0.995, 0.05, f"~{wraps:.0f} turns across the band",
                transform=ax.transAxes, ha="right", fontsize=7, color="0.3")
    axes[-1, 0].set_xlabel("frequency (MHz)")
    fig.suptitle(title, fontsize=11)
    fig.supylabel("raw phase (rad, wrapped)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# 5. Per-antenna gain phase + delay fit
# ---------------------------------------------------------------------------

def plot_gain_delay_fits(cal_file, ref_ant, out_png, out_csv, tag=""):
    """Solved gain phase vs frequency per antenna with the fitted delay line.

    Reads the cal h5 (so it also runs when a run reuses an existing cal).
    Fit is ``run_cal_diff.py``'s brute-force delay search over +/-300 ns.
    """
    import h5py
    import pandas as pd
    plt = _plt()

    with h5py.File(cal_file, "r") as f:
        gains = f["gains"][:]
        ants = [int(a) for a in f["ant_ids"][:]]
        flags = np.asarray(f["flags"][:], bool)
        freq = f["freqs_mhz"][:]
    if ref_ant not in ants:
        raise ValueError(f"ref_ant {ref_ant} not in cal {cal_file}")
    good = flags & (np.abs(gains).min(axis=0) > 0)
    phase = np.angle(gains * np.conj(gains[ants.index(ref_ant)])[None, :])

    m = fit_mask(freq, good)
    rows = []
    ncol = int(np.ceil(np.sqrt(len(ants))))
    nrow = int(np.ceil(len(ants) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for k, a in enumerate(ants):
        ax = axes.flat[k]
        ph = np.where(m, phase[k], np.nan)
        ax.plot(freq, ph, ".", ms=1.0, color="tab:blue")
        title = f"ant {a}"
        if m.sum() >= 50:
            tau, resid, coh = delay_fit(phase[k], freq, m)
            phi0 = delay_fit_phase0(phase[k], freq, m, tau)
            model = np.angle(np.exp(1j * (2 * np.pi * tau * 1e-9 * freq * 1e6
                                          + phi0)))
            ax.plot(freq[m], model[m], ".", ms=0.8, color="tab:red", alpha=0.6)
            rows.append(dict(ant=a, nchan=int(m.sum()), delay_ns=float(tau),
                             resid_rms_rad=resid, coh_after_delay=coh))
            title += f"\n{tau:+.1f} ns, res {resid:.2f} rad"
        else:
            rows.append(dict(ant=a, nchan=int(m.sum())))
            title += "\ntoo few good channels"
        ax.set_title(title, fontsize=8)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(alpha=0.3)
    for k in range(len(ants), nrow * ncol):
        axes.flat[k].axis("off")
    tab = pd.DataFrame(rows)
    tab.to_csv(out_csv, index=False)
    med = (float(np.nanmedian(tab["resid_rms_rad"]))
           if "resid_rms_rad" in tab else float("nan"))
    fig.suptitle(f"Solved gain phase vs frequency with fitted delay (red), ref "
                 f"ant {ref_ant}{(' - ' + tag) if tag else ''};\n"
                 f"median residual rms {med:.2f} rad", fontsize=11)
    fig.supxlabel("frequency (MHz)")
    fig.supylabel("gain phase (rad)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png, tab


# ---------------------------------------------------------------------------
# 6. Rank-1 ratio vs frequency (svd_census/make_plots_aug19.py template)
# ---------------------------------------------------------------------------

def plot_rank1(cases, out_png, title):
    """Rank-1 ratio vs frequency.

    cases: list of (freq, rank1, label, colour); the first is primary.
    """
    plt = _plt()
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


# ---------------------------------------------------------------------------
# 7. Singular values vs frequency (run_census_aug19.py template)
# ---------------------------------------------------------------------------

def plot_svd_vs_freq(cal, out_png, title, n_show=6):
    """sigma_1..k of the solve matrix vs frequency + the rank-1 fraction.

    ``cal['singular_values']`` is (n_chan, n_ant): the singular values of the
    same per-channel PHASE_ONLY matrix the gains came from (unit-modulus
    entries, zero diagonal), so sigma_1 -> n_ant-1 for a perfectly coherent
    point source and the rank-1 fraction -> 1.
    """
    plt = _plt()
    sv = np.asarray(cal["singular_values"], dtype=float)   # (n_chan, n_ant)
    freq = np.asarray(cal["freqs_mhz"])
    n = sv.shape[1]
    tot = np.nansum(sv, axis=1)
    frac = np.where(tot > 0, sv[:, 0] / np.where(tot > 0, tot, 1.0), np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0), constrained_layout=True)
    for k in range(min(n_show, n)):
        ax[0].plot(freq, sv[:, k], lw=0.5, label=rf"$\sigma_{{{k+1}}}$")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("frequency (MHz)")
    ax[0].set_ylabel(r"$\sigma_k$")
    ax[0].axhline(n - 1, color="k", ls="--", lw=0.8,
                  label=rf"$\sigma_1\to${n - 1} when perfectly coherent")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f"SVD of the {n}-antenna phase-only solve matrix",
                    fontsize=10)
    ax[1].plot(freq, frac, lw=0.5, color="C3")
    # Perfect rank-1 with a zeroed diagonal is J - I: singular values
    # (n-1, 1, 1, ...), so the ideal fraction is (n-1)/(2(n-1)) = 0.5.
    ax[1].axhline(0.5, color="k", ls="--", lw=0.8,
                  label="perfectly coherent limit 0.5")
    ax[1].axhline(1.0 / n, color="k", ls=":", lw=0.8,
                  label=f"incoherent floor 1/{n} = {1.0/n:.3f}")
    ax[1].axvspan(SUNBAND[0], SUNBAND[1], color="gold", alpha=0.15,
                  label=f"{SUNBAND[0]:.0f}-{SUNBAND[1]:.0f} MHz")
    ax[1].set_xlabel("frequency (MHz)")
    ax[1].set_ylabel(r"$\sigma_1/\Sigma\sigma$")
    ax[1].set_ylim(0, 1)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    ax[1].set_title("rank-1 fraction", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)
    return out_png, dict(
        sigma1_median=float(np.nanmedian(sv[:, 0])),
        sigma1_perfect=float(n - 1),
        rank1_fraction_median=float(np.nanmedian(frac)),
        rank1_fraction_inband=float(np.nanmedian(frac[band_mask(freq)])),
        rank1_fraction_perfect=0.5,
        incoherent_floor=float(1.0 / n))


# ---------------------------------------------------------------------------
# 8. Fringe-stop waterfalls
# ---------------------------------------------------------------------------

def plot_fringe_stopped(fs, mapping, out_dir, max_baselines, tz=LOCAL_TZ):
    """Fringe-stopped visibility waterfalls via the casm_vis_analysis helper.

    Figures are rendered in memory (``output_dir=None``) so the x-axis can be
    relabelled to local time and only the figures produced by THIS run are
    written and returned.
    """
    from casm_vis_analysis.plotting.fringe_diag import (plot_fringe_diagnostic,
                                                        group_by_snap_pair)
    plt = _plt()
    target_aids = list(fs["target_aids"])
    target_labels = list(fs["target_labels"])
    target_snaps = [mapping.snap_adc(a)[0] for a in target_aids]
    ref_snap = mapping.snap_adc(int(fs["ref_ant"]))[0]
    time_unix = np.asarray(fs["time_unix"])
    panels = [("Raw phase", np.asarray(fs["vis"])),
              ("Geometric", np.asarray(fs["geometric_phase"])),
              ("Fringe-stopped", np.asarray(fs["vis_stopped"]))]
    figs = plot_fringe_diagnostic(panels, time_unix,
                                  np.asarray(fs["freq_mhz"]), target_labels,
                                  target_snaps, ref_snap, output_dir=None,
                                  split_max=max_baselines,
                                  freq_mask=fs.get("freq_mask"), time_tz=tz)
    # Reproduce the helper's own file naming, in the same order it yields.
    names = []
    for (rs, ts), indices in sorted(group_by_snap_pair(ref_snap,
                                                       target_snaps).items()):
        n_parts = int(np.ceil(len(indices) / max_baselines))
        for p in range(n_parts):
            names.append(f"fringe_diag_snap{rs}_to_{ts}"
                         + (f"_part{p + 1}" if n_parts > 1 else "") + ".png")
    pngs = []
    for fig, name in zip(figs, names):
        localize_hours_axis(fig, time_unix[0], tz=tz)
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        pngs.append(path)
    print(f"  fringe-stopped waterfalls: {len(pngs)} figure(s) in {out_dir}",
          flush=True)
    return pngs


# ---------------------------------------------------------------------------
# 9. Cal diff vs a previous cal (run_cal_diff.py)
# ---------------------------------------------------------------------------

def plot_cal_diff(cal_file, prev_cal_file, ref_ant, out_png, out_csv):
    """Per-antenna phase & delay diff vs a previous cal."""
    import h5py
    import pandas as pd
    plt = _plt()

    def _load(path):
        with h5py.File(path, "r") as f:
            g = f["gains"][:]
            ants = [int(a) for a in f["ant_ids"][:]]
            flags = f["flags"][:].astype(bool)
            freq = f["freqs_mhz"][:]
        if ref_ant not in ants:
            raise ValueError(f"ref_ant {ref_ant} not in {path}")
        ph = np.angle(g * np.conj(g[ants.index(ref_ant)])[None, :])
        return dict(ants=ants, freq=freq, phase=ph,
                    good=flags & (np.abs(g).min(axis=0) > 0))

    A, B = _load(prev_cal_file), _load(cal_file)
    if A["freq"].shape != B["freq"].shape or not np.allclose(A["freq"],
                                                             B["freq"]):
        print(f"  cal diff SKIPPED: frequency axes differ "
              f"({A['freq'].shape} vs {B['freq'].shape})", flush=True)
        return None, None
    common = [a for a in B["ants"] if a in A["ants"]]
    if not common:
        print("  cal diff SKIPPED: no antenna in common with "
              f"{prev_cal_file}", flush=True)
        return None, None
    freq = B["freq"]
    rows, dphi_all = [], np.full((len(common), len(freq)), np.nan)
    for i, a in enumerate(common):
        d = np.angle(np.exp(1j * (B["phase"][B["ants"].index(a)]
                                  - A["phase"][A["ants"].index(a)])))
        m = fit_mask(freq, A["good"], B["good"])
        dphi_all[i] = np.where(m, d, np.nan)
        if m.sum() < 50:
            rows.append(dict(ant=a, nchan=int(m.sum())))
            continue
        tau, resid, coh = delay_fit(d, freq, m)
        rows.append(dict(ant=a, nchan=int(m.sum()),
                         mean_abs_dphi_rad=float(np.mean(np.abs(d[m]))),
                         circ_coherence=float(np.abs(np.exp(1j * d[m]).mean())),
                         delay_ns=float(tau), resid_rms_rad=resid,
                         coh_after_delay=coh))
    tab = pd.DataFrame(rows)
    tab.to_csv(out_csv, index=False)
    m = fit_mask(freq, A["good"], B["good"])
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


# ---------------------------------------------------------------------------
# 10. Beamformed response on a bright source (run_task1_cyga.py)
# ---------------------------------------------------------------------------

def _catalog_key(name):
    return name.lower().replace("-", "_").replace(" ", "_").replace("+", "_")


def _cal_phase(cal_file, ants, freq):
    """Per-antenna gain phase from a cal h5, aligned to ``ants`` and ``freq``."""
    import h5py
    with h5py.File(cal_file, "r") as f:
        cal_ants = [int(a) for a in f["ant_ids"][:]]
        g = f["gains"][:]
        fq = f["freqs_mhz"][:]
        fl = np.asarray(f["flags"][:], bool)
    if fq.shape != freq.shape or not np.allclose(fq, freq):
        raise ValueError(f"{cal_file}: frequency axis does not match the data")
    missing = [a for a in ants if a not in cal_ants]
    if missing:
        raise ValueError(f"{cal_file}: missing antennas {missing}")
    ph = np.array([np.angle(g[cal_ants.index(a)]) for a in ants])
    good = fl & (np.abs(g).min(axis=0) > 0)
    return ph, good


def beamform_source_check(cal_file, layout_csv, antennas, source, window,
                          out_png, prev_cal_file=None, off_alt_deg=-25.0,
                          min_alt_deg=10.0, tz=LOCAL_TZ, fmt_name="layout_64ant"):
    """Coherent-beam response on a bright reference source, with the new cal.

    Visibility-domain beamforming, conventions per casm-wiki
    ``vis-beamforming-conventions`` and ``run_task1_cyga.py``:
    ``w_i(f) = conj(g_i) exp(+2i pi f (r_i.s)/c)``,
    ``P = 2 Re sum_{i<j} w_i w_j^* V_ij`` on the stored upper triangle, and
    ``coherence = P / (2 sum_{i<j} |V_ij|)`` so 1.0 is a perfectly phased
    point source and 0.0 is noise.

    Returns a dict with the figure path and the band-averaged numbers, or
    ``dict(skipped=<reason>)`` when the data or the source is not available.
    """
    from casm_io.correlator import read_visibilities, load_format, AntennaMapping
    from casm_io.correlator.baselines import triu_flat_index
    from casm_vis_analysis.sources import source_altaz
    plt = _plt()

    ants = sorted(int(a) for a in antennas)
    mapping = AntennaMapping.load(layout_csv)
    df = mapping.dataframe
    pkt = np.array([mapping.packet_index(a) for a in ants])
    pos = np.array([df.loc[df["antenna_id"] == a,
                           ["x_m", "y_m", "z_m"]].values[0] for a in ants])

    try:
        d = read_visibilities(window[0], window[1], time_tz="UTC",
                              data_root="/mnt", fmt=load_format(fmt_name),
                              verbose=False)
    except Exception as exc:
        return dict(skipped=f"no visibilities for {window[0]} -> {window[1]} "
                            f"UTC ({type(exc).__name__}: {exc})")
    freq = np.asarray(d["freq_mhz"])
    tu = np.asarray(d["time_unix"])
    vis = np.asarray(d["vis"])
    nt, nf, n_bl = vis.shape
    n_inputs = int((-1 + (1 + 8 * n_bl) ** 0.5) / 2)

    alt, az = source_altaz(_catalog_key(source), tu)
    alt = np.asarray(alt, float)
    az = np.asarray(az, float)
    if alt.max() < min_alt_deg:
        return dict(skipped=f"{source} never rises above {min_alt_deg} deg in "
                            f"{window[0]} -> {window[1]} UTC "
                            f"(max {alt.max():.1f} deg)")

    try:
        ph_new, good_new = _cal_phase(cal_file, ants, freq)
    except ValueError as exc:
        return dict(skipped=str(exc))
    cases = {"new cal": (ph_new, "tab:green")}
    good = good_new
    if prev_cal_file:
        try:
            ph_old, good_old = _cal_phase(prev_cal_file, ants, freq)
            cases["previous cal"] = (ph_old, "tab:red")
            good = good & good_old
        except ValueError as exc:
            print(f"  beam check: previous cal not comparable ({exc})",
                  flush=True)
    good = fit_mask(freq, good)
    if good.sum() < 50:
        return dict(skipped=f"only {int(good.sum())} usable channels")

    n = len(ants)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    bl = np.array([triu_flat_index(n_inputs, pkt[i], pkt[j]) for i, j in pairs])
    wi = np.array([i for i, _ in pairs])
    wj = np.array([j for _, j in pairs])
    V = vis[:, :, bl].astype(np.complex128)
    del d, vis
    denom = 2.0 * np.abs(V).sum(axis=2)

    def steer(cal_phase, alt_deg, az_deg):
        a, A = np.deg2rad(alt_deg), np.deg2rad(az_deg)
        s = np.array([np.cos(a) * np.sin(A), np.cos(a) * np.cos(A), np.sin(a)])
        geo = pos @ s / C_M_S
        return np.exp(-1j * cal_phase) * np.exp(
            2j * np.pi * np.outer(geo, freq * 1e6))

    def beam(Vt, w):
        return 2 * np.real(Vt * (w[wi] * np.conj(w[wj])).T).sum(axis=1)

    coh = {}
    for label, (cp, _col) in cases.items():
        Pt = np.empty((nt, nf))
        for t in range(nt):
            Pt[t] = beam(V[t], steer(cp, alt[t], az[t]))
        coh[label] = Pt / denom
    Pt = np.empty((nt, nf))
    for t in range(nt):
        Pt[t] = beam(V[t], steer(ph_new, alt[t] + off_alt_deg, az[t]))
    coh[f"null: {off_alt_deg:+.0f} deg off"] = Pt / denom
    colours = dict((k, v[1]) for k, v in cases.items())
    colours[f"null: {off_alt_deg:+.0f} deg off"] = "grey"
    del V

    summary = {}
    for label, c in coh.items():
        per_int = np.nanmean(c[:, good], axis=1)
        sd = float(np.nanstd(per_int, ddof=1)) if nt > 1 else float("nan")
        summary[label] = dict(coh=float(np.nanmean(per_int)),
                              scatter=sd,
                              err=sd / np.sqrt(nt) if nt > 1 else float("nan"))

    times = local_datetimes(tu, tz)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), constrained_layout=True)
    ax = axes[0]
    for label, c in coh.items():
        ax.plot(times, np.nanmean(c[:, good], axis=1), "-o", ms=3, lw=1.6,
                color=colours[label],
                label=f"{label}: {summary[label]['coh']:+.4f}")
    ax.axhline(0, color="k", lw=0.7)
    k = int(np.argmax(alt))
    ax.axvline(times[k], color="k", ls="--", lw=0.8,
               label=f"max altitude {alt[k]:.1f} deg")
    format_local_axis(ax, tz)
    ax.set_ylabel(r"coherence  $P / 2\Sigma|V|$")
    ax.set_title(f"{source}: coherent-beam response vs time "
                 f"({int(good.sum())} channels)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for label, c in coh.items():
        cm = np.where(good, np.nanmean(c, axis=0), np.nan)
        ax.plot(freq, cm, ".", ms=1.0, alpha=0.3, color=colours[label])
        ax.plot(freq, _roll(cm, 48, 12), lw=1.8, color=colours[label],
                label=label)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("coherence")
    ax.set_title(f"{source}: coherence vs frequency ({nt} integrations)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"Beamformed response on {source} with the new cal - "
                 f"{utc_local_title(tu, tz)}, alt "
                 f"{alt.min():.1f}-{alt.max():.1f} deg", fontsize=12)
    fig.savefig(out_png, dpi=125, facecolor="white")
    plt.close(fig)
    return dict(png=out_png, source=source, window=list(window),
                n_integrations=int(nt), n_channels=int(good.sum()),
                alt_range=[float(alt.min()), float(alt.max())],
                summary=summary)


# ---------------------------------------------------------------------------
# 11. Grid map + source-transit coverage
# ---------------------------------------------------------------------------

def plot_grid_map(weights_file, out_png, near, title):
    """Beam-grid alt/az scatter (no all-sky image)."""
    import h5py
    plt = _plt()
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


def plot_transit_coverage(weights_file, sources, date, out_png, tz=LOCAL_TZ):
    """Source tracks through the beam grid: ``bf_weights_generator.plot_transit``.

    Same machinery as the ``casm-bf-source-transit`` CLI, so the figure in the
    notebook is the schedule the operator would print for the day.
    """
    from bf_weights_generator.plot_transit import plot_source_transit
    plot_source_transit(weights_file, sources=list(sources), date=date,
                        time_tz=tz, time_start="00:00", time_end="23:59",
                        output_path=out_png)
    return out_png
