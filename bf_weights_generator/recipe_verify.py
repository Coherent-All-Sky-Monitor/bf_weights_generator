"""Product checks for the canonical recipe: pointing fit + cal-band health.

Vendored (unchanged numerics) from
``/mnt/nvme5/solar0819/pointing_verify/verify_pointing.py`` so a recipe run
does not depend on a scratch directory that can be swept at any time.

Pointing check
--------------
``w_stored = cal * exp(-i 2 pi f tau)`` with ``tau = -(r.s)/c`` (generator
convention, ``weights.py`` / ``snap_weights.py``). Divide the cal out of the
stored int8 weights and grid-search the direction ``s`` that maximises the
residual coherence: a healthy file peaks at the pointing its own table claims
(``coh_fit ~= coh_stated ~= 1``); a broken one peaks somewhere else or nowhere.

Cal-band check
--------------
``subband_flag_table`` reports, per 512-channel subband, the fraction of
channels with a solved, non-zero gain. A whole subband at zero means the
F-engine delivered no data there (see casm-wiki ``missing-subbands-diagnostic``)
and the beamformer is running on a fraction of the band.
"""

from __future__ import annotations

import numpy as np

C = 299792458.0

#: Channels per subband in the deployed 6 x 512 layout.
SUBBAND_SIZE = 512


# ---------------------------------------------------------------------------
# Pointing verification (verbatim numerics from verify_pointing.py)
# ---------------------------------------------------------------------------

def load_file(path):
    """Read the geometry + pointing table out of an int8 weights HDF5."""
    import h5py
    with h5py.File(path) as h:
        d = dict(
            active=h['array_config/active_mask'][:],
            ant_ids=h['array_config/antenna_ids'][:],
            pos=h['array_config/positions_enu'][:],
            freqs=h['frequencies_hz'][:],
            alt=h['pointings/alt_deg'][:],
            az=h['pointings/az_deg'][:],
            attrs=dict(h.attrs),
        )
    return d


def beam_complex(path, beam, slots, chan_step=8):
    """Return complex weights (n_chan_sub, n_slot) for pol 0."""
    import h5py
    with h5py.File(path) as h:
        W = h['weights_int8']
        re = W[0, ::chan_step, 0, beam, :].astype(np.float32)
        im = W[1, ::chan_step, 0, beam, :].astype(np.float32)
    w = (re + 1j * im)[:, slots]
    return w


def cal_aligned(calw, ant_ids_at_slot, freqs_desc, chan_step=8):
    """cal weights aligned to slots, frequency axis matched to file (descending)."""
    cal_ids = np.asarray(calw.ant_ids).astype(int)
    # calw.weights are ascending in freq; file freqs are descending
    cw = calw.weights[:, ::-1]
    flags = calw.flags[::-1]
    rows = [np.where(cal_ids == int(a))[0][0] for a in ant_ids_at_slot]
    return cw[rows][:, ::chan_step].T, flags[::chan_step]


def geo_weights(pos_act, freqs, alt_deg, az_deg):
    """Geometric steering weights for a set of directions."""
    alt = np.deg2rad(np.atleast_1d(alt_deg))
    az = np.deg2rad(np.atleast_1d(az_deg))
    l = np.cos(alt) * np.sin(az)
    m = np.cos(alt) * np.cos(az)
    n = np.sin(alt)
    # (n_dir, n_ant)
    path = (np.outer(l, pos_act[:, 0]) + np.outer(m, pos_act[:, 1])
            + np.outer(n, pos_act[:, 2]))
    tau = -path / C
    # phases = 2 pi f tau ; w = exp(-i phases)
    ph = 2 * np.pi * tau[:, :, None] * freqs[None, None, :]   # (ndir,nant,nchan)
    return np.exp(-1j * ph)


def coherence(resid_norm, pos_act, freqs, alts, azs):
    """resid_norm: (nchan, nant) unit-modulus. Returns coherence per direction."""
    g = geo_weights(pos_act, freqs, alts, azs)        # (ndir, nant, nchan)
    r = resid_norm.T[None]                             # (1, nant, nchan)
    N = r.shape[1] * r.shape[2]
    return np.abs((r * np.conj(g)).sum(axis=(1, 2))) / N


def fit_beam(path, beam, meta, calw, chan_step=8):
    """Grid-search the direction one beam actually points at."""
    slots = np.where(meta['active'])[0]
    w = beam_complex(path, beam, slots, chan_step)
    ca, flags = cal_aligned(calw, meta['ant_ids'][slots], meta['freqs'],
                            chan_step)
    freqs = meta['freqs'][::chan_step]
    good = (np.abs(w).sum(axis=1) > 0) & flags & (np.abs(ca).min(axis=1) > 0)
    w = w[good]
    ca = ca[good]
    freqs = freqs[good]
    if len(freqs) == 0:
        raise RuntimeError(
            f"beam {beam}: no channel has both a non-zero int8 payload and a "
            f"solved cal gain; the weights file cannot be pointing-verified")
    resid = w * np.conj(ca)
    resid /= np.abs(resid)
    pos_act = meta['pos'][slots]

    # coarse full-sky grid in l,m, then two refinement passes
    best = None
    for step, span in [(2.0, None), (0.25, 6.0), (0.03, 0.8)]:
        if span is None:
            alts = np.arange(0.0, 90.001, step)
            azs = np.arange(0.0, 360.0, step)
        else:
            alts = np.arange(max(0.0, best[0] - span),
                             min(90.0, best[0] + span) + 1e-9, step)
            azs = np.arange(best[1] - span * 3, best[1] + span * 3 + 1e-9, step)
        A, Z = np.meshgrid(alts, azs, indexing='ij')
        co = np.zeros(A.size)
        af = A.ravel()
        zf = Z.ravel()
        chunk = 4000
        for i in range(0, af.size, chunk):
            co[i:i + chunk] = coherence(resid, pos_act, freqs,
                                        af[i:i + chunk], zf[i:i + chunk])
        k = int(np.argmax(co))
        best = (af[k], zf[k] % 360.0, co[k])
    # coherence at the STATED pointing
    c_stated = coherence(resid, pos_act, freqs,
                         [meta['alt'][beam]], [meta['az'][beam]])[0]
    return dict(beam=beam, stated_alt=meta['alt'][beam],
                stated_az=meta['az'][beam], fit_alt=best[0], fit_az=best[1],
                coh_fit=best[2], coh_stated=c_stated, nchan=len(freqs),
                nant=len(slots))


# ---------------------------------------------------------------------------
# Cal band health
# ---------------------------------------------------------------------------

def subband_flag_table(cal_file, subband_size=SUBBAND_SIZE):
    """Per-subband fraction of channels with a solved, non-zero gain.

    Reads the cal h5 directly so the check also runs when a run reuses an
    existing cal (``cal_path``) and never touches the visibilities.

    Returns a list of dicts, one per subband, ordered as stored in the file.
    """
    import h5py
    with h5py.File(cal_file, "r") as f:
        gains = f["gains"][:]
        flags = np.asarray(f["flags"][:], dtype=bool)
        freq = f["freqs_mhz"][:]
    nonzero = flags & (np.abs(gains).min(axis=0) > 0)
    nchan = len(freq)
    rows = []
    for k in range(int(np.ceil(nchan / subband_size))):
        sl = slice(k * subband_size, min((k + 1) * subband_size, nchan))
        n = int(nonzero[sl].sum())
        tot = int(freq[sl].size)
        rows.append(dict(subband=k, chan_lo=sl.start, chan_hi=sl.stop - 1,
                         freq_lo_mhz=float(freq[sl].min()),
                         freq_hi_mhz=float(freq[sl].max()),
                         n_good=n, n_chan=tot,
                         good_frac=float(n / tot) if tot else 0.0))
    return rows
