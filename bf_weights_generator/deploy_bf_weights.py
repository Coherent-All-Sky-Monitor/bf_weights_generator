#!/usr/bin/env python3
"""
Deploy int8 beamformer weights to the CASM real-time pipeline.

Converts an HDF5 int8 weights file into DADA-format files (one per 512-channel
subband) and optionally writes them to the beamformer named pipes for live
application.

Now supports BOTH coherent-beam (CB) and incoherent-beam (IB) weights with
format-type safeguards so you can't accidentally send a CB file to the IB
FIFOs (or vice versa).

Modes:
  - Default: write direct.dada.{0-5} files to --output-dir (current dir)
  - --upload: write to named pipes on corr1/corr2 to apply live
  - --save-defaults: copy the staged files to the restart-defaults directory
    /data/casm/default_weights_64ant_512beam on both machines, as
    direct.dada.{stream} (CB) and incoh.dada.{stream} (IB). Both kinds live in
    that ONE directory — those are the exact names bfcorr loads on restart.
    Any failure (missing staged file, scp/ssh error, size mismatch) aborts the
    run with a non-zero exit; it is never reported and skipped.

Examples:
    # CB only (existing behavior, unchanged)
    python deploy_bf_weights.py weights.h5 --upload

    # CB + IB together (recommended for daytime obs that need IB subtraction)
    python deploy_bf_weights.py weights_cb.h5 \\
        --ib-weights weights_ib.h5 --upload

    # Different SCALE_IB from SCALE_CB (advanced)
    python deploy_bf_weights.py weights_cb.h5 \\
        --ib-weights weights_ib.h5 --upload --scale 32 --ib-scale 32

    # Dry run (show what would happen, for both CB and IB)
    python deploy_bf_weights.py weights_cb.h5 \\
        --ib-weights weights_ib.h5 --upload --dry-run

    # Upload at a future time
    python deploy_bf_weights.py weights_cb.h5 --upload \\
        --utc-start 2026-05-16-18:30:00
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone

import h5py
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────

HDR_SIZE = 4096
N_SUBBANDS = 6
SUBBAND_WIDTH = 512
# Raw SNAP signal-slot count per subband, as the IB kernel sees it
# (= nsig_per_snap * nsnap = 12 * 11 = 132 for the current CASM layout).
# This is DIFFERENT from CB's K-dim of 64 (which is the GEMM antenna axis
# the CUTLASS kernel reduces over). The IB kernel iterates over raw SNAP
# signal slots and expects this many entries per channel.
N_SIG_SLOTS = 132                   # IB input dimension per channel
N_CB_ANTS   = 64                    # CB weights' antenna-axis size (GEMM K-dim)

# Stream -> host mapping (used for BOTH CB and IB uploads)
STREAM_HOST = {
    0: "casm-corr1", 1: "casm-corr1", 2: "casm-corr1",
    3: "casm-corr2", 4: "casm-corr2", 5: "casm-corr2",
}

# Named-pipe paths on the corr machines.
# IB path source: live config files in this repo:
#   medusa_antenna.cfg:                  INCOH_WEIGHTS_FIFO  /tmp/incohweights.fifo
#   casm_bf_weights.py:                  INCOH_WEIGHTS_FIFO + f".{self.stream_id}"
#   ldunn_scratch/vishnu_weights/apply_weights.sh:
#       cat incoh_mask.dada.0 > /tmp/incohweights.fifo.0   (and .1, .2)
PIPE_PATH_FMT    = "/tmp/bfweights.fifo.{stream}"        # CB (coherent beams)
IB_PIPE_PATH_FMT = "/tmp/incohweights.fifo.{stream}"     # IB (incoherent beam)
# Restart defaults. bfcorr loads BOTH CB and IB defaults out of the SAME
# directory and appends ".<stream_id>" to the configured basename. Ground truth:
#   fourier-space/sources/casm/backend/ovro_64ant_512beam/medusa_antenna.cfg
#     117: BFCORR_DEFAULT_BF_WEIGHTS     /data/casm/default_weights_64ant_512beam/direct.dada
#     118: BFCORR_DEFAULT_INCOH_WEIGHTS  /data/casm/default_weights_64ant_512beam/incoh.dada
#   fourier-space/sources/casm/src/python/casm_bfcorr.py:86,91
#     weights_fname = f"{...}.{self.stream_id}"  -> cat into the matching FIFO
# There is NO separate IB defaults directory. (Until 2026-08-14 this script
# wrote IB defaults to /data/casm/default_ib_weights_64ant/direct_ib.dada.N,
# a path nothing reads and that never existed on either node — see
# casm-wiki incidents, 2026-08-04/2026-08-07.)
DEFAULT_WEIGHTS_DIR = "/data/casm/default_weights_64ant_512beam"
DEFAULT_BASENAME = {"cb": "direct.dada", "ib": "incoh.dada"}

# Local staging filenames written by this script (NOT what bfcorr reads).
LOCAL_BASENAME = {"cb": "direct.dada", "ib": "direct_ib.dada"}

# HDF5 format_type tags written by bf_weights_generator
FORMAT_CB = "int8_snap_weights"
FORMAT_IB = "int8_incoh_bf_weights"

UTC_FMT = "%Y-%m-%d-%H:%M:%S"
# Named pipes we are willing to write to on the corr machines.
_ALLOWED_PIPE_RE = re.compile(r"^/tmp/(bfweights|incohweights)\.fifo\.[0-5]$")


def _validate_utc_start(utc_start):
    """Reject anything that isn't a clean %Y-%m-%d-%H:%M:%S string.

    Guards against header injection (e.g. embedded newlines adding rogue DADA
    keys like a second SCALE line) and silent typos in the live header.
    """
    s = str(utc_start)
    try:
        datetime.strptime(s, UTC_FMT)
    except ValueError:
        raise ValueError(
            f"UTC_START {utc_start!r} is not in the required format {UTC_FMT!r} "
            f"(no newlines/extra fields allowed)"
        )
    return s


def build_header(file_size, utc_start, scale, nbit=8):
    """Build a 4096-byte NUL-padded ASCII header."""
    utc_start = _validate_utc_start(utc_start)
    if int(scale) <= 0:
        raise ValueError(f"SCALE must be a positive integer, got {scale!r}")
    lines = [
        f"FILE_SIZE {file_size}",
        f"RESOLUTION {file_size}",
        f"UTC_START {utc_start}",
        f"NBIT {nbit}",
        f"SCALE {scale}",
    ]
    header_str = "\n".join(lines) + "\n"
    header_bytes = header_str.encode("ascii")
    if len(header_bytes) > HDR_SIZE:
        raise ValueError(
            f"Header exceeds {HDR_SIZE} bytes ({len(header_bytes)})"
        )
    return header_bytes.ljust(HDR_SIZE, b"\x00")


def write_dada_file(path, header, data_bytes):
    """Write a DADA-format file (header + data)."""
    with open(path, "wb") as f:
        f.write(header)
        f.write(data_bytes)


def write_to_pipe_path(host, pipe_path, header, data_bytes, dry_run=False):
    """Write header + data to a specific named pipe via SSH.

    The pipe path is allowlisted and shell-quoted: the third ssh argument is
    interpreted by the remote shell, so an unsanitized path would be a command
    injection vector into a live correlator host.
    """
    if not _ALLOWED_PIPE_RE.match(pipe_path):
        raise ValueError(
            f"refusing to write to non-allowlisted pipe path {pipe_path!r} "
            f"(expected /tmp/bfweights.fifo.N or /tmp/incohweights.fifo.N, N in 0-5)"
        )
    if host not in STREAM_HOST.values():
        raise ValueError(f"refusing to ssh to unknown host {host!r}")
    payload = header + data_bytes
    total_size = len(payload)
    if dry_run:
        print(f"  [DRY RUN] Would write {total_size:,} bytes to "
              f"{host}:{pipe_path}")
        return
    print(f"  Writing {total_size:,} bytes to {host}:{pipe_path} ...",
          end=" ", flush=True)
    proc = subprocess.run(
        ["ssh", host, "cat > " + shlex.quote(pipe_path)],
        input=payload, capture_output=True,
    )
    if proc.returncode != 0:
        print("FAILED")
        print(f"  stderr: {proc.stderr.decode()}", file=sys.stderr)
        raise RuntimeError(
            f"Failed to write to {host}:{pipe_path}: {proc.stderr.decode()}"
        )
    print("OK")


def write_to_pipe(host, stream_id, header, data_bytes, dry_run=False):
    """Backwards-compatible CB-FIFO writer."""
    write_to_pipe_path(host, PIPE_PATH_FMT.format(stream=stream_id),
                       header, data_bytes, dry_run=dry_run)


# ── Default-weights paths (pure, unit-testable, no I/O) ───────────────────

def _check_kind(kind):
    if kind not in ("cb", "ib"):
        raise ValueError(f"kind must be 'cb' or 'ib', got {kind!r}")
    return kind


def _check_stream(stream_id):
    if stream_id not in STREAM_HOST:
        raise ValueError(
            f"unknown stream id {stream_id!r} (expected one of "
            f"{sorted(STREAM_HOST)})"
        )
    return stream_id


def local_staged_name(kind, stream_id):
    """Filename this script writes locally for (kind, stream)."""
    _check_kind(kind)
    _check_stream(stream_id)
    return f"{LOCAL_BASENAME[kind]}.{stream_id}"


def default_dest_path(kind, stream_id):
    """Absolute path on the corr machine that bfcorr loads on restart.

    kind='cb' -> /data/casm/default_weights_64ant_512beam/direct.dada.N
    kind='ib' -> /data/casm/default_weights_64ant_512beam/incoh.dada.N
    """
    _check_kind(kind)
    _check_stream(stream_id)
    return f"{DEFAULT_WEIGHTS_DIR}/{DEFAULT_BASENAME[kind]}.{stream_id}"


def default_copy_plan(kind, streams, output_dir):
    """(host, local_src, remote_dst) triples for saving defaults.

    Pure: builds paths only, touches nothing. Used by --save-defaults and by
    the unit tests, so the tests exercise the exact paths the deploy uses.
    """
    _check_kind(kind)
    plan = []
    for stream_id in streams:
        _check_stream(stream_id)
        plan.append((
            STREAM_HOST[stream_id],
            os.path.join(output_dir, local_staged_name(kind, stream_id)),
            default_dest_path(kind, stream_id),
        ))
    return plan


def _run_checked(argv, what):
    """Run a subprocess and raise RuntimeError on any non-zero exit."""
    proc = subprocess.run(argv, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{what} FAILED (exit {proc.returncode}): "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout.decode(errors="replace").strip()


def save_defaults(kind, streams, output_dir, dry_run=False):
    """Copy staged DADA files to the restart-defaults path on corr1/corr2.

    Every failure — missing local file, scp failure, post-copy size mismatch —
    raises RuntimeError. Nothing is printed-and-skipped: a run that returns
    normally means all requested streams are on disk on the right host, at the
    path bfcorr reads, with the right size.
    """
    label = "CB" if kind == "cb" else "IB"
    print(f"\nSaving {label} as default weights "
          f"({DEFAULT_BASENAME[kind]}.N in {DEFAULT_WEIGHTS_DIR}):")
    for host, src, dst in default_copy_plan(kind, streams, output_dir):
        if dry_run:
            print(f"  [DRY RUN] Would copy {src} to {host}:{dst}")
            continue
        if not os.path.isfile(src):
            raise RuntimeError(
                f"staged {label} file {src} does not exist — cannot save "
                f"defaults for {host}:{dst}"
            )
        local_size = os.path.getsize(src)
        print(f"  Copying {os.path.basename(src)} to {host}:{dst} ...",
              end=" ", flush=True)
        try:
            _run_checked(["scp", "-p", src, f"{host}:{dst}"],
                         f"scp {src} -> {host}:{dst}")
            remote_size = _run_checked(
                ["ssh", host, "stat -c %s " + shlex.quote(dst)],
                f"stat {host}:{dst}",
            )
            if int(remote_size) != local_size:
                raise RuntimeError(
                    f"{host}:{dst} is {remote_size} bytes but the local file "
                    f"is {local_size} bytes — copy is incomplete"
                )
        except RuntimeError:
            print("FAILED")
            raise
        print(f"OK ({local_size:,} bytes verified)")


# ── Safeguards ────────────────────────────────────────────────────────────

def _check_format(h5_path, expected_format, kind_label):
    """Open an HDF5 weights file and verify it's the correct kind (CB vs IB).

    REFUSES to proceed if the file's format_type attr says it is the OTHER
    kind. This prevents accidentally posting a CB file to the IB FIFOs (which
    would corrupt the live pipeline since shapes differ) or vice versa.
    """
    with h5py.File(h5_path, "r") as f:
        actual = str(f.attrs.get("format_type", "")).strip()
    if actual and actual != expected_format:
        raise SystemExit(
            f"\nERROR: {h5_path}\n"
            f"  appears to be a {actual!r} file, but you passed it as "
            f"the {kind_label} weights argument (expected "
            f"{expected_format!r}).\n"
            f"  Refusing to upload — this would write the wrong data to "
            f"the {kind_label} FIFOs and could destabilize the live "
            f"pipeline.\n"
            f"  Swap the --weights / --ib-weights args or pass the matching "
            f"file."
        )
    if not actual:
        print(f"  WARNING: {h5_path} has no 'format_type' attribute; "
              f"cannot verify it is a {kind_label} file. Proceeding anyway.")
    return actual


# ── IB upload helpers ─────────────────────────────────────────────────────

def load_ib_weights(ib_h5_path, expected_n_chan):
    """Load IB weights array and validate its shape against expected_n_chan."""
    _check_format(ib_h5_path, FORMAT_IB, kind_label="IB (incoherent)")
    with h5py.File(ib_h5_path, "r") as f:
        w_ib = f["weights_int8"][:]   # (n_chan, n_inputs=64) uint8
        ib_n_chan = f.attrs.get("n_channels", w_ib.shape[0])
        ib_n_inputs = f.attrs.get("n_inputs", w_ib.shape[1])
        ib_scale_meta = f.attrs.get("scale_factor", None)
    if w_ib.shape != (expected_n_chan, N_SIG_SLOTS):
        raise SystemExit(
            f"\nERROR: IB weights shape {w_ib.shape} does not match "
            f"expected ({expected_n_chan}, {N_SIG_SLOTS}). The IB kernel "
            f"expects nsig_per_snap*nsnap = 12*11 = 132 signal slots per "
            f"channel, NOT the CB GEMM antenna count. Rebuild the IB H5 "
            f"with width {N_SIG_SLOTS}."
        )
    if w_ib.dtype != np.uint8:
        raise SystemExit(
            f"\nERROR: IB weights dtype is {w_ib.dtype}, expected uint8."
        )
    print(f"  Shape: {w_ib.shape}  dtype: {w_ib.dtype}")
    print(f"  {int(ib_n_chan)} channels, {int(ib_n_inputs)} SNAP inputs")
    n_nonzero = int((w_ib > 0).sum())
    print(f"  Active entries (nonzero): {n_nonzero:,} / "
          f"{w_ib.size:,}  "
          f"({100.0 * n_nonzero / w_ib.size:.1f}%)")
    if ib_scale_meta is not None:
        print(f"  IB scale_factor metadata: {ib_scale_meta}")
    return w_ib


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deploy int8 beamformer weights to the CASM pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "weights_h5",
        help="Path to CB int8 weights HDF5 file (format_type='int8_snap_weights')",
    )
    parser.add_argument(
        "--ib-weights", default=None,
        help="Optional path to IB int8 weights HDF5 file "
             "(format_type='int8_incoh_bf_weights'). If provided, IB weights "
             "will be uploaded in parallel with CB weights.",
    )
    parser.add_argument(
        "--output-dir", "-o", default=".",
        help="Directory for DADA files (default: current dir)",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Write to named pipes on corr1/corr2 (apply live)",
    )
    parser.add_argument(
        "--save-defaults", action="store_true",
        help="Write to default weights directory on corr1 and corr2",
    )
    parser.add_argument(
        "--utc-start", default=None,
        help="UTC_START for header (%%Y-%%m-%%d-%%H:%%M:%%S). "
             "Default: current time (apply immediately). "
             "Same value is used for CB and IB.",
    )
    parser.add_argument(
        "--scale", type=int, default=32,
        help="SCALE header value for CB (default: 32)",
    )
    parser.add_argument(
        "--ib-scale", type=int, default=32,
        help="SCALE header value for IB (default: 32, matching SCALE_CB). "
             "The |CB|/|IB| ratio = (SCALE_IB/SCALE_CB) * |w_CB|^2 / (bf^2 * w_IB). "
             "bfcorr now runs bf_scale_factor=127 (changed from 64 on 2026-05-27), "
             "so SCALE_IB=SCALE_CB=32 gives |CB|/|IB| = (32/32)*127^2/127^2 = 1.0 "
             "for clean CB-IB subtraction. (Old convention was SCALE_IB=8 at bf=64, "
             "which gave 0.985; that is WRONG at bf=127, where it yields 0.25.) "
             "Override only if bf_scale_factor changes or you change w_IB values.",
    )
    parser.add_argument(
        "--ib-pipe-fmt", default=IB_PIPE_PATH_FMT,
        help=f"Named-pipe path format for IB streams (default: {IB_PIPE_PATH_FMT})",
    )
    parser.add_argument(
        "--streams", default="0,1,2,3,4,5",
        help="Comma-separated stream IDs to write (default: 0,1,2,3,4,5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without executing",
    )
    args = parser.parse_args(argv)

    streams = [int(s) for s in args.streams.split(",")]

    # ── Load CB weights (validates format type) ──────────────────────────
    print(f"Loading CB weights: {args.weights_h5}")
    _check_format(args.weights_h5, FORMAT_CB, kind_label="CB (coherent)")
    with h5py.File(args.weights_h5, "r") as f:
        w = f["weights_int8"][:]
        n_chan = f.attrs.get("n_channels", w.shape[1])
        n_beams = f.attrs.get("n_beams", w.shape[3])
        n_ant = f.attrs.get("n_antennas", w.shape[4])
    print(f"  Shape: {w.shape}  dtype: {w.dtype}")
    print(f"  {n_chan} channels, {n_beams} beams, {n_ant} antenna slots")
    if n_chan != N_SUBBANDS * SUBBAND_WIDTH:
        print(f"ERROR: expected {N_SUBBANDS * SUBBAND_WIDTH} channels, "
              f"got {n_chan}", file=sys.stderr)
        sys.exit(1)

    # ── Load IB weights if requested ─────────────────────────────────────
    w_ib = None
    if args.ib_weights:
        print(f"\nLoading IB weights: {args.ib_weights}")
        w_ib = load_ib_weights(args.ib_weights, expected_n_chan=n_chan)
        ib_scale = args.ib_scale
        print(f"  IB SCALE header: {ib_scale}  (CB SCALE: {args.scale})")

    # ── Compute CB sizes ─────────────────────────────────────────────────
    subband_data = w[:, :SUBBAND_WIDTH, :, :, :]
    cb_data_size = subband_data.nbytes
    print(f"\n  CB per-subband data: {cb_data_size:,} bytes")
    print(f"  CB total file size:  {HDR_SIZE + cb_data_size:,} bytes "
          f"(header {HDR_SIZE} + data {cb_data_size:,})")

    if w_ib is not None:
        ib_data_size = SUBBAND_WIDTH * N_SIG_SLOTS   # 67,584 bytes
        print(f"  IB per-subband data: {ib_data_size:,} bytes")
        print(f"  IB total file size:  {HDR_SIZE + ib_data_size:,} bytes")

    # ── Build headers ────────────────────────────────────────────────────
    if args.utc_start:
        utc_start = args.utc_start
    else:
        utc_start = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H:%M:%S")

    cb_header = build_header(cb_data_size, utc_start, args.scale)
    print(f"\n  CB header (SCALE={args.scale}):")
    for line in cb_header.decode("ascii").strip("\x00").strip().split("\n"):
        print(f"    {line}")

    if w_ib is not None:
        ib_header = build_header(ib_data_size, utc_start, ib_scale)
        print(f"\n  IB header (SCALE={ib_scale}):")
        for line in ib_header.decode("ascii").strip("\x00").strip().split("\n"):
            print(f"    {line}")

    # ── Write DADA files (CB only — IB on-disk default is optional) ─────
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nWriting CB DADA files to: {args.output_dir}")
    for stream_id in streams:
        chan0 = stream_id * SUBBAND_WIDTH
        subband = w[:, chan0:chan0 + SUBBAND_WIDTH, :, :, :]
        fname = f"direct.dada.{stream_id}"
        fpath = os.path.join(args.output_dir, fname)
        if args.dry_run:
            print(f"  [DRY RUN] Would write {fname} "
                  f"(channels {chan0}-{chan0 + SUBBAND_WIDTH - 1})")
        else:
            write_dada_file(fpath, cb_header, subband.tobytes())
            print(f"  {fname}: {os.path.getsize(fpath):,} bytes "
                  f"(channels {chan0}-{chan0 + SUBBAND_WIDTH - 1})")

    if w_ib is not None:
        print(f"\nWriting IB DADA files to: {args.output_dir}")
        for stream_id in streams:
            chan0 = stream_id * SUBBAND_WIDTH
            ib_subband = w_ib[chan0:chan0 + SUBBAND_WIDTH, :]
            fname = f"direct_ib.dada.{stream_id}"
            fpath = os.path.join(args.output_dir, fname)
            if args.dry_run:
                print(f"  [DRY RUN] Would write {fname} "
                      f"(channels {chan0}-{chan0 + SUBBAND_WIDTH - 1})")
            else:
                write_dada_file(fpath, ib_header, ib_subband.tobytes())
                print(f"  {fname}: {os.path.getsize(fpath):,} bytes "
                      f"(channels {chan0}-{chan0 + SUBBAND_WIDTH - 1})")

    # ── Upload to pipes ──────────────────────────────────────────────────
    if args.upload:
        print(f"\nUploading CB to live pipeline (UTC_START={utc_start}):")
        for stream_id in streams:
            host = STREAM_HOST[stream_id]
            chan0 = stream_id * SUBBAND_WIDTH
            subband = w[:, chan0:chan0 + SUBBAND_WIDTH, :, :, :]
            write_to_pipe(host, stream_id, cb_header, subband.tobytes(),
                          dry_run=args.dry_run)

        if w_ib is not None:
            print(f"\nUploading IB to live pipeline (UTC_START={utc_start}):")
            for stream_id in streams:
                host = STREAM_HOST[stream_id]
                chan0 = stream_id * SUBBAND_WIDTH
                ib_subband = w_ib[chan0:chan0 + SUBBAND_WIDTH, :]
                pipe_path = args.ib_pipe_fmt.format(stream=stream_id)
                write_to_pipe_path(host, pipe_path, ib_header,
                                   ib_subband.tobytes(),
                                   dry_run=args.dry_run)

    # ── Save as defaults ─────────────────────────────────────────────────
    if args.save_defaults:
        try:
            save_defaults("cb", streams, args.output_dir,
                          dry_run=args.dry_run)
            if w_ib is not None:
                save_defaults("ib", streams, args.output_dir,
                              dry_run=args.dry_run)
            else:
                print("\nNOTE: no --ib-weights given, so IB defaults "
                      "(incoh.dada.N) were NOT refreshed; the next restart "
                      "will load whatever IB defaults are already on the "
                      "corr machines.")
        except RuntimeError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            print("Default weights are NOT fully saved. The next obs restart "
                  "would load stale defaults — fix the failure and re-run "
                  "with --save-defaults before restarting.", file=sys.stderr)
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
