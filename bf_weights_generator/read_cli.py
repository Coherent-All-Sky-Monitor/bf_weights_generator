"""
CLI tool to read and inspect beamforming weight HDF5 files.

Usage:
    casm-bf-weights weights.h5                  # Print file summary
    casm-bf-weights weights.h5 --beam 3         # Print alt/az for beam 3
    casm-bf-weights weights.h5 --list-beams     # Table of all beam coordinates
    casm-bf-weights weights.h5 --info           # Full metadata dump
"""

import argparse
import os
import sys

try:
    import h5py
except ImportError:
    h5py = None

try:
    import numpy as np
except ImportError:
    np = None


def _detect_format(f):
    """Detect HDF5 format type from file contents."""
    fmt = f.attrs.get("format_type", None)
    if fmt is not None:
        return str(fmt)
    # v2 stationary/tracking format
    if "version" in f.attrs and "weights" in f:
        is_stationary = f.attrs.get("is_stationary", False)
        return "v2_stationary" if is_stationary else "v2_tracking"
    return "unknown"


def _format_label(fmt):
    """Human-readable format label."""
    labels = {
        "int8_snap_weights": "int8_snap_weights",
        "combined_complex64_snap_weights": "combined_complex64_snap_weights",
        "v2_stationary": "v2_stationary",
        "v2_tracking": "v2_tracking",
    }
    return labels.get(fmt, fmt)


def _get_n_beams(f, fmt):
    """Get number of beams from file."""
    if "n_beams" in f.attrs:
        return int(f.attrs["n_beams"])
    if "pointings" in f and "alt_deg" in f["pointings"]:
        return f["pointings"]["alt_deg"].shape[0]
    if "phase_centers" in f and "ra_deg" in f["phase_centers"]:
        return f["phase_centers"]["ra_deg"].shape[0]
    return 0


def _get_n_channels(f):
    """Get number of channels."""
    if "n_channels" in f.attrs:
        return int(f.attrs["n_channels"])
    if "frequencies_hz" in f:
        return f["frequencies_hz"].shape[0]
    return 0


def _get_n_antennas(f):
    """Get number of antennas."""
    if "n_antennas" in f.attrs:
        return int(f.attrs["n_antennas"])
    return 0


def _is_tracking(f, fmt):
    """Check if this is a tracking file (RA/Dec instead of Alt/Az)."""
    return fmt == "v2_tracking" or "phase_centers" in f


def _get_freq_range(f):
    """Get frequency range in MHz."""
    if "frequencies_hz" in f:
        freqs = f["frequencies_hz"][:]
        return freqs.min() / 1e6, freqs.max() / 1e6
    if "freq_config" in f:
        fc = f["freq_config"]
        end = float(fc.attrs.get("freq_end_voltage_mhz", 0))
        bw = float(fc.attrs.get("total_bw_mhz", 0))
        if end and bw:
            return end - bw, end
    return None, None


def _get_beam_fwhm(f):
    """Compute beam FWHM (E-W, N-S) in degrees from antenna positions in the file."""
    if np is None:
        return None, None

    # Find antenna positions and active mask
    positions_enu = None
    for grp_name in ("array_config", "compute_array_config"):
        if grp_name in f:
            grp = f[grp_name]
            positions_enu = grp["positions_enu"][:]
            if "active_mask" in grp:
                positions_enu = positions_enu[grp["active_mask"][:]]
            elif "antenna_flags" in grp:
                positions_enu = positions_enu[grp["antenna_flags"][:].astype(bool)]
            break

    if positions_enu is None:
        return None, None

    # Center frequency
    freq_hz = 437.5e6
    if "frequencies_hz" in f:
        freqs = f["frequencies_hz"][:]
        freq_hz = float(freqs.mean())

    wavelength = 299792458.0 / freq_hz
    d_ew = positions_enu[:, 0].max() - positions_enu[:, 0].min()
    d_ns = positions_enu[:, 1].max() - positions_enu[:, 1].min()

    fwhm_ew = float(np.rad2deg(wavelength / d_ew)) if d_ew > 0 else 180.0
    fwhm_ns = float(np.rad2deg(wavelength / d_ns)) if d_ns > 0 else 180.0
    return fwhm_ew, fwhm_ns


def cmd_summary(f, filepath, fmt):
    """Print compact file summary (default)."""
    n_beams = _get_n_beams(f, fmt)
    n_chan = _get_n_channels(f)
    n_ant = _get_n_antennas(f)
    freq_lo, freq_hi = _get_freq_range(f)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

    print(f"File: {os.path.basename(filepath)}")
    print(f"Format: {_format_label(fmt)}")
    if _is_tracking(f, fmt):
        print(f"Beams: {n_beams} (tracking, RA/Dec), Channels: {n_chan}, Antennas: {n_ant}")
    else:
        print(f"Beams: {n_beams}, Channels: {n_chan}, Antennas: {n_ant}")
    if freq_lo is not None:
        print(f"Freq range: {freq_lo:.2f} - {freq_hi:.2f} MHz")
    print(f"File size: {file_size_mb:.2f} MB")


def cmd_beam(f, filepath, fmt, beam_idx):
    """Print coordinates for a single beam."""
    n_beams = _get_n_beams(f, fmt)
    if beam_idx < 0 or beam_idx >= n_beams:
        print(f"Error: beam index {beam_idx} out of range (0-{n_beams - 1})", file=sys.stderr)
        sys.exit(1)

    if _is_tracking(f, fmt):
        ra = f["phase_centers"]["ra_deg"][beam_idx]
        dec = f["phase_centers"]["dec_deg"][beam_idx]
        print(f"Beam {beam_idx}: RA = {ra:.4f} deg, Dec = {dec:.4f} deg")
    else:
        alt = f["pointings"]["alt_deg"][beam_idx]
        az = f["pointings"]["az_deg"][beam_idx]
        print(f"Beam {beam_idx}: Alt = {alt:.2f} deg, Az = {az:.2f} deg")


def cmd_list_beams(f, filepath, fmt):
    """Print table of all beam coordinates."""
    n_beams = _get_n_beams(f, fmt)
    fwhm_ew, fwhm_ns = _get_beam_fwhm(f)
    print(f"Beams: {n_beams}")
    if fwhm_ew is not None:
        print(f"Beam FWHM: {fwhm_ew:.1f}\u00b0 (E-W) \u00d7 {fwhm_ns:.1f}\u00b0 (N-S)")

    if _is_tracking(f, fmt):
        ra = f["phase_centers"]["ra_deg"][:]
        dec = f["phase_centers"]["dec_deg"][:]
        print(f"  {'Index':>5}  {'RA (deg)':>9}  {'Dec (deg)':>9}")
        for i in range(n_beams):
            print(f"  {i:>5}  {ra[i]:>9.4f}  {dec[i]:>9.4f}")
    else:
        alt = f["pointings"]["alt_deg"][:]
        az = f["pointings"]["az_deg"][:]
        print(f"  {'Index':>5}  {'Alt (deg)':>9}  {'Az (deg)':>9}")
        for i in range(n_beams):
            print(f"  {i:>5}  {alt[i]:>9.2f}  {az[i]:>9.2f}")


def cmd_info(f, filepath, fmt):
    """Print full metadata dump."""
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

    fwhm_ew, fwhm_ns = _get_beam_fwhm(f)

    print(f"File: {filepath}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Format: {_format_label(fmt)}")
    if fwhm_ew is not None:
        print(f"Beam FWHM: {fwhm_ew:.1f}\u00b0 (E-W) \u00d7 {fwhm_ns:.1f}\u00b0 (N-S)")
    print()

    # Root attributes
    print("Attributes:")
    for key in sorted(f.attrs.keys()):
        val = f.attrs[key]
        print(f"  {key} = {val}")
    print()

    # Datasets and groups
    print("Contents:")

    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            attrs = dict(obj.attrs)
            if attrs:
                attr_str = ", ".join(f"{k}={v}" for k, v in list(attrs.items())[:5])
                if len(attrs) > 5:
                    attr_str += f", ... ({len(attrs)} total)"
                print(f"  {name}/  [{attr_str}]")
            else:
                print(f"  {name}/")

    f.visititems(_visitor)

    # Weights array info
    print("\nWeights:")
    for name in ("weights", "weights_int8"):
        if name in f:
            ds = f[name]
            print(f"  Dataset: {name}")
            print(f"  Shape: {ds.shape}")
            print(f"  Dtype: {ds.dtype}")
            size_mb = ds.nbytes / (1024 * 1024)
            print(f"  Uncompressed size: {size_mb:.2f} MB")
            break

    # Frequency axis
    freq_lo, freq_hi = _get_freq_range(f)
    n_chan = _get_n_channels(f)
    print(f"\nFrequency axis:")
    print(f"  Channels: {n_chan}")
    if freq_lo is not None:
        print(f"  Range: {freq_lo:.2f} - {freq_hi:.2f} MHz")
        if n_chan > 1:
            chan_bw = (freq_hi - freq_lo) / (n_chan - 1)
            print(f"  Channel width: {chan_bw * 1e3:.2f} kHz")
    if "freq_config" in f:
        fc = f["freq_config"]
        print(f"  Total BW: {fc.attrs.get('total_bw_mhz', '?')} MHz")
        print(f"  Total channels (full band): {fc.attrs.get('total_n_chan', '?')}")

    # Pointing summary
    n_beams = _get_n_beams(f, fmt)
    if n_beams > 0:
        print(f"\nBeams: {n_beams}")
        if _is_tracking(f, fmt) and "phase_centers" in f:
            ra = f["phase_centers"]["ra_deg"][:]
            dec = f["phase_centers"]["dec_deg"][:]
            print(f"  RA range:  {ra.min():.4f} - {ra.max():.4f} deg")
            print(f"  Dec range: {dec.min():.4f} - {dec.max():.4f} deg")
        elif "pointings" in f:
            alt = f["pointings"]["alt_deg"][:]
            az = f["pointings"]["az_deg"][:]
            print(f"  Alt range: {alt.min():.2f} - {alt.max():.2f} deg")
            print(f"  Az range:  {az.min():.2f} - {az.max():.2f} deg")


def read_main():
    """Entry point for casm-bf-weights CLI."""
    parser = argparse.ArgumentParser(
        prog="casm-bf-weights",
        description="Read and inspect beamforming weight HDF5 files.",
    )
    parser.add_argument("file", help="HDF5 weight file to read")
    parser.add_argument("--beam", type=int, metavar="N", help="Print alt/az for beam index N")
    parser.add_argument(
        "--list-beams", action="store_true", help="Print table of all beam coordinates"
    )
    parser.add_argument("--info", action="store_true", help="Full metadata dump")

    args = parser.parse_args()

    if h5py is None:
        print("Error: h5py is required. Install with: pip install h5py", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        with h5py.File(args.file, "r") as f:
            fmt = _detect_format(f)

            if args.beam is not None:
                cmd_beam(f, args.file, fmt, args.beam)
            elif args.list_beams:
                cmd_list_beams(f, args.file, fmt)
            elif args.info:
                cmd_info(f, args.file, fmt)
            else:
                cmd_summary(f, args.file, fmt)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    read_main()
