#!/usr/bin/env python3
"""
Example: Read and inspect saved beamformer weights file.

This script reads HDF5 or NPZ weight files and prints their contents
to verify the data was saved correctly.

Usage:
    python read_weights_file.py weights.h5
    python read_weights_file.py weights.npz
    python read_weights_file.py weights.h5 --verbose
"""

import sys
import argparse
import numpy as np

sys.path.insert(0, '/home/casm/software/vishnu/bf_weights_generator')


def read_hdf5(filepath: str, verbose: bool = False):
    """Read and display HDF5 weight file contents."""
    import h5py

    print(f"\nReading HDF5 file: {filepath}")
    print("=" * 60)

    with h5py.File(filepath, 'r') as f:
        # Print structure
        print("\nFile structure:")
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"  {indent}{name}: {obj.shape} {obj.dtype}")
            else:
                print(f"  {indent}{name}/")
        f.visititems(print_structure)

        # Print main arrays
        print("\n" + "-" * 60)
        print("Main arrays:")
        print("-" * 60)

        if 'weights' in f:
            weights = f['weights'][:]
            print(f"\n  weights:")
            print(f"    Shape: {weights.shape}")
            print(f"    Dtype: {weights.dtype}")
            print(f"    Size: {weights.nbytes / 1e6:.2f} MB")

            if weights.ndim == 3:
                print(f"    -> (n_beams={weights.shape[0]}, n_ant={weights.shape[1]}, n_chan={weights.shape[2]})")
            elif weights.ndim == 4:
                print(f"    -> (n_beams={weights.shape[0]}, n_times={weights.shape[1]}, n_ant={weights.shape[2]}, n_chan={weights.shape[3]})")

        if 'frequencies_hz' in f:
            freqs = f['frequencies_hz'][:]
            print(f"\n  frequencies_hz:")
            print(f"    Shape: {freqs.shape}")
            print(f"    Range: {freqs[0]/1e6:.2f} - {freqs[-1]/1e6:.2f} MHz")

        # Print attributes
        print("\n" + "-" * 60)
        print("Attributes:")
        print("-" * 60)
        for key, val in f.attrs.items():
            print(f"  {key}: {val}")

        # Print pointing/phase center info
        if 'pointings' in f:
            print("\n" + "-" * 60)
            print("Pointings (stationary beam):")
            print("-" * 60)
            pg = f['pointings']
            if 'alt_deg' in pg:
                alt = pg['alt_deg'][:]
                az = pg['az_deg'][:]
                for i in range(min(len(alt), 10)):
                    print(f"  Beam {i}: Alt={alt[i]:.1f}°, Az={az[i]:.1f}°")
                if len(alt) > 10:
                    print(f"  ... and {len(alt) - 10} more beams")

        if 'phase_centers' in f:
            print("\n" + "-" * 60)
            print("Phase centers (tracking beam):")
            print("-" * 60)
            pc = f['phase_centers']
            if 'ra_deg' in pc:
                ra = pc['ra_deg'][:]
                dec = pc['dec_deg'][:]
                for i in range(min(len(ra), 10)):
                    print(f"  Beam {i}: RA={ra[i]:.4f}°, Dec={dec[i]:.4f}°")

        # Verbose: print sample weight values
        if verbose and 'weights' in f:
            print("\n" + "-" * 60)
            print("Sample weight values (beam 0, ant 0, first 5 channels):")
            print("-" * 60)
            w = f['weights'][0, 0, :5] if f['weights'].ndim == 3 else f['weights'][0, 0, 0, :5]
            for i, val in enumerate(w):
                print(f"  Chan {i}: {val:.6f} (amp={np.abs(val):.4f}, phase={np.angle(val):.4f} rad)")

    print("\n" + "=" * 60)


def read_npz(filepath: str, verbose: bool = False):
    """Read and display NPZ weight file contents."""
    print(f"\nReading NPZ file: {filepath}")
    print("=" * 60)

    data = np.load(filepath, allow_pickle=True)

    print("\nArrays in file:")
    print("-" * 60)
    for key in sorted(data.files):
        arr = data[key]
        if isinstance(arr, np.ndarray):
            if arr.ndim == 0:
                # Scalar
                print(f"  {key}: {arr.item()}")
            else:
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  {key}: {arr}")

    # Print main arrays
    print("\n" + "-" * 60)
    print("Main arrays:")
    print("-" * 60)

    if 'weights' in data:
        weights = data['weights']
        print(f"\n  weights:")
        print(f"    Shape: {weights.shape}")
        print(f"    Dtype: {weights.dtype}")
        print(f"    Size: {weights.nbytes / 1e6:.2f} MB")

        if weights.ndim == 3:
            print(f"    -> (n_beams={weights.shape[0]}, n_ant={weights.shape[1]}, n_chan={weights.shape[2]})")
        elif weights.ndim == 4:
            print(f"    -> (n_beams={weights.shape[0]}, n_times={weights.shape[1]}, n_ant={weights.shape[2]}, n_chan={weights.shape[3]})")

    if 'frequencies_mhz' in data:
        freqs = data['frequencies_mhz']
        print(f"\n  frequencies_mhz:")
        print(f"    Shape: {freqs.shape}")
        print(f"    Range: {freqs[0]:.2f} - {freqs[-1]:.2f} MHz")

    # Print pointing info
    if 'alt_deg' in data:
        print("\n" + "-" * 60)
        print("Pointing (stationary beam):")
        print("-" * 60)
        print(f"  Alt: {data['alt_deg']}°")
        print(f"  Az: {data['az_deg']}°")
        if 'l' in data:
            print(f"  Direction cosines: l={data['l']}, m={data['m']}, n={data['n']}")

    if 'source_ra_deg' in data:
        print("\n" + "-" * 60)
        print("Phase center (tracking beam):")
        print("-" * 60)
        print(f"  RA: {data['source_ra_deg']}°")
        print(f"  Dec: {data['source_dec_deg']}°")
        if 'source_name' in data:
            print(f"  Name: {data['source_name']}")

    # Verbose: print sample values
    if verbose and 'weights' in data:
        print("\n" + "-" * 60)
        print("Sample weight values (beam 0, ant 0, first 5 channels):")
        print("-" * 60)
        w = data['weights']
        w_sample = w[0, 0, :5] if w.ndim == 3 else w[0, 0, 0, :5]
        for i, val in enumerate(w_sample):
            print(f"  Chan {i}: {val:.6f} (amp={np.abs(val):.4f}, phase={np.angle(val):.4f} rad)")

    print("\n" + "=" * 60)
    data.close()


def main():
    parser = argparse.ArgumentParser(
        description='Read and inspect beamformer weights file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python read_weights_file.py zenith.h5
    python read_weights_file.py zenith.npz
    python read_weights_file.py beam_grid.h5 --verbose
        """
    )

    parser.add_argument('filepath', type=str, help='Path to weights file (.h5 or .npz)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print sample weight values')

    args = parser.parse_args()

    filepath = args.filepath

    if filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        read_hdf5(filepath, args.verbose)
    elif filepath.endswith('.npz'):
        read_npz(filepath, args.verbose)
    else:
        print(f"Unknown file format: {filepath}")
        print("Supported formats: .h5, .hdf5, .npz")
        sys.exit(1)


if __name__ == "__main__":
    main()
