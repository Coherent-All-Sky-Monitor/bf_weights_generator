#!/usr/bin/env python3
"""
Example: Read SNAP beamformer weights file.

Usage:
    # Activate virtual environment first:
    source /home/casm/software/vishnu/venv/bf_weights_generator/bin/activate

    # Run this script:
    python examples/read_weights_example.py weights.h5
"""

import sys
from pathlib import Path

# Add package to path if not installed
PACKAGE_DIR = Path(__file__).parent.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from bf_weights_generator import load_int8_weights_hdf5


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_weights_example.py <weights.h5>")
        sys.exit(1)

    filepath = sys.argv[1]
    weights = load_int8_weights_hdf5(filepath)

    # Print summary
    print(f"Loaded: {weights}")
    print(f"\nBeams ({weights.n_beams}):")
    for i, p in enumerate(weights.pointings):
        print(f"  {i}: Alt={p.alt_deg:.1f}°, Az={p.az_deg:.1f}° ({p.name})")

    print(f"\nFrequencies: {weights.frequencies_hz[0]/1e6:.2f} - {weights.frequencies_hz[-1]/1e6:.2f} MHz")
    print(f"Active antennas: {weights.array_config.n_active}/64")

    # Convert to complex64
    complex_w = weights.to_complex64()
    print(f"\nComplex weights shape: {complex_w.shape}")


if __name__ == "__main__":
    main()
