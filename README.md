# CASM Geometric Beamformer Weights Generator

A Python package for computing geometric beamformer weights for the CASM (Coherent All-Sky Monitor) array at OVRO (Owens Valley Radio Observatory).

## Features

- **Stationary beams**: Fixed in Alt/Az, time-independent weights for FRB transit search
- **Tracking beams**: Follow RA/Dec sources, time-dependent weights
- **Coherent beamforming**: Phase-align signals toward specified directions
- **Incoherent beamforming**: Total power beams (no phase steering)
- **Beam grid generation**: Automatically tile the sky with configurable spacing
- **Frequency-dependent phases**: Correct across 3072 channels (375-469 MHz)
- **Antenna flagging**: Exclude bad antennas from weight computation
- **HDF5 & NPZ output**: Compressed file formats with full metadata

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install with all dependencies
pip install -e ".[full]"
```

### Dependencies

- **Required**: `numpy>=1.20.0`
- **Optional**: `astropy>=5.0` (coordinate transforms), `h5py>=3.0` (HDF5 support)

## Quick Start

```python
from bf_weights_generator import (
    GeometricBeamformer,
    PhaseCenter,
    StationaryPointing,
    generate_beam_grid_altaz,
    save_weights_hdf5,
)

bf = GeometricBeamformer()

# === STATIONARY BEAMS (FRB Transit Search) ===
# Fixed in Alt/Az, time-independent weights
pointings = generate_beam_grid_altaz(spacing_deg=4.0)
stationary = bf.compute_stationary_weights(pointings, mode='coherent')
# Shape: (n_beams, n_ant, n_chan) - NO time axis

# === TRACKING BEAMS (Source Monitoring) ==
# Follow RA/Dec, time-dependent weights
cyg_a = PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA")
tracking = bf.compute_tracking_weights(
    phase_centers=[cyg_a],
    start_time=1700000000.0,
    duration_sec=3600,
    cadence_sec=10.0
)
# Shape: (n_beams, n_times, n_ant, n_chan) - HAS time axis

save_weights_hdf5(stationary, "stationary_weights.h5")
save_weights_hdf5(tracking, "tracking_weights.h5")
```

## Beamforming Modes

### Stationary vs Tracking

| Property | Stationary | Tracking |
|----------|-----------|----------|
| Pointing frame | Alt/Az (local horizon) | RA/Dec (celestial) |
| Weights | Time-independent | Time-dependent |
| Weight shape | `(n_beams, n_ant, n_chan)` | `(n_beams, n_times, n_ant, n_chan)` |


### Coherent vs Incoherent

| Property | Coherent | Incoherent |
|----------|----------|------------|
| Phase steering | Yes: `w = exp(-2πi·f·τ)` | No: weights are unity |
| Output | `beam = \|W·v\|²` | `beam = v*·v = \|v\|²` |
| Sensitivity | Scales as N antennas | Scales as sqrt(N) |

## Stationary Beam Examples

### Generate beam grid for FRB search

```python
from bf_weights_generator import generate_beam_grid_altaz, GeometricBeamformer

# Generate 682 beams covering Alt 30°-90°, full azimuth
pointings = generate_beam_grid_altaz(
    alt_min_deg=30.0,
    alt_max_deg=90.0,
    spacing_deg=4.0  # ~4° beam spacing (tunable)
)

bf = GeometricBeamformer()
weights = bf.compute_stationary_weights(pointings, mode='coherent')
print(f"Shape: {weights.shape}")  # (682, 13, 3072)
```

### Custom beam pointings

```python
from bf_weights_generator import StationaryPointing

pointings = [
    StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith"),
    StationaryPointing(alt_deg=60.0, az_deg=0.0, name="north_60"),
    StationaryPointing(alt_deg=60.0, az_deg=90.0, name="east_60"),
    # Can also specify by direction cosines
    StationaryPointing(l=0.3, m=0.4, name="custom"),
]

weights = bf.compute_stationary_weights(pointings, mode='coherent')
```

### Single stationary beam at specific Alt/Az

```python
from bf_weights_generator import StationaryPointing, GeometricBeamformer

bf = GeometricBeamformer()

# Zenith beam
zenith = StationaryPointing(alt_deg=90.0, az_deg=0.0, name="Zenith")
weights = bf.compute_stationary_weights([zenith], mode='coherent')
print(f"Shape: {weights.weights.shape}")  # (1, 13, 3072)

# South at 60° altitude
south60 = StationaryPointing(alt_deg=60.0, az_deg=180.0, name="South60")
weights = bf.compute_stationary_weights([south60], mode='coherent')

# Get the geometric delays
delays = bf.compute_stationary_delays(zenith)
print(f"Delays (ns): {delays * 1e9}")
```

### Stationary incoherent beams

```python
# 8 identical total-power beams (no phase steering)
weights = bf.compute_stationary_weights(mode='incoherent', n_beams=8)
# All weights are unity; at runtime: beam = v* × v = |v|²
```

## Tracking Beam Examples

### Track a celestial source

```python
from bf_weights_generator import PhaseCenter

cyg_a = PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA")

weights = bf.compute_tracking_weights(
    phase_centers=[cyg_a],
    start_time=1700000000.0,
    duration_sec=3600,
    cadence_sec=10.0,
    mode='coherent'
)
print(f"Shape: {weights.shape}")  # (1, 360, 13, 3072)
```


## Configuration

### Array Configuration

```python
from bf_weights_generator import ArrayConfig

arr = ArrayConfig()
arr.flag_antennas([3, 12])  # Flag bad antennas

bf = GeometricBeamformer(array_config=arr)
```

### Frequency Configuration

```python
from bf_weights_generator import FrequencyConfig

freq = FrequencyConfig(
    n_chan=1024,              # Number of channels (default: 3072)
    total_bw_mhz=125.0,       # Total bandwidth
    total_n_chan=4096,        # System channels
    freq_end_voltage_mhz=468.75
)

bf = GeometricBeamformer(freq_config=freq)
```

## File I/O

### Python API

```python
from bf_weights_generator import save_weights_hdf5, load_weights_hdf5, inspect_weights_file

# Save to HDF5
save_weights_hdf5(weights, "weights.h5", compression_opts=4)

# Save to NPZ
from bf_weights_generator import save_weights_npz
save_weights_npz(weights, "weights.npz")

# Inspect without loading
info = inspect_weights_file("weights.h5")
print(f"Type: {'stationary' if info['is_stationary'] else 'tracking'}")
print(f"Shape: {info['weights_shape']}")

# Load
weights = load_weights_hdf5("weights.h5")
```

### Command-Line: Dump Stationary Weights

Use `examples/dump_stationary_weights.py` to generate and save stationary beam weights:

```bash
# Single beam at zenith (HDF5/NPZ format)
python examples/dump_stationary_weights.py --alt 90 --az 0 --name Zenith -o zenith.h5

# Incoherent weights (unity)
python examples/dump_stationary_weights.py --alt 90 --az 0 --mode incoherent -o incoh.h5

# Custom grid parameters
python examples/dump_stationary_weights.py --grid --alt-min 45 --alt-max 90 --alt-step 15 --az-step 30 -o grid.h5
```

**Options:**
| Option | Description |
|--------|-------------|
| `--alt` | Altitude in degrees (0=horizon, 90=zenith) |
| `--az` | Azimuth in degrees (0=North, 90=East, 180=South) |
| `--name` | Beam name for metadata |
| `--mode` | `coherent` (default) or `incoherent` |
| `--grid` | Generate multiple beams in a grid |
| `-o` | Output file (.h5 for HDF5, .npz for NumPy) |

### Command-Line: Read and Inspect Weights Files

Use `examples/read_weights_file.py` to inspect saved weight files:

```bash
# Read HDF5 file
python examples/read_weights_file.py zenith.h5

# Read NPZ file
python examples/read_weights_file.py zenith.npz

# Verbose mode (show sample weight values)
python examples/read_weights_file.py zenith.h5 --verbose
```

**Example output:**
```
weights:
  Shape: (1, 7, 3072)
  Dtype: complex64
  Size: 0.17 MB
  -> (n_beams=1, n_ant=7, n_chan=3072)

frequencies_hz:
  Shape: (3072,)
  Range: 468.73 - 375.02 MHz

Pointings (stationary beam):
  Beam 0: Alt=90.0°, Az=0.0°
```

### Command-Line: Compute Delays for DADA Files

Use `examples/compute_delays_for_dada.py` for working with voltage data:

```bash
# Compute tracking delays for Cas A
python examples/compute_delays_for_dada.py --source casa --time "2026-01-21T01:16:44"

# Compute stationary delays at Alt/Az
python examples/compute_delays_for_dada.py --alt 60 --az 180 --name South60

# Save delays to file
python examples/compute_delays_for_dada.py --source cyga --output delays.npz
```

## HDF5 File Structure

```
/
├── weights              # Complex64 array
├── frequencies_hz       # Float64 (n_chan,)
├── antenna_indices      # Active antenna indices
├── unix_times           # (tracking only) Float64 (n_times,)
├── pointings/           # (stationary only) Alt/Az and l,m,n
│   ├── alt_deg, az_deg, l, m, n
├── phase_centers/       # (tracking only) RA/Dec
│   ├── ra_deg, dec_deg
├── array_config/        # Antenna positions
└── freq_config/         # Frequency setup
```

## CASM Array Configuration

13 antennas (current), expandable to 256 (43 SNAPs × 6 antennas).

| Antenna | East (m) | North (m) | Up (m) |
|---------|----------|-----------|--------|
| 0-5     | 0.0-2.03 | 0.0       | 0.0    |
| 6-11    | 0.0-2.03 | -10.5     | 0.0    |
| 12      | -4.553   | -5.5      | -0.279 |


## Beam Spacing

Adjust `spacing_deg` based on your array size and desired overlap. Default is `4 deg`.

## Example Scripts

The `examples/` directory contains ready-to-use scripts:

| Script | Description |
|--------|-------------|
| `basic_usage.py` | Python API examples for all beamforming modes |
| `dump_stationary_weights.py` | CLI to generate and save stationary beam weights |
| `read_weights_file.py` | CLI to inspect saved HDF5/NPZ weight files |
| `compute_delays_for_dada.py` | CLI for computing delays and processing DADA voltage files |
| `generate_example_weights.py` | Generate example weights for all 4 beamforming modes |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## TODO

- [ ] Cable delay calibration pipeline (in progress, see `tests/calibrate_cable_delays.py`)
- [ ] Validate calibration with bright source observations

## License

MIT License
