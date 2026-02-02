# CASM Beamformer Weights Generator

Generate geometric beamformer weights for the CASM phased array at OVRO.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install
pip install -e ".[full]"
```

**Dependencies:** numpy, h5py, astropy (optional)

## Quick Start: Generate SNAP Weights

### Command Line

```bash
# Generate 8 beams covering the sky
python examples/generate_snap_weights.py \
    --layout casm_antenna_layout1.csv \
    --output weights.h5 \
    --n-beams 8

# Custom beam spacing (degrees)
python examples/generate_snap_weights.py \
    --layout casm_antenna_layout1.csv \
    --output weights.h5 \
    --spacing 20

# Specific beam positions (alt:az format)
python examples/generate_snap_weights.py \
    --layout casm_antenna_layout1.csv \
    --output weights.h5 \
    --beams "90:0,70:0,70:90,70:180"

# Transit survey preset (8 beams for FRB search)
python examples/generate_snap_weights.py \
    --layout casm_antenna_layout1.csv \
    --output weights.h5 \
    --beams transit
```

### Python API

```python
from bf_weights_generator import (
    Array64Config,
    SnapWeightsGenerator,
    generate_beam_grid,
    save_int8_weights_hdf5,
)

# Load antenna layout
array = Array64Config.from_csv("casm_antenna_layout1.csv")
gen = SnapWeightsGenerator(array)

# Generate beams and compute weights
beams = generate_beam_grid(n_beams=8)
weights = gen.compute_int8_weights(beams)

# Save
save_int8_weights_hdf5(weights, "weights.h5")
```

## Standard Stationary Weights (13 antennas)

For the default 13-antenna array without SNAP reordering:

```python
from bf_weights_generator import (
    GeometricBeamformer, StationaryPointing, save_weights_hdf5
)

bf = GeometricBeamformer()

pointings = [
    StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith"),
    StationaryPointing(alt_deg=60.0, az_deg=45.0, name="ne60"),
]

weights = bf.compute_stationary_weights(pointings)  # (n_beams, 13, 3072)
save_weights_hdf5(weights, "weights.h5")
```

## Reading Weight Files

```python
from bf_weights_generator import load_int8_weights_hdf5

weights = load_int8_weights_hdf5("weights.h5")

# Int8 data: shape (2, 3072, 2, n_beams, 64)
# Axes: [real/imag, channel, pol, beam, antenna]
data = weights.weights_int8

# Convert to complex
complex_weights = weights.to_complex64()  # (n_beams, 64, n_chan)

# List beam pointings
for i, p in enumerate(weights.pointings):
    print(f"Beam {i}: Alt={p.alt_deg:.1f}°, Az={p.az_deg:.1f}°")

# Frequencies (Hz)
freqs = weights.frequencies_hz
```

## Plotting Beams

```bash
python examples/plot_beams.py --weights weights.h5 --output beams.png
```

## Beam Grid Options

| Method | Example |
|--------|---------|
| Auto-spacing for N beams | `generate_beam_grid(n_beams=8)` |
| Fixed spacing | `generate_beam_grid(spacing_deg=20.0)` |
| Altitude limits | `generate_beam_grid(n_beams=8, alt_min_deg=45.0)` |

## Transit Survey Beams

The preset `transit` pattern provides 8 beams for FRB detection:

| Beam | Alt | Az | Position |
|------|-----|-----|----------|
| 0 | 90° | 0° | Zenith |
| 1 | 70° | 0° | North |
| 2 | 70° | 90° | East |
| 3 | 70° | 180° | South |
| 4 | 70° | 270° | West |
| 5 | 50° | 45° | NE |
| 6 | 50° | 135° | SE |
| 7 | 50° | 225° | SW |

## CSV Layout Format

Your antenna layout CSV needs these columns:

| Column | Description |
|--------|-------------|
| `ant64` | Slot index (0-63) |
| `x_east_m`, `y_north_m`, `z_up_m` | ENU coordinates |
| `snap_A`, `adc_A` | SNAP board and ADC channel |
| `include_in_beamforming` | Boolean |
| `pos_type` | Must be "antenna" |

## Output Format

**Shape:** `(2, n_chan, 2, n_beams, 64)`

| Axis | Meaning |
|------|---------|
| 0 | Real/Imaginary (0=real, 1=imag) |
| 1 | Channels (reversed, low-to-high) |
| 2 | Polarization (both identical) |
| 3 | Beams |
| 4 | Antennas (SNAP input order) |

**Quantization:** `int8_value = round(complex_weight * 127)`

## Running Tests

```bash
pytest tests/ -v
```

## TODO

- Tracking beams (follow RA/Dec sources)
- Cable delay calibration integration

## License

MIT
