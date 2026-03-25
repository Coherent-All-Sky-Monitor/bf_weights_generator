# bf_weights_generator

Generate beamformer weights for the CASM phased array at OVRO. Takes calibration weights from `casm-svd-calibrate` and produces SNAP-ready weight files with geometric steering + delay calibration combined.

## Install

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/bf_weights_generator
pip install -e ".[full]"
```

## How to generate beamforming weights

### 1. Generate calibration weights (casm_calibrator)

```bash
casm-svd-calibrate \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs 2026-03-20-05:55:45 \
  --source sun \
  --layout ~/software/dev/antenna_layouts/antenna_layout_mar21.csv \
  --ref-ant 3 \
  --time-start '2026-03-21 10:00:00' --time-end '2026-03-21 15:00:00' --time-tz US/Pacific \
  --output cal_weights.npz \
  --plots cal_diagnostics.pdf
```

### 2. Generate 512-beam stationary weights (this repo)

Tile the sky with 512 beams in alt-az, combining geometric steering with calibration delays, and quantize to int8 for the SNAP hardware:

```python
from bf_weights_generator import (
    Array64Config, load_calibration_weights,
    save_int8_weights_hdf5, generate_beam_grid, SnapWeightsGenerator,
)

# Load antenna layout and calibration weights
layout = Array64Config.from_csv("antenna_layout_mar21.csv")
cal = load_calibration_weights("cal_weights.npz")

# Generate 512 beams tiling the sky (alt-az grid)
beams = generate_beam_grid(n_beams=512, array_config=layout)

# Quantize to int8 and save for SNAP hardware
gen = SnapWeightsGenerator(layout)
int8_weights = gen.compute_int8_weights(beams, cal_weights=cal)
save_int8_weights_hdf5(int8_weights, "weights_512beam.h5")
```

## CLI Commands

### casm-bf-weights

Read and inspect weight files.

```bash
# Print summary (format, n_beams, n_channels, freq range)
casm-bf-weights weights_512beam.h5

# List all beam pointings (alt/az)
casm-bf-weights weights_512beam.h5 --list-beams

# Show details for a specific beam
casm-bf-weights weights_512beam.h5 --beam 42

# Full metadata dump
casm-bf-weights weights_512beam.h5 --info
```

### casm-bf-plotter

Plot beam positions on a sky map (zenithal equidistant projection with FWHM ellipses).

```bash
# Plot beam layout
casm-bf-plotter weights_512beam.h5 -o beam_layout.png

# Custom frequency for FWHM calculation
casm-bf-plotter weights_512beam.h5 --freq 400e6 -o beams_400mhz.png
```

## Testing

```bash
pytest tests/ -v
```

## Detailed API reference

See [docs/api_reference.md](docs/api_reference.md) for the full Python API, output formats, CSV layout spec, and weight file structure.
