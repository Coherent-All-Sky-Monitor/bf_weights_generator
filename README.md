# bf_weights_generator

Generate beamformer weights for the CASM phased array at OVRO. Takes
calibration weights from `casm-svd-calibrate` and produces SNAP-ready
weight files with geometric steering and delay calibration combined.

## Install

```bash
source ~/software/dev/casm_venvs/casm_refactor_env/bin/activate
cd /home/casm/software/dev/bf_weights_generator
pip install -e ".[full]"
```

## Concepts

- **`Array64Config`** — Per-SNAP-slot antenna layout. Despite the name,
  arrays are sized to `n_snaps * n_adc` (currently `6 * 12 = 72`) so
  every slot in real CAsMan hardware is addressable. The class exposes
  `positions_enu`, `active_mask`, `antenna_ids` and the derived
  `active_indices` / `active_positions` / `n_active`. Build via
  `Array64Config.from_csv(path)` or `Array64Config.from_antenna_mapping(ant)`
  (where `ant` is a `casm_io.AntennaMapping`).
- **`FrequencyConfig`** — 3072 channels of a 4096-channel voltage system,
  125 MHz bandwidth. **The default tracks the legacy `layout_32ant`
  band (channel-0 upper edge 468.75 MHz)** for reproducibility of
  historical weights. Use `FrequencyConfig.layout_64ant()` for the
  post-Jan-27-2026 band (channel-0 center 484.375 MHz), or
  `FrequencyConfig.from_format(fmt)` to pin to a casm_io
  `VisibilityFormat`.
- **`CalibrationWeights`** — input from `casm_calibrator`; ant-aligned
  via `ant_ids`.
- **`Int8StationaryWeights`** — output container. Quantized weights of
  shape `(2, n_chan, 2, n_beams, n_slots)` = (real/imag, chan, pol,
  beam, snap_input_idx); always written in descending-frequency SNAP
  native order.
- **`CombinedWeights`** — complex64 container preserved before the int8
  quantization step (shape `(n_beams, n_slots, n_chan)`).
- **`StationaryPointing`** — fixed (alt, az) beam direction.
- **`generate_beam_grid_altaz(spacing_deg=...)`** — returns
  `List[StationaryPointing]` tiling the sky on an alt-az grid.

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

### 2. Generate the int8 weight file (this repo)

Two equivalent entry points:

```python
import numpy as np
from bf_weights_generator import (
    Array64Config,
    FrequencyConfig,
    SnapWeightsGenerator,
    StationaryPointing,
    generate_beam_grid_altaz,
    load_calibration_weights,
    save_int8_weights_hdf5,
)

# Layout + cal
layout = Array64Config.from_csv("antenna_layout_mar21.csv")
cal    = load_calibration_weights("cal_weights.npz")
freq   = FrequencyConfig.layout_64ant()   # post-Jan-27-2026 band

# 256-beam alt-az grid
pointings = generate_beam_grid_altaz(spacing_deg=4.0)

# Int8 weights, ready for SNAP firmware
gen = SnapWeightsGenerator(layout, freq_config=freq)
int8 = gen.compute_int8_weights(pointings=pointings, cal_weights=cal)

save_int8_weights_hdf5(int8, "weights_alt_az.h5")
```

Or, if you want the intermediate complex64 representation:

```python
from bf_weights_generator import (
    generate_combined_weights,
    save_combined_weights_hdf5,
)

combined = generate_combined_weights(
    pointing=pointings,
    array_config=layout,
    cal_weights=cal,
    freq_config=freq,
    freq_order="descending",   # or "ascending"; default descending = SNAP-native
)
save_combined_weights_hdf5(combined, "combined.h5", overwrite=True)

# When ready for hardware, quantize:
int8 = combined.to_int8()
save_int8_weights_hdf5(int8, "weights_alt_az.h5", overwrite=True)
```

### 3. Saved HDF5 schema (`save_int8_weights_hdf5`)

Root attributes:

| key | meaning |
|---|---|
| `format_type` | `"int8_snap_weights"` |
| `version` | `"2.0"` (current schema) |
| `n_beams`, `n_channels`, `n_pol`, `n_antennas` | mirror the array shape |
| `scale_factor` | int8 scale (default 127.0) |
| `created_utc` | ISO timestamp |

Datasets:

| path | shape | dtype | notes |
|---|---|---|---|
| `weights_int8` | `(2, n_chan, 2, n_beams, n_slots)` | int8 | real/imag, chan, pol, beam, snap_input_idx |
| `frequencies_hz` | `(n_chan,)` | float64 | descending |
| `pointings/alt_deg`, `pointings/az_deg` | `(n_beams,)` | float64 | with JSON `names` attribute |
| `array_config/positions_enu` | `(n_slots, 3)` | float64 |
| `array_config/active_mask` | `(n_slots,)` | bool |
| `array_config/antenna_ids` | `(n_slots,)` | int32 | v2.0 (replaces v1.0 `snap_to_ant64`/`ant64_to_snap` reorder pair) |

The v1.0 reader path still loads files written with `snap_to_ant64`;
the slot count is taken from the dataset length, so any future hardware
expansion past 64 slots loads correctly too.

## CLI commands

### `casm-bf-weights` — read / inspect

```bash
casm-bf-weights weights.h5                # summary
casm-bf-weights weights.h5 --list-beams   # all beam pointings
casm-bf-weights weights.h5 --beam 42      # detail for one beam
casm-bf-weights weights.h5 --info         # full metadata dump
```

### `casm-bf-plotter` — sky map

```bash
casm-bf-plotter weights.h5 -o beam_layout.png
casm-bf-plotter weights.h5 --freq 400e6 -o beams_400mhz.png
```

## Testing

```bash
pytest tests/ -v
```

## Detailed API reference

See [docs/api_reference.md](docs/api_reference.md) for the full Python
API, output formats, CSV layout spec, and weight file structure.
