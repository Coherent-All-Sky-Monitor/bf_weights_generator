# bf_weights_generator

Generates SNAP-ready int8-quantized beamforming weights and incoherent-beam (IB)
masks for the CASM phased array at OVRO. Takes calibration weights from
`casm_calibrator` and antenna layout from `casm_io`, then writes `.h5` weight
files consumed by the online beamformer (hella / bfcorr).

## Installation

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
pip install -e "/home/casm/software/dev/bf_weights_generator[full]"
```

`[full]` pulls in `h5py` and `astropy`. Both are required for production use.

## Primary example

```python
from bf_weights_generator import (
    Array64Config, FrequencyConfig, generate_beam_grid_altaz,
    generate_combined_weights, load_calibration_weights,
    save_int8_weights_hdf5,
)
from casm_io.correlator import AntennaMapping

# No path -> the canonical $CASM_LAYOUT_DIR/current layout.
ant = AntennaMapping.load().with_inactive([3])
array_cfg = Array64Config.from_antenna_mapping(ant)
freq_cfg = FrequencyConfig.layout_64ant()          # recommended for current band

cal = load_calibration_weights("/path/to/cal.h5")  # casm_calibrator output

beams = generate_beam_grid_altaz(spacing_deg=4.0)  # 4° default recommended
combined = generate_combined_weights(
    pointing=beams,
    array_config=array_cfg,
    cal_weights=cal,
    freq_config=freq_cfg,
    freq_order="descending",                       # required for CASM native order
)
weights_int8 = combined.to_int8()
save_int8_weights_hdf5(weights_int8, "/tmp/weights.h5", overwrite=False)
```

## Key concepts

**`Array64Config`** holds 72 slots (6 SNAPs x 12 ADCs = `n_snaps * n_adc`). The
class name is a legacy artefact; slot count is never hard-coded. Size all
allocations from `len(array_config.positions_enu)`. Build via
`Array64Config.from_antenna_mapping(ant)`, not manual construction.

**`FrequencyConfig`** defaults to the legacy `layout_32ant` band (channel-0
upper edge 468.75 MHz). This default is frozen for reproducibility of historical
weights. For the current `layout_64ant` (post-Jan-27-2026) band, use
`FrequencyConfig.layout_64ant()` explicitly.

**`freq_order="descending"` is required** in `generate_combined_weights` to match
CASM-native channel order. Ascending output is byte-different and breaks bit-exact
reproduction of deployed weights.

**Two distinct scale numbers exist; do not conflate them.** The HDF5 attribute
`scale_factor=127` is the int8 quantization normalizer. The DADA header `SCALE`
written at FIFO upload time is 32 (CB) or 8 (IB). See
[docs/int8_weights.md](docs/int8_weights.md) for the full derivation.

## Module overview

| Module | Purpose |
|--------|---------|
| `snap_weights.py` | `Array64Config`, `SnapWeightsGenerator`, `Int8StationaryWeights`, `CombinedWeights`, `CalibrationWeights`, `generate_combined_weights` |
| `config.py` | `FrequencyConfig`, `ArrayConfig`, observatory constants, `compute_beam_fwhm` |
| `weights.py` | `GeometricBeamformer`, `StationaryPointing`, `generate_beam_grid_altaz` |
| `io.py` | `save_int8_weights_hdf5`, `load_int8_weights_hdf5`, `save_combined_weights_hdf5`, `load_combined_weights_hdf5` |
| `coordinates.py` | LST, direction cosines, geometric delay utilities |
| `deploy_bf_weights.py` | DADA-file generation and FIFO upload for the live pipeline |

## CLI commands

```bash
casm-bf-weights weights.h5               # summary inspection
casm-bf-weights weights.h5 --list-beams  # all beam pointings
casm-bf-weights weights.h5 --beam 42     # one beam detail
casm-bf-plotter weights.h5 -o beams.png  # sky map
casm-bf-inspect weights.h5               # full metadata dump
```

## Detailed documentation

| Topic | File |
|-------|------|
| Array64Config, FrequencyConfig | [docs/array_and_frequency_config.md](docs/array_and_frequency_config.md) |
| Beam grid generation, StationaryPointing | [docs/beam_grid.md](docs/beam_grid.md) |
| Int8 weights, HDF5 schema, two-scale clarification | [docs/int8_weights.md](docs/int8_weights.md) |
| IB binary mask format and FIFO upload | [docs/ib_weights.md](docs/ib_weights.md) |
| Common workflows and recipes | [docs/recipes.md](docs/recipes.md) |

## Testing

```bash
pytest tests/ -v
# 95 pass, 12 skip (skips require optional voltage-data fixtures)
```

## Repository boundary

- **Consumes**: `CalibrationWeights` from `casm_calibrator`; antenna layout via `casm_io.AntennaMapping`
- **Produces**: `.h5` weight files for hella / bfcorr; FIFO upload via `deploy_bf_weights.py`
- **Does not**: perform SVD calibration, apply RFI masks, or manage the deployment FIFO lifecycle
