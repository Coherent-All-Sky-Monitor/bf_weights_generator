# Incoherent beam (IB) weights

## Binary mask shape

The IB binary mask has shape `(n_chan, 132)`, where:

```
132 = nsig_per_snap (12) * nsnap (11)
```

This is the exact format the online pipeline (hella / bfcorr) expects. The mask
indicates which SNAP signal slots are active for the incoherent beam; active slots
are 1, inactive (unwired) slots are 0. The mask has no beam dimension because the
IB is a single total-power sum over all active antennas.

The 132-slot dimension maps to `snap_input_idx = snap_id * 12 + adc` for 11
SNAPs. The current hardware uses 6 SNAPs (72 slots); the remaining 60 entries
in the 132-slot vector are zero.

## DADA file format

The FIFO upload sends a 4096-byte ASCII header followed by the weight data, one
DADA file per 512-channel subband (6 subbands for 3072 channels). The header
includes:

```
FILE_SIZE <data_bytes>
RESOLUTION <data_bytes>
UTC_START <YYYY-MM-DD-HH:MM:SS>
NBIT 8
SCALE <value>
```

The `SCALE` field is **32 for CB** and **8 for IB**. This is the gain factor
that bfcorr/hella applies at runtime. See [int8_weights.md](int8_weights.md) for
the two-scale derivation and the 0.985 |CB|/|IB| ratio.

## deploy_bf_weights.py

The deployment script converts an HDF5 weight file into per-subband DADA files
and optionally uploads them to the live pipeline via SSH to the named FIFOs on
casm-corr1 and casm-corr2.

### FIFO paths

```
/tmp/bfweights.fifo.{stream}      streams 0-2 -> casm-corr1
                                  streams 3-5 -> casm-corr2
```

### DADA file output

Local DADA files are written as `direct.dada.{0..5}` (CB) or
`direct_ib.dada.{0..5}` (IB) in the output directory.

### Usage

```bash
# Write DADA files to current directory (no live upload)
python bf_weights_generator/deploy_bf_weights.py weights.h5

# Upload to live pipeline immediately
python bf_weights_generator/deploy_bf_weights.py weights.h5 --upload

# Upload at a scheduled UTC time
python bf_weights_generator/deploy_bf_weights.py weights.h5 --upload \
    --utc-start 2026-04-01-16:00:00

# Upload and save as new observation defaults
python bf_weights_generator/deploy_bf_weights.py weights.h5 --upload --save-defaults

# Dry run
python bf_weights_generator/deploy_bf_weights.py weights.h5 --upload --dry-run
```

### Key flags

| Flag | Default | Notes |
|------|---------|-------|
| `--output-dir` | `.` | Directory for local DADA files |
| `--upload` | off | Write to FIFOs on corr1/corr2 |
| `--save-defaults` | off | Copy to `/data/casm/default_weights_64ant_512beam/` on both machines |
| `--utc-start` | current time | Apply immediately if omitted |
| `--scale` | **32** | DADA `SCALE` header value. Use 32 for CB, 8 for IB. |
| `--streams` | `0,1,2,3,4,5` | Comma-separated subband IDs |
| `--dry-run` | off | Print actions without executing |

The script verifies the `format_type` attribute in the HDF5 file before upload.
Passing a CB file as an IB argument (or vice versa) raises an error and aborts,
preventing corrupt data from reaching the live pipeline.

## format_type attribute

`save_int8_weights_hdf5` writes `format_type = "int8_snap_weights"` as a root
HDF5 attribute. The deploy script uses this to verify that the right file is
being uploaded to the right FIFO. If the attribute is absent (older files), a
warning is printed and the upload proceeds.

## Gotcha: SCALE in DADA vs scale_factor in HDF5

The `--scale 32` argument to `deploy_bf_weights.py` controls only the DADA
header field. It does not affect the int8 values stored in the HDF5 file (those
are always normalized by `scale_factor=127`). Passing `--scale 8` for a CB
upload would halve the apparent CB gain relative to IB, breaking the 0.985 ratio.
