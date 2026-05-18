# Changelog

## [Unreleased] — 2026-05-17

### Fixed

- **`Array64Config` 72-slot mismatch**: allocations in `compute_int8_weights`, `generate_combined_weights`, and the legacy v1 HDF5 reader in `io.py` used the literal 64 instead of `len(array_config.positions_enu)`. Any antenna wired at SNAP 5, ADC 4-11 (slot index >= 64) would have produced a silent `IndexError`. All three sites now derive the slot count from the layout object.
- **`FrequencyConfig.freq_end_voltage_mhz` default reverted to 468.75 MHz**: the default had been silently advanced to 484.39 MHz (the `layout_64ant` upper edge), shifting all channel center frequencies by ~15.6 MHz for any caller using `FrequencyConfig()` without arguments. This broke bit-exact reproducibility of previously deployed int8 weights. The default is restored to 468.75 (the `layout_32ant` upper edge). Callers that need the current band must now call `FrequencyConfig.layout_64ant()` or `FrequencyConfig.from_format("layout_64ant")` explicitly.
- **`load_calibration_weights` NPZ fallback now uses `allow_pickle=False`**: the legacy NPZ path previously defaulted to `allow_pickle=True`. Every key consumed (weights, flags, ant_ids, freqs_hz / freqs_mhz, ref_ant_id, source) is a plain ndarray or scalar, so pickle is not needed.

### Changed

- **Docstrings updated to reflect 72-slot reality**: references to `snap_to_ant64 mapping` replaced with `antenna_ids mapping` throughout `snap_weights.py` and `io.py`. The `Array64Config` class docstring now explicitly calls out the legacy name and states that no separate "ant64" slot space exists.
- **Legacy examples moved to `examples/legacy/`**: scripts that referenced the removed `snap_to_ant64` / `ant64_to_snap` attributes and the old 64-slot allocation are now under `examples/legacy/` with a `README.md` explaining that they will not run against the current API.

### Added

- **`FrequencyConfig.layout_64ant()` classmethod**: returns a `FrequencyConfig` pinned to the post-Jan-27-2026 band (channel-0 center at 484.375 MHz). Use this instead of passing a raw `freq_end_voltage_mhz` value.
- **`CLAUDE.md`**: project rules, module layout, key signatures, conventions, and known gotchas for Claude Code and new contributors.
