"""Regression test: end-to-end weights pipeline on a sparse antenna layout.

Before the v2 ID-space refactor, ``_apply_calibration_weights`` assumed
``cal_weights.ant_ids - 1`` was an ant64 slot index that matched
``Array64Config.active_indices``. This was true for dense-sequential
layouts (antenna IDs 1..N filling slots 0..N-1) and silently wrong for
sparse layouts. Production hit this when ``with_inactive([2, 3, 12])``
was applied to a 23-antenna layout, leaving active IDs
``[1, 7, 9, 10, 14, 15, 18, 19, 21, 22, 23, 24, 30, 32, 33, 36, 38, 40, 42, 44, 45]``.

This test exercises a small synthetic sparse layout end-to-end through
``generate_combined_weights`` so the bug stays fixed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casm_io.correlator.mapping import AntennaMapping
from bf_weights_generator import (
    Array64Config,
    CalibrationWeights,
    StationaryPointing,
    generate_combined_weights,
)


def _write_sparse_csv(tmp_path: Path) -> Path:
    """Synthesize a canonical-schema CSV with sparse antenna IDs.

    Five antennas at antenna_id ∈ {1, 7, 9, 10, 14}, mapped to
    snap_input_idx ∈ {0, 6, 8, 9, 13}. Plus an unwired-but-present
    placeholder row at slot 1 so we exercise the wired-but-not-active
    path (functional=0).
    """
    df = pd.DataFrame({
        "antenna_id":   [1, 7, 9, 10, 14, 99],
        "snap_id":      [0, 0, 0, 0,  1,  0],
        "adc":          [0, 6, 8, 9,  1,  1],
        "packet_index": [0, 6, 8, 9, 13,  1],
        "x_m":          [1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
        "y_m":          [10.0, 20.0, 30.0, 40.0, 50.0, 0.0],
        "z_m":          [0.1, 0.2, 0.3, 0.4, 0.5, 0.0],
        "functional":             [1, 1, 1, 1, 1, 0],   # antenna 99 = inactive
        "include_in_beamforming": [1, 1, 1, 1, 1, 0],
    })
    csv = tmp_path / "sparse_layout.csv"
    df.to_csv(csv, index=False)
    return csv


def _make_cal(ant_ids, n_chan=8):
    """Build a tiny CalibrationWeights with phase-only unit-amplitude entries."""
    rng = np.random.default_rng(42)
    weights = np.exp(1j * rng.uniform(-np.pi, np.pi,
                                       size=(len(ant_ids), n_chan))).astype(np.complex64)
    flags = np.ones(n_chan, dtype=bool)
    freqs_hz = np.linspace(400e6, 500e6, n_chan)   # ascending
    return CalibrationWeights(
        weights=weights,
        flags=flags,
        frequencies_hz=freqs_hz,
        ant_ids=np.asarray(ant_ids, dtype=int),
        ref_ant_id=int(ant_ids[0]),
        source="test",
    )


def test_sparse_layout_active_indices_are_snap_input_indices(tmp_path):
    csv = _write_sparse_csv(tmp_path)
    ant = AntennaMapping.load(str(csv))
    arr = Array64Config.from_antenna_mapping(ant)

    # 5 active antennas (the 6th has functional=0 so it drops out).
    assert arr.n_active == 5
    # active_indices returns snap_input_idx values, NOT a sequential
    # 0..4 range. This is the contract the cell-34 bug violated.
    np.testing.assert_array_equal(arr.active_indices, [0, 6, 8, 9, 13])
    # antenna_ids[snap_idx] gives the real antenna_id at each slot.
    expected_ids = np.full(72, -1, dtype=np.int32)
    expected_ids[0]  = 1
    expected_ids[1]  = 99    # wired but inactive, still has antenna_id
    expected_ids[6]  = 7
    expected_ids[8]  = 9
    expected_ids[9]  = 10
    expected_ids[13] = 14
    np.testing.assert_array_equal(arr.antenna_ids, expected_ids)


def test_sparse_layout_combined_weights_align_to_snap_inputs(tmp_path):
    """generate_combined_weights writes the cal-multiplied geo weights into
    exactly the snap_input_idx slots specified by the layout."""
    csv = _write_sparse_csv(tmp_path)
    ant = AntennaMapping.load(str(csv))
    arr = Array64Config.from_antenna_mapping(ant)

    # Build cal that uses freq_config's channel count (default 3072) so
    # the channel-count check inside _apply_calibration_weights passes.
    from bf_weights_generator.config import FrequencyConfig
    fc = FrequencyConfig()
    cal_ids = [1, 7, 9, 10, 14]
    cal = _make_cal(cal_ids, n_chan=fc.n_chan)

    result = generate_combined_weights(
        pointing=StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith"),
        array_config=arr,
        cal_weights=cal,
        freq_config=fc,
    )

    # Combined output is (n_beams=1, n_slots=72, n_chan).
    # n_slots = n_snaps * n_adc = 6 * 12 = 72 (the full CAsMan slot count).
    assert result.weights.shape == (1, 72, fc.n_chan)
    # The five active SNAP inputs should carry non-zero combined weights.
    for snap_idx in [0, 6, 8, 9, 13]:
        assert np.any(result.weights[0, snap_idx, :] != 0), (
            f"snap_input_idx {snap_idx} should be non-zero"
        )
    # Every other slot (including the wired-but-inactive antenna 99 at
    # slot 1) must be exactly zero.
    inactive_snap_idx = [i for i in range(72) if i not in {0, 6, 8, 9, 13}]
    for snap_idx in inactive_snap_idx:
        assert np.all(result.weights[0, snap_idx, :] == 0), (
            f"snap_input_idx {snap_idx} should be zero"
        )


def test_sparse_layout_missing_cal_antenna_raises_with_real_id(tmp_path):
    """When a cal_weights set lacks a real antenna_id we need, the error
    message must name the missing real ID — not an opaque slot index."""
    csv = _write_sparse_csv(tmp_path)
    ant = AntennaMapping.load(str(csv))
    arr = Array64Config.from_antenna_mapping(ant)

    from bf_weights_generator.config import FrequencyConfig
    fc = FrequencyConfig()
    # Cal has antenna 14 missing. The error should call out antenna 14.
    cal = _make_cal([1, 7, 9, 10], n_chan=fc.n_chan)
    with pytest.raises(ValueError, match=r"antenna 14"):
        generate_combined_weights(
            pointing=StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith"),
            array_config=arr,
            cal_weights=cal,
            freq_config=fc,
        )
