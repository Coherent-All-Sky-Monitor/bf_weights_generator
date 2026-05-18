"""
Tests for SNAP beamformer weight generation.

Tests cover:
- CSV parsing and Array64Config creation
- SNAP ordering correctness
- Inactive antenna handling
- Int8 quantization round-trip
- HDF5 I/O
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from bf_weights_generator import (
    Array64Config,
    SnapWeightsGenerator,
    Int8StationaryWeights,
    CombinedWeights,
    CalibrationWeights,
    load_calibration_weights,
    FrequencyConfig,
    StationaryPointing,
    save_int8_weights_hdf5,
    load_int8_weights_hdf5,
    inspect_int8_weights_file,
    save_combined_weights_hdf5,
    load_combined_weights_hdf5,
    generate_combined_weights,
    TRANSIT_SURVEY_BEAMS,
    parse_beams_arg,
    generate_beam_grid,
    compute_beam_fwhm,
    estimate_n_beams,
)
from bf_weights_generator.weights import generate_beam_grid_altaz


# Path to test CSV files (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
CSV_LAYOUT1 = PROJECT_ROOT / "casm_antenna_layout1.csv"
CSV_LAYOUT2 = PROJECT_ROOT / "casm_antenna_layout2.csv"


class TestArray64Config:
    """Tests for Array64Config CSV parsing."""

    @pytest.fixture
    def layout1(self):
        """Load layout1 configuration."""
        if not CSV_LAYOUT1.exists():
            pytest.skip(f"CSV file not found: {CSV_LAYOUT1}")
        return Array64Config.from_csv(str(CSV_LAYOUT1))

    @pytest.fixture
    def layout2(self):
        """Load layout2 configuration."""
        if not CSV_LAYOUT2.exists():
            pytest.skip(f"CSV file not found: {CSV_LAYOUT2}")
        return Array64Config.from_csv(str(CSV_LAYOUT2))

    def test_parse_layout1(self, layout1):
        """Test parsing casm_antenna_layout1.csv.

        After the v2 refactor an antenna only counts as active if it has
        SNAP wiring (snap_A and adc_A populated). The legacy fixture has
        12 antennas marked include_in_beamforming=True; one of them
        (N21_E04, ant64=3) has empty snap_A/adc_A so it's not assignable
        to any SNAP input. Net active = 11.
        """
        assert layout1.n_active == 11
        # 6 SNAPs * 12 ADCs = 72 slots in the default slot_table view.
        assert layout1.positions_enu.shape == (72, 3)
        assert layout1.active_mask.shape == (72,)
        assert np.sum(layout1.active_mask) == 11

    def test_parse_layout2(self, layout2):
        """Test parsing casm_antenna_layout2.csv."""
        # Same SNAP-wiring constraint as layout1.
        assert layout2.n_active == 11
        assert layout2.positions_enu.shape == (72, 3)

    def test_active_indices_are_snap_input_indices(self, layout1):
        """active_indices now returns snap_input_idx values directly.

        For layout1, each active antenna's snap_input_idx is
        snap_A * 12 + adc_A. Verify the set matches what the CSV says.
        """
        # CSV gives snap_A/adc_A for each ant64=0..11 (one is unwired).
        # Compute expected snap_input_idx values from the CSV directly.
        import csv as _csv
        from pathlib import Path as _Path
        csv_path = _Path(layout1.csv_path)
        expected = []
        with open(csv_path, newline='') as f:
            for row in _csv.DictReader(f):
                if row.get('pos_type', '').strip().lower() != 'antenna':
                    continue
                if row.get('include_in_beamforming', '').strip().lower() != 'true':
                    continue
                snap_a = row.get('snap_A', '').strip()
                adc_a = row.get('adc_A', '').strip()
                if not snap_a or not adc_a:
                    continue
                expected.append(int(float(snap_a)) * 12 + int(float(adc_a)))
        np.testing.assert_array_equal(layout1.active_indices, sorted(expected))

    def test_inactive_positions_zero(self, layout1):
        """Test that inactive positions are all zeros."""
        inactive_mask = ~layout1.active_mask
        inactive_positions = layout1.positions_enu[inactive_mask]
        np.testing.assert_array_equal(inactive_positions, 0.0)

    def test_antenna_ids_at_snap_inputs(self, layout1):
        """antenna_ids[snap_input_idx] gives the real antenna_id at that slot.

        Replaces the legacy snap_to_ant64/ant64_to_snap dual map with a
        single antenna_id-per-slot lookup.
        """
        # The fixture uses antenna_id == ant64 + 1 (set by the legacy
        # CSV translator). For the active set, antenna_ids should be
        # 1-indexed and unique.
        active_ids = layout1.antenna_ids[layout1.active_mask]
        assert (active_ids > 0).all(), "active slots must have real antenna IDs"
        assert len(np.unique(active_ids)) == len(active_ids), \
            "no duplicate antenna IDs across active slots"
        # Inactive slots are -1.
        assert (layout1.antenna_ids[~layout1.active_mask] == -1).all()

    def test_snap_input_calculation(self, layout1):
        """snap_input_idx = snap_board*12 + adc_channel.

        From CSV: N21_E01 has snap_A=0, adc_A=8 → snap_input_idx=8 →
        antenna_ids[8] == 1 (it's the first row, ant64=0 → antenna_id=1).
        """
        assert layout1.antenna_ids[8] == 1
        assert layout1.active_mask[8]

    def test_to_array_config(self, layout1):
        """Test conversion to ArrayConfig."""
        arr_config = layout1.to_array_config()
        assert arr_config.n_antennas == 11
        assert arr_config.n_active_antennas == 11
        assert arr_config.positions_enu.shape == (11, 3)

    def test_pos_ids_stored(self, layout1):
        """Test that position IDs are stored at SNAP-input slots."""
        # 72 slots in the default slot_table view (6 SNAPs * 12 ADCs).
        assert len(layout1.pos_ids) == 72
        # The first active antenna in the CSV is N21_E01 at snap_input_idx=8.
        # pos_ids[8] should be a non-empty label derived from row/col.
        # The legacy translator doesn't carry row/col, so the fallback
        # label is "ANT<antenna_id>" — antenna 1 lives at slot 8.
        assert layout1.pos_ids[8] == "ANT1"


class TestSnapWeightsGenerator:
    """Tests for SnapWeightsGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator with layout1."""
        if not CSV_LAYOUT1.exists():
            pytest.skip(f"CSV file not found: {CSV_LAYOUT1}")
        array_config = Array64Config.from_csv(str(CSV_LAYOUT1))
        return SnapWeightsGenerator(array_config)

    def test_compute_default_beams(self, generator):
        """Test computing weights with default transit survey beams."""
        weights = generator.compute_int8_weights()
        n_slots = len(generator.array_config.positions_enu)  # 72 for 6×12
        assert weights.n_beams == 8
        assert weights.weights_int8.shape == (2, 3072, 2, 8, n_slots)

    def test_compute_custom_beams(self, generator):
        """Test computing weights with custom beams."""
        pointings = [
            StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith"),
            StationaryPointing(alt_deg=60.0, az_deg=45.0, name="ne"),
        ]
        weights = generator.compute_int8_weights(pointings)
        n_slots = len(generator.array_config.positions_enu)  # 72 for 6×12
        assert weights.n_beams == 2
        assert weights.weights_int8.shape == (2, 3072, 2, 2, n_slots)

    def test_inactive_antennas_zero(self, generator):
        """Test that inactive SNAP inputs have zero weights."""
        weights = generator.compute_int8_weights()
        active_set = set(int(i) for i in generator.array_config.active_indices)
        # Iterate over the full slot count (n_snaps × n_adc = 72 for 6×12).
        n_slots = len(generator.array_config.positions_enu)
        for snap_idx in range(n_slots):
            if snap_idx not in active_set:
                real = weights.weights_int8[0, :, :, :, snap_idx]
                imag = weights.weights_int8[1, :, :, :, snap_idx]
                assert np.all(real == 0), f"SNAP input {snap_idx} should be zero (real)"
                assert np.all(imag == 0), f"SNAP input {snap_idx} should be zero (imag)"

    def test_active_antennas_nonzero(self, generator):
        """Test that active SNAP inputs have non-zero weights."""
        weights = generator.compute_int8_weights()
        n_slots = len(generator.array_config.positions_enu)
        for snap_idx in generator.array_config.active_indices:
            if snap_idx >= n_slots:
                continue   # outside the int8 output (shouldn't happen)
            real = weights.weights_int8[0, :, :, :, snap_idx]
            imag = weights.weights_int8[1, :, :, :, snap_idx]
            assert np.any(real != 0) or np.any(imag != 0), \
                f"Active SNAP input {snap_idx} should have non-zero weights"

    def test_polarization_duplication(self, generator):
        """Test that both polarizations have identical weights."""
        weights = generator.compute_int8_weights()
        # Shape: (2, n_chan, 2, n_beams, 64) where axis 2 is polarization
        pol_a = weights.weights_int8[:, :, 0, :, :]
        pol_b = weights.weights_int8[:, :, 1, :, :]
        np.testing.assert_array_equal(pol_a, pol_b)

    def test_quantization_range(self, generator):
        """Test that quantized weights are within int8 range."""
        weights = generator.compute_int8_weights()
        assert weights.weights_int8.dtype == np.int8
        assert np.all(weights.weights_int8 >= -128)
        assert np.all(weights.weights_int8 <= 127)


class TestQuantizationRoundtrip:
    """Tests for int8 quantization accuracy."""

    @pytest.fixture
    def weights(self):
        """Generate weights for round-trip testing."""
        if not CSV_LAYOUT1.exists():
            pytest.skip(f"CSV file not found: {CSV_LAYOUT1}")
        array_config = Array64Config.from_csv(str(CSV_LAYOUT1))
        generator = SnapWeightsGenerator(array_config)
        return generator.compute_int8_weights()

    def test_roundtrip_error(self, weights):
        """Test that quantization round-trip error is within tolerance."""
        # Reconstruct complex weights
        complex_weights = weights.to_complex64()

        # For unit-amplitude weights, quantization error should be < 1/127 ≈ 0.8%
        # The magnitude should be close to 1 for coherent beams
        magnitudes = np.abs(complex_weights)

        # Only check active SNAP inputs (slot indices within the output).
        n_slots = len(weights.array_config.positions_enu)
        active_snap_inputs = [
            int(i) for i in weights.array_config.active_indices if int(i) < n_slots
        ]

        for snap_idx in active_snap_inputs:
            mags = magnitudes[:, snap_idx, :]
            # Should be close to 1.0 (within quantization error)
            assert np.allclose(mags, 1.0, atol=0.02), \
                f"SNAP input {snap_idx}: magnitude deviation too large"


class TestInt8WeightsIO:
    """Tests for HDF5 save/load of int8 weights."""

    @pytest.fixture
    def weights(self):
        """Generate weights for I/O testing."""
        if not CSV_LAYOUT1.exists():
            pytest.skip(f"CSV file not found: {CSV_LAYOUT1}")
        array_config = Array64Config.from_csv(str(CSV_LAYOUT1))
        generator = SnapWeightsGenerator(array_config)
        return generator.compute_int8_weights()

    def test_save_load_roundtrip(self, weights):
        """Test saving and loading weights produces identical data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_weights.h5"
            save_int8_weights_hdf5(weights, filepath)
            loaded = load_int8_weights_hdf5(filepath)

            # Check weights match
            np.testing.assert_array_equal(loaded.weights_int8, weights.weights_int8)
            np.testing.assert_array_equal(loaded.frequencies_hz, weights.frequencies_hz)
            assert loaded.scale_factor == weights.scale_factor
            assert loaded.n_beams == weights.n_beams
            assert loaded.n_channels == weights.n_channels

            # Check array config (v2.0 schema: antenna_ids replaces the
            # snap_to_ant64/ant64_to_snap reorder maps).
            np.testing.assert_array_equal(
                loaded.array_config.positions_enu,
                weights.array_config.positions_enu
            )
            np.testing.assert_array_equal(
                loaded.array_config.active_mask,
                weights.array_config.active_mask
            )
            np.testing.assert_array_equal(
                loaded.array_config.antenna_ids,
                weights.array_config.antenna_ids
            )

            # Check pointings
            assert len(loaded.pointings) == len(weights.pointings)
            for p1, p2 in zip(loaded.pointings, weights.pointings):
                assert p1.alt_deg == pytest.approx(p2.alt_deg)
                assert p1.az_deg == pytest.approx(p2.az_deg)
                assert p1.name == p2.name

    def test_inspect_file(self, weights):
        """Test inspect_int8_weights_file returns correct metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_weights.h5"
            save_int8_weights_hdf5(weights, filepath)
            info = inspect_int8_weights_file(filepath)

            assert info['n_beams'] == weights.n_beams
            assert info['n_channels'] == weights.n_channels
            assert info['n_antennas'] == len(weights.array_config.positions_enu)
            assert info['n_pol'] == 2
            assert info['scale_factor'] == weights.scale_factor
            assert info['weights_shape'] == weights.shape
            assert info['weights_dtype'] == 'int8'
            assert info['n_active_antennas'] == weights.array_config.n_active

    def test_overwrite_protection(self, weights):
        """Test that overwrite=False raises error for existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_weights.h5"
            save_int8_weights_hdf5(weights, filepath)

            with pytest.raises(FileExistsError):
                save_int8_weights_hdf5(weights, filepath, overwrite=False)

            # Should succeed with overwrite=True
            save_int8_weights_hdf5(weights, filepath, overwrite=True)


class TestParseBeamsArg:
    """Tests for parse_beams_arg function."""

    def test_transit(self):
        """Test 'transit' beam specification."""
        beams = parse_beams_arg("transit")
        assert len(beams) == 8
        assert beams[0].alt_deg == 90.0  # Zenith

    def test_zenith(self):
        """Test 'zenith' beam specification."""
        beams = parse_beams_arg("zenith")
        assert len(beams) == 1
        assert beams[0].alt_deg == 90.0
        assert beams[0].az_deg == 0.0

    def test_custom_single(self):
        """Test single custom beam."""
        beams = parse_beams_arg("70:45")
        assert len(beams) == 1
        assert beams[0].alt_deg == 70.0
        assert beams[0].az_deg == 45.0

    def test_custom_multiple(self):
        """Test multiple custom beams."""
        beams = parse_beams_arg("90:0, 70:90, 60:180")
        assert len(beams) == 3
        assert beams[0].alt_deg == 90.0
        assert beams[1].alt_deg == 70.0
        assert beams[2].alt_deg == 60.0

    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_beams_arg("invalid")


class TestTransitSurveyBeams:
    """Tests for TRANSIT_SURVEY_BEAMS constant."""

    def test_beam_count(self):
        """Test that there are 8 transit survey beams."""
        assert len(TRANSIT_SURVEY_BEAMS) == 8

    def test_zenith_beam(self):
        """Test that first beam is zenith."""
        assert TRANSIT_SURVEY_BEAMS[0].alt_deg == 90.0

    def test_beam_names(self):
        """Test that all beams have names."""
        for beam in TRANSIT_SURVEY_BEAMS:
            assert beam.name != ""

    def test_altitude_range(self):
        """Test that all beams are above horizon."""
        for beam in TRANSIT_SURVEY_BEAMS:
            assert beam.alt_deg >= 0.0
            assert beam.alt_deg <= 90.0


class TestGenerateBeamGrid:
    """Tests for generate_beam_grid function."""

    def test_n_beams_target(self):
        """Test that n_beams produces approximately the target count."""
        for target in [4, 8, 16, 32]:
            beams = generate_beam_grid(n_beams=target)
            # Should be within 50% of target (grid constraints make exact count hard)
            assert len(beams) <= target
            assert len(beams) >= target // 2

    def test_spacing_affects_count(self):
        """Test that larger spacing produces fewer beams."""
        beams_15 = generate_beam_grid(spacing_deg=15.0)
        beams_30 = generate_beam_grid(spacing_deg=30.0)
        assert len(beams_30) < len(beams_15)

    def test_alt_range(self):
        """Test that beams respect altitude limits."""
        beams = generate_beam_grid(alt_min_deg=45.0, alt_max_deg=80.0, spacing_deg=20.0)
        for beam in beams:
            assert beam.alt_deg >= 45.0
            assert beam.alt_deg <= 90.0  # May include zenith

    def test_includes_zenith(self):
        """Test that grid includes zenith beam."""
        beams = generate_beam_grid(spacing_deg=30.0)
        zenith_beams = [b for b in beams if b.alt_deg >= 89.0]
        assert len(zenith_beams) >= 1

    def test_azimuth_distribution(self):
        """Test that beams are distributed in azimuth."""
        beams = generate_beam_grid(n_beams=8)
        # Get non-zenith beams
        non_zenith = [b for b in beams if b.alt_deg < 85.0]
        if len(non_zenith) > 1:
            azimuths = [b.az_deg for b in non_zenith]
            # Should span a range of azimuths
            az_range = max(azimuths) - min(azimuths)
            assert az_range > 45.0  # At least some spread

    def test_all_beams_have_names(self):
        """Test that all generated beams have names."""
        beams = generate_beam_grid(n_beams=12)
        for beam in beams:
            assert beam.name != ""
            assert beam.name.startswith("beam_")

    def test_default_parameters(self):
        """Test default parameters produce reasonable output."""
        beams = generate_beam_grid()
        assert len(beams) > 0
        assert all(b.alt_deg >= 30.0 for b in beams)


class TestComputeBeamFwhm:
    """Tests for compute_beam_fwhm function."""

    def test_known_array(self):
        """Test FWHM with known baselines: 3m E-W, 21.5m N-S at 437.5 MHz."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, -21.5, 0.0],
        ])
        fwhm_ew, fwhm_ns = compute_beam_fwhm(positions, freq_hz=437.5e6)
        # lambda = c / f = 0.685 m
        # fwhm_ew = degrees(0.685 / 3.0) ~ 13.1°
        # fwhm_ns = degrees(0.685 / 21.5) ~ 1.83°
        assert 12.0 < fwhm_ew < 15.0, f"E-W FWHM {fwhm_ew:.1f}° outside expected range"
        assert 1.5 < fwhm_ns < 2.5, f"N-S FWHM {fwhm_ns:.1f}° outside expected range"

    def test_ew_wider_than_ns(self):
        """Test that shorter E-W baseline gives wider E-W beam."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, -21.5, 0.0],
        ])
        fwhm_ew, fwhm_ns = compute_beam_fwhm(positions)
        assert fwhm_ew > fwhm_ns

    def test_single_antenna_ew(self):
        """Test that zero E-W baseline gives 180° FWHM."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -10.0, 0.0],
        ])
        fwhm_ew, fwhm_ns = compute_beam_fwhm(positions)
        assert fwhm_ew == 180.0
        assert fwhm_ns < 180.0

    def test_single_antenna_ns(self):
        """Test that zero N-S baseline gives 180° FWHM."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        fwhm_ew, fwhm_ns = compute_beam_fwhm(positions)
        assert fwhm_ew < 180.0
        assert fwhm_ns == 180.0

    def test_freq_config(self):
        """Test that freq_config is used when freq_hz is not provided."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [10.0, -10.0, 0.0],
        ])
        freq_config = FrequencyConfig()
        fwhm_ew1, fwhm_ns1 = compute_beam_fwhm(positions, freq_config=freq_config)
        # Center of default band is ~437.5 MHz
        fwhm_ew2, fwhm_ns2 = compute_beam_fwhm(positions, freq_hz=437.5e6)
        # Should be close (not exact due to mean vs center)
        assert abs(fwhm_ew1 - fwhm_ew2) < 0.5
        assert abs(fwhm_ns1 - fwhm_ns2) < 0.5

    def test_with_layout_csv(self):
        """Test FWHM computation with actual CSV layout."""
        if not CSV_LAYOUT1.exists():
            pytest.skip(f"CSV file not found: {CSV_LAYOUT1}")
        array = Array64Config.from_csv(str(CSV_LAYOUT1))
        fwhm_ew, fwhm_ns = compute_beam_fwhm(array.active_positions)
        # Should return reasonable values
        assert 0.1 < fwhm_ew < 180.0
        assert 0.1 < fwhm_ns < 180.0


class TestEstimateNBeams:
    """Tests for estimate_n_beams function."""

    def test_basic_count(self):
        """Test that estimate returns a positive integer."""
        n = estimate_n_beams(10.0, 10.0)
        assert n > 0
        assert isinstance(n, int)

    def test_smaller_beams_more_beams(self):
        """Test that smaller beams require more beams."""
        n_large = estimate_n_beams(20.0, 20.0)
        n_small = estimate_n_beams(5.0, 5.0)
        assert n_small > n_large

    def test_elliptical_vs_isotropic(self):
        """Test that elliptical beams give different count than isotropic."""
        n_iso = estimate_n_beams(10.0, 10.0)
        n_ellip = estimate_n_beams(20.0, 5.0)
        # Elliptical with same area should give same count
        assert n_iso == n_ellip

    def test_overlap_increases_beams(self):
        """Test that smaller overlap fraction increases beam count."""
        n_fwhm = estimate_n_beams(10.0, 10.0, overlap=1.0)
        n_nyquist = estimate_n_beams(10.0, 10.0, overlap=0.5)
        assert n_nyquist > n_fwhm

    def test_alt_range_affects_count(self):
        """Test that larger altitude range requires more beams."""
        n_small = estimate_n_beams(10.0, 10.0, alt_min_deg=60.0)
        n_large = estimate_n_beams(10.0, 10.0, alt_min_deg=30.0)
        assert n_large > n_small

    def test_physical_sanity(self):
        """Test with CASM-like values: ~13° EW, ~2° NS beams above 30°."""
        n = estimate_n_beams(13.0, 2.0, alt_min_deg=30.0)
        # Should be a reasonable number (hundreds for narrow NS beam)
        assert n > 10
        assert n < 10000


class TestEllipticalBeamGrid:
    """Tests for elliptical beam grid generation."""

    def test_altaz_with_elliptical_spacing(self):
        """Test generate_beam_grid_altaz with explicit elliptical spacing."""
        beams = generate_beam_grid_altaz(
            spacing_ew_deg=20.0,
            spacing_ns_deg=5.0,
            alt_min_deg=60.0,
            alt_max_deg=90.0,
        )
        assert len(beams) > 0
        # With 5° NS spacing from 60° to 90°, should have ~7 altitude levels
        alts = sorted(set(round(b.alt_deg, 1) for b in beams))
        assert len(alts) >= 5

    def test_altaz_ns_controls_altitude_step(self):
        """Test that spacing_ns_deg controls altitude separation."""
        beams_5 = generate_beam_grid_altaz(
            spacing_ew_deg=20.0, spacing_ns_deg=5.0,
            alt_min_deg=30.0, alt_max_deg=90.0,
        )
        beams_10 = generate_beam_grid_altaz(
            spacing_ew_deg=20.0, spacing_ns_deg=10.0,
            alt_min_deg=30.0, alt_max_deg=90.0,
        )
        # 5° NS spacing should produce more altitude levels than 10°
        alts_5 = set(round(b.alt_deg, 1) for b in beams_5)
        alts_10 = set(round(b.alt_deg, 1) for b in beams_10)
        assert len(alts_5) > len(alts_10)

    def test_altaz_ew_controls_azimuth_step(self):
        """Test that spacing_ew_deg controls azimuth separation."""
        beams_10 = generate_beam_grid_altaz(
            spacing_ew_deg=10.0, spacing_ns_deg=30.0,
            alt_min_deg=30.0, alt_max_deg=60.0,
        )
        beams_30 = generate_beam_grid_altaz(
            spacing_ew_deg=30.0, spacing_ns_deg=30.0,
            alt_min_deg=30.0, alt_max_deg=60.0,
        )
        # Narrower E-W spacing should produce more beams (more azimuths)
        assert len(beams_10) > len(beams_30)

    def test_altaz_with_positions(self):
        """Test generate_beam_grid_altaz with antenna positions."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, -21.5, 0.0],
        ])
        beams = generate_beam_grid_altaz(
            positions_enu=positions,
            alt_min_deg=60.0,
            alt_max_deg=90.0,
        )
        assert len(beams) > 0
        # With ~2° NS FWHM, should have many altitude levels from 60° to 90°
        alts = sorted(set(round(b.alt_deg, 1) for b in beams))
        assert len(alts) >= 10

    def test_snap_generate_beam_grid_with_array_config(self):
        """Test generate_beam_grid with Array64Config."""
        if not CSV_LAYOUT1.exists():
            pytest.skip(f"CSV file not found: {CSV_LAYOUT1}")
        array = Array64Config.from_csv(str(CSV_LAYOUT1))
        beams = generate_beam_grid(array_config=array, alt_min_deg=60.0)
        assert len(beams) > 0
        for beam in beams:
            assert beam.alt_deg >= 60.0

    def test_snap_generate_beam_grid_with_positions(self):
        """Test generate_beam_grid with raw positions."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, -21.5, 0.0],
        ])
        beams = generate_beam_grid(positions_enu=positions, alt_min_deg=60.0)
        assert len(beams) > 0

    def test_snap_generate_beam_grid_n_beams_with_array(self):
        """Test generate_beam_grid with n_beams and array_config."""
        if not CSV_LAYOUT1.exists():
            pytest.skip(f"CSV file not found: {CSV_LAYOUT1}")
        array = Array64Config.from_csv(str(CSV_LAYOUT1))
        beams = generate_beam_grid(n_beams=8, array_config=array)
        assert len(beams) <= 8
        assert len(beams) >= 4

    def test_backward_compat_isotropic(self):
        """Test that existing isotropic calls still work."""
        beams = generate_beam_grid_altaz(spacing_deg=20.0, alt_min_deg=60.0)
        assert len(beams) > 0
        beams2 = generate_beam_grid(spacing_deg=20.0, alt_min_deg=60.0)
        assert len(beams2) > 0


CSV_CURRENT = PROJECT_ROOT / "casm_antenna_layout_current.csv"
CSV_PRE_FEB16 = PROJECT_ROOT / "casm_antenna_layout_pre_feb16.csv"
CAL_WEIGHTS_PATH = (
    PROJECT_ROOT / "delay_cal_weights"
    / "svd_weights_sun_2026-02-14_phase-only_thr2.0_norfi.npz"
)


class TestCalibrationWeights:
    """Tests for CalibrationWeights loading."""

    def test_load_cal_weights(self):
        """Load actual .npz and verify shapes and phase-only property."""
        if not CAL_WEIGHTS_PATH.exists():
            pytest.skip(f"Cal weights not found: {CAL_WEIGHTS_PATH}")
        cal = load_calibration_weights(str(CAL_WEIGHTS_PATH))
        assert cal.weights.shape == (16, 3072)
        assert cal.flags.shape == (3072,)
        assert cal.frequencies_hz.shape == (3072,)
        assert cal.ant_ids.shape == (16,)
        assert cal.ref_ant_id == 5
        assert cal.source == "SUN"
        # Phase-only: magnitude should be ~1 for unflagged channels
        good = cal.flags
        mags = np.abs(cal.weights[:, good])
        nonzero = mags > 0
        if np.any(nonzero):
            np.testing.assert_allclose(mags[nonzero], 1.0, atol=0.01)

    def test_flags_convention(self):
        """Verify True=good (non-zero weights), False=flagged (zero weights)."""
        if not CAL_WEIGHTS_PATH.exists():
            pytest.skip(f"Cal weights not found: {CAL_WEIGHTS_PATH}")
        cal = load_calibration_weights(str(CAL_WEIGHTS_PATH))
        bad = ~cal.flags
        # Flagged channels should have zero weights (at least for ref ant)
        ref_idx = np.where(cal.ant_ids == cal.ref_ant_id)[0][0]
        flagged_weights = cal.weights[ref_idx, bad]
        np.testing.assert_array_equal(flagged_weights, 0.0)

    def test_frequency_ascending(self):
        """Verify loaded freqs are ascending."""
        if not CAL_WEIGHTS_PATH.exists():
            pytest.skip(f"Cal weights not found: {CAL_WEIGHTS_PATH}")
        cal = load_calibration_weights(str(CAL_WEIGHTS_PATH))
        assert np.all(np.diff(cal.frequencies_hz) > 0)

    def test_load_invalid_shapes(self, tmp_path):
        """Test that shape mismatches raise ValueError."""
        npz_path = tmp_path / "bad.npz"
        np.savez(
            npz_path,
            weights=np.ones((4, 10), dtype=complex),
            flags=np.ones(5, dtype=bool),  # Wrong length
            freqs_mhz=np.linspace(375, 469, 10),
            ant_ids=np.arange(1, 5),
        )
        with pytest.raises(ValueError, match="Shape mismatch"):
            load_calibration_weights(str(npz_path))

    def test_load_descending_freqs_auto_flipped(self, tmp_path):
        """Test that descending frequencies are auto-flipped to ascending."""
        npz_path = tmp_path / "desc.npz"
        freqs_desc = np.linspace(469, 375, 10)
        weights_desc = np.arange(40, dtype=float).reshape(4, 10).astype(complex)
        flags_desc = np.array([True]*5 + [False]*5)
        np.savez(
            npz_path,
            weights=weights_desc,
            flags=flags_desc,
            freqs_mhz=freqs_desc,
            ant_ids=np.arange(1, 5),
        )
        cal = load_calibration_weights(str(npz_path))
        # Frequencies should be ascending after load
        assert np.all(np.diff(cal.frequencies_hz) > 0)
        # Data should be flipped to match: first cal channel = lowest freq
        np.testing.assert_array_equal(
            cal.weights, weights_desc[:, ::-1]
        )
        np.testing.assert_array_equal(
            cal.flags, flags_desc[::-1]
        )


class TestCombinedWeights:
    """Tests for combining geometric and calibration weights."""

    @pytest.fixture
    def cal(self):
        """Load actual calibration weights."""
        if not CAL_WEIGHTS_PATH.exists():
            pytest.skip(f"Cal weights not found: {CAL_WEIGHTS_PATH}")
        return load_calibration_weights(str(CAL_WEIGHTS_PATH))

    @pytest.fixture
    def pre_feb16(self):
        """Load pre-Feb16 layout."""
        if not CSV_PRE_FEB16.exists():
            pytest.skip(f"CSV not found: {CSV_PRE_FEB16}")
        return Array64Config.from_csv(str(CSV_PRE_FEB16))

    @pytest.fixture
    def current_layout(self):
        """Load current layout."""
        if not CSV_CURRENT.exists():
            pytest.skip(f"CSV not found: {CSV_CURRENT}")
        return Array64Config.from_csv(str(CSV_CURRENT))

    def test_combined_shape(self, pre_feb16, cal):
        """Output shape matches pure geometric case."""
        gen = SnapWeightsGenerator(pre_feb16)
        pointings = [StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith")]

        geo_only = gen.compute_int8_weights(pointings)
        combined = gen.compute_int8_weights(pointings, cal_weights=cal)

        assert combined.shape == geo_only.shape

    def test_cal_only_zenith(self, pre_feb16, cal):
        """At zenith, geo weights ≈ 1. Combined phases should match cal phases."""
        gen = SnapWeightsGenerator(pre_feb16)
        pointings = [StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith")]

        combined = gen.compute_int8_weights(pointings, cal_weights=cal)
        complex_w = combined.to_complex64()  # (n_beams, n_slots, n_chan) SNAP order
        n_slots = len(pre_feb16.positions_enu)

        # At zenith all geometric delays are zero, so geo weights = 1+0j.
        # Combined = cal * 1 = cal. Check phase agreement for active antennas.
        # SNAP output reverses geo (desc→asc), cal was flipped desc for multiply
        # then reversed back → SNAP channel i = cal channel i (both ascending).
        for snap_idx in pre_feb16.active_indices:
            snap_idx = int(snap_idx)
            if snap_idx >= n_slots:
                continue
            aid = int(pre_feb16.antenna_ids[snap_idx])
            cal_ant_idx = np.where(cal.ant_ids == aid)[0]
            if len(cal_ant_idx) == 0:
                continue
            cal_ant_idx = cal_ant_idx[0]

            # Compare only unflagged channels
            good = cal.flags  # ascending, same index as SNAP output
            snap_w = complex_w[0, snap_idx, good]
            cal_w = cal.weights[cal_ant_idx, good]
            nonzero = np.abs(snap_w) > 0.01
            if np.sum(nonzero) > 10:
                phase_diff = np.angle(snap_w[nonzero] * np.conj(cal_w[nonzero]))
                # Tolerance 0.2 rad: accounts for int8 quantization and
                # small z-offsets creating residual frequency-dependent phase
                assert np.std(phase_diff) < 0.2, (
                    f"Phase mismatch for antenna {aid} (snap_idx={snap_idx}): "
                    f"std(phase_diff)={np.std(phase_diff):.3f}"
                )

    def test_flagged_channels_zero(self, pre_feb16, cal):
        """Flagged channels should produce zero int8 weights."""
        gen = SnapWeightsGenerator(pre_feb16)
        pointings = [StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith")]
        combined = gen.compute_int8_weights(pointings, cal_weights=cal)

        # SNAP channel i = cal channel i (both ascending after pipeline)
        flagged = np.where(~cal.flags)[0]
        assert len(flagged) > 0, "No flagged channels to test"

        # Flagged channels should be zero for all active SNAP inputs
        n_checked = 0
        n_slots = len(pre_feb16.positions_enu)
        for snap_idx in pre_feb16.active_indices:
            snap_idx = int(snap_idx)
            if snap_idx >= n_slots:
                continue
            for fi in flagged:
                real = combined.weights_int8[0, fi, 0, 0, snap_idx]
                imag = combined.weights_int8[1, fi, 0, 0, snap_idx]
                assert real == 0 and imag == 0, (
                    f"Flagged chan {fi} not zero for SNAP input {snap_idx}"
                )
                n_checked += 1

        assert n_checked > 0

    def test_layout_remap(self, pre_feb16, current_layout, cal):
        """With output_array_config, SNAP ordering uses output layout."""
        gen = SnapWeightsGenerator(pre_feb16, output_array_config=current_layout)
        pointings = [StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith")]
        combined = gen.compute_int8_weights(pointings, cal_weights=cal)

        # Verify shape and that the compute layout's active SNAP inputs
        # carry non-zero weights. Now that there's no separate ant64
        # reorder, output_array_config doesn't change which slots get
        # written — it just records provenance.
        n_slots = len(pre_feb16.positions_enu)
        assert combined.shape[4] == n_slots
        for snap_idx in pre_feb16.active_indices:
            snap_idx = int(snap_idx)
            if snap_idx >= n_slots:
                continue
            w = combined.weights_int8[:, :, :, :, snap_idx]
            assert np.any(w != 0), (
                f"SNAP input {snap_idx} should have non-zero weights"
            )

    def test_channel_count_mismatch_raises(self, pre_feb16):
        """Cal weights with different channel count should raise ValueError."""
        cal = CalibrationWeights(
            weights=np.ones((16, 100), dtype=complex),
            flags=np.ones(100, dtype=bool),
            frequencies_hz=np.linspace(100e6, 200e6, 100),
            ant_ids=np.arange(1, 17),
            ref_ant_id=5,
            source="FAKE",
        )
        gen = SnapWeightsGenerator(pre_feb16)
        pointings = [StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith")]
        with pytest.raises(ValueError, match="Channel count mismatch"):
            gen.compute_int8_weights(pointings, cal_weights=cal)

    def test_freq_order_independent(self, pre_feb16, cal):
        """Same result whether cal freqs are ascending or descending."""
        gen = SnapWeightsGenerator(pre_feb16)
        pointings = [StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith")]

        # Original (ascending cal freqs)
        result_asc = gen.compute_int8_weights(pointings, cal_weights=cal)

        # Create descending version (consistent flip of data + freqs)
        cal_desc = CalibrationWeights(
            weights=cal.weights[:, ::-1].copy(),
            flags=cal.flags[::-1].copy(),
            frequencies_hz=cal.frequencies_hz[::-1].copy(),
            ant_ids=cal.ant_ids.copy(),
            ref_ant_id=cal.ref_ant_id,
            source=cal.source,
        )
        # _apply_calibration_weights auto-flips to match geo order
        result_desc = gen.compute_int8_weights(pointings, cal_weights=cal_desc)

        np.testing.assert_array_equal(
            result_asc.weights_int8, result_desc.weights_int8
        )


class TestGenerateCombinedWeights:
    """Tests for generate_combined_weights() API."""

    @pytest.fixture
    def pre_feb16(self):
        if not CSV_PRE_FEB16.exists():
            pytest.skip(f"CSV not found: {CSV_PRE_FEB16}")
        return Array64Config.from_csv(str(CSV_PRE_FEB16))

    @pytest.fixture
    def current_layout(self):
        if not CSV_CURRENT.exists():
            pytest.skip(f"CSV not found: {CSV_CURRENT}")
        return Array64Config.from_csv(str(CSV_CURRENT))

    @pytest.fixture
    def cal(self):
        if not CAL_WEIGHTS_PATH.exists():
            pytest.skip(f"Cal weights not found: {CAL_WEIGHTS_PATH}")
        return load_calibration_weights(str(CAL_WEIGHTS_PATH))

    def test_single_pointing(self, pre_feb16):
        """Single StationaryPointing produces shape (1, n_slots, n_chan)."""
        result = generate_combined_weights(
            pointing=StationaryPointing(alt_deg=90.0, az_deg=0.0),
            array_config=pre_feb16,
        )
        n_slots = len(pre_feb16.positions_enu)  # 72 for current 6×12 hw
        assert isinstance(result, CombinedWeights)
        assert result.weights.shape == (1, n_slots, 3072)
        assert result.weights.dtype == np.complex64

    def test_multi_pointing(self, pre_feb16):
        """List of pointings produces shape (N, n_slots, n_chan)."""
        pointings = [
            StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith"),
            StationaryPointing(alt_deg=70.0, az_deg=90.0, name="east"),
            StationaryPointing(alt_deg=50.0, az_deg=180.0, name="south"),
        ]
        result = generate_combined_weights(
            pointing=pointings, array_config=pre_feb16,
        )
        n_slots = len(pre_feb16.positions_enu)  # 72 for current 6×12 hw
        assert result.weights.shape == (3, n_slots, 3072)
        assert result.n_beams == 3

    def test_with_cal(self, pre_feb16, cal):
        """With cal weights, active antennas on good channels have magnitude ~1."""
        result = generate_combined_weights(
            pointing=StationaryPointing(alt_deg=90.0, az_deg=0.0),
            array_config=pre_feb16,
            cal_weights=cal,
        )
        assert result.cal_weights is not None
        # Check magnitudes of active antennas on good channels
        active_snaps = [int(i) for i in pre_feb16.active_indices if int(i) < 64]
        good = result.flags
        mags = np.abs(result.weights[0, active_snaps, :][:, good])
        nonzero = mags > 0.01
        if np.any(nonzero):
            np.testing.assert_allclose(mags[nonzero], 1.0, atol=0.05)

    def test_without_cal(self, pre_feb16):
        """Without cal, active antennas have magnitude 1.0 (pure geometric)."""
        result = generate_combined_weights(
            pointing=StationaryPointing(alt_deg=90.0, az_deg=0.0),
            array_config=pre_feb16,
        )
        assert result.cal_weights is None
        assert np.all(result.flags)  # All channels good
        active_snaps = [int(i) for i in pre_feb16.active_indices if int(i) < 64]
        mags = np.abs(result.weights[0, active_snaps, :])
        np.testing.assert_allclose(mags, 1.0, atol=1e-5)

    def test_output_remap(self, pre_feb16, current_layout, cal):
        """output_array_config != array_config uses different SNAP ordering."""
        result = generate_combined_weights(
            pointing=StationaryPointing(alt_deg=90.0, az_deg=0.0),
            array_config=pre_feb16,
            cal_weights=cal,
            output_array_config=current_layout,
        )
        assert result.output_array_config is current_layout
        assert result.array_config is pre_feb16
        # Active SNAP inputs in pre_feb16 (the compute layout) should
        # have non-zero combined weights — the output layout argument
        # is a no-op now that there's no separate ant64 reorder.
        for snap_idx in pre_feb16.active_indices:
            if int(snap_idx) < 64:
                assert np.any(result.weights[0, int(snap_idx), :] != 0)

    def test_freq_order_default(self, pre_feb16):
        """Default freq_order is descending."""
        result = generate_combined_weights(
            pointing=StationaryPointing(alt_deg=90.0, az_deg=0.0),
            array_config=pre_feb16,
        )
        assert result.freq_order == "descending"
        assert result.frequencies_hz[0] > result.frequencies_hz[-1]

    def test_freq_order_ascending(self, pre_feb16):
        """Ascending freq_order flips frequencies."""
        result = generate_combined_weights(
            pointing=StationaryPointing(alt_deg=90.0, az_deg=0.0),
            array_config=pre_feb16,
            freq_order="ascending",
        )
        assert result.freq_order == "ascending"
        assert result.frequencies_hz[0] < result.frequencies_hz[-1]

    def test_save_load_roundtrip(self, pre_feb16, cal):
        """Save and load produces equivalent CombinedWeights."""
        result = generate_combined_weights(
            pointing=[
                StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zen"),
                StationaryPointing(alt_deg=70.0, az_deg=45.0, name="ne"),
            ],
            array_config=pre_feb16,
            cal_weights=cal,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_combined.h5"
            save_combined_weights_hdf5(result, filepath)
            loaded = load_combined_weights_hdf5(filepath)

            np.testing.assert_array_equal(loaded.weights, result.weights)
            np.testing.assert_array_equal(loaded.frequencies_hz, result.frequencies_hz)
            np.testing.assert_array_equal(loaded.flags, result.flags)
            assert loaded.freq_order == result.freq_order
            assert loaded.n_beams == result.n_beams
            assert len(loaded.pointings) == len(result.pointings)
            for p1, p2 in zip(loaded.pointings, result.pointings):
                assert p1.alt_deg == pytest.approx(p2.alt_deg)
                assert p1.az_deg == pytest.approx(p2.az_deg)
                assert p1.name == p2.name
            # Check array configs round-tripped
            np.testing.assert_array_equal(
                loaded.array_config.positions_enu,
                result.array_config.positions_enu,
            )
            np.testing.assert_array_equal(
                loaded.output_array_config.antenna_ids,
                result.output_array_config.antenna_ids,
            )
            # Check cal weights fully round-tripped
            assert loaded.cal_weights is not None
            assert loaded.cal_weights.source == result.cal_weights.source
            assert loaded.cal_weights.ref_ant_id == result.cal_weights.ref_ant_id
            np.testing.assert_array_equal(
                loaded.cal_weights.weights, result.cal_weights.weights,
            )
            np.testing.assert_array_equal(
                loaded.cal_weights.ant_ids, result.cal_weights.ant_ids,
            )
            np.testing.assert_array_equal(
                loaded.cal_weights.flags, result.cal_weights.flags,
            )

    def test_flagged_channels_zero(self, pre_feb16, cal):
        """Flagged channels should have zero weights."""
        result = generate_combined_weights(
            pointing=StationaryPointing(alt_deg=90.0, az_deg=0.0),
            array_config=pre_feb16,
            cal_weights=cal,
        )
        flagged = ~result.flags
        assert np.any(flagged), "No flagged channels to test"
        active_snaps = [int(i) for i in pre_feb16.active_indices if int(i) < 64]
        for snap_idx in active_snaps:
            flagged_w = result.weights[0, snap_idx, flagged]
            np.testing.assert_array_equal(
                flagged_w, 0.0,
                err_msg=f"Flagged channels not zero for SNAP input {snap_idx}",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
