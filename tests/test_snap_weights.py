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
    FrequencyConfig,
    StationaryPointing,
    save_int8_weights_hdf5,
    load_int8_weights_hdf5,
    inspect_int8_weights_file,
    TRANSIT_SURVEY_BEAMS,
    parse_beams_arg,
    generate_beam_grid,
)


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
        """Test parsing casm_antenna_layout1.csv."""
        # Should have 12 active antennas (excluding out-trigger)
        assert layout1.n_active == 12
        assert layout1.positions_enu.shape == (64, 3)
        assert layout1.active_mask.shape == (64,)
        assert np.sum(layout1.active_mask) == 12

    def test_parse_layout2(self, layout2):
        """Test parsing casm_antenna_layout2.csv."""
        assert layout2.n_active == 12
        assert layout2.positions_enu.shape == (64, 3)

    def test_active_indices(self, layout1):
        """Test that active indices are 0-11."""
        expected = np.arange(12)
        np.testing.assert_array_equal(layout1.active_indices, expected)

    def test_inactive_positions_zero(self, layout1):
        """Test that inactive positions are all zeros."""
        inactive_mask = ~layout1.active_mask
        inactive_positions = layout1.positions_enu[inactive_mask]
        np.testing.assert_array_equal(inactive_positions, 0.0)

    def test_snap_ordering(self, layout1):
        """Test SNAP input ordering is computed correctly."""
        # Check that snap_to_ant64 and ant64_to_snap are inverses
        for snap_idx in range(64):
            ant64_idx = layout1.snap_to_ant64[snap_idx]
            if ant64_idx >= 0:
                # If this SNAP input maps to an antenna, verify reverse mapping
                assert layout1.ant64_to_snap[ant64_idx] == snap_idx

    def test_snap_input_calculation(self, layout1):
        """Test that SNAP input index = snap_board * 12 + adc_channel."""
        # From CSV: ant64=0 has snap_A=0, adc_A=8 -> snap_input = 8
        # Check that snap_to_ant64[8] == 0
        assert layout1.snap_to_ant64[8] == 0

    def test_to_array_config(self, layout1):
        """Test conversion to ArrayConfig."""
        arr_config = layout1.to_array_config()
        assert arr_config.n_antennas == 12
        assert arr_config.n_active_antennas == 12
        assert arr_config.positions_enu.shape == (12, 3)

    def test_pos_ids_stored(self, layout1):
        """Test that position IDs are stored."""
        assert len(layout1.pos_ids) == 64
        # First active antenna should have a pos_id
        assert layout1.pos_ids[0] == "N21_E01"


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
        assert weights.n_beams == 8
        assert weights.weights_int8.shape == (2, 3072, 2, 8, 64)

    def test_compute_custom_beams(self, generator):
        """Test computing weights with custom beams."""
        pointings = [
            StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith"),
            StationaryPointing(alt_deg=60.0, az_deg=45.0, name="ne"),
        ]
        weights = generator.compute_int8_weights(pointings)
        assert weights.n_beams == 2
        assert weights.weights_int8.shape == (2, 3072, 2, 2, 64)

    def test_inactive_antennas_zero(self, generator):
        """Test that inactive antennas have zero weights."""
        weights = generator.compute_int8_weights()

        # Find inactive SNAP inputs
        active_snap_inputs = set()
        for snap_idx in range(64):
            if generator.array_config.snap_to_ant64[snap_idx] >= 0:
                active_snap_inputs.add(snap_idx)

        # Check that inactive SNAP inputs have all-zero weights
        for snap_idx in range(64):
            if snap_idx not in active_snap_inputs:
                real = weights.weights_int8[0, :, :, :, snap_idx]
                imag = weights.weights_int8[1, :, :, :, snap_idx]
                assert np.all(real == 0), f"SNAP input {snap_idx} should be zero (real)"
                assert np.all(imag == 0), f"SNAP input {snap_idx} should be zero (imag)"

    def test_active_antennas_nonzero(self, generator):
        """Test that active antennas have non-zero weights."""
        weights = generator.compute_int8_weights()

        active_snap_inputs = set()
        for snap_idx in range(64):
            if generator.array_config.snap_to_ant64[snap_idx] >= 0:
                active_snap_inputs.add(snap_idx)

        # Check that at least some active SNAP inputs have non-zero weights
        for snap_idx in active_snap_inputs:
            real = weights.weights_int8[0, :, :, :, snap_idx]
            imag = weights.weights_int8[1, :, :, :, snap_idx]
            # Complex weights should have |w|=1, so real^2 + imag^2 should be non-zero
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

        # Only check active antennas
        active_snap_inputs = []
        for snap_idx in range(64):
            if weights.array_config.snap_to_ant64[snap_idx] >= 0:
                active_snap_inputs.append(snap_idx)

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

            # Check array config
            np.testing.assert_array_equal(
                loaded.array_config.positions_enu,
                weights.array_config.positions_enu
            )
            np.testing.assert_array_equal(
                loaded.array_config.active_mask,
                weights.array_config.active_mask
            )
            np.testing.assert_array_equal(
                loaded.array_config.snap_to_ant64,
                weights.array_config.snap_to_ant64
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
            assert info['n_antennas'] == 64
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
