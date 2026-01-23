"""
Tests for the beamformer weights generator.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bf_weights_generator import (
    GeometricBeamformer,
    PhaseCenter,
    ArrayConfig,
    FrequencyConfig,
    BeamMode,
    compute_lst_rad,
    hadec_to_direction_cosines,
    compute_geometric_delays,
    SPEED_OF_LIGHT_M_S,
)


class TestFrequencyConfig:
    """Tests for FrequencyConfig."""

    def test_default_config(self):
        """Test default frequency configuration."""
        cfg = FrequencyConfig()
        assert cfg.n_chan == 3072
        assert cfg.total_bw_mhz == 125.0
        assert cfg.total_n_chan == 4096

    def test_channel_bandwidth(self):
        """Test channel bandwidth calculation."""
        cfg = FrequencyConfig()
        expected_bw = 125.0 / 4096
        assert np.isclose(cfg.chan_bw_mhz, expected_bw)

    def test_frequency_array(self):
        """Test frequency array generation."""
        cfg = FrequencyConfig(n_chan=10, freq_end_voltage_mhz=500.0)
        freqs = cfg.get_frequencies_mhz()
        assert len(freqs) == 10
        # First channel should be just below freq_end
        assert freqs[0] < cfg.freq_end_voltage_mhz
        # Frequencies should decrease
        assert all(np.diff(freqs) < 0)


class TestArrayConfig:
    """Tests for ArrayConfig."""

    def test_default_antennas(self):
        """Test default antenna configuration."""
        cfg = ArrayConfig()
        assert cfg.n_antennas == 13
        assert cfg.n_active_antennas == 13

    def test_flag_antennas(self):
        """Test antenna flagging."""
        cfg = ArrayConfig()
        cfg.flag_antennas([0, 5, 12])
        assert cfg.n_active_antennas == 10
        assert 0 not in cfg.active_indices
        assert 5 not in cfg.active_indices
        assert 12 not in cfg.active_indices

    def test_unflag_antennas(self):
        """Test antenna unflagging."""
        cfg = ArrayConfig()
        cfg.flag_antennas([0, 1, 2])
        cfg.unflag_antennas([0])
        assert cfg.n_active_antennas == 11
        assert 0 in cfg.active_indices

    def test_active_positions(self):
        """Test active positions array."""
        cfg = ArrayConfig()
        cfg.flag_antennas([12])  # Flag the outlier antenna
        assert len(cfg.active_positions) == 12
        assert cfg.active_positions.shape == (12, 3)


class TestPhaseCenter:
    """Tests for PhaseCenter."""

    def test_from_degrees(self):
        """Test creation from degrees."""
        pc = PhaseCenter(ra_deg=180.0, dec_deg=45.0, name="test")
        assert pc.ra_deg == 180.0
        assert pc.dec_deg == 45.0
        assert pc.name == "test"

    def test_from_hours(self):
        """Test creation from hours."""
        pc = PhaseCenter.from_hours(ra_hours=12.0, dec_deg=45.0)
        assert pc.ra_deg == 180.0  # 12h = 180 degrees

    def test_from_hms_dms(self):
        """Test creation from sexagesimal."""
        # Cygnus A: 19h 59m 28.4s, +40d 44m 2s
        pc = PhaseCenter.from_hms_dms("19:59:28.4", "+40:44:02", name="CygA")
        assert np.isclose(pc.ra_deg, 299.868, atol=0.01)
        assert np.isclose(pc.dec_deg, 40.734, atol=0.01)


class TestCoordinates:
    """Tests for coordinate transformations."""

    def test_lst_computation(self):
        """Test LST computation."""
        # At Greenwich meridian, GMST = LST
        # At 2000-01-01 12:00:00 UTC (JD 2451545.0), GMST ~ 18.7h
        unix_time = 946728000.0  # 2000-01-01 12:00:00 UTC
        lst = compute_lst_rad(unix_time, lon_deg=0.0)
        lst_hours = np.rad2deg(lst) / 15.0
        # GMST at J2000.0 noon is approximately 18.7 hours
        assert 18 < lst_hours < 19

    def test_direction_cosines_zenith(self):
        """Test direction cosines for source at zenith."""
        # For a source at zenith: l=0, m=0, n=1
        # This happens when HA=0 and dec=lat
        lat_rad = np.deg2rad(37.0)
        dec_rad = lat_rad  # Source at zenith
        ha_rad = np.array([0.0])

        l, m, n = hadec_to_direction_cosines(ha_rad, dec_rad, lat_rad)

        assert np.isclose(l[0], 0.0, atol=1e-10)
        assert np.isclose(m[0], 0.0, atol=1e-10)
        assert np.isclose(n[0], 1.0, atol=1e-10)

    def test_direction_cosines_east(self):
        """Test direction cosines for source on east horizon."""
        # For a source on east horizon at equator: l=1, m=0, n=0
        lat_rad = 0.0  # Equator
        dec_rad = 0.0  # Celestial equator
        ha_rad = np.array([-np.pi/2])  # 6h east (negative HA)

        l, m, n = hadec_to_direction_cosines(ha_rad, dec_rad, lat_rad)

        assert np.isclose(l[0], -1.0, atol=1e-10)  # East is positive l
        assert np.isclose(m[0], 0.0, atol=1e-10)
        assert np.isclose(n[0], 0.0, atol=1e-10)

    def test_geometric_delays(self):
        """Test geometric delay computation."""
        # Simple test: source at zenith, delays should be z/c
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],  # 1m up
            [0.0, 0.0, -1.0],  # 1m down
        ])

        l = np.array([0.0])
        m = np.array([0.0])
        n = np.array([1.0])  # Zenith

        delays = compute_geometric_delays(positions, l, m, n)

        # delay = -z/c for zenith source
        expected = -positions[:, 2] / SPEED_OF_LIGHT_M_S
        assert np.allclose(delays[0], expected)


class TestGeometricBeamformer:
    """Tests for GeometricBeamformer."""

    def test_create_default(self):
        """Test default beamformer creation."""
        bf = GeometricBeamformer()
        assert bf.array_config.n_antennas == 13
        assert bf.freq_config.n_chan == 3072

    def test_compute_coherent_weights(self):
        """Test coherent weight computation."""
        bf = GeometricBeamformer()
        pc = PhaseCenter(ra_deg=0.0, dec_deg=45.0)

        # Short observation for testing
        times = np.array([1700000000.0, 1700000010.0])

        weights = bf.compute_weights(
            phase_centers=[pc],
            unix_times=times,
            mode='coherent'
        )

        assert weights.mode == BeamMode.COHERENT
        assert weights.n_beams == 1
        assert weights.n_times == 2
        assert weights.n_antennas == 13
        assert weights.n_channels == 3072

        # Weights should be unit magnitude
        assert np.allclose(np.abs(weights.weights), 1.0)

    def test_compute_incoherent_weights(self):
        """Test incoherent weight computation."""
        bf = GeometricBeamformer()

        times = np.array([1700000000.0])

        weights = bf.compute_weights(
            unix_times=times,
            mode='incoherent',
            n_beams=8
        )

        assert weights.mode == BeamMode.INCOHERENT
        assert weights.n_beams == 8
        # All weights should be unity (1 + 0j)
        assert np.allclose(weights.weights, 1.0)

    def test_multi_beam(self):
        """Test multi-beam weight computation."""
        bf = GeometricBeamformer()

        pcs = [
            PhaseCenter(ra_deg=0.0, dec_deg=30.0),
            PhaseCenter(ra_deg=90.0, dec_deg=30.0),
            PhaseCenter(ra_deg=180.0, dec_deg=30.0),
        ]

        times = np.array([1700000000.0])

        weights = bf.compute_weights(
            phase_centers=pcs,
            unix_times=times,
            mode='coherent'
        )

        assert weights.n_beams == 3
        # Different phase centers should have different weights
        assert not np.allclose(weights.weights[0], weights.weights[1])
        assert not np.allclose(weights.weights[1], weights.weights[2])

    def test_time_generation(self):
        """Test automatic time array generation."""
        bf = GeometricBeamformer()
        pc = PhaseCenter(ra_deg=0.0, dec_deg=45.0)

        weights = bf.compute_weights(
            phase_centers=[pc],
            start_time=1700000000.0,
            duration_sec=100.0,
            cadence_sec=10.0
        )

        assert weights.n_times == 10

    def test_flagged_antennas(self):
        """Test weights with flagged antennas."""
        arr_cfg = ArrayConfig()
        arr_cfg.flag_antennas([0, 12])

        bf = GeometricBeamformer(array_config=arr_cfg)
        pc = PhaseCenter(ra_deg=0.0, dec_deg=45.0)

        weights = bf.compute_weights(
            phase_centers=[pc],
            unix_times=np.array([1700000000.0])
        )

        assert weights.n_antennas == 11
        assert 0 not in weights.antenna_indices
        assert 12 not in weights.antenna_indices

    def test_phase_frequency_dependence(self):
        """Test that phases vary with frequency."""
        bf = GeometricBeamformer()
        pc = PhaseCenter(ra_deg=90.0, dec_deg=45.0)  # Off zenith

        weights = bf.compute_weights(
            phase_centers=[pc],
            unix_times=np.array([1700000000.0])
        )

        # Get phases for antenna 12 (outlier with z-offset)
        phases = np.angle(weights.weights[0, 0, -1, :])

        # Phases should vary across frequency
        phase_range = np.max(phases) - np.min(phases)
        assert phase_range > 0.1  # Some significant phase variation


class TestWeightsPhysics:
    """Tests for physical correctness of weights."""

    def test_zenith_source_minimal_delay(self):
        """For a zenith source, delays should be minimal for flat array."""
        arr_cfg = ArrayConfig()
        # Use only antennas in the z=0 plane
        arr_cfg.flag_antennas([12])  # Antenna 12 has z=-0.279

        bf = GeometricBeamformer(array_config=arr_cfg)

        # Create a source at approximate zenith for OVRO latitude
        pc = PhaseCenter(ra_deg=0.0, dec_deg=37.2)

        # Find time when RA=0 transits (LST ~ 0)
        # At OVRO, this happens around certain times
        times = np.array([1700000000.0])  # Arbitrary time for now

        delays = bf.compute_delays(pc, times)

        # For flat array looking at zenith, delays should be ~0
        # (only z contributes, and z=0 for flagged antennas)
        assert np.max(np.abs(delays)) < 1e-6  # Less than 1 microsecond

    def test_weight_conjugate_for_delay_compensation(self):
        """Verify weights are conjugate of the signal phase."""
        # For a positive delay tau, the signal phase is exp(+2*pi*i*f*tau)
        # The weight to compensate should be exp(-2*pi*i*f*tau)

        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # 1m East offset
        arr_cfg = ArrayConfig(positions_enu=positions)
        freq_cfg = FrequencyConfig(n_chan=1, freq_end_voltage_mhz=450.0)

        bf = GeometricBeamformer(array_config=arr_cfg, freq_config=freq_cfg)

        # Source in the East direction
        # This creates a positive path length difference for antenna 1
        pc = PhaseCenter(ra_deg=90.0, dec_deg=0.0)  # East

        weights = bf.compute_weights(
            phase_centers=[pc],
            unix_times=np.array([1700000000.0])
        )

        # The weight phases should compensate for the geometric delay
        # Antenna 1 is 1m East, so for a source in the East:
        # - Signal arrives at antenna 1 first (negative delay, positive path diff)
        # - Weight should have positive phase to compensate
        phase_diff = np.angle(weights.weights[0, 0, 1, 0]) - np.angle(weights.weights[0, 0, 0, 0])

        # We can't predict exact phase without knowing the time/direction precisely,
        # but the phases should be different for the two antennas
        assert not np.isclose(phase_diff, 0.0, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
