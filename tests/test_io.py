"""
Tests for file I/O functionality.
"""

import numpy as np
import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bf_weights_generator import (
    GeometricBeamformer,
    PhaseCenter,
    save_weights_npz,
    load_weights_npz,
    inspect_weights_file,
)

# Conditionally import HDF5 functions
try:
    from bf_weights_generator import save_weights_hdf5, load_weights_hdf5
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


def create_test_weights():
    """Helper to create test weights."""
    bf = GeometricBeamformer()
    pc = PhaseCenter(ra_deg=45.0, dec_deg=30.0, name="TestSource")
    return bf.compute_weights(
        phase_centers=[pc],
        unix_times=np.array([1700000000.0, 1700000010.0, 1700000020.0])
    )


class TestNpzIO:
    """Tests for NPZ file I/O."""

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip."""
        weights = create_test_weights()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name

        try:
            save_weights_npz(weights, filepath)
            loaded = load_weights_npz(filepath)

            # Check data matches
            assert np.allclose(weights.weights, loaded.weights)
            assert np.allclose(weights.unix_times, loaded.unix_times)
            assert np.allclose(weights.frequencies_hz, loaded.frequencies_hz)
            assert np.array_equal(weights.antenna_indices, loaded.antenna_indices)

            # Check metadata matches
            assert weights.mode == loaded.mode
            assert weights.n_beams == loaded.n_beams
            assert weights.n_times == loaded.n_times
            assert weights.n_antennas == loaded.n_antennas
            assert weights.n_channels == loaded.n_channels

            # Check phase centers
            assert len(weights.phase_centers) == len(loaded.phase_centers)
            for orig, load in zip(weights.phase_centers, loaded.phase_centers):
                assert orig.ra_deg == load.ra_deg
                assert orig.dec_deg == load.dec_deg
                assert orig.name == load.name

        finally:
            os.unlink(filepath)

    def test_inspect_npz(self):
        """Test file inspection."""
        weights = create_test_weights()

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name

        try:
            save_weights_npz(weights, filepath)
            info = inspect_weights_file(filepath)

            assert info['format'] == 'npz'
            assert info['mode'] == 'coherent'
            assert info['n_beams'] == 1
            assert info['n_times'] == 3
            assert info['n_antennas'] == 13
            assert info['n_channels'] == 3072

        finally:
            os.unlink(filepath)


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not installed")
class TestHdf5IO:
    """Tests for HDF5 file I/O."""

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip."""
        weights = create_test_weights()

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            save_weights_hdf5(weights, filepath, overwrite=True)
            loaded = load_weights_hdf5(filepath)

            # Check data matches
            assert np.allclose(weights.weights, loaded.weights)
            assert np.allclose(weights.unix_times, loaded.unix_times)
            assert np.allclose(weights.frequencies_hz, loaded.frequencies_hz)
            assert np.array_equal(weights.antenna_indices, loaded.antenna_indices)

            # Check metadata
            assert weights.mode == loaded.mode
            assert weights.n_beams == loaded.n_beams

        finally:
            os.unlink(filepath)

    def test_compression(self):
        """Test that compression works."""
        weights = create_test_weights()

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            save_weights_hdf5(weights, filepath, compression_opts=9, overwrite=True)
            loaded = load_weights_hdf5(filepath)
            assert np.allclose(weights.weights, loaded.weights)

        finally:
            os.unlink(filepath)

    def test_no_overwrite_error(self):
        """Test that overwrite=False raises error for existing file."""
        weights = create_test_weights()

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            save_weights_hdf5(weights, filepath, overwrite=True)

            with pytest.raises(FileExistsError):
                save_weights_hdf5(weights, filepath, overwrite=False)

        finally:
            os.unlink(filepath)

    def test_inspect_hdf5(self):
        """Test HDF5 file inspection."""
        weights = create_test_weights()

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            filepath = f.name

        try:
            save_weights_hdf5(weights, filepath, overwrite=True)
            info = inspect_weights_file(filepath)

            assert info['format'] == 'hdf5'
            assert info['mode'] == 'coherent'
            assert info['weights_dtype'] == 'complex64'

        finally:
            os.unlink(filepath)


class TestIncoherentIO:
    """Tests for incoherent mode I/O."""

    def test_incoherent_npz_roundtrip(self):
        """Test incoherent weights save/load."""
        bf = GeometricBeamformer()
        weights = bf.compute_weights(
            unix_times=np.array([1700000000.0]),
            mode='incoherent',
            n_beams=8
        )

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name

        try:
            save_weights_npz(weights, filepath)
            loaded = load_weights_npz(filepath)

            assert loaded.mode.value == 'incoherent'
            assert loaded.n_beams == 8
            assert np.allclose(loaded.weights, 1.0)

        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
