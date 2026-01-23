"""
File I/O for beamformer weights.

Supports HDF5 format for efficient storage and retrieval of large weight arrays.
Handles both tracking (time-dependent) and stationary (time-independent) weights.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timezone
import json

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from .weights import (
    TrackingBeamWeights,
    StationaryBeamWeights,
    BeamMode,
    BeamType,
    PhaseCenter,
    StationaryPointing,
)
from .config import ArrayConfig, FrequencyConfig


def _check_h5py():
    """Raise error if h5py is not available."""
    if not HDF5_AVAILABLE:
        raise ImportError(
            "h5py is required for HDF5 file operations. "
            "Install with: pip install h5py"
        )


def save_weights_hdf5(weights: Union[TrackingBeamWeights, StationaryBeamWeights],
                       filepath: Union[str, Path],
                       compression: str = "gzip",
                       compression_opts: int = 4,
                       overwrite: bool = False) -> None:
    """
    Save beamformer weights to HDF5 file.

    Handles both tracking (time-dependent) and stationary (time-independent) weights.

    Parameters
    ----------
    weights : TrackingBeamWeights or StationaryBeamWeights
        Beamformer weights object to save.
    filepath : str or Path
        Output file path.
    compression : str, optional
        Compression algorithm. Default is "gzip".
    compression_opts : int, optional
        Compression level (1-9). Default is 4.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is False.
    """
    _check_h5py()

    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File exists: {filepath}. Use overwrite=True to replace.")

    filepath.parent.mkdir(parents=True, exist_ok=True)

    is_stationary = isinstance(weights, StationaryBeamWeights)

    with h5py.File(filepath, 'w') as f:
        # Store main data arrays
        f.create_dataset('weights', data=weights.weights,
                         compression=compression, compression_opts=compression_opts)
        f.create_dataset('frequencies_hz', data=weights.frequencies_hz)
        f.create_dataset('antenna_indices', data=weights.antenna_indices)

        # Store root attributes
        f.attrs['mode'] = weights.mode.value
        f.attrs['beam_type'] = weights.beam_type.value
        f.attrs['is_stationary'] = is_stationary
        f.attrs['n_beams'] = weights.n_beams
        f.attrs['n_antennas'] = weights.n_antennas
        f.attrs['n_channels'] = weights.n_channels
        f.attrs['created_utc'] = datetime.now(timezone.utc).isoformat()
        f.attrs['version'] = '2.0'

        if is_stationary:
            # Stationary weights: store pointings
            pt_grp = f.create_group('pointings')
            pt_grp.create_dataset('alt_deg', data=[p.alt_deg for p in weights.pointings])
            pt_grp.create_dataset('az_deg', data=[p.az_deg for p in weights.pointings])
            pt_grp.create_dataset('l', data=[p.l for p in weights.pointings])
            pt_grp.create_dataset('m', data=[p.m for p in weights.pointings])
            pt_grp.create_dataset('n', data=[p.n for p in weights.pointings])
            pt_grp.attrs['names'] = json.dumps([p.name for p in weights.pointings])
        else:
            # Tracking weights: store phase centers and times
            f.create_dataset('unix_times', data=weights.unix_times)
            f.attrs['n_times'] = weights.n_times

            pc_grp = f.create_group('phase_centers')
            pc_grp.create_dataset('ra_deg', data=[pc.ra_deg for pc in weights.phase_centers])
            pc_grp.create_dataset('dec_deg', data=[pc.dec_deg for pc in weights.phase_centers])
            pc_grp.attrs['names'] = json.dumps([pc.name for pc in weights.phase_centers])

        # Store array configuration
        arr_grp = f.create_group('array_config')
        arr_grp.create_dataset('positions_enu', data=weights.array_config.positions_enu)
        arr_grp.create_dataset('antenna_flags', data=weights.array_config.antenna_flags)
        arr_grp.attrs['lat_deg'] = weights.array_config.lat_deg
        arr_grp.attrs['lon_deg'] = weights.array_config.lon_deg
        arr_grp.attrs['alt_m'] = weights.array_config.alt_m

        # Store frequency configuration
        freq_grp = f.create_group('freq_config')
        freq_grp.attrs['n_chan'] = weights.freq_config.n_chan
        freq_grp.attrs['total_bw_mhz'] = weights.freq_config.total_bw_mhz
        freq_grp.attrs['total_n_chan'] = weights.freq_config.total_n_chan
        freq_grp.attrs['freq_end_voltage_mhz'] = weights.freq_config.freq_end_voltage_mhz


def load_weights_hdf5(filepath: Union[str, Path]) -> Union[TrackingBeamWeights, StationaryBeamWeights]:
    """
    Load beamformer weights from HDF5 file.

    Automatically detects whether the file contains tracking or stationary weights.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    TrackingBeamWeights or StationaryBeamWeights
        Loaded beamformer weights object.
    """
    _check_h5py()

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        # Load main data arrays
        weights_data = f['weights'][:]
        frequencies_hz = f['frequencies_hz'][:]
        antenna_indices = f['antenna_indices'][:]

        # Load mode and type
        mode = BeamMode(f.attrs['mode'])
        beam_type = BeamType(f.attrs.get('beam_type', 'tracking'))
        is_stationary = f.attrs.get('is_stationary', False)

        # Load array configuration
        arr_grp = f['array_config']
        array_config = ArrayConfig(
            positions_enu=arr_grp['positions_enu'][:],
            antenna_flags=arr_grp['antenna_flags'][:],
            lat_deg=arr_grp.attrs['lat_deg'],
            lon_deg=arr_grp.attrs['lon_deg'],
            alt_m=arr_grp.attrs['alt_m']
        )

        # Load frequency configuration
        freq_grp = f['freq_config']
        freq_config = FrequencyConfig(
            n_chan=int(freq_grp.attrs['n_chan']),
            total_bw_mhz=float(freq_grp.attrs['total_bw_mhz']),
            total_n_chan=int(freq_grp.attrs['total_n_chan']),
            freq_end_voltage_mhz=float(freq_grp.attrs['freq_end_voltage_mhz'])
        )

        if is_stationary:
            # Load stationary pointings
            pt_grp = f['pointings']
            alt_deg = pt_grp['alt_deg'][:]
            az_deg = pt_grp['az_deg'][:]
            names = json.loads(pt_grp.attrs['names'])
            pointings = [
                StationaryPointing(alt_deg=alt, az_deg=az, name=name)
                for alt, az, name in zip(alt_deg, az_deg, names)
            ]

            return StationaryBeamWeights(
                weights=weights_data,
                mode=mode,
                beam_type=beam_type,
                pointings=pointings,
                frequencies_hz=frequencies_hz,
                antenna_indices=antenna_indices,
                array_config=array_config,
                freq_config=freq_config
            )
        else:
            # Load tracking phase centers and times
            unix_times = f['unix_times'][:]

            pc_grp = f['phase_centers']
            ra_deg = pc_grp['ra_deg'][:]
            dec_deg = pc_grp['dec_deg'][:]
            names = json.loads(pc_grp.attrs['names'])
            phase_centers = [
                PhaseCenter(ra_deg=ra, dec_deg=dec, name=name)
                for ra, dec, name in zip(ra_deg, dec_deg, names)
            ]

            return TrackingBeamWeights(
                weights=weights_data,
                mode=mode,
                beam_type=beam_type,
                phase_centers=phase_centers,
                unix_times=unix_times,
                frequencies_hz=frequencies_hz,
                antenna_indices=antenna_indices,
                array_config=array_config,
                freq_config=freq_config
            )


def save_weights_npz(weights: Union[TrackingBeamWeights, StationaryBeamWeights],
                      filepath: Union[str, Path],
                      compressed: bool = True) -> None:
    """
    Save beamformer weights to NumPy npz file.

    Parameters
    ----------
    weights : TrackingBeamWeights or StationaryBeamWeights
        Beamformer weights object to save.
    filepath : str or Path
        Output file path.
    compressed : bool, optional
        Whether to use compression. Default is True.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    is_stationary = isinstance(weights, StationaryBeamWeights)

    # Prepare metadata as JSON string
    metadata = {
        'mode': weights.mode.value,
        'beam_type': weights.beam_type.value,
        'is_stationary': is_stationary,
        'n_beams': weights.n_beams,
        'n_antennas': weights.n_antennas,
        'n_channels': weights.n_channels,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'version': '2.0',
        'array_config': {
            'lat_deg': weights.array_config.lat_deg,
            'lon_deg': weights.array_config.lon_deg,
            'alt_m': weights.array_config.alt_m,
        },
        'freq_config': {
            'n_chan': weights.freq_config.n_chan,
            'total_bw_mhz': weights.freq_config.total_bw_mhz,
            'total_n_chan': weights.freq_config.total_n_chan,
            'freq_end_voltage_mhz': weights.freq_config.freq_end_voltage_mhz,
        }
    }

    if is_stationary:
        metadata['pointings'] = [
            {'alt_deg': p.alt_deg, 'az_deg': p.az_deg, 'l': p.l, 'm': p.m, 'n': p.n, 'name': p.name}
            for p in weights.pointings
        ]
    else:
        metadata['n_times'] = weights.n_times
        metadata['phase_centers'] = [
            {'ra_deg': pc.ra_deg, 'dec_deg': pc.dec_deg, 'name': pc.name}
            for pc in weights.phase_centers
        ]

    save_func = np.savez_compressed if compressed else np.savez

    save_data = {
        'weights': weights.weights,
        'frequencies_hz': weights.frequencies_hz,
        'antenna_indices': weights.antenna_indices,
        'positions_enu': weights.array_config.positions_enu,
        'antenna_flags': weights.array_config.antenna_flags,
        'metadata': np.array(json.dumps(metadata)),
    }

    if not is_stationary:
        save_data['unix_times'] = weights.unix_times

    save_func(filepath, **save_data)


def load_weights_npz(filepath: Union[str, Path]) -> Union[TrackingBeamWeights, StationaryBeamWeights]:
    """
    Load beamformer weights from NumPy npz file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    TrackingBeamWeights or StationaryBeamWeights
        Loaded beamformer weights object.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.load(filepath, allow_pickle=False)
    metadata = json.loads(str(data['metadata']))

    # Reconstruct array config
    array_config = ArrayConfig(
        positions_enu=data['positions_enu'],
        antenna_flags=data['antenna_flags'],
        **metadata['array_config']
    )

    # Reconstruct freq config
    freq_config = FrequencyConfig(**metadata['freq_config'])

    is_stationary = metadata.get('is_stationary', False)
    mode = BeamMode(metadata['mode'])
    beam_type = BeamType(metadata.get('beam_type', 'tracking'))

    if is_stationary:
        # Reconstruct stationary pointings
        pointings = [
            StationaryPointing(
                alt_deg=p['alt_deg'], az_deg=p['az_deg'], name=p['name']
            )
            for p in metadata['pointings']
        ]

        return StationaryBeamWeights(
            weights=data['weights'],
            mode=mode,
            beam_type=beam_type,
            pointings=pointings,
            frequencies_hz=data['frequencies_hz'],
            antenna_indices=data['antenna_indices'],
            array_config=array_config,
            freq_config=freq_config
        )
    else:
        # Reconstruct tracking phase centers
        phase_centers = [
            PhaseCenter(ra_deg=pc['ra_deg'], dec_deg=pc['dec_deg'], name=pc['name'])
            for pc in metadata['phase_centers']
        ]

        return TrackingBeamWeights(
            weights=data['weights'],
            mode=mode,
            beam_type=beam_type,
            phase_centers=phase_centers,
            unix_times=data['unix_times'],
            frequencies_hz=data['frequencies_hz'],
            antenna_indices=data['antenna_indices'],
            array_config=array_config,
            freq_config=freq_config
        )


def inspect_weights_file(filepath: Union[str, Path]) -> dict:
    """
    Inspect a weights file without loading the full data.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict
        Dictionary containing file metadata and array shapes.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == '.h5' or suffix == '.hdf5':
        _check_h5py()
        with h5py.File(filepath, 'r') as f:
            is_stationary = f.attrs.get('is_stationary', False)
            info = {
                'format': 'hdf5',
                'mode': f.attrs['mode'],
                'beam_type': f.attrs.get('beam_type', 'tracking'),
                'is_stationary': is_stationary,
                'n_beams': int(f.attrs['n_beams']),
                'n_antennas': int(f.attrs['n_antennas']),
                'n_channels': int(f.attrs['n_channels']),
                'created_utc': f.attrs.get('created_utc', 'unknown'),
                'version': f.attrs.get('version', 'unknown'),
                'weights_shape': f['weights'].shape,
                'weights_dtype': str(f['weights'].dtype),
                'file_size_mb': filepath.stat().st_size / 1e6,
            }
            if not is_stationary:
                info['n_times'] = int(f.attrs.get('n_times', f['unix_times'].shape[0]))
    elif suffix == '.npz':
        data = np.load(filepath, allow_pickle=False)
        metadata = json.loads(str(data['metadata']))
        is_stationary = metadata.get('is_stationary', False)
        info = {
            'format': 'npz',
            'mode': metadata['mode'],
            'beam_type': metadata.get('beam_type', 'tracking'),
            'is_stationary': is_stationary,
            'n_beams': metadata['n_beams'],
            'n_antennas': metadata['n_antennas'],
            'n_channels': metadata['n_channels'],
            'created_utc': metadata.get('created_utc', 'unknown'),
            'version': metadata.get('version', 'unknown'),
            'weights_shape': data['weights'].shape,
            'weights_dtype': str(data['weights'].dtype),
            'file_size_mb': filepath.stat().st_size / 1e6,
        }
        if not is_stationary:
            info['n_times'] = metadata.get('n_times', len(data['unix_times']))
    else:
        raise ValueError(f"Unknown file format: {suffix}")

    return info
