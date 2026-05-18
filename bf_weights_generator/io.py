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

# Import snap_weights classes lazily to avoid circular imports
def _get_snap_weights_classes():
    """Lazy import of snap_weights module classes."""
    from .snap_weights import Int8StationaryWeights, Array64Config
    return Int8StationaryWeights, Array64Config


def _get_combined_weights_classes():
    """Lazy import of CombinedWeights and related classes."""
    from .snap_weights import (
        CombinedWeights, Array64Config, CalibrationWeights,
    )
    return CombinedWeights, Array64Config, CalibrationWeights


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


# =============================================================================
# Int8 Weights I/O (for SNAP beamformer)
# =============================================================================

def save_int8_weights_hdf5(
    weights,  # Int8StationaryWeights - type hint omitted to avoid circular import
    filepath: Union[str, Path],
    compression: str = "gzip",
    compression_opts: int = 4,
    overwrite: bool = False,
) -> None:
    """
    Save int8-quantized beamformer weights to HDF5 file.

    Parameters
    ----------
    weights : Int8StationaryWeights
        Quantized weights object to save.
    filepath : str or Path
        Output file path.
    compression : str, optional
        Compression algorithm. Default is "gzip".
    compression_opts : int, optional
        Compression level (1-9). Default is 4.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is False.

    Notes
    -----
    HDF5 structure (schema v2.0):
        /
        ├── weights_int8          # int8, shape (2, n_chan, 2, n_beams, 64)
        │                         #   axis 4 is in SNAP input order
        │                         #   (snap_input_idx = snap_id * 12 + adc)
        ├── frequencies_hz        # float64, shape (n_chan,)
        ├── pointings/
        │   ├── alt_deg, az_deg   # float, shape (n_beams,)
        │   └── names             # string attribute
        ├── array_config/
        │   ├── positions_enu     # float64, shape (64, 3) — by snap_input_idx
        │   ├── active_mask       # bool,    shape (64,)   — by snap_input_idx
        │   └── antenna_ids       # int,     shape (64,)   — real antenna_id at
        │                         #   each snap_input_idx slot, -1 if unwired
        └── Attributes: scale_factor, n_beams, n_channels, n_pol, n_antennas

    The reader supports v1.0 files (with the legacy ``snap_to_ant64`` /
    ``ant64_to_snap`` datasets) and synthesizes ``antenna_ids`` on load.
    """
    _check_h5py()

    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"File exists: {filepath}. Use overwrite=True to replace.")

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        # Store main data
        f.create_dataset('weights_int8', data=weights.weights_int8,
                         compression=compression, compression_opts=compression_opts)
        f.create_dataset('frequencies_hz', data=weights.frequencies_hz)

        # Store root attributes.
        # n_antennas mirrors the SNAP slot count (n_snaps * n_adc — 72 on
        # current 6×12 hardware). Read from the array_config so that it
        # tracks reality even if the slot count changes.
        f.attrs['scale_factor'] = weights.scale_factor
        f.attrs['n_beams'] = weights.n_beams
        f.attrs['n_channels'] = weights.n_channels
        f.attrs['n_pol'] = 2
        f.attrs['n_antennas'] = len(weights.array_config.positions_enu)
        f.attrs['created_utc'] = datetime.now(timezone.utc).isoformat()
        f.attrs['version'] = '2.0'
        f.attrs['format_type'] = 'int8_snap_weights'

        # Store pointings
        pt_grp = f.create_group('pointings')
        pt_grp.create_dataset('alt_deg', data=[p.alt_deg for p in weights.pointings])
        pt_grp.create_dataset('az_deg', data=[p.az_deg for p in weights.pointings])
        pt_grp.attrs['names'] = json.dumps([p.name for p in weights.pointings])

        # Store array configuration (v2.0: antenna_ids replaces the two
        # reorder maps because every internal index is already a
        # snap_input_idx).
        arr_grp = f.create_group('array_config')
        arr_grp.create_dataset('positions_enu', data=weights.array_config.positions_enu)
        arr_grp.create_dataset('active_mask', data=weights.array_config.active_mask)
        arr_grp.create_dataset('antenna_ids', data=weights.array_config.antenna_ids)
        arr_grp.attrs['csv_path'] = weights.array_config.csv_path
        arr_grp.attrs['pos_ids'] = json.dumps(weights.array_config.pos_ids)

        # Store frequency configuration
        freq_grp = f.create_group('freq_config')
        freq_grp.attrs['n_chan'] = weights.freq_config.n_chan
        freq_grp.attrs['total_bw_mhz'] = weights.freq_config.total_bw_mhz
        freq_grp.attrs['total_n_chan'] = weights.freq_config.total_n_chan
        freq_grp.attrs['freq_end_voltage_mhz'] = weights.freq_config.freq_end_voltage_mhz


def load_int8_weights_hdf5(filepath: Union[str, Path]):
    """
    Load int8-quantized beamformer weights from HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    Int8StationaryWeights
        Loaded quantized weights object.
    """
    _check_h5py()
    Int8StationaryWeights, Array64Config = _get_snap_weights_classes()

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        # Load main data
        weights_int8 = f['weights_int8'][:]
        frequencies_hz = f['frequencies_hz'][:]
        scale_factor = float(f.attrs['scale_factor'])

        # Load pointings
        pt_grp = f['pointings']
        alt_deg = pt_grp['alt_deg'][:]
        az_deg = pt_grp['az_deg'][:]
        names = json.loads(pt_grp.attrs['names'])
        pointings = [
            StationaryPointing(alt_deg=alt, az_deg=az, name=name)
            for alt, az, name in zip(alt_deg, az_deg, names)
        ]

        # Load array configuration (supports v2.0 antenna_ids and v1.0
        # legacy snap_to_ant64/ant64_to_snap reorder maps).
        arr_grp = f['array_config']
        array_config = _array64_from_h5_group(arr_grp, Array64Config)

        # Load frequency configuration
        freq_grp = f['freq_config']
        freq_config = FrequencyConfig(
            n_chan=int(freq_grp.attrs['n_chan']),
            total_bw_mhz=float(freq_grp.attrs['total_bw_mhz']),
            total_n_chan=int(freq_grp.attrs['total_n_chan']),
            freq_end_voltage_mhz=float(freq_grp.attrs['freq_end_voltage_mhz']),
        )

        return Int8StationaryWeights(
            weights_int8=weights_int8,
            pointings=pointings,
            frequencies_hz=frequencies_hz,
            array_config=array_config,
            freq_config=freq_config,
            scale_factor=scale_factor,
        )


def inspect_int8_weights_file(filepath: Union[str, Path]) -> dict:
    """
    Inspect an int8 weights file without loading the full data.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict
        Dictionary containing file metadata and array shapes.
    """
    _check_h5py()

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        info = {
            'format': 'hdf5',
            'format_type': f.attrs.get('format_type', 'unknown'),
            'scale_factor': float(f.attrs['scale_factor']),
            'n_beams': int(f.attrs['n_beams']),
            'n_channels': int(f.attrs['n_channels']),
            'n_pol': int(f.attrs['n_pol']),
            'n_antennas': int(f.attrs['n_antennas']),
            'created_utc': f.attrs.get('created_utc', 'unknown'),
            'version': f.attrs.get('version', 'unknown'),
            'weights_shape': f['weights_int8'].shape,
            'weights_dtype': str(f['weights_int8'].dtype),
            'file_size_mb': filepath.stat().st_size / 1e6,
        }

        # Add array config info
        arr_grp = f['array_config']
        info['n_active_antennas'] = int(np.sum(arr_grp['active_mask'][:]))
        info['csv_path'] = arr_grp.attrs.get('csv_path', 'unknown')

        # Add pointing info
        pt_grp = f['pointings']
        names = json.loads(pt_grp.attrs['names'])
        info['beam_names'] = names

    return info


# =============================================================================
# Combined Weights I/O (complex64 geo+cal)
# =============================================================================

def _array64_from_h5_group(grp, Array64ConfigCls):
    """Construct Array64Config from an HDF5 group (v1.0 or v2.0).

    v2.0 stores ``antenna_ids`` directly. v1.0 stored ``snap_to_ant64``
    and ``ant64_to_snap`` reorder maps; we synthesize ``antenna_ids`` by
    treating the legacy ``ant64`` slot index as ``antenna_id - 1`` (the
    convention the v1.0 fixtures and tests used). This matches what the
    pre-refactor pipeline did, so existing weights HDF5 files keep
    loading correctly.
    """
    positions_enu = grp['positions_enu'][:]
    active_mask = grp['active_mask'][:]
    pos_ids = json.loads(grp.attrs['pos_ids'])
    csv_path = grp.attrs.get('csv_path', '')

    if 'antenna_ids' in grp:
        antenna_ids = grp['antenna_ids'][:].astype(np.int32)
    elif 'snap_to_ant64' in grp:
        # v1.0 legacy: ant64 was an arbitrary slot; the convention used
        # by every test fixture and by save_calibration's writer was
        # antenna_id == ant64 + 1. Reconstruct on that basis. The legacy
        # files were written at 64 slots; size the reconstructed array
        # from the source dataset so we don't truncate.
        snap_to_ant64 = grp['snap_to_ant64'][:]
        n_slots = len(snap_to_ant64)
        antenna_ids = np.full(n_slots, -1, dtype=np.int32)
        # Walk SNAP inputs; the ant64 index at that input gives the
        # antenna_id (after +1) that lives in this slot.
        for snap_idx in range(n_slots):
            ant64_idx = int(snap_to_ant64[snap_idx])
            if ant64_idx >= 0:
                antenna_ids[snap_idx] = ant64_idx + 1
    else:
        raise ValueError(
            "array_config group has neither 'antenna_ids' (v2.0) nor "
            "'snap_to_ant64' (v1.0); cannot reconstruct slot mapping."
        )

    return Array64ConfigCls(
        positions_enu=positions_enu,
        active_mask=active_mask,
        antenna_ids=antenna_ids,
        pos_ids=pos_ids,
        csv_path=csv_path,
    )


def _save_array64_config_group(grp, config):
    """Save Array64Config into an HDF5 group (v2.0 schema)."""
    grp.create_dataset('positions_enu', data=config.positions_enu)
    grp.create_dataset('active_mask', data=config.active_mask)
    grp.create_dataset('antenna_ids', data=config.antenna_ids)
    grp.attrs['csv_path'] = config.csv_path
    grp.attrs['pos_ids'] = json.dumps(config.pos_ids)
    grp.attrs['n_active'] = config.n_active


def _load_array64_config_group(grp):
    """Load Array64Config from an HDF5 group (v1.0 or v2.0)."""
    _, Array64Config, _ = _get_combined_weights_classes()
    return _array64_from_h5_group(grp, Array64Config)


def save_combined_weights_hdf5(
    weights,  # CombinedWeights - type hint omitted to avoid circular import
    filepath: Union[str, Path],
    compression: str = "gzip",
    compression_opts: int = 4,
    overwrite: bool = False,
) -> None:
    """
    Save combined geo+cal beamformer weights to HDF5.

    Parameters
    ----------
    weights : CombinedWeights
        Combined weights object to save.
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

    with h5py.File(filepath, 'w') as f:
        # Main data
        f.create_dataset(
            'weights', data=weights.weights,
            compression=compression, compression_opts=compression_opts,
        )
        f.create_dataset('frequencies_hz', data=weights.frequencies_hz)
        f.create_dataset('channel_flags', data=weights.flags)

        # Root attributes. n_antennas mirrors the SNAP slot count
        # (n_snaps * n_adc — 72 on current 6×12 hardware).
        f.attrs['format_type'] = 'combined_complex64_snap_weights'
        f.attrs['version'] = '1.0'
        f.attrs['created_utc'] = datetime.now(timezone.utc).isoformat()
        f.attrs['n_beams'] = weights.n_beams
        f.attrs['n_antennas'] = len(weights.array_config.positions_enu)
        f.attrs['n_channels'] = weights.n_channels
        f.attrs['n_good_channels'] = weights.n_good_channels
        f.attrs['weights_dtype'] = str(weights.weights.dtype)
        f.attrs['freq_order'] = weights.freq_order
        f.attrs['antenna_order'] = 'snap_input'

        # Pointings
        pt_grp = f.create_group('pointings')
        pt_grp.create_dataset('alt_deg', data=[p.alt_deg for p in weights.pointings])
        pt_grp.create_dataset('az_deg', data=[p.az_deg for p in weights.pointings])
        pt_grp.create_dataset('l', data=[p.l for p in weights.pointings])
        pt_grp.create_dataset('m', data=[p.m for p in weights.pointings])
        pt_grp.create_dataset('n', data=[p.n for p in weights.pointings])
        pt_grp.attrs['names'] = json.dumps([p.name for p in weights.pointings])

        # Compute array config
        _save_array64_config_group(
            f.create_group('compute_array_config'), weights.array_config
        )

        # Output array config
        _save_array64_config_group(
            f.create_group('output_array_config'), weights.output_array_config
        )

        # Frequency config
        freq_grp = f.create_group('freq_config')
        freq_grp.attrs['n_chan'] = weights.freq_config.n_chan
        freq_grp.attrs['total_bw_mhz'] = weights.freq_config.total_bw_mhz
        freq_grp.attrs['total_n_chan'] = weights.freq_config.total_n_chan
        freq_grp.attrs['freq_end_voltage_mhz'] = weights.freq_config.freq_end_voltage_mhz

        # Calibration metadata
        if weights.cal_weights is not None:
            cal = weights.cal_weights
            cal_grp = f.create_group('calibration')
            cal_grp.attrs['source'] = cal.source
            cal_grp.attrs['ref_ant_id'] = cal.ref_ant_id
            cal_grp.attrs['n_cal_antennas'] = len(cal.ant_ids)
            cal_grp.attrs['n_good_channels'] = int(np.sum(cal.flags))
            cal_grp.attrs['n_total_channels'] = len(cal.flags)
            cal_grp.create_dataset('ant_ids', data=cal.ant_ids)
            cal_grp.create_dataset('flags', data=cal.flags)
            cal_grp.create_dataset('frequencies_hz', data=cal.frequencies_hz)
            cal_grp.create_dataset(
                'weights', data=cal.weights,
                compression=compression, compression_opts=compression_opts,
            )
        else:
            f.attrs['calibration'] = 'none (geometric only)'


def load_combined_weights_hdf5(filepath: Union[str, Path]):
    """
    Load combined geo+cal beamformer weights from HDF5.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    CombinedWeights
        Loaded combined weights object.
    """
    _check_h5py()
    CombinedWeights, Array64Config, CalibrationWeights = _get_combined_weights_classes()

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        weights_data = f['weights'][:]
        frequencies_hz = f['frequencies_hz'][:]
        flags = f['channel_flags'][:]
        freq_order = f.attrs.get('freq_order', 'descending')

        # Pointings
        pt_grp = f['pointings']
        alt_deg = pt_grp['alt_deg'][:]
        az_deg = pt_grp['az_deg'][:]
        names = json.loads(pt_grp.attrs['names'])
        pointings = [
            StationaryPointing(alt_deg=alt, az_deg=az, name=name)
            for alt, az, name in zip(alt_deg, az_deg, names)
        ]

        # Array configs
        array_config = _load_array64_config_group(f['compute_array_config'])
        output_array_config = _load_array64_config_group(f['output_array_config'])

        # Frequency config
        freq_grp = f['freq_config']
        freq_config = FrequencyConfig(
            n_chan=int(freq_grp.attrs['n_chan']),
            total_bw_mhz=float(freq_grp.attrs['total_bw_mhz']),
            total_n_chan=int(freq_grp.attrs['total_n_chan']),
            freq_end_voltage_mhz=float(freq_grp.attrs['freq_end_voltage_mhz']),
        )

        # Calibration metadata (optional)
        cal_weights = None
        if 'calibration' in f:
            cal_grp = f['calibration']
            cal_weights = CalibrationWeights(
                weights=cal_grp['weights'][:],
                flags=cal_grp['flags'][:],
                frequencies_hz=cal_grp['frequencies_hz'][:],
                ant_ids=cal_grp['ant_ids'][:],
                ref_ant_id=int(cal_grp.attrs['ref_ant_id']),
                source=str(cal_grp.attrs['source']),
            )

        return CombinedWeights(
            weights=weights_data,
            frequencies_hz=frequencies_hz,
            flags=flags,
            pointings=pointings,
            array_config=array_config,
            output_array_config=output_array_config,
            freq_config=freq_config,
            cal_weights=cal_weights,
            freq_order=freq_order,
        )
