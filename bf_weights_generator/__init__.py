"""
CASM Geometric Beamformer Weights Generator
============================================

A Python package for computing geometric beamformer weights for the CASM
array at OVRO (Owens Valley Radio Observatory).

Main Classes
------------
GeometricBeamformer
    Main class for computing beamformer weights.
PhaseCenter
    Specification of sky position for tracking beams (RA/Dec).
StationaryPointing
    Specification of fixed beam direction for stationary beams (Alt/Az).
TrackingBeamWeights
    Container for time-dependent weights (tracking beams).
StationaryBeamWeights
    Container for time-independent weights (stationary beams).
ArrayConfig
    Configuration for antenna positions and observatory location.
FrequencyConfig
    Configuration for frequency channels.

Beamforming Modes
-----------------
Coherent
    Phase-aligns signals from all antennas toward a specified direction.
    Weights are complex exponentials that compensate for geometric delays.

Incoherent
    No phase steering; all beams are identical total-power beams.
    Weights are unity; the actual weighting (data conjugate) is applied at runtime.

Beam Types
----------
Tracking
    Beams follow celestial sources (RA/Dec). Weights are time-dependent
    because the source direction changes in the local frame as Earth rotates.

Stationary
    Beams are fixed in the local horizon frame (Alt/Az). Weights are
    time-independent. The sky drifts through the beams as Earth rotates.
    Ideal for FRB transit searches.

Quick Start
-----------
>>> import numpy as np
>>> from bf_weights_generator import GeometricBeamformer, PhaseCenter, StationaryPointing
>>>
>>> bf = GeometricBeamformer()
>>>
>>> # Tracking beam (follows source as Earth rotates)
>>> cyg_a = PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA")
>>> tracking = bf.compute_tracking_weights(
...     phase_centers=[cyg_a],
...     start_time=1700000000.0,
...     duration_sec=3600,
...     cadence_sec=10.0
... )
>>>
>>> # Stationary beam grid (fixed in local sky, for FRB transit search)
>>> from bf_weights_generator import generate_beam_grid_altaz
>>> pointings = generate_beam_grid_altaz(spacing_deg=4.0)
>>> stationary = bf.compute_stationary_weights(pointings)
"""

from typing import TypedDict

import numpy as np


__version__ = "0.3.0"
__author__ = "CASM Team"


class BFWeights(TypedDict, total=False):
    """Beamforming-weights container.

    Returned by ``generate_bf_weights`` (forthcoming) and accepted by
    ``deploy_bf_weights`` / ``inspect_snap_weights`` /
    ``plot_source_transit``. Compose-friendly TypedDict; the legacy
    HDF5/NPZ loaders return rich objects that are convertible to this
    dict via ``.to_bfweights_dict()`` (planned).
    """
    int8_weights: np.ndarray
    freqs_hz: np.ndarray
    ant64_to_snap: np.ndarray
    snap_to_ant64: np.ndarray
    active_mask: np.ndarray
    source: str
    cal_source_path: str
    layout_version: dict

# Core classes
from .weights import (
    GeometricBeamformer,
    PhaseCenter,
    StationaryPointing,
    TrackingBeamWeights,
    StationaryBeamWeights,
    BeamformerWeights,  # Backward compatibility alias
    BeamMode,
    BeamType,
    generate_beam_grid_altaz,
    generate_beam_grid_lm,
)

# Configuration
from .config import (
    ArrayConfig,
    FrequencyConfig,
    OVRO_LAT_DEG,
    OVRO_LON_DEG,
    OVRO_ALT_M,
    DEFAULT_ANTENNA_POSITIONS,
    DEFAULT_N_ANTENNAS,
    SPEED_OF_LIGHT_M_S,
    compute_beam_fwhm,
    estimate_n_beams,
)

# Coordinate utilities
from .coordinates import (
    compute_lst_rad,
    radec_to_hadec,
    hadec_to_direction_cosines,
    radec_to_direction_cosines,
    compute_geometric_delays,
)

# File I/O
from .io import (
    save_weights_hdf5,
    load_weights_hdf5,
    save_weights_npz,
    load_weights_npz,
    inspect_weights_file,
    save_int8_weights_hdf5,
    load_int8_weights_hdf5,
    inspect_int8_weights_file,
    save_combined_weights_hdf5,
    load_combined_weights_hdf5,
)

# SNAP 64-antenna weights
from .snap_weights import (
    Array64Config,
    SnapWeightsGenerator,
    Int8StationaryWeights,
    CombinedWeights,
    CalibrationWeights,
    load_calibration_weights,
    generate_combined_weights,
    TRANSIT_SURVEY_BEAMS,
    parse_beams_arg,
    generate_beam_grid,
)

# Compose-friendly inspect API (Phase 4)
from .inspect import inspect_snap_weights

__all__ = [
    # Version
    "__version__",
    # Core classes
    "GeometricBeamformer",
    "PhaseCenter",
    "StationaryPointing",
    "TrackingBeamWeights",
    "StationaryBeamWeights",
    "BeamformerWeights",
    "BeamMode",
    "BeamType",
    "generate_beam_grid_altaz",
    "generate_beam_grid_lm",
    # Configuration
    "ArrayConfig",
    "FrequencyConfig",
    "OVRO_LAT_DEG",
    "OVRO_LON_DEG",
    "OVRO_ALT_M",
    "DEFAULT_ANTENNA_POSITIONS",
    "DEFAULT_N_ANTENNAS",
    "SPEED_OF_LIGHT_M_S",
    "compute_beam_fwhm",
    "estimate_n_beams",
    # Coordinate utilities
    "compute_lst_rad",
    "radec_to_hadec",
    "hadec_to_direction_cosines",
    "radec_to_direction_cosines",
    "compute_geometric_delays",
    # File I/O
    "save_weights_hdf5",
    "load_weights_hdf5",
    "save_weights_npz",
    "load_weights_npz",
    "inspect_weights_file",
    "save_int8_weights_hdf5",
    "load_int8_weights_hdf5",
    "inspect_int8_weights_file",
    "save_combined_weights_hdf5",
    "load_combined_weights_hdf5",
    # SNAP 64-antenna weights
    "Array64Config",
    "SnapWeightsGenerator",
    "Int8StationaryWeights",
    "CombinedWeights",
    "CalibrationWeights",
    "load_calibration_weights",
    "generate_combined_weights",
    "TRANSIT_SURVEY_BEAMS",
    "parse_beams_arg",
    "generate_beam_grid",
    # Compose API (Phase 4)
    "BFWeights",
    "inspect_snap_weights",
]
