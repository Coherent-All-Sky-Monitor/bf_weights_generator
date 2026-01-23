"""
Core beamformer weight computation for CASM.

Supports:
- Coherent and incoherent beamforming modes
- Tracking beams (RA/Dec, time-dependent weights)
- Stationary beams (Alt/Az, time-independent weights)
- Beam grid generation for transit surveys
"""

import numpy as np
from typing import Optional, Union, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import warnings

from .config import ArrayConfig, FrequencyConfig, SPEED_OF_LIGHT_M_S
from .coordinates import (
    radec_to_direction_cosines,
    radec_to_direction_cosines_astropy,
    compute_geometric_delays,
    ASTROPY_AVAILABLE
)


class BeamMode(Enum):
    """Beamforming mode enumeration."""
    COHERENT = "coherent"
    INCOHERENT = "incoherent"


class BeamType(Enum):
    """Beam type enumeration."""
    TRACKING = "tracking"      # RA/Dec, time-dependent weights
    STATIONARY = "stationary"  # Alt/Az, time-independent weights


@dataclass
class PhaseCenter:
    """
    Phase center specification for tracking beams (RA/Dec).

    Attributes
    ----------
    ra_deg : float
        Right Ascension in degrees (J2000).
    dec_deg : float
        Declination in degrees (J2000).
    name : str, optional
        Optional source name for reference.
    """
    ra_deg: float
    dec_deg: float
    name: str = ""

    @classmethod
    def from_hours(cls, ra_hours: float, dec_deg: float, name: str = "") -> "PhaseCenter":
        """Create PhaseCenter from RA in hours."""
        return cls(ra_deg=ra_hours * 15.0, dec_deg=dec_deg, name=name)

    @classmethod
    def from_hms_dms(cls, ra_hms: str, dec_dms: str, name: str = "") -> "PhaseCenter":
        """
        Create PhaseCenter from sexagesimal strings.

        Parameters
        ----------
        ra_hms : str
            RA in format "HH:MM:SS.S" or "HHhMMmSS.Ss"
        dec_dms : str
            Dec in format "+DD:MM:SS.S" or "+DDdMMmSS.Ss"
        """
        # Parse RA
        ra_hms = ra_hms.replace('h', ':').replace('m', ':').replace('s', '')
        ra_parts = ra_hms.split(':')
        ra_hours = float(ra_parts[0]) + float(ra_parts[1]) / 60 + float(ra_parts[2]) / 3600
        ra_deg = ra_hours * 15.0

        # Parse Dec
        dec_dms = dec_dms.replace('d', ':').replace('m', ':').replace('s', '').replace("'", ':').replace('"', '')
        dec_parts = dec_dms.split(':')
        sign = -1 if dec_parts[0].startswith('-') else 1
        dec_deg = sign * (abs(float(dec_parts[0])) + float(dec_parts[1]) / 60 + float(dec_parts[2]) / 3600)

        return cls(ra_deg=ra_deg, dec_deg=dec_deg, name=name)

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
        return f"PhaseCenter(ra={self.ra_deg:.6f}°, dec={self.dec_deg:.6f}°{name_str})"


@dataclass
class StationaryPointing:
    """
    Pointing specification for stationary beams (Alt/Az or direction cosines).

    Stationary beams are fixed in the local horizon frame. The sky drifts
    through them as Earth rotates.

    Attributes
    ----------
    alt_deg : float, optional
        Altitude in degrees (0 = horizon, 90 = zenith).
    az_deg : float, optional
        Azimuth in degrees (0 = North, 90 = East).
    l : float, optional
        Direction cosine East component.
    m : float, optional
        Direction cosine North component.
    n : float, optional
        Direction cosine Up component (toward zenith).
    name : str
        Optional beam name for reference.

    Notes
    -----
    Specify either (alt_deg, az_deg) OR (l, m, n). If both are provided,
    direction cosines take precedence. If only (l, m) are provided,
    n is computed as sqrt(1 - l² - m²).

    Direction cosines in ENU frame:
        l = cos(alt) * sin(az)   [East]
        m = cos(alt) * cos(az)   [North]
        n = sin(alt)             [Up]
    """
    alt_deg: Optional[float] = None
    az_deg: Optional[float] = None
    l: Optional[float] = None
    m: Optional[float] = None
    n: Optional[float] = None
    name: str = ""

    def __post_init__(self):
        """Validate and compute direction cosines."""
        if self.l is not None and self.m is not None:
            # Direction cosines provided
            if self.n is None:
                # Compute n from l, m
                n_squared = 1.0 - self.l**2 - self.m**2
                if n_squared < 0:
                    raise ValueError(f"Invalid direction cosines: l²+m² > 1 (l={self.l}, m={self.m})")
                self.n = np.sqrt(n_squared)
            # Compute alt/az for reference
            self.alt_deg = np.rad2deg(np.arcsin(self.n))
            self.az_deg = np.rad2deg(np.arctan2(self.l, self.m))
            if self.az_deg < 0:
                self.az_deg += 360.0
        elif self.alt_deg is not None and self.az_deg is not None:
            # Alt/Az provided, compute direction cosines
            alt_rad = np.deg2rad(self.alt_deg)
            az_rad = np.deg2rad(self.az_deg)
            self.l = np.cos(alt_rad) * np.sin(az_rad)
            self.m = np.cos(alt_rad) * np.cos(az_rad)
            self.n = np.sin(alt_rad)
        else:
            raise ValueError("Must provide either (alt_deg, az_deg) or (l, m) or (l, m, n)")

    @classmethod
    def from_altaz(cls, alt_deg: float, az_deg: float, name: str = "") -> "StationaryPointing":
        """Create StationaryPointing from altitude and azimuth."""
        return cls(alt_deg=alt_deg, az_deg=az_deg, name=name)

    @classmethod
    def from_direction_cosines(cls, l: float, m: float, n: Optional[float] = None,
                                name: str = "") -> "StationaryPointing":
        """Create StationaryPointing from direction cosines."""
        return cls(l=l, m=m, n=n, name=name)

    @classmethod
    def from_zenith_angle(cls, za_deg: float, az_deg: float, name: str = "") -> "StationaryPointing":
        """Create StationaryPointing from zenith angle and azimuth."""
        return cls(alt_deg=90.0 - za_deg, az_deg=az_deg, name=name)

    @property
    def direction_cosines(self) -> Tuple[float, float, float]:
        """Return (l, m, n) direction cosines."""
        return (self.l, self.m, self.n)

    @property
    def zenith_angle_deg(self) -> float:
        """Return zenith angle in degrees."""
        return 90.0 - self.alt_deg

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
        return f"StationaryPointing(alt={self.alt_deg:.2f}°, az={self.az_deg:.2f}°{name_str})"


def generate_beam_grid_altaz(
    alt_min_deg: float = 30.0,
    alt_max_deg: float = 90.0,
    az_min_deg: float = 0.0,
    az_max_deg: float = 360.0,
    spacing_deg: float = 4.0,
    exclude_horizon: bool = True
) -> List[StationaryPointing]:
    """
    Generate a grid of stationary beam pointings in Alt/Az coordinates.

    Parameters
    ----------
    alt_min_deg : float
        Minimum altitude in degrees (default: 30°).
    alt_max_deg : float
        Maximum altitude in degrees (default: 90° = zenith).
    az_min_deg : float
        Minimum azimuth in degrees (default: 0° = North).
    az_max_deg : float
        Maximum azimuth in degrees (default: 360°).
    spacing_deg : float
        Approximate beam spacing in degrees (default: 4°).
        At higher altitudes, fewer azimuth beams are placed to maintain
        roughly uniform sky coverage.
    exclude_horizon : bool
        If True, exclude beams below alt_min_deg (default: True).

    Returns
    -------
    List[StationaryPointing]
        List of beam pointings covering the specified region.

    Notes
    -----
    The default spacing of 4° is appropriate for CASM's current configuration
    (~13 antennas, ~10m baseline). Adjust based on your array's beam size:
        beam_FWHM ≈ λ/D ≈ 0.75m / 10m ≈ 4° at 400 MHz
    """
    pointings = []
    beam_idx = 0

    # Generate altitude levels
    alt_values = np.arange(alt_min_deg, alt_max_deg + spacing_deg/2, spacing_deg)

    for alt in alt_values:
        if alt > 90.0:
            alt = 90.0

        # At higher altitudes, the azimuth circle is smaller
        # Adjust number of azimuth points to maintain ~uniform spacing
        if alt >= 89.0:
            # Near zenith, just one beam
            az_values = [0.0]
        else:
            # Circumference at this altitude: 2π * cos(alt)
            # Number of beams: circumference / spacing
            cos_alt = np.cos(np.deg2rad(alt))
            az_spacing = spacing_deg / cos_alt if cos_alt > 0.1 else 360.0

            # Handle partial azimuth range
            az_range = az_max_deg - az_min_deg
            if az_range >= 360.0:
                n_az = max(1, int(np.round(360.0 / az_spacing)))
                az_values = np.linspace(0, 360, n_az, endpoint=False)
            else:
                n_az = max(1, int(np.ceil(az_range / az_spacing)))
                az_values = np.linspace(az_min_deg, az_max_deg, n_az)

        for az in az_values:
            pointing = StationaryPointing(
                alt_deg=alt,
                az_deg=az % 360.0,
                name=f"beam_{beam_idx:03d}"
            )
            pointings.append(pointing)
            beam_idx += 1

    return pointings


def generate_beam_grid_lm(
    l_min: float = -0.8,
    l_max: float = 0.8,
    m_min: float = -0.8,
    m_max: float = 0.8,
    spacing: float = 0.07,
    exclude_below_horizon: bool = True
) -> List[StationaryPointing]:
    """
    Generate a grid of stationary beam pointings in direction cosine space.

    This creates a regular grid in (l, m) space, which projects to a
    roughly uniform grid on the sky near zenith.

    Parameters
    ----------
    l_min, l_max : float
        Range of l (East) direction cosine (default: -0.8 to 0.8).
    m_min, m_max : float
        Range of m (North) direction cosine (default: -0.8 to 0.8).
    spacing : float
        Grid spacing in direction cosine units (default: 0.07 ≈ 4°).
    exclude_below_horizon : bool
        If True, exclude beams where l² + m² > 1 (default: True).

    Returns
    -------
    List[StationaryPointing]
        List of beam pointings covering the specified region.

    Notes
    -----
    Direction cosine spacing of 0.07 corresponds to ~4° at small angles.
    For reference: sin(4°) ≈ 0.07
    """
    pointings = []
    beam_idx = 0

    l_values = np.arange(l_min, l_max + spacing/2, spacing)
    m_values = np.arange(m_min, m_max + spacing/2, spacing)

    for l in l_values:
        for m in m_values:
            # Check if above horizon
            if exclude_below_horizon and (l**2 + m**2 >= 1.0):
                continue

            try:
                pointing = StationaryPointing(
                    l=l, m=m,
                    name=f"beam_{beam_idx:03d}"
                )
                pointings.append(pointing)
                beam_idx += 1
            except ValueError:
                # Skip invalid direction cosines
                continue

    return pointings


@dataclass
class TrackingBeamWeights:
    """
    Container for tracking beam weights (time-dependent).

    Shape: (n_beams, n_times, n_ant, n_chan)
    """
    weights: np.ndarray
    mode: BeamMode
    beam_type: BeamType
    phase_centers: List[PhaseCenter]
    unix_times: np.ndarray
    frequencies_hz: np.ndarray
    antenna_indices: np.ndarray
    array_config: ArrayConfig
    freq_config: FrequencyConfig

    @property
    def n_beams(self) -> int:
        return len(self.phase_centers)

    @property
    def n_times(self) -> int:
        return len(self.unix_times)

    @property
    def n_antennas(self) -> int:
        return len(self.antenna_indices)

    @property
    def n_channels(self) -> int:
        return len(self.frequencies_hz)

    @property
    def shape(self) -> tuple:
        return self.weights.shape

    @property
    def is_stationary(self) -> bool:
        return False

    def __repr__(self) -> str:
        return (f"TrackingBeamWeights(mode={self.mode.value}, "
                f"shape={self.shape}, "
                f"n_beams={self.n_beams}, "
                f"n_times={self.n_times}, "
                f"n_ant={self.n_antennas}, "
                f"n_chan={self.n_channels})")


@dataclass
class StationaryBeamWeights:
    """
    Container for stationary beam weights (time-independent).

    Shape: (n_beams, n_ant, n_chan) - no time axis
    """
    weights: np.ndarray
    mode: BeamMode
    beam_type: BeamType
    pointings: List[StationaryPointing]
    frequencies_hz: np.ndarray
    antenna_indices: np.ndarray
    array_config: ArrayConfig
    freq_config: FrequencyConfig

    @property
    def n_beams(self) -> int:
        return len(self.pointings)

    @property
    def n_antennas(self) -> int:
        return len(self.antenna_indices)

    @property
    def n_channels(self) -> int:
        return len(self.frequencies_hz)

    @property
    def shape(self) -> tuple:
        return self.weights.shape

    @property
    def is_stationary(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (f"StationaryBeamWeights(mode={self.mode.value}, "
                f"shape={self.shape}, "
                f"n_beams={self.n_beams}, "
                f"n_ant={self.n_antennas}, "
                f"n_chan={self.n_channels})")


# Backward compatibility alias
BeamformerWeights = TrackingBeamWeights


class GeometricBeamformer:
    """
    Geometric beamformer weight calculator for CASM.

    This class computes complex beamformer weights that phase-align signals
    from multiple antennas toward specified sky directions.

    Supports both:
    - Tracking beams: Follow celestial sources (RA/Dec), weights vary with time
    - Stationary beams: Fixed in local frame (Alt/Az), weights are time-independent

    Parameters
    ----------
    array_config : ArrayConfig, optional
        Array configuration. Uses default CASM configuration if not specified.
    freq_config : FrequencyConfig, optional
        Frequency configuration. Uses default CASM configuration if not specified.
    use_astropy : bool, optional
        Use astropy for coordinate transformations (more accurate but slower).
        Default is True if astropy is available.

    Examples
    --------
    >>> from bf_weights_generator import GeometricBeamformer, PhaseCenter, StationaryPointing
    >>> import numpy as np
    >>>
    >>> bf = GeometricBeamformer()
    >>>
    >>> # Tracking beam (follows Cygnus A as Earth rotates)
    >>> cyg_a = PhaseCenter(ra_deg=299.868, dec_deg=40.734, name="CygA")
    >>> tracking_weights = bf.compute_tracking_weights(
    ...     phase_centers=[cyg_a],
    ...     start_time=1700000000.0,
    ...     duration_sec=3600,
    ...     cadence_sec=10.0
    ... )
    >>>
    >>> # Stationary beam grid (fixed in local sky, for FRB transit search)
    >>> from bf_weights_generator import generate_beam_grid_altaz
    >>> pointings = generate_beam_grid_altaz(spacing_deg=4.0)
    >>> stationary_weights = bf.compute_stationary_weights(pointings)
    """

    def __init__(self,
                 array_config: Optional[ArrayConfig] = None,
                 freq_config: Optional[FrequencyConfig] = None,
                 use_astropy: bool = True):

        self.array_config = array_config or ArrayConfig()
        self.freq_config = freq_config or FrequencyConfig()
        self.use_astropy = use_astropy and ASTROPY_AVAILABLE

        if use_astropy and not ASTROPY_AVAILABLE:
            warnings.warn(
                "astropy not available, falling back to simplified coordinate transforms. "
                "Install astropy for higher accuracy: pip install astropy"
            )

    def _compute_direction_cosines_tracking(self, phase_center: PhaseCenter,
                                             unix_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute time-varying direction cosines for a tracking beam."""
        if self.use_astropy:
            return radec_to_direction_cosines_astropy(
                ra_deg=phase_center.ra_deg,
                dec_deg=phase_center.dec_deg,
                unix_times=unix_times,
                lat_deg=self.array_config.lat_deg,
                lon_deg=self.array_config.lon_deg,
                alt_m=self.array_config.alt_m
            )
        else:
            return radec_to_direction_cosines(
                ra_deg=phase_center.ra_deg,
                dec_deg=phase_center.dec_deg,
                unix_times=unix_times,
                lat_deg=self.array_config.lat_deg,
                lon_deg=self.array_config.lon_deg
            )

    # =========================================================================
    # Stationary Beams (Alt/Az, time-independent)
    # =========================================================================

    def compute_stationary_delays(self, pointing: StationaryPointing) -> np.ndarray:
        """
        Compute geometric delays for a stationary beam pointing.

        Parameters
        ----------
        pointing : StationaryPointing
            Fixed beam direction in local frame.

        Returns
        -------
        delays : np.ndarray
            Geometric delays in seconds, shape (n_active_ant,).
            Time-independent since pointing is fixed.
        """
        l, m, n = pointing.direction_cosines
        positions = self.array_config.active_positions

        # Compute delays: τ = -(x*l + y*m + z*n) / c
        # positions: (n_ant, 3) -> x, y, z = E, N, U
        x = positions[:, 0]  # East
        y = positions[:, 1]  # North
        z = positions[:, 2]  # Up

        path_length = x * l + y * m + z * n
        delays = -path_length / SPEED_OF_LIGHT_M_S

        return delays

    def compute_stationary_phases(self, pointing: StationaryPointing) -> np.ndarray:
        """
        Compute geometric phases for a stationary beam at all frequencies.

        Parameters
        ----------
        pointing : StationaryPointing
            Fixed beam direction in local frame.

        Returns
        -------
        phases : np.ndarray
            Geometric phases in radians, shape (n_active_ant, n_chan).
            Time-independent.
        """
        delays = self.compute_stationary_delays(pointing)  # (n_ant,)
        freqs = self.freq_config.get_frequencies_hz()  # (n_chan,)

        # phases = 2π * f * τ
        # delays: (n_ant,), freqs: (n_chan,)
        # Result: (n_ant, n_chan)
        phases = 2.0 * np.pi * delays[:, np.newaxis] * freqs[np.newaxis, :]

        return phases

    def compute_stationary_coherent_weights(self,
                                             pointings: List[StationaryPointing]) -> np.ndarray:
        """
        Compute coherent weights for stationary beams.

        Parameters
        ----------
        pointings : List[StationaryPointing]
            List of fixed beam directions.

        Returns
        -------
        weights : np.ndarray
            Complex weights, shape (n_beams, n_ant, n_chan).
            Time-independent.
        """
        n_beams = len(pointings)
        n_ant = self.array_config.n_active_antennas
        n_chan = self.freq_config.n_chan

        weights = np.zeros((n_beams, n_ant, n_chan), dtype=np.complex64)

        for i, pointing in enumerate(pointings):
            phases = self.compute_stationary_phases(pointing)  # (n_ant, n_chan)
            weights[i] = np.exp(-1j * phases).astype(np.complex64)

        return weights

    def compute_stationary_incoherent_weights(self, n_beams: int) -> np.ndarray:
        """
        Compute incoherent weights for stationary beams.

        Parameters
        ----------
        n_beams : int
            Number of identical incoherent beams.

        Returns
        -------
        weights : np.ndarray
            Unity weights, shape (n_beams, n_ant, n_chan).
        """
        n_ant = self.array_config.n_active_antennas
        n_chan = self.freq_config.n_chan

        return np.ones((n_beams, n_ant, n_chan), dtype=np.complex64)

    def compute_stationary_weights(
        self,
        pointings: Optional[List[StationaryPointing]] = None,
        mode: Union[BeamMode, str] = BeamMode.COHERENT,
        n_beams: int = 1,
        # Beam grid parameters (used if pointings is None)
        grid_type: str = "altaz",
        alt_min_deg: float = 30.0,
        alt_max_deg: float = 90.0,
        spacing_deg: float = 4.0,
    ) -> StationaryBeamWeights:
        """
        Compute stationary (time-independent) beamformer weights.

        Stationary beams are fixed in the local horizon frame (Alt/Az).
        The sky drifts through them as Earth rotates. Weights are computed
        once and applied continuously.

        Parameters
        ----------
        pointings : List[StationaryPointing], optional
            Explicit list of beam pointings. If None, generates a beam grid.
        mode : BeamMode or str
            'coherent' for phase-steered beams, 'incoherent' for total power.
        n_beams : int
            Number of beams for incoherent mode (ignored for coherent).
        grid_type : str
            Type of beam grid to generate: 'altaz' or 'lm' (default: 'altaz').
        alt_min_deg : float
            Minimum altitude for beam grid (default: 30°).
        alt_max_deg : float
            Maximum altitude for beam grid (default: 90°).
        spacing_deg : float
            Beam spacing in degrees (default: 4°).

        Returns
        -------
        StationaryBeamWeights
            Container with time-independent weights, shape (n_beams, n_ant, n_chan).
        """
        if isinstance(mode, str):
            mode = BeamMode(mode.lower())

        # Generate beam grid if not provided
        if pointings is None and mode == BeamMode.COHERENT:
            if grid_type == "altaz":
                pointings = generate_beam_grid_altaz(
                    alt_min_deg=alt_min_deg,
                    alt_max_deg=alt_max_deg,
                    spacing_deg=spacing_deg
                )
            else:  # lm
                spacing_lm = np.sin(np.deg2rad(spacing_deg))
                pointings = generate_beam_grid_lm(spacing=spacing_lm)

        frequencies_hz = self.freq_config.get_frequencies_hz()
        antenna_indices = self.array_config.active_indices

        if mode == BeamMode.COHERENT:
            weights = self.compute_stationary_coherent_weights(pointings)
        else:  # INCOHERENT
            weights = self.compute_stationary_incoherent_weights(n_beams)
            # Create dummy pointings for incoherent beams
            pointings = [
                StationaryPointing(alt_deg=90.0, az_deg=0.0, name=f"incoherent_{i}")
                for i in range(n_beams)
            ]

        return StationaryBeamWeights(
            weights=weights,
            mode=mode,
            beam_type=BeamType.STATIONARY,
            pointings=pointings,
            frequencies_hz=frequencies_hz,
            antenna_indices=antenna_indices,
            array_config=self.array_config,
            freq_config=self.freq_config
        )

    # =========================================================================
    # Tracking Beams (RA/Dec, time-dependent)
    # =========================================================================

    def compute_tracking_delays(self, phase_center: PhaseCenter,
                                 unix_times: np.ndarray) -> np.ndarray:
        """
        Compute geometric delays for a tracking beam.

        Parameters
        ----------
        phase_center : PhaseCenter
            Celestial position to track.
        unix_times : np.ndarray
            Unix timestamps, shape (n_times,).

        Returns
        -------
        delays : np.ndarray
            Geometric delays in seconds, shape (n_times, n_active_ant).
        """
        l, m, n = self._compute_direction_cosines_tracking(phase_center, unix_times)
        positions = self.array_config.active_positions
        delays = compute_geometric_delays(positions, l, m, n, SPEED_OF_LIGHT_M_S)
        return delays

    def compute_tracking_phases(self, phase_center: PhaseCenter,
                                 unix_times: np.ndarray) -> np.ndarray:
        """
        Compute geometric phases for a tracking beam at all frequencies.

        Parameters
        ----------
        phase_center : PhaseCenter
            Celestial position to track.
        unix_times : np.ndarray
            Unix timestamps, shape (n_times,).

        Returns
        -------
        phases : np.ndarray
            Geometric phases in radians, shape (n_times, n_active_ant, n_chan).
        """
        delays = self.compute_tracking_delays(phase_center, unix_times)
        freqs = self.freq_config.get_frequencies_hz()

        # phases = 2π * f * τ
        phases = 2.0 * np.pi * delays[:, :, np.newaxis] * freqs[np.newaxis, np.newaxis, :]

        return phases

    def compute_tracking_coherent_weights(self, phase_centers: List[PhaseCenter],
                                           unix_times: np.ndarray) -> np.ndarray:
        """
        Compute coherent weights for tracking beams.

        Parameters
        ----------
        phase_centers : List[PhaseCenter]
            List of celestial positions to track.
        unix_times : np.ndarray
            Unix timestamps, shape (n_times,).

        Returns
        -------
        weights : np.ndarray
            Complex weights, shape (n_beams, n_times, n_ant, n_chan).
        """
        n_beams = len(phase_centers)
        n_times = len(unix_times)
        n_ant = self.array_config.n_active_antennas
        n_chan = self.freq_config.n_chan

        weights = np.zeros((n_beams, n_times, n_ant, n_chan), dtype=np.complex64)

        for i, pc in enumerate(phase_centers):
            phases = self.compute_tracking_phases(pc, unix_times)
            weights[i] = np.exp(-1j * phases).astype(np.complex64)

        return weights

    def compute_tracking_incoherent_weights(self, n_beams: int,
                                             unix_times: np.ndarray) -> np.ndarray:
        """
        Compute incoherent weights for tracking beams.

        Parameters
        ----------
        n_beams : int
            Number of identical incoherent beams.
        unix_times : np.ndarray
            Unix timestamps, shape (n_times,).

        Returns
        -------
        weights : np.ndarray
            Unity weights, shape (n_beams, n_times, n_ant, n_chan).
        """
        n_times = len(unix_times)
        n_ant = self.array_config.n_active_antennas
        n_chan = self.freq_config.n_chan

        return np.ones((n_beams, n_times, n_ant, n_chan), dtype=np.complex64)

    def compute_tracking_weights(
        self,
        phase_centers: Union[List[PhaseCenter], PhaseCenter, None] = None,
        unix_times: Optional[np.ndarray] = None,
        start_time: Optional[float] = None,
        duration_sec: Optional[float] = None,
        cadence_sec: float = 1.0,
        mode: Union[BeamMode, str] = BeamMode.COHERENT,
        n_beams: int = 1
    ) -> TrackingBeamWeights:
        """
        Compute tracking (time-dependent) beamformer weights.

        Tracking beams follow celestial sources (RA/Dec) as Earth rotates.
        Weights must be recomputed at each time step.

        Parameters
        ----------
        phase_centers : List[PhaseCenter], PhaseCenter, or None
            Celestial positions for coherent beams. Required for coherent mode.
        unix_times : np.ndarray, optional
            Explicit Unix timestamps. If None, generated from start_time/duration.
        start_time : float, optional
            Start time as Unix timestamp.
        duration_sec : float, optional
            Observation duration in seconds.
        cadence_sec : float
            Time cadence in seconds (default: 1.0).
        mode : BeamMode or str
            'coherent' for phase-steered beams, 'incoherent' for total power.
        n_beams : int
            Number of beams for incoherent mode (ignored for coherent).

        Returns
        -------
        TrackingBeamWeights
            Container with time-dependent weights, shape (n_beams, n_times, n_ant, n_chan).
        """
        if isinstance(mode, str):
            mode = BeamMode(mode.lower())

        # Generate time array
        if unix_times is None:
            if start_time is None or duration_sec is None:
                raise ValueError("Must provide unix_times or both start_time and duration_sec")
            n_times = int(np.ceil(duration_sec / cadence_sec))
            unix_times = start_time + np.arange(n_times) * cadence_sec
        else:
            unix_times = np.atleast_1d(unix_times).astype(np.float64)

        frequencies_hz = self.freq_config.get_frequencies_hz()
        antenna_indices = self.array_config.active_indices

        if mode == BeamMode.COHERENT:
            if phase_centers is None:
                raise ValueError("phase_centers required for coherent mode")
            if isinstance(phase_centers, PhaseCenter):
                phase_centers = [phase_centers]
            weights = self.compute_tracking_coherent_weights(phase_centers, unix_times)
        else:  # INCOHERENT
            weights = self.compute_tracking_incoherent_weights(n_beams, unix_times)
            phase_centers = [
                PhaseCenter(ra_deg=0.0, dec_deg=0.0, name=f"incoherent_{i}")
                for i in range(n_beams)
            ]

        return TrackingBeamWeights(
            weights=weights,
            mode=mode,
            beam_type=BeamType.TRACKING,
            phase_centers=phase_centers,
            unix_times=unix_times,
            frequencies_hz=frequencies_hz,
            antenna_indices=antenna_indices,
            array_config=self.array_config,
            freq_config=self.freq_config
        )

    # =========================================================================
    # Backward-compatible interface
    # =========================================================================

    def compute_delays(self, phase_center: PhaseCenter,
                       unix_times: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for compute_tracking_delays."""
        return self.compute_tracking_delays(phase_center, unix_times)

    def compute_phases(self, phase_center: PhaseCenter,
                       unix_times: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for compute_tracking_phases."""
        return self.compute_tracking_phases(phase_center, unix_times)

    def compute_coherent_weights(self, phase_centers: List[PhaseCenter],
                                  unix_times: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for compute_tracking_coherent_weights."""
        return self.compute_tracking_coherent_weights(phase_centers, unix_times)

    def compute_incoherent_weights(self, n_beams: int,
                                    unix_times: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for compute_tracking_incoherent_weights."""
        return self.compute_tracking_incoherent_weights(n_beams, unix_times)

    def compute_weights(self, *args, **kwargs) -> TrackingBeamWeights:
        """Backward-compatible alias for compute_tracking_weights."""
        return self.compute_tracking_weights(*args, **kwargs)

    def __repr__(self) -> str:
        return (f"GeometricBeamformer(array={self.array_config}, "
                f"freq={self.freq_config}, "
                f"use_astropy={self.use_astropy})")
