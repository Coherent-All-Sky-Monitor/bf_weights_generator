"""
Configuration constants and array parameters for CASM beamformer.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

# =============================================================================
# OVRO Observatory Location
# =============================================================================
OVRO_LAT_DEG = 37.2339  # degrees North
OVRO_LON_DEG = -118.2820  # degrees East (negative for West)
OVRO_ALT_M = 1222.0  # meters above sea level

# =============================================================================
# Speed of light
# =============================================================================
SPEED_OF_LIGHT_M_S = 299792458.0  # m/s

# =============================================================================
# Default CASM Antenna Positions (ENU coordinates in meters)
# =============================================================================
DEFAULT_ANTENNA_POSITIONS = np.array([
    [0.0, 0.0, 0.0],        # Antenna 0
    [0.38, 0.0, 0.0],       # Antenna 1
    [0.825, 0.0, 0.0],      # Antenna 2
    [1.205, 0.0, 0.0],      # Antenna 3
    [1.65, 0.0, 0.0],       # Antenna 4
    [2.03, 0.0, 0.0],       # Antenna 5
    [0.0, -10.5, 0.0],      # Antenna 6
    [0.38, -10.5, 0.0],     # Antenna 7
    [0.825, -10.5, 0.0],    # Antenna 8
    [1.205, -10.5, 0.0],    # Antenna 9
    [1.65, -10.5, 0.0],     # Antenna 10
    [2.03, -10.5, 0.0],     # Antenna 11
    [-4.553, -5.5, -0.279], # Antenna 12
], dtype=np.float64)

DEFAULT_N_ANTENNAS = 13


@dataclass
class FrequencyConfig:
    """
    Configuration for CASM frequency channels.

    The default configuration uses 3072 channels from a 4096-channel system
    with 125 MHz total bandwidth. Frequencies are computed from the high end
    of the voltage band going downward.

    Attributes
    ----------
    n_chan : int
        Number of frequency channels to use (default: 3072)
    total_bw_mhz : float
        Total system bandwidth in MHz (default: 125.0)
    total_n_chan : int
        Total number of channels in the system (default: 4096)
    freq_end_voltage_mhz : float
        Upper frequency edge of the voltage data in MHz (default: 468.75)
    """
    n_chan: int = 3072
    total_bw_mhz: float = 125.0
    total_n_chan: int = 4096
    freq_end_voltage_mhz: float = 468.75

    @property
    def chan_bw_mhz(self) -> float:
        """Channel bandwidth in MHz."""
        return self.total_bw_mhz / self.total_n_chan

    @property
    def chan_bw_hz(self) -> float:
        """Channel bandwidth in Hz."""
        return self.chan_bw_mhz * 1e6

    def get_frequencies_mhz(self) -> np.ndarray:
        """
        Compute center frequencies for all channels.

        Returns
        -------
        np.ndarray
            Array of shape (n_chan,) containing center frequencies in MHz.
            Frequencies are ordered from high to low.
        """
        return self.freq_end_voltage_mhz - self.chan_bw_mhz * (np.arange(self.n_chan) + 0.5)

    def get_frequencies_hz(self) -> np.ndarray:
        """
        Compute center frequencies for all channels in Hz.

        Returns
        -------
        np.ndarray
            Array of shape (n_chan,) containing center frequencies in Hz.
        """
        return self.get_frequencies_mhz() * 1e6

    def __repr__(self) -> str:
        freqs = self.get_frequencies_mhz()
        return (f"FrequencyConfig(n_chan={self.n_chan}, "
                f"freq_range=[{freqs[-1]:.3f}, {freqs[0]:.3f}] MHz, "
                f"chan_bw={self.chan_bw_mhz*1e3:.3f} kHz)")


@dataclass
class ArrayConfig:
    """
    Configuration for the CASM antenna array.

    Attributes
    ----------
    positions_enu : np.ndarray
        Antenna positions in ENU (East-North-Up) coordinates, shape (n_ant, 3).
        Units are meters.
    antenna_flags : np.ndarray
        Boolean array indicating which antennas are active (True = active).
    lat_deg : float
        Observatory latitude in degrees.
    lon_deg : float
        Observatory longitude in degrees (negative for West).
    alt_m : float
        Observatory altitude in meters above sea level.
    """
    positions_enu: np.ndarray = field(default_factory=lambda: DEFAULT_ANTENNA_POSITIONS.copy())
    antenna_flags: Optional[np.ndarray] = None
    lat_deg: float = OVRO_LAT_DEG
    lon_deg: float = OVRO_LON_DEG
    alt_m: float = OVRO_ALT_M

    def __post_init__(self):
        """Initialize antenna flags if not provided."""
        if self.antenna_flags is None:
            self.antenna_flags = np.ones(len(self.positions_enu), dtype=bool)
        self.positions_enu = np.asarray(self.positions_enu, dtype=np.float64)
        self.antenna_flags = np.asarray(self.antenna_flags, dtype=bool)

    @property
    def n_antennas(self) -> int:
        """Total number of antennas."""
        return len(self.positions_enu)

    @property
    def n_active_antennas(self) -> int:
        """Number of active (unflagged) antennas."""
        return np.sum(self.antenna_flags)

    @property
    def active_positions(self) -> np.ndarray:
        """Positions of active antennas only."""
        return self.positions_enu[self.antenna_flags]

    @property
    def active_indices(self) -> np.ndarray:
        """Indices of active antennas."""
        return np.where(self.antenna_flags)[0]

    def flag_antennas(self, indices: List[int]) -> None:
        """
        Flag (disable) specific antennas.

        Parameters
        ----------
        indices : List[int]
            List of antenna indices to flag.
        """
        for idx in indices:
            if 0 <= idx < self.n_antennas:
                self.antenna_flags[idx] = False

    def unflag_antennas(self, indices: Optional[List[int]] = None) -> None:
        """
        Unflag (enable) specific antennas or all antennas.

        Parameters
        ----------
        indices : List[int], optional
            List of antenna indices to unflag. If None, unflag all.
        """
        if indices is None:
            self.antenna_flags[:] = True
        else:
            for idx in indices:
                if 0 <= idx < self.n_antennas:
                    self.antenna_flags[idx] = True

    @property
    def lat_rad(self) -> float:
        """Observatory latitude in radians."""
        return np.deg2rad(self.lat_deg)

    @property
    def lon_rad(self) -> float:
        """Observatory longitude in radians."""
        return np.deg2rad(self.lon_deg)

    def __repr__(self) -> str:
        return (f"ArrayConfig(n_antennas={self.n_antennas}, "
                f"n_active={self.n_active_antennas}, "
                f"location=({self.lat_deg:.4f}°N, {abs(self.lon_deg):.4f}°W))")
