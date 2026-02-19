"""
SNAP beamformer weight generation for 64-antenna arrays.

This module generates int8-quantized stationary beamformer weights for SNAP
hardware. It reads antenna layouts from CSV files and outputs weights in
SNAP input order.

Key classes:
- Array64Config: 64-slot antenna array loaded from CSV
- SnapWeightsGenerator: Computes int8 weights for SNAP beamformer
- Int8StationaryWeights: Container for quantized weights
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import csv

from .config import ArrayConfig, FrequencyConfig
from .weights import (
    GeometricBeamformer,
    StationaryPointing,
    StationaryBeamWeights,
    BeamMode,
)


# =============================================================================
# Transit Survey Beam Pattern (8 beams)
# =============================================================================
TRANSIT_SURVEY_BEAMS = [
    StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith"),
    StationaryPointing(alt_deg=70.0, az_deg=0.0, name="north_high"),
    StationaryPointing(alt_deg=70.0, az_deg=90.0, name="east_high"),
    StationaryPointing(alt_deg=70.0, az_deg=180.0, name="south_high"),
    StationaryPointing(alt_deg=70.0, az_deg=270.0, name="west_high"),
    StationaryPointing(alt_deg=50.0, az_deg=45.0, name="ne_mid"),
    StationaryPointing(alt_deg=50.0, az_deg=135.0, name="se_mid"),
    StationaryPointing(alt_deg=50.0, az_deg=225.0, name="sw_mid"),
]


@dataclass
class Array64Config:
    """
    64-slot antenna array configuration loaded from CSV.

    The SNAP beamformer expects weights for 64 antenna inputs regardless of
    how many antennas are actually installed. This class manages the mapping
    between physical antenna positions and SNAP input indices.

    Attributes
    ----------
    positions_enu : np.ndarray
        (64, 3) array of ENU positions. Inactive slots have [0, 0, 0].
    active_mask : np.ndarray
        (64,) boolean array. True for slots with installed antennas.
    snap_to_ant64 : np.ndarray
        (64,) array mapping SNAP input index to ant64 slot.
    ant64_to_snap : np.ndarray
        (64,) array mapping ant64 slot to SNAP input index. -1 if inactive.
    pos_ids : List[str]
        Position IDs from CSV for each of the 64 slots ("" if inactive).
    csv_path : str
        Path to the CSV file this was loaded from.
    """
    positions_enu: np.ndarray
    active_mask: np.ndarray
    snap_to_ant64: np.ndarray
    ant64_to_snap: np.ndarray
    pos_ids: List[str] = field(default_factory=list)
    csv_path: str = ""

    @classmethod
    def from_csv(cls, csv_path: str) -> 'Array64Config':
        """
        Parse antenna layout CSV and build 64-slot array configuration.

        CSV must contain columns:
        - ant64: slot index (0-63)
        - x_east_m, y_north_m, z_up_m: ENU coordinates
        - snap_A, adc_A: SNAP board and ADC channel for Pol A
        - include_in_beamforming: boolean filter
        - pos_type: must be 'antenna' to be included

        Parameters
        ----------
        csv_path : str
            Path to the antenna layout CSV file.

        Returns
        -------
        Array64Config
            Configured 64-slot array.
        """
        csv_path = str(csv_path)

        # Initialize 64-slot arrays
        positions_enu = np.zeros((64, 3), dtype=np.float64)
        active_mask = np.zeros(64, dtype=bool)
        snap_to_ant64 = np.full(64, -1, dtype=np.int32)
        ant64_to_snap = np.full(64, -1, dtype=np.int32)
        pos_ids = [""] * 64

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Filter: only include antennas marked for beamforming
                if row.get('pos_type', '').strip().lower() != 'antenna':
                    continue
                if row.get('include_in_beamforming', '').strip().lower() != 'true':
                    continue

                # Get ant64 slot index
                ant64_str = row.get('ant64', '').strip()
                if not ant64_str:
                    continue
                ant64_idx = int(float(ant64_str))
                if ant64_idx < 0 or ant64_idx >= 64:
                    continue

                # Get position
                x = float(row.get('x_east_m', 0))
                y = float(row.get('y_north_m', 0))
                z = float(row.get('z_up_m', 0))

                positions_enu[ant64_idx] = [x, y, z]
                active_mask[ant64_idx] = True
                pos_ids[ant64_idx] = row.get('pos_id', '')

                # Get SNAP mapping (Pol A)
                snap_a_str = row.get('snap_A', '').strip()
                adc_a_str = row.get('adc_A', '').strip()
                if snap_a_str and adc_a_str:
                    snap_board = int(float(snap_a_str))
                    adc_channel = int(float(adc_a_str))
                    snap_input_idx = snap_board * 12 + adc_channel

                    if 0 <= snap_input_idx < 64:
                        snap_to_ant64[snap_input_idx] = ant64_idx
                        ant64_to_snap[ant64_idx] = snap_input_idx

        return cls(
            positions_enu=positions_enu,
            active_mask=active_mask,
            snap_to_ant64=snap_to_ant64,
            ant64_to_snap=ant64_to_snap,
            pos_ids=pos_ids,
            csv_path=csv_path,
        )

    @property
    def n_active(self) -> int:
        """Number of active (installed) antennas."""
        return int(np.sum(self.active_mask))

    @property
    def active_indices(self) -> np.ndarray:
        """Indices of active antenna slots (0-63)."""
        return np.where(self.active_mask)[0]

    @property
    def active_positions(self) -> np.ndarray:
        """Positions of active antennas only, shape (n_active, 3)."""
        return self.positions_enu[self.active_mask]

    def to_array_config(self) -> ArrayConfig:
        """
        Create an ArrayConfig with only active antennas.

        This is used to interface with GeometricBeamformer which expects
        only the active antenna positions.

        Returns
        -------
        ArrayConfig
            Configuration with active antenna positions only.
        """
        return ArrayConfig(
            positions_enu=self.active_positions,
            antenna_flags=np.ones(self.n_active, dtype=bool),
        )

    def __repr__(self) -> str:
        return (f"Array64Config(n_active={self.n_active}/64, "
                f"csv='{Path(self.csv_path).name}')")


@dataclass
class Int8StationaryWeights:
    """
    Container for int8-quantized stationary beamformer weights.

    The weights are stored in the format expected by SNAP hardware:
    - Shape: (2, n_chan, 2, n_beams, 64) = (real/imag, chan, pol, beam, ant)
    - Antenna dimension is in SNAP input order
    - Channels are in reversed order (low-to-high frequency)

    Attributes
    ----------
    weights_int8 : np.ndarray
        Quantized weights, shape (2, n_chan, 2, n_beams, 64).
        First axis: 0=real, 1=imag.
    pointings : List[StationaryPointing]
        Beam pointing directions.
    frequencies_hz : np.ndarray
        Channel frequencies in Hz (in SNAP order, i.e., reversed).
    array_config : Array64Config
        Array configuration used to generate weights.
    freq_config : FrequencyConfig
        Frequency configuration.
    scale_factor : float
        Scale factor used for quantization (default: 127.0).
    """
    weights_int8: np.ndarray
    pointings: List[StationaryPointing]
    frequencies_hz: np.ndarray
    array_config: Array64Config
    freq_config: FrequencyConfig
    scale_factor: float = 127.0

    @property
    def n_beams(self) -> int:
        """Number of beams."""
        return len(self.pointings)

    @property
    def n_channels(self) -> int:
        """Number of frequency channels."""
        return len(self.frequencies_hz)

    @property
    def shape(self) -> tuple:
        """Shape of weights array."""
        return self.weights_int8.shape

    def to_complex64(self) -> np.ndarray:
        """
        Reconstruct complex64 weights from int8 representation.

        Returns
        -------
        np.ndarray
            Complex weights, shape (n_beams, 64, n_chan).
            Note: This is in SNAP order, channels reversed from original.
        """
        real = self.weights_int8[0].astype(np.float32) / self.scale_factor
        imag = self.weights_int8[1].astype(np.float32) / self.scale_factor
        # Shape: (n_chan, 2, n_beams, 64) -> transpose to (n_beams, 64, n_chan)
        # Take pol 0 since both pols are identical
        complex_weights = real[:, 0, :, :] + 1j * imag[:, 0, :, :]
        # Transpose from (n_chan, n_beams, 64) to (n_beams, 64, n_chan)
        return complex_weights.transpose(1, 2, 0)

    def __repr__(self) -> str:
        return (f"Int8StationaryWeights(shape={self.shape}, "
                f"n_beams={self.n_beams}, n_chan={self.n_channels})")


class SnapWeightsGenerator:
    """
    Generate int8-quantized beamformer weights for SNAP hardware.

    This class wraps GeometricBeamformer to produce weights in the format
    expected by the SNAP beamformer:
    - 64 antenna inputs (inactive = 0)
    - Antenna ordering matches SNAP input order
    - Channels reversed (low-to-high frequency)
    - Quantized to int8

    Parameters
    ----------
    array_config : Array64Config
        64-slot array configuration loaded from CSV.
    freq_config : FrequencyConfig, optional
        Frequency configuration. Uses default CASM configuration if not specified.

    Examples
    --------
    >>> from bf_weights_generator.snap_weights import Array64Config, SnapWeightsGenerator
    >>> array = Array64Config.from_csv("casm_antenna_layout1.csv")
    >>> gen = SnapWeightsGenerator(array)
    >>> weights = gen.compute_int8_weights()  # Uses default transit survey beams
    >>> weights.shape
    (2, 3072, 2, 8, 64)
    """

    def __init__(
        self,
        array_config: Array64Config,
        freq_config: Optional[FrequencyConfig] = None,
    ):
        self.array_config = array_config
        self.freq_config = freq_config or FrequencyConfig()

        # Create beamformer with only active antennas
        self._beamformer = GeometricBeamformer(
            array_config=array_config.to_array_config(),
            freq_config=self.freq_config,
        )

    def compute_int8_weights(
        self,
        pointings: Optional[List[StationaryPointing]] = None,
        scale_factor: float = 127.0,
    ) -> Int8StationaryWeights:
        """
        Compute int8-quantized stationary beamformer weights.

        Processing steps:
        1. Compute complex64 weights for active antennas
        2. Expand to 64 slots (inactive = 0)
        3. Reorder antennas to SNAP input order
        4. Reverse channel order
        5. Quantize to int8
        6. Add polarization dimension (Pol A = Pol B)

        Parameters
        ----------
        pointings : List[StationaryPointing], optional
            Beam pointing directions. If None, uses TRANSIT_SURVEY_BEAMS.
        scale_factor : float
            Scale factor for quantization. Default is 127.0.

        Returns
        -------
        Int8StationaryWeights
            Quantized weights ready for SNAP hardware.
        """
        if pointings is None:
            pointings = TRANSIT_SURVEY_BEAMS

        # Step 1: Compute complex64 weights for active antennas
        stationary = self._beamformer.compute_stationary_weights(
            pointings=pointings,
            mode=BeamMode.COHERENT,
        )
        # Shape: (n_beams, n_active_ant, n_chan)
        active_weights = stationary.weights

        n_beams = len(pointings)
        n_chan = self.freq_config.n_chan

        # Step 2: Expand to 64 slots (inactive = 0)
        weights_64 = np.zeros((n_beams, 64, n_chan), dtype=np.complex64)
        active_indices = self.array_config.active_indices
        for i, ant64_idx in enumerate(active_indices):
            weights_64[:, ant64_idx, :] = active_weights[:, i, :]

        # Step 3: Reorder antennas to SNAP input order
        weights_snap_order = np.zeros((n_beams, 64, n_chan), dtype=np.complex64)
        for snap_idx in range(64):
            ant64_idx = self.array_config.snap_to_ant64[snap_idx]
            if ant64_idx >= 0:
                weights_snap_order[:, snap_idx, :] = weights_64[:, ant64_idx, :]
            # else: stays zero (inactive SNAP input)

        # Step 4: Reverse channel order
        weights_reversed = weights_snap_order[:, :, ::-1]

        # Step 5: Quantize to int8
        real_scaled = np.round(weights_reversed.real * scale_factor)
        imag_scaled = np.round(weights_reversed.imag * scale_factor)
        real_int8 = np.clip(real_scaled, -128, 127).astype(np.int8)
        imag_int8 = np.clip(imag_scaled, -128, 127).astype(np.int8)

        # Step 6: Reshape to (2, n_chan, n_beams, 64) and add pol dimension
        # Current shape: (n_beams, 64, n_chan)
        # Target shape: (2, n_chan, 2, n_beams, 64)
        real_reshaped = real_int8.transpose(2, 0, 1)  # (n_chan, n_beams, 64)
        imag_reshaped = imag_int8.transpose(2, 0, 1)  # (n_chan, n_beams, 64)

        # Add polarization dimension (duplicate for both pols)
        # Shape: (n_chan, 2, n_beams, 64)
        real_with_pol = np.stack([real_reshaped, real_reshaped], axis=1)
        imag_with_pol = np.stack([imag_reshaped, imag_reshaped], axis=1)

        # Stack real and imag as first dimension
        # Shape: (2, n_chan, 2, n_beams, 64)
        weights_int8 = np.stack([real_with_pol, imag_with_pol], axis=0)

        # Compute reversed frequencies
        frequencies_hz = self.freq_config.get_frequencies_hz()[::-1]

        return Int8StationaryWeights(
            weights_int8=weights_int8,
            pointings=pointings,
            frequencies_hz=frequencies_hz,
            array_config=self.array_config,
            freq_config=self.freq_config,
            scale_factor=scale_factor,
        )

    def __repr__(self) -> str:
        return (f"SnapWeightsGenerator(array={self.array_config}, "
                f"freq={self.freq_config})")


def generate_beam_grid(
    n_beams: Optional[int] = None,
    spacing_deg: float = 15.0,
    alt_min_deg: float = 30.0,
    alt_max_deg: float = 90.0,
    az_min_deg: float = 0.0,
    az_max_deg: float = 360.0,
) -> List[StationaryPointing]:
    """
    Generate a grid of beam pointings covering the sky.

    The grid is generated with uniform spacing in altitude, and azimuth
    spacing adjusted at each altitude to maintain roughly uniform sky coverage
    (fewer azimuth beams near zenith where the circles are smaller).

    Parameters
    ----------
    n_beams : int, optional
        Target number of beams. If specified, spacing_deg is computed
        automatically to achieve approximately this many beams.
    spacing_deg : float
        Beam spacing in degrees (default: 15.0). Ignored if n_beams is specified.
    alt_min_deg : float
        Minimum altitude in degrees (default: 30.0).
    alt_max_deg : float
        Maximum altitude in degrees (default: 90.0).
    az_min_deg : float
        Minimum azimuth in degrees (default: 0.0).
    az_max_deg : float
        Maximum azimuth in degrees (default: 360.0).

    Returns
    -------
    List[StationaryPointing]
        List of beam pointings.

    Examples
    --------
    >>> beams = generate_beam_grid(n_beams=8)  # ~8 beams
    >>> beams = generate_beam_grid(spacing_deg=20)  # 20 deg spacing
    >>> beams = generate_beam_grid(n_beams=16, alt_min_deg=45)  # 16 beams above 45 deg
    """
    # If n_beams specified, estimate spacing to achieve target count
    if n_beams is not None:
        # Approximate: solid angle covered = 2*pi*(1 - cos(90-alt_min))
        # Each beam covers ~(spacing)^2 steradians
        # So n_beams ~ solid_angle / beam_area
        alt_min_rad = np.deg2rad(alt_min_deg)
        solid_angle = 2 * np.pi * (1 - np.sin(alt_min_rad))  # steradians above alt_min
        beam_area = (np.deg2rad(spacing_deg)) ** 2
        estimated_n = solid_angle / beam_area

        # Adjust spacing iteratively to get close to target
        if n_beams > 1:
            # Scale spacing to match target
            scale = np.sqrt(estimated_n / n_beams)
            spacing_deg = spacing_deg * scale
            # Clamp to reasonable range
            spacing_deg = max(5.0, min(60.0, spacing_deg))

    pointings = []
    beam_idx = 0

    # Generate altitude levels
    alt_values = np.arange(alt_min_deg, alt_max_deg + spacing_deg / 2, spacing_deg)

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
                alt_deg=float(alt),
                az_deg=float(az) % 360.0,
                name=f"beam_{beam_idx:03d}"
            )
            pointings.append(pointing)
            beam_idx += 1

    # If n_beams was specified and we overshot, trim to closest match
    if n_beams is not None and len(pointings) > n_beams:
        # Keep beams closest to zenith (highest altitude first)
        pointings.sort(key=lambda p: -p.alt_deg)
        pointings = pointings[:n_beams]
        # Re-sort by altitude then azimuth for consistent ordering
        pointings.sort(key=lambda p: (-p.alt_deg, p.az_deg))
        # Rename beams
        for i, p in enumerate(pointings):
            p.name = f"beam_{i:03d}"

    return pointings


def parse_beams_arg(beams_str: str) -> List[StationaryPointing]:
    """
    Parse beam specification string into list of StationaryPointing.

    Formats:
    - "transit": Use TRANSIT_SURVEY_BEAMS (8 beams)
    - "zenith": Single zenith beam
    - "alt:az,alt:az,...": Custom beams with alt:az pairs

    Parameters
    ----------
    beams_str : str
        Beam specification string.

    Returns
    -------
    List[StationaryPointing]
        List of beam pointings.

    Examples
    --------
    >>> parse_beams_arg("transit")  # 8 transit survey beams
    >>> parse_beams_arg("zenith")  # Single zenith beam
    >>> parse_beams_arg("90:0,70:0,70:90")  # Custom: zenith + 2 at 70deg
    """
    beams_str = beams_str.strip().lower()

    if beams_str == "transit":
        return list(TRANSIT_SURVEY_BEAMS)  # Return a copy

    if beams_str == "zenith":
        return [StationaryPointing(alt_deg=90.0, az_deg=0.0, name="zenith")]

    # Parse custom format: alt:az,alt:az,...
    pointings = []
    for i, pair in enumerate(beams_str.split(",")):
        pair = pair.strip()
        if ":" in pair:
            alt_str, az_str = pair.split(":")
            alt = float(alt_str)
            az = float(az_str)
            pointings.append(
                StationaryPointing(alt_deg=alt, az_deg=az, name=f"beam_{i}")
            )
        else:
            raise ValueError(f"Invalid beam format: '{pair}'. Expected 'alt:az'")

    return pointings
