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

from .config import ArrayConfig, FrequencyConfig, compute_beam_fwhm
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


@dataclass
class CalibrationWeights:
    """
    Container for delay calibration weights loaded from SVD pipeline output.

    These weights correct for instrumental phase offsets (cable delays,
    electronics, etc.). They are derived by fringe-stopping visibilities
    toward a bright source and extracting per-antenna complex gains via SVD.

    The stored weights are conj(gain), ready for direct multiplication with
    geometric weights: w_total = w_cal * w_geo.

    Attributes
    ----------
    weights : np.ndarray
        (n_ant, n_chan) complex array, frequency ascending.
    flags : np.ndarray
        (n_chan,) boolean array. True = good channel, False = flagged.
    frequencies_hz : np.ndarray
        (n_chan,) float64, ascending order.
    ant_ids : np.ndarray
        (n_ant,) int array, 1-indexed antenna IDs.
    ref_ant_id : int
        Reference antenna ID used in SVD calibration.
    source : str
        Calibrator source name (e.g. "SUN").
    """
    weights: np.ndarray
    flags: np.ndarray
    frequencies_hz: np.ndarray
    ant_ids: np.ndarray
    ref_ant_id: int
    source: str = ""


def load_calibration_weights(npz_path: str) -> CalibrationWeights:
    """
    Load SVD calibration weights from .npz file.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file produced by the SVD calibration pipeline.

    Returns
    -------
    CalibrationWeights
        Loaded calibration weights with metadata.

    Raises
    ------
    ValueError
        If the file has inconsistent shapes or non-ascending frequencies.
    FileNotFoundError
        If the file does not exist.
    """
    data = np.load(npz_path, allow_pickle=True)

    weights = data['weights']
    flags = data['flags']
    ant_ids = data['ant_ids']

    # Get frequencies (prefer Hz, fall back to MHz)
    if 'freqs_hz' in data:
        frequencies_hz = data['freqs_hz'].astype(np.float64)
    elif 'freqs_mhz' in data:
        frequencies_hz = data['freqs_mhz'].astype(np.float64) * 1e6
    else:
        raise ValueError("Cal weights file must contain 'freqs_hz' or 'freqs_mhz'")

    # Validate shapes
    n_ant, n_chan = weights.shape
    if len(flags) != n_chan:
        raise ValueError(
            f"Shape mismatch: weights has {n_chan} channels but flags has {len(flags)}"
        )
    if len(frequencies_hz) != n_chan:
        raise ValueError(
            f"Shape mismatch: weights has {n_chan} channels but frequencies has "
            f"{len(frequencies_hz)}"
        )
    if len(ant_ids) != n_ant:
        raise ValueError(
            f"Shape mismatch: weights has {n_ant} antennas but ant_ids has "
            f"{len(ant_ids)}"
        )

    # Ensure frequencies are in ascending order (flip if descending)
    if n_chan > 1 and frequencies_hz[1] < frequencies_hz[0]:
        frequencies_hz = frequencies_hz[::-1]
        weights = weights[:, ::-1]
        flags = flags[::-1]

    ref_ant_id = int(data['ref_ant_id']) if 'ref_ant_id' in data else 0
    source = str(data['source']) if 'source' in data else ""

    return CalibrationWeights(
        weights=weights,
        flags=flags,
        frequencies_hz=frequencies_hz,
        ant_ids=ant_ids,
        ref_ant_id=ref_ant_id,
        source=source,
    )


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
        64-slot array configuration loaded from CSV. Used for computing
        geometric weights (antenna positions).
    freq_config : FrequencyConfig, optional
        Frequency configuration. Uses default CASM configuration if not specified.
    output_array_config : Array64Config, optional
        Array configuration for SNAP output ordering. If provided, the SNAP
        reordering step uses this layout's snap_to_ant64 mapping instead of
        array_config's. This is useful when the calibration data was taken with
        a different SNAP board assignment than the current one.

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
        output_array_config: Optional[Array64Config] = None,
    ):
        self.array_config = array_config
        self.output_array_config = output_array_config or array_config
        self.freq_config = freq_config or FrequencyConfig()

        # Create beamformer with only active antennas
        self._beamformer = GeometricBeamformer(
            array_config=array_config.to_array_config(),
            freq_config=self.freq_config,
        )

    def _apply_calibration_weights(
        self,
        geo_weights: np.ndarray,
        cal_weights: 'CalibrationWeights',
    ) -> np.ndarray:
        """
        Combine geometric weights with calibration weights.

        Performs element-wise multiplication: w_total = w_cal * w_geo.
        Cal weights are auto-flipped to match the geometric frequency order
        (descending) if they are in ascending order. Antenna mapping uses
        ant_id → ant64 conversion.

        Parameters
        ----------
        geo_weights : np.ndarray
            Geometric weights, shape (n_beams, n_active, n_chan).
            Frequencies in FrequencyConfig order (descending).
        cal_weights : CalibrationWeights
            Calibration weights. Frequency axis is auto-flipped to match
            geometric order if ascending.

        Returns
        -------
        np.ndarray
            Combined weights, same shape as geo_weights. Flagged channels
            are zeroed out.

        Raises
        ------
        ValueError
            If channel counts differ or antenna mapping fails.
        """
        n_geo_chan = geo_weights.shape[2]
        n_cal_chan = cal_weights.weights.shape[1]

        if n_geo_chan != n_cal_chan:
            raise ValueError(
                f"Channel count mismatch: geo has {n_geo_chan}, "
                f"cal has {n_cal_chan}"
            )

        # Get cal data — flip to descending if ascending
        cal_freqs = cal_weights.frequencies_hz
        is_ascending = len(cal_freqs) > 1 and cal_freqs[1] > cal_freqs[0]

        if is_ascending:
            cal_w = cal_weights.weights[:, ::-1]    # (n_ant, n_chan) descending
            cal_flags = cal_weights.flags[::-1]      # (n_chan,) descending
        else:
            cal_w = cal_weights.weights
            cal_flags = cal_weights.flags

        # --- Antenna mapping: cal ant_ids (1-indexed) → ant64 (0-indexed) ---
        active_indices = self.array_config.active_indices
        n_active = len(active_indices)
        cal_ant64 = cal_weights.ant_ids - 1

        cal_ant_map = np.full(n_active, -1, dtype=np.int32)
        for i, ant64_idx in enumerate(active_indices):
            matches = np.where(cal_ant64 == ant64_idx)[0]
            if len(matches) == 1:
                cal_ant_map[i] = matches[0]
            elif len(matches) == 0:
                raise ValueError(
                    f"Active antenna ant64={ant64_idx} not found in cal weights "
                    f"ant_ids={cal_weights.ant_ids}"
                )

        # --- Combine: w_total = w_cal * w_geo ---
        # Build cal array aligned to active antenna ordering: (n_active, n_chan)
        cal_aligned = np.zeros((n_active, n_geo_chan), dtype=np.complex64)
        for i in range(n_active):
            ci = cal_ant_map[i]
            if ci >= 0:
                cal_aligned[i, :] = cal_w[ci, :]

        # Zero flagged channels in cal
        cal_aligned[:, ~cal_flags] = 0.0

        # Multiply: broadcast over beams
        combined = geo_weights.copy()
        # Where cal is zero (flagged), result is zero. Where non-zero, multiply.
        combined *= cal_aligned[np.newaxis, :, :]

        return combined

    def compute_int8_weights(
        self,
        pointings: Optional[List[StationaryPointing]] = None,
        scale_factor: float = 127.0,
        cal_weights: Optional['CalibrationWeights'] = None,
    ) -> Int8StationaryWeights:
        """
        Compute int8-quantized stationary beamformer weights.

        Processing steps:
        1. Compute complex64 geometric weights for active antennas
        1.5. (Optional) Combine with calibration weights
        2. Expand to 64 slots (inactive = 0)
        3. Reorder antennas to SNAP input order (using output_array_config)
        4. Reverse channel order
        5. Quantize to int8
        6. Add polarization dimension (Pol A = Pol B)

        Parameters
        ----------
        pointings : List[StationaryPointing], optional
            Beam pointing directions. If None, uses TRANSIT_SURVEY_BEAMS.
        scale_factor : float
            Scale factor for quantization. Default is 127.0.
        cal_weights : CalibrationWeights, optional
            Delay calibration weights. If provided, combined with geometric
            weights before quantization. Flagged channels are zeroed out.

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

        # Step 1.5: Combine with calibration weights if provided
        if cal_weights is not None:
            active_weights = self._apply_calibration_weights(
                active_weights, cal_weights
            )

        n_beams = len(pointings)
        n_chan = self.freq_config.n_chan

        # Step 2: Expand to 64 slots (inactive = 0)
        weights_64 = np.zeros((n_beams, 64, n_chan), dtype=np.complex64)
        active_indices = self.array_config.active_indices
        for i, ant64_idx in enumerate(active_indices):
            weights_64[:, ant64_idx, :] = active_weights[:, i, :]

        # Step 3: Reorder antennas to SNAP input order (using output layout)
        weights_snap_order = np.zeros((n_beams, 64, n_chan), dtype=np.complex64)
        snap_mapping = self.output_array_config.snap_to_ant64
        for snap_idx in range(64):
            ant64_idx = snap_mapping[snap_idx]
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
        parts = [f"array={self.array_config}", f"freq={self.freq_config}"]
        if self.output_array_config is not self.array_config:
            parts.append(f"output_array={self.output_array_config}")
        return f"SnapWeightsGenerator({', '.join(parts)})"


@dataclass
class CombinedWeights:
    """
    Container for combined geometric + calibration beamformer weights.

    Holds complex64 weights in SNAP input order, ready for beamforming.
    Self-contained result with all metadata needed for file I/O and analysis.

    Attributes
    ----------
    weights : np.ndarray
        (n_beams, 64, n_chan) complex64 weights in SNAP input order.
    frequencies_hz : np.ndarray
        (n_chan,) float64 channel frequencies in Hz.
    flags : np.ndarray
        (n_chan,) bool, True = good channel.
    pointings : List[StationaryPointing]
        Beam pointing directions.
    array_config : Array64Config
        Layout used for geometric weight computation (antenna positions).
    output_array_config : Array64Config
        Layout used for SNAP input ordering.
    freq_config : FrequencyConfig
        Frequency configuration.
    cal_weights : CalibrationWeights or None
        Calibration weights used, or None if geometric only.
    freq_order : str
        "descending" or "ascending" — ordering of the frequency axis.
    """
    weights: np.ndarray
    frequencies_hz: np.ndarray
    flags: np.ndarray
    pointings: List[StationaryPointing]
    array_config: Array64Config
    output_array_config: Array64Config
    freq_config: FrequencyConfig
    cal_weights: Optional[CalibrationWeights]
    freq_order: str = "descending"

    @property
    def n_beams(self) -> int:
        return len(self.pointings)

    @property
    def n_channels(self) -> int:
        return len(self.frequencies_hz)

    @property
    def n_good_channels(self) -> int:
        return int(np.sum(self.flags))

    def __repr__(self) -> str:
        cal_str = f"cal={self.cal_weights.source}" if self.cal_weights else "geo-only"
        return (
            f"CombinedWeights(n_beams={self.n_beams}, n_chan={self.n_channels}, "
            f"good_chan={self.n_good_channels}, {cal_str}, "
            f"freq_order={self.freq_order!r})"
        )


def generate_combined_weights(
    pointing,
    array_config: Array64Config,
    cal_weights: Optional[CalibrationWeights] = None,
    output_array_config: Optional[Array64Config] = None,
    freq_config: Optional[FrequencyConfig] = None,
    freq_order: str = "descending",
) -> CombinedWeights:
    """
    Generate combined geometric + calibration beamformer weights.

    Computes complex64 weights that combine geometric steering phases with
    optional SVD-derived delay calibration corrections. Returns weights in
    SNAP input order, ready for beamforming.

    Parameters
    ----------
    pointing : StationaryPointing or List[StationaryPointing]
        Beam pointing direction(s).
    array_config : Array64Config
        Antenna layout used for geometric weight computation (positions).
    cal_weights : CalibrationWeights, optional
        Delay calibration weights. If provided, combined with geometric
        weights. Flagged channels are zeroed out.
    output_array_config : Array64Config, optional
        Layout used for SNAP input ordering. Defaults to array_config.
        Use this when the calibration data was taken with a different SNAP
        board assignment than the current beamformer.
    freq_config : FrequencyConfig, optional
        Frequency configuration. Defaults to CASM 3072-channel setup.
    freq_order : str
        Output frequency ordering: "descending" (default, 469->375 MHz)
        or "ascending" (375->469 MHz).

    Returns
    -------
    CombinedWeights
        Combined weights with full metadata.
    """
    # Wrap single pointing in list
    if isinstance(pointing, StationaryPointing):
        pointings = [pointing]
    else:
        pointings = list(pointing)

    if output_array_config is None:
        output_array_config = array_config
    if freq_config is None:
        freq_config = FrequencyConfig()

    gen = SnapWeightsGenerator(
        array_config, freq_config=freq_config,
        output_array_config=output_array_config,
    )

    # Step 1: geometric weights for active antennas (descending freq)
    stationary = gen._beamformer.compute_stationary_weights(
        pointings=pointings, mode=BeamMode.COHERENT,
    )
    active_weights = stationary.weights  # (n_beams, n_active, n_chan)

    # Step 2: apply calibration if provided
    if cal_weights is not None:
        active_weights = gen._apply_calibration_weights(active_weights, cal_weights)

    n_beams = len(pointings)
    n_chan = freq_config.n_chan
    active_indices = array_config.active_indices

    # Step 3: expand to 64 slots
    weights_64 = np.zeros((n_beams, 64, n_chan), dtype=np.complex64)
    for i, ant64_idx in enumerate(active_indices):
        weights_64[:, ant64_idx, :] = active_weights[:, i, :]

    # Step 4: reorder to SNAP input order
    weights_snap = np.zeros((n_beams, 64, n_chan), dtype=np.complex64)
    snap_mapping = output_array_config.snap_to_ant64
    for snap_idx in range(64):
        ant64_idx = snap_mapping[snap_idx]
        if ant64_idx >= 0:
            weights_snap[:, snap_idx, :] = weights_64[:, ant64_idx, :]

    # Step 5: build frequencies (descending, matching geo convention)
    frequencies_hz = freq_config.get_frequencies_hz()  # descending

    # Step 6: build flags
    flags = np.ones(n_chan, dtype=bool)
    if cal_weights is not None:
        cal_flags = cal_weights.flags
        is_ascending = (
            len(cal_weights.frequencies_hz) > 1
            and cal_weights.frequencies_hz[1] > cal_weights.frequencies_hz[0]
        )
        if is_ascending:
            flags = cal_flags[::-1].copy()  # flip to descending
        else:
            flags = cal_flags.copy()

    # Step 7: flip to ascending if requested
    actual_order = "descending"
    if freq_order == "ascending":
        weights_snap = weights_snap[:, :, ::-1]
        frequencies_hz = frequencies_hz[::-1]
        flags = flags[::-1]
        actual_order = "ascending"

    return CombinedWeights(
        weights=weights_snap,
        frequencies_hz=frequencies_hz,
        flags=flags,
        pointings=pointings,
        array_config=array_config,
        output_array_config=output_array_config,
        freq_config=freq_config,
        cal_weights=cal_weights,
        freq_order=actual_order,
    )


def generate_beam_grid(
    n_beams: Optional[int] = None,
    spacing_deg: float = 15.0,
    alt_min_deg: float = 30.0,
    alt_max_deg: float = 90.0,
    az_min_deg: float = 0.0,
    az_max_deg: float = 360.0,
    positions_enu: Optional[np.ndarray] = None,
    array_config: Optional['Array64Config'] = None,
    freq_hz: Optional[float] = None,
) -> List[StationaryPointing]:
    """
    Generate a grid of beam pointings covering the sky.

    The grid is generated with spacing in altitude and azimuth adjusted at
    each altitude to maintain roughly uniform sky coverage. Supports both
    isotropic and elliptical (array-aware) beam spacing.

    Parameters
    ----------
    n_beams : int, optional
        Target number of beams. If specified, spacing is scaled
        automatically to achieve approximately this many beams.
    spacing_deg : float
        Isotropic beam spacing in degrees (default: 15.0). Ignored if
        n_beams is specified or if array positions are provided.
    alt_min_deg : float
        Minimum altitude in degrees (default: 30.0).
    alt_max_deg : float
        Maximum altitude in degrees (default: 90.0).
    az_min_deg : float
        Minimum azimuth in degrees (default: 0.0).
    az_max_deg : float
        Maximum azimuth in degrees (default: 360.0).
    positions_enu : np.ndarray, optional
        Antenna positions in ENU coordinates, shape (n_ant, 3). If provided,
        auto-computes elliptical spacing from beam FWHM (lambda/D per axis).
    array_config : Array64Config, optional
        Array configuration. If provided, uses its active positions for
        FWHM computation. Overrides ``positions_enu``.
    freq_hz : float, optional
        Reference frequency in Hz for FWHM computation (default: 437.5 MHz).
        Only used when positions are provided.

    Returns
    -------
    List[StationaryPointing]
        List of beam pointings.

    Examples
    --------
    >>> beams = generate_beam_grid(n_beams=8)  # ~8 beams
    >>> beams = generate_beam_grid(spacing_deg=20)  # 20 deg spacing
    >>> beams = generate_beam_grid(n_beams=16, alt_min_deg=45)  # 16 beams above 45 deg
    >>> # Array-aware elliptical spacing:
    >>> from bf_weights_generator import Array64Config
    >>> arr = Array64Config.from_csv("layout.csv")
    >>> beams = generate_beam_grid(array_config=arr)
    """
    # Determine per-axis spacing from array positions if provided
    if array_config is not None:
        pos = array_config.active_positions
        spacing_ew, spacing_ns = compute_beam_fwhm(pos, freq_hz=freq_hz)
    elif positions_enu is not None:
        spacing_ew, spacing_ns = compute_beam_fwhm(positions_enu, freq_hz=freq_hz)
    else:
        spacing_ew = spacing_deg
        spacing_ns = spacing_deg

    # If n_beams specified, estimate scaling to achieve target count
    if n_beams is not None:
        alt_min_rad = np.deg2rad(alt_min_deg)
        alt_max_rad = np.deg2rad(alt_max_deg)
        solid_angle = 2 * np.pi * (np.sin(alt_max_rad) - np.sin(alt_min_rad))
        beam_area = np.deg2rad(spacing_ew) * np.deg2rad(spacing_ns)
        if beam_area > 0:
            estimated_n = solid_angle / beam_area
        else:
            estimated_n = 1

        if n_beams > 1 and estimated_n > 0:
            scale = np.sqrt(estimated_n / n_beams)
            spacing_ew = spacing_ew * scale
            spacing_ns = spacing_ns * scale
            # Clamp to reasonable range
            spacing_ew = max(0.5, min(60.0, spacing_ew))
            spacing_ns = max(0.5, min(60.0, spacing_ns))

    pointings = []
    beam_idx = 0

    # Generate altitude levels (N-S spacing controls altitude step)
    alt_values = np.arange(alt_min_deg, alt_max_deg + spacing_ns / 2, spacing_ns)

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
            # Number of beams: circumference / spacing_ew
            cos_alt = np.cos(np.deg2rad(alt))
            az_spacing = spacing_ew / cos_alt if cos_alt > 0.1 else 360.0

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
