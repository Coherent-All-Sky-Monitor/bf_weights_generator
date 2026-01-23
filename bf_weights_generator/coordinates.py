"""
Coordinate transformations for geometric beamforming.

Handles conversions between:
- RA/Dec (J2000) -> Hour Angle / Dec
- Hour Angle / Dec -> Direction Cosines (l, m, n)
- Direction Cosines -> Geometric Delays
"""

import numpy as np
from typing import Union, Tuple
from datetime import datetime

try:
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
    from astropy.time import Time
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

from .config import OVRO_LAT_DEG, OVRO_LON_DEG, OVRO_ALT_M


def _check_astropy():
    """Raise error if astropy is not available."""
    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "astropy is required for coordinate transformations. "
            "Install with: pip install astropy"
        )


def compute_lst_rad(unix_time: Union[float, np.ndarray], lon_deg: float) -> Union[float, np.ndarray]:
    """
    Compute Local Sidereal Time (LST) from Unix timestamp.

    Uses a simplified formula that's accurate to ~1 second over typical
    observation durations. For higher accuracy, use astropy directly.

    Parameters
    ----------
    unix_time : float or np.ndarray
        Unix timestamp(s) in seconds since 1970-01-01 00:00:00 UTC.
    lon_deg : float
        Observatory longitude in degrees (negative for West).

    Returns
    -------
    float or np.ndarray
        Local Sidereal Time in radians.
    """
    # Julian Date from Unix time
    # Unix epoch (1970-01-01 00:00:00 UTC) = JD 2440587.5
    jd = unix_time / 86400.0 + 2440587.5

    # Julian centuries from J2000.0
    T = (jd - 2451545.0) / 36525.0

    # Greenwich Mean Sidereal Time (GMST) in degrees
    # Using IAU 1982 expression
    gmst_deg = (280.46061837 +
                360.98564736629 * (jd - 2451545.0) +
                0.000387933 * T**2 -
                T**3 / 38710000.0)

    # Convert to LST by adding longitude
    lst_deg = gmst_deg + lon_deg

    # Normalize to [0, 360)
    lst_deg = lst_deg % 360.0

    return np.deg2rad(lst_deg)


def radec_to_hadec(ra_rad: float, dec_rad: float, lst_rad: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], float]:
    """
    Convert RA/Dec to Hour Angle / Dec.

    Parameters
    ----------
    ra_rad : float
        Right Ascension in radians.
    dec_rad : float
        Declination in radians.
    lst_rad : float or np.ndarray
        Local Sidereal Time in radians.

    Returns
    -------
    ha_rad : float or np.ndarray
        Hour Angle in radians.
    dec_rad : float
        Declination in radians (unchanged).
    """
    ha_rad = lst_rad - ra_rad
    return ha_rad, dec_rad


def hadec_to_direction_cosines(ha_rad: Union[float, np.ndarray], dec_rad: float,
                                lat_rad: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Hour Angle / Dec to direction cosines (l, m, n) in ENU frame.

    The direction cosines point FROM the array TOWARD the source:
    - l: East component
    - m: North component
    - n: Up component (toward zenith)

    For a source at (HA, Dec) observed from latitude (lat):
    - l = cos(dec) * sin(ha)
    - m = cos(lat) * sin(dec) - sin(lat) * cos(dec) * cos(ha)
    - n = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(ha)

    Parameters
    ----------
    ha_rad : float or np.ndarray
        Hour Angle in radians.
    dec_rad : float
        Declination in radians.
    lat_rad : float
        Observatory latitude in radians.

    Returns
    -------
    l, m, n : np.ndarray
        Direction cosines (East, North, Up). Shape matches ha_rad.
    """
    ha_rad = np.atleast_1d(ha_rad)

    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_ha = np.cos(ha_rad)
    sin_ha = np.sin(ha_rad)

    # Direction cosines in ENU frame
    l = cos_dec * sin_ha
    m = cos_lat * sin_dec - sin_lat * cos_dec * cos_ha
    n = sin_lat * sin_dec + cos_lat * cos_dec * cos_ha

    return l, m, n


def radec_to_direction_cosines(ra_deg: float, dec_deg: float,
                                unix_times: np.ndarray,
                                lat_deg: float = OVRO_LAT_DEG,
                                lon_deg: float = OVRO_LON_DEG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RA/Dec (J2000) to direction cosines at given times.

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees (J2000).
    dec_deg : float
        Declination in degrees (J2000).
    unix_times : np.ndarray
        Unix timestamps in seconds.
    lat_deg : float
        Observatory latitude in degrees.
    lon_deg : float
        Observatory longitude in degrees.

    Returns
    -------
    l, m, n : np.ndarray
        Direction cosines (East, North, Up). Shape (n_times,).
    """
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    lat_rad = np.deg2rad(lat_deg)

    # Compute LST for all times
    lst_rad = compute_lst_rad(unix_times, lon_deg)

    # Convert to HA/Dec
    ha_rad, _ = radec_to_hadec(ra_rad, dec_rad, lst_rad)

    # Convert to direction cosines
    l, m, n = hadec_to_direction_cosines(ha_rad, dec_rad, lat_rad)

    return l, m, n


def radec_to_direction_cosines_astropy(ra_deg: float, dec_deg: float,
                                        unix_times: np.ndarray,
                                        lat_deg: float = OVRO_LAT_DEG,
                                        lon_deg: float = OVRO_LON_DEG,
                                        alt_m: float = OVRO_ALT_M) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RA/Dec (J2000) to direction cosines using astropy.

    This provides higher accuracy than the simplified formula, accounting for
    precession, nutation, and atmospheric refraction.

    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees (J2000).
    dec_deg : float
        Declination in degrees (J2000).
    unix_times : np.ndarray
        Unix timestamps in seconds.
    lat_deg : float
        Observatory latitude in degrees.
    lon_deg : float
        Observatory longitude in degrees.
    alt_m : float
        Observatory altitude in meters.

    Returns
    -------
    l, m, n : np.ndarray
        Direction cosines (East, North, Up). Shape (n_times,).
    """
    _check_astropy()

    # Create observatory location
    location = EarthLocation(lat=lat_deg * u.deg,
                              lon=lon_deg * u.deg,
                              height=alt_m * u.m)

    # Create sky coordinate (J2000)
    source = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')

    # Create time array
    times = Time(unix_times, format='unix')

    # Transform to AltAz frame
    altaz_frame = AltAz(obstime=times, location=location)
    source_altaz = source.transform_to(altaz_frame)

    # Get altitude and azimuth
    alt_rad = source_altaz.alt.rad
    az_rad = source_altaz.az.rad

    # Convert Alt/Az to direction cosines (ENU)
    # Az is measured from North toward East
    # l (East) = cos(alt) * sin(az)
    # m (North) = cos(alt) * cos(az)
    # n (Up) = sin(alt)
    l = np.cos(alt_rad) * np.sin(az_rad)
    m = np.cos(alt_rad) * np.cos(az_rad)
    n = np.sin(alt_rad)

    return l, m, n


def compute_geometric_delays(positions_enu: np.ndarray,
                              l: np.ndarray, m: np.ndarray, n: np.ndarray,
                              c: float = 299792458.0) -> np.ndarray:
    """
    Compute geometric delays for each antenna at each time.

    The geometric delay is the extra path length (in time) that a wavefront
    traveling from direction (l, m, n) must travel to reach each antenna
    relative to the phase center (origin).

    delay = -(x*l + y*m + z*n) / c

    Negative sign: a positive delay means the signal arrives LATER at that
    antenna compared to the origin.

    Parameters
    ----------
    positions_enu : np.ndarray
        Antenna positions in ENU coordinates, shape (n_ant, 3).
    l, m, n : np.ndarray
        Direction cosines, shape (n_times,).
    c : float
        Speed of light in m/s.

    Returns
    -------
    delays : np.ndarray
        Geometric delays in seconds, shape (n_times, n_ant).
    """
    # positions_enu: (n_ant, 3) -> x, y, z = E, N, U
    # l, m, n: (n_times,)

    x = positions_enu[:, 0]  # East (n_ant,)
    y = positions_enu[:, 1]  # North (n_ant,)
    z = positions_enu[:, 2]  # Up (n_ant,)

    # Compute dot product for each time and antenna
    # Result shape: (n_times, n_ant)
    path_length = (np.outer(l, x) + np.outer(m, y) + np.outer(n, z))

    # Delay is negative path length divided by speed of light
    # (signal arrives earlier at antennas in the direction of the source)
    delays = -path_length / c

    return delays
