"""Geographical helpers for CALIOP orbit data."""

from enum import IntEnum

import numpy as np


def format_calipso_utc_time(utc_time):
    """
    Transform UTC time CALIPSO format ('yymmdd.ffffffff') in a more readable format:
    'YYYY-MM-DD HH:MM:SS'
    """
    date_int = int(utc_time)
    date = f"{date_int:06d}"
    year = "20" + date[:2]
    month = date[2:4]
    day = date[4:6]

    fraction_of_day = utc_time - date_int
    total_seconds = int(fraction_of_day * 86400)
    hour, remainder = divmod(total_seconds, 3600)
    minute, second = divmod(remainder, 60)

    return f"{year}-{month}-{day} {hour:02d}:{minute:02d}:{second:02d}"


def lon_m180_180_to_0_360(lon_m180_180):
    """
    Transform longitude(s) from the -180/180 convention to the 0/360 convention.

    Accepts a scalar, a list, or a (optionally masked) numpy array.
    """
    return np.ma.asarray(lon_m180_180) % 360


def get_prof_min_max_indexes_from_lon(lon, lon_prof_min, lon_prof_max):
    """
    Return indices of profiles closest to lon_prof_min and lon_prof_max
    """
    # Transform lon -180/180 to lon 0/360
    lon = lon_m180_180_to_0_360(lon)
    lon_prof_min = lon_m180_180_to_0_360(lon_prof_min)
    lon_prof_max = lon_m180_180_to_0_360(lon_prof_max)

    index_prof_min = int(np.argmin(np.abs(lon - lon_prof_min)))
    index_prof_max = int(np.argmin(np.abs(lon - lon_prof_max)))

    if index_prof_max <= index_prof_min:
        raise ValueError(
            f"index_prof_max (= {index_prof_max}) <= index_prof_min "
            f"(= {index_prof_min}); please check lon_prof_min and lon_prof_max"
        )

    return index_prof_min, index_prof_max


class _LonCrossing(IntEnum):
    NONE = 0
    ZERO = 1  # crosses 0° only
    ANTIMERIDIAN = 2  # crosses 180° only
    ZERO_THEN_ANTIMERIDIAN = 3
    ANTIMERIDIAN_THEN_ZERO = 4


_NEAR_0_DEG = 10
_NEAR_180_DEG = 170


def unwrap_lon(lon):
    """Transform lon -180/180 to lon -360/360 (to avoid bump in map when crossing)"""

    crossing = _LonCrossing.NONE
    i_first_cross = 0
    i_second_cross = 0

    for i in np.arange(lon.size - 1):
        sign_changed = np.sign(lon[i + 1]) != np.sign(lon[i])

        if sign_changed and np.abs(lon[i]) < _NEAR_0_DEG:
            if crossing == _LonCrossing.ANTIMERIDIAN:
                crossing = _LonCrossing.ANTIMERIDIAN_THEN_ZERO
                i_second_cross = i
                break
            crossing = _LonCrossing.ZERO
            i_first_cross = i
        elif sign_changed and np.abs(lon[i]) > _NEAR_180_DEG:
            if crossing == _LonCrossing.ZERO:
                crossing = _LonCrossing.ZERO_THEN_ANTIMERIDIAN
                i_second_cross = i
                break
            crossing = _LonCrossing.ANTIMERIDIAN
            i_first_cross = i

    mono_lon = np.copy(lon)

    if crossing == _LonCrossing.ANTIMERIDIAN:
        mono_lon[mono_lon < 0] += 360
    elif crossing == _LonCrossing.ZERO_THEN_ANTIMERIDIAN:
        if mono_lon[i_second_cross + 1] > 0:  # neg-pos-neg
            mono_lon[i_second_cross + 1 :] -= 360
        else:  # pos-neg-pos
            mono_lon[i_second_cross + 1 :] += 360
    elif crossing == _LonCrossing.ANTIMERIDIAN_THEN_ZERO:
        if mono_lon[i_first_cross + 1] > 0:  # neg-pos-neg
            mono_lon[: i_first_cross + 1] += 360
        else:  # pos-neg-pos
            mono_lon[: i_first_cross + 1] -= 360

    return mono_lon
