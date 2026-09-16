"""Geographical helpers for CALIOP orbit data."""

import numpy as np


def UTC_time_CALIPSO(utc_time):
    """
    Transform UTC time CALIPSO format ('yymmdd.ffffffff') in a more readable format:
    'YYYY-MM-DD HH:MM:SS
    '"""
    date = "%06d" % utc_time
    fraction_hour = utc_time - int(date)

    year = "20" + date[:2]
    month = date[2:4]
    day = date[4:6]

    hour = int(fraction_hour * 24)
    minute = int((fraction_hour * 24 - hour) * 60)
    second = int((fraction_hour * 24 * 60 - hour * 60 - minute) * 60)

    utc_time_string = "%s-%s-%s %02d:%02d:%02d" % (
        year,
        month,
        day,
        hour,
        minute,
        second,
    )

    return utc_time_string


def lon_m180_180_to_0_360(lon_m180_180):
    """
    Transform list of longitudes from -180 to 180 to longitudes from 0 to 360
    """
    lon = np.ma.copy(lon_m180_180)
    lon[lon < 0] = lon[lon < 0] + 360

    return lon


def get_prof_min_max_indexes_from_lon(lon, lon_prof_min, lon_prof_max):
    """
    Return indices of profiles closest to lon_prof_min and lon_prof_max
    """
    # Transform lon -180/180 to lon 0/360
    lon = lon_m180_180_to_0_360(lon)
    lon_prof_min = lon_m180_180_to_0_360(lon_prof_min)
    lon_prof_max = lon_m180_180_to_0_360(lon_prof_max)

    # Initialization
    diff_lon_min = 9999
    diff_lon_max = 9999
    index_prof_min = 0
    index_prof_max = 0

    for i in np.arange(lon.size):

        current_diff_lon_min = np.abs(lon_prof_min - lon[i])
        if current_diff_lon_min < diff_lon_min:
            diff_lon_min = current_diff_lon_min
            index_prof_min = i

        current_diff_lon_max = np.abs(lon_prof_max - lon[i])
        if current_diff_lon_max < diff_lon_max:
            diff_lon_max = current_diff_lon_max
            index_prof_max = i

    if index_prof_max <= index_prof_min:
        raise ValueError(
            f"Error: index_prof_max (= {index_prof_max}) <= index_prof_min "
            f"(= {index_prof_min}); please check lon_prof_min and lon_prof_max\n"
        )

    return index_prof_min, index_prof_max


def get_monotical_lon(lon):
    """Transform lon -180/180 to lon -360/360 (to avoid bump in map when crossing)"""

    # Initialization
    cross_0 = False
    cross_180 = False
    cross_combination = 0  # 0: no cross
    # 1: cross 0°
    # 2: cross 180°
    # 3: cross 0° first then 180°
    # 4: cross 180° first then 0°
    i_first_cross = 0
    i_second_cross = 0
    for i in np.arange(lon.size - 1):
        # If cross 0°
        if (np.sign(lon[i + 1]) != np.sign(lon[i])) and (np.abs(lon[i]) < 10):
            if cross_180 == True:
                cross_combination = 4
                i_second_cross = i
                break
            else:
                cross_0 = True
                cross_combination = 1
                i_first_cross = i
        # If cross 180°
        if (np.sign(lon[i + 1]) != np.sign(lon[i])) and (np.abs(lon[i]) > 170):
            # If already cross 0°
            if cross_0 == True:
                cross_combination = 3
                i_second_cross = i
                break
            else:
                cross_180 = True
                cross_combination = 2
                i_first_cross = i

    mono_lon = np.copy(lon)

    if cross_combination == 2:  # cross 180°
        mono_lon[mono_lon < 0] = mono_lon[mono_lon < 0] + 360
    elif cross_combination == 3:  # cross 0° first then 180°
        if mono_lon[i_second_cross + 1] > 0:  # neg-pos-neg
            mono_lon[i_second_cross + 1 :] = mono_lon[i_second_cross + 1 :] - 360
        else:  # pos-neg-pos
            mono_lon[i_second_cross + 1 :] = mono_lon[i_second_cross + 1 :] + 360
    elif cross_combination == 4:  # cross 180° first then 0°
        if mono_lon[i_first_cross + 1] > 0:  # neg-pos-neg
            mono_lon[: i_first_cross + 1] = mono_lon[: i_first_cross + 1] + 360
        else:  # pos-neg-pos
            mono_lon[: i_first_cross + 1] = mono_lon[: i_first_cross + 1] - 360

    return mono_lon
