"""Geographical helpers for CALIOP orbit data."""

from math import acos, cos, sin

import numpy as np
from scipy import interpolate
from numba import jit


def geo_distance(lat1, lon1, lat2, lon2):
    """
    Compute geometrical distance between to coordinates using spherical model (accurate to
    around 0.3 %).
    """
    earth_mean_radius = 6371.  # km

    # Distance in rad
    dist_rad = acos(sin(np.deg2rad(lat1)) * sin(np.deg2rad(lat2)) + \
                    cos(np.deg2rad(lat1)) * cos(np.deg2rad(lat2)) * \
                    cos(np.deg2rad(lon1) - \
                        np.deg2rad(lon2)))

    # Distance in km
    dist_km = dist_rad * earth_mean_radius

    return dist_km


def UTC_time_CALIPSO(utc_time):
    """
    Transform UTC time CALIPSO format ('yymmdd.ffffffff') in a more readable format:
    'YYYY-MM-DD HH:MM:SS
    '"""
    date = '%06d' % utc_time
    fraction_hour = utc_time - int(date)

    year = '20' + date[:2]
    month = date[2:4]
    day = date[4:6]

    hour = int(fraction_hour * 24)
    minute = int((fraction_hour * 24 - hour) * 60)
    second = int((fraction_hour * 24 * 60 - hour * 60 - minute) * 60)

    utc_time_string = "%s-%s-%s %02d:%02d:%02d" % (year, month, day, hour, minute, second)

    return utc_time_string


def granule_date_decomposition(granule_date):

    year = int(granule_date[:4])
    month = int(granule_date[5:7])
    day = int(granule_date[8:10])
    hour = int(granule_date[11:13])
    minute = int(granule_date[14:16])
    second = int(granule_date[17:19])
    day_night_flag = granule_date[19:21]

    return year, month, day, hour, minute, second, day_night_flag

    
def lon_m180_180_to_0_360(lon_m180_180):
    """
    Transform list of longitudes from -180 to 180 to longitudes from 0 to 360
    """
    lon = np.ma.copy(lon_m180_180)
    lon[lon < 0] = lon[lon < 0] + 360

    return lon


def lon_0_360_to_m180_180(lon_0_360):
    """
    Transform list of longitudes from 0 to 360 to longitudes from -180 to 180
    """
    lon = np.ma.copy(lon_0_360)
    lon[lon > 180] = lon[lon > 180] - 360

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
        raise ValueError(f"Error: index_prof_max (= {index_prof_max}) <= index_prof_min "
                         f"(= {index_prof_min}); please check lon_prof_min and lon_prof_max\n")

    return index_prof_min, index_prof_max


def get_prof_min_max_indexes_from_latminmax(lat, lat_min, lat_max):
    """
    Return indices of start and end profiles that would remove extremities outside
    the lat_min-lat_max range
    """
    
    # Get indices where latitudes are within the range
    lat_min = -100 if lat_min is None else lat_min
    lat_max = 100 if lat_max is None else lat_max
    indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]

    if indices.size > 0:
        index_prof_min = indices[0]
        index_prof_max = indices[-1]
    else:
        raise ValueError(f"No values found between lat_min={lat_min} and lat_max={lat_max}")
    
    return index_prof_min, index_prof_max


def change_map_grid_resolution(old_grid_data, old_lat, old_lon, new_lat, new_lon):
    """
    Interpolate data to new grid resolution.
    
    :param old_grid_data: data in original grid resolution
    :param old_grid_resolution: the old map (lat, lon) grid coordinates
    :param new_grid_resolution: the new map (lat, lon) grid coordinates
    :return: data in new grid resolution
    """
    f = interpolate.interp2d(old_lon, old_lat, old_grid_data, kind='linear')
    new_grid_data = f(new_lon, new_lat)
    # if new_grid_resolution[0] > old_grid_resolution[0] and new_grid_resolution[1] > old_grid_resolution[1]:
    #     print("coucou")
    # else:
    #     sys.stderr.write(f"Error: this case of change in map grid resolution is not implemented yet.")
    #     sys.exit(1)
    
    return new_grid_data


def extrapolate_latlon(data, nb_times_higher_resolution):
    """
    Extrapolate a vector of continuous latitudes or longitudes to a higher resolution supposing
    the initial values are the center of the higher resolution grid. Then, only odd value for
    nb_times_higher_resolution are authorized. For exemple if nb_times_higher_resolution = 3, it
    means each original coordinate is extended to 3 coordinates with the one in the center equal
    to the original.
    
    :param data: vector of latitudes or longitudes
    :return: extrapolted vector of latitudes or longitudes
    """
    if (nb_times_higher_resolution % 2) != 1:
        raise Exception("Error: nb_times_higher_resolution should be an odd number.\n")
    

@jit(nopython=True)
def neighbors(shape, p):
    """Get neighbors of a pixel"""

    v = []

    if p[0] != shape[0]-1: # if not extreme right
        v.append( (p[0]+1, p[1]) ) # add right neighbor

    if p[0] != 0: # if not extreme left
        v.append( (p[0]-1, p[1]) ) # add left neighbor

    if p[1] != shape[1]-1: # if not extreme top
        v.append( (p[0], p[1]+1) ) # add top neighbor

    if p[1] != 0: # if not extreme bottom
        v.append( (p[0], p[1]-1) ) # add bottom neighbor

    return v
        

def get_monotical_lon(lon):
    """Transform lon -180/180 to lon -360/360 (to avoid bump in map when crossing)"""
    
    # Initialization
    cross_0 = False
    cross_180 = False
    cross_combination = 0 # 0: no cross
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
        mono_lon[mono_lon < 0] = \
            mono_lon[mono_lon < 0] + 360
    elif cross_combination == 3:  # cross 0° first then 180°
        if mono_lon[i_second_cross+1] > 0: # neg-pos-neg
            mono_lon[i_second_cross+1:] = mono_lon[i_second_cross+1:] - 360
        else: # pos-neg-pos
            mono_lon[i_second_cross+1:] = mono_lon[i_second_cross+1:] + 360
    elif cross_combination == 4:  # cross 180° first then 0°
        if mono_lon[i_first_cross+1] > 0: # neg-pos-neg
            mono_lon[:i_first_cross+1] = mono_lon[:i_first_cross+1] + 360
        else: # pos-neg-pos
            mono_lon[:i_first_cross+1] = mono_lon[:i_first_cross+1] - 360
        
    return mono_lon
        
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    if False:
        lon_m180_180 = np.array((-180, -179, -150, -100, -50, 0, 50, 100, 150))
        lon_0_360 = lon_m180_180_to_0_360(lon_m180_180)
        print(lon_0_360)
        lon_m180_180 = lon_0_360_to_m180_180(lon_0_360)
        print(lon_m180_180)
    
        print(get_prof_min_max_indexes_from_lon(lon_m180_180, -54, 29))
    
    if False:
        import os
        from readers.netcdf_reader import NetCDFReader
        ERAI_PATH = os.path.join("/home", "thibault", "Documents", "Pro", "Recherche", "codes", "DATA",
                                 "ERAI", "GLOBAL_075", "1xmonthly")
        skt_filename = f"skt.2010.asmei.GLOBAL_075.nc"
        with NetCDFReader(os.path.join(ERAI_PATH, "AN_SF", "2010", skt_filename)) as data_reader:
            skin_temp = data_reader.get_data('skt')[:,:,:] # (month, lat, lon)
            lat = data_reader.get_data('lat')[:]
            lon = data_reader.get_data('lon')[:]
        
        new_lat = np.arange(-85, 89, 10)
        new_lon = np.arange(5, 359, 10)
        
        new_skin_temp = change_map_grid_resolution(skin_temp[0,:,:], lat, lon, new_lat, new_lon)
        
        plt.subplot(211)
        plt.pcolormesh(lon, lat, skin_temp[0,:,:])
        plt.subplot(212)
        plt.pcolormesh(new_lon, new_lat, new_skin_temp)
        plt.show()
    
    if False:
        # lon = np.array((-10, -7, -5, -3, 3, 5, 7, 10)) # cross_combination 1
        # lon = np.array((150, 160, 170, 179, -175, -165, -155, -145)) # cross_combination 2
        # lon = np.array((-20, -10, -5, 5, 10, 50, 100, 150, 160, 170, 179, -175, -165, -155, -145)) # cross_combination 3 neg-pos-neg
        # lon = np.array((20, 10, 5, -5, -10, -50, -100, -150, -160, -170, -179, 175, 165, 155, 145)) # cross_combination 3 neg-pos-neg
        # lon = np.array((170, 179, -175, -165, -155, -145, -20, -10, -5, 5, 10, 50, 100, 150, 160)) # cross_combination 4 pos-neg-pos
        lon = np.array((-170, -179, 175, 165, 155, 145, 20, 10, 5, -5, -10, -50, -100, -150, -160)) # cross_combination 4 neg-pos-neg
        lon_mono = get_monotical_lon(lon)
        for i in np.arange(lon.size):
            print(f"{lon[i]:+04d} {lon_mono[i]:+04d}")

    if True:
        granule_date = "2023-05-03T03-55-06ZN"
        year, month, day, hour, minute, second, day_night_flag = granule_date_decomposition(granule_date)
        print(year, month, day, hour, minute, second, day_night_flag)
