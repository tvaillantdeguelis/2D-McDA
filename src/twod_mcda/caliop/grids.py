"""CALIOP vertical and horizontal grid transformations."""

import numpy as np

from twod_mcda.caliop.constants import (
    FILL_VALUE_FLOAT,
    LAYER_ALTITUDE_R1_INDEX_RANGE,
    LAYER_ALTITUDE_R2_INDEX_RANGE,
    LAYER_ALTITUDE_R3_INDEX_RANGE,
    LAYER_ALTITUDE_R4_INDEX_RANGE,
    LAYER_ALTITUDE_R5_INDEX_RANGE,
    NUMBER_OF_VERTICAL_BINS,
    NUMBER_OF_VFM_VERTICAL_BINS,
    N_30M_BINS_PER_BIN_R1,
    N_30M_BINS_PER_BIN_R2,
    N_30M_BINS_PER_BIN_R3,
    N_30M_BINS_PER_BIN_R4,
    N_30M_BINS_PER_BIN_R5,
    N_333M_BINS_PER_BIN_R3,
    N_333M_BINS_PER_BIN_R4,
    N_BINS_R1,
    N_BINS_R2,
    N_BINS_R3,
    N_BINS_R4,
    N_BINS_R5,
    N_LASER_PULSES_PER_1km,
    N_LASER_PULSES_PER_5km,
    N_PROFILES_VFM_R2,
    N_PROFILES_VFM_R3,
    N_PROFILES_VFM_R4,
)


def alt_to_regular_30m_vertical_grid(alt, reverse_altitude=True):
    """
    Put alt in regular grid (30 m)
    reverse_altitude: if True, return array from bottom to top
    """

    # Initialization
    nb_vert_levels = get_nb_regular_30m_vertical_levels()
    reg_grid_alt = np.ones(nb_vert_levels)*FILL_VALUE_FLOAT
    reg_grid_r5_index_range = (0, N_BINS_R5*N_30M_BINS_PER_BIN_R5)
    reg_grid_r4_index_range = (reg_grid_r5_index_range[1],
                               reg_grid_r5_index_range[1] + N_BINS_R4*N_30M_BINS_PER_BIN_R4)
    reg_grid_r3_index_range = (reg_grid_r4_index_range[1],
                               reg_grid_r4_index_range[1] + N_BINS_R3*N_30M_BINS_PER_BIN_R3)
    reg_grid_r2_index_range = (reg_grid_r3_index_range[1],
                               reg_grid_r3_index_range[1] + N_BINS_R2*N_30M_BINS_PER_BIN_R2)
    reg_grid_r1_index_range = (reg_grid_r2_index_range[1],
                               reg_grid_r2_index_range[1] + N_BINS_R1*N_30M_BINS_PER_BIN_R1)
    
    # Copy region R2 (30 m resolution)
    reg_grid_alt[reg_grid_r2_index_range[0]:reg_grid_r2_index_range[1]] = \
        alt[LAYER_ALTITUDE_R2_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1]+1]
    step_30m = alt[LAYER_ALTITUDE_R2_INDEX_RANGE[1]-2] - alt[LAYER_ALTITUDE_R2_INDEX_RANGE[1]-1]
    
    # Complete R1 by decreasing by nb_bins×step_30m the lowest bin altitude of R2
    start_index = reg_grid_r1_index_range[0]
    end_index = reg_grid_r1_index_range[1]
    reg_grid_alt[start_index:end_index] = reg_grid_alt[start_index-1] - (np.arange(end_index-start_index)+1)*step_30m
    
    # Complete R3, R4, and R5 by increasing by nb_bins×step_30m the highest bin altitude of R2
    end_index = reg_grid_r3_index_range[1]
    reg_grid_alt[end_index-1::-1] = reg_grid_alt[end_index] + (np.arange(end_index)+1)*step_30m

    if reverse_altitude:
        reg_grid_alt = reg_grid_alt[::-1]

    return reg_grid_alt


def shape_to_regular_30m_vertical_grid(data, reverse_altitude=True):
    """
    Duplicate and/or average CALIOP data to get a regular 30 m vertical resolution grid.
    reverse_altitude: if True, return array from bottom to top
    """
    
    # Initialization
    nb_vert_levels = get_nb_regular_30m_vertical_levels()
    reg_grid_r5_index_range = (0, N_BINS_R5*N_30M_BINS_PER_BIN_R5)
    reg_grid_r4_index_range = (reg_grid_r5_index_range[1],
                               reg_grid_r5_index_range[1] + N_BINS_R4*N_30M_BINS_PER_BIN_R4)
    reg_grid_r3_index_range = (reg_grid_r4_index_range[1],
                               reg_grid_r4_index_range[1] + N_BINS_R3*N_30M_BINS_PER_BIN_R3)
    reg_grid_r2_index_range = (reg_grid_r3_index_range[1],
                               reg_grid_r3_index_range[1] + N_BINS_R2*N_30M_BINS_PER_BIN_R2)
    reg_grid_r1_index_range = (reg_grid_r2_index_range[1],
                               reg_grid_r2_index_range[1] + N_BINS_R1*N_30M_BINS_PER_BIN_R1)

    reg_grid_data = np.ma.ones((data.shape[0], nb_vert_levels))*FILL_VALUE_FLOAT

    # Duplicate data
    reg_grid_data[:, reg_grid_r5_index_range[0]:reg_grid_r5_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R5_INDEX_RANGE[0]:LAYER_ALTITUDE_R5_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R5, axis=1)
    reg_grid_data[:, reg_grid_r4_index_range[0]:reg_grid_r4_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R4_INDEX_RANGE[0]:LAYER_ALTITUDE_R4_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R4, axis=1)
    reg_grid_data[:, reg_grid_r3_index_range[0]:reg_grid_r3_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R3_INDEX_RANGE[0]:LAYER_ALTITUDE_R3_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R3, axis=1)
    reg_grid_data[:, reg_grid_r2_index_range[0]:reg_grid_r2_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R2_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R2, axis=1)
    reg_grid_data[:, reg_grid_r1_index_range[0]:reg_grid_r1_index_range[1]] = \
        np.repeat(data[:, LAYER_ALTITUDE_R1_INDEX_RANGE[0]:LAYER_ALTITUDE_R1_INDEX_RANGE[1]+1],
                  N_30M_BINS_PER_BIN_R1, axis=1)
    
    if reverse_altitude:
        reg_grid_data = reg_grid_data[:, ::-1]
    
    return reg_grid_data


def get_nb_regular_30m_vertical_levels():
    
    nb_30m_vert_levels = N_BINS_R1*N_30M_BINS_PER_BIN_R1 + \
                         N_BINS_R2*N_30M_BINS_PER_BIN_R2 + \
                         N_BINS_R3*N_30M_BINS_PER_BIN_R3 + \
                         N_BINS_R4*N_30M_BINS_PER_BIN_R4 + \
                         N_BINS_R5*N_30M_BINS_PER_BIN_R5
    
    return nb_30m_vert_levels


def unfold_vfm(vfm, put_in_all_alt_grid=False):
    """
    Unfold VFM in regular grid with 545 levels.
    
    :param vfm: folded VFM
    :param put_in_all_alt_grid: (optional) put in grid with 583 levels (-2 to 40 km). Zero where no data.
                                default: False
    :return: unfolded VFM
    """
    # Number of VFM masks in the file
    nb_vfm = vfm.shape[0]

    # Initialization
    vfm_unfolded = np.zeros((nb_vfm*N_LASER_PULSES_PER_5km, NUMBER_OF_VFM_VERTICAL_BINS),
                            dtype='uint16')
    r4_index_range = (0, 0+N_BINS_R4) # Regions R1 and R5 are not in VFM
    r3_index_range = (r4_index_range[1], r4_index_range[1]+N_BINS_R3)
    r2_index_range = (r3_index_range[1], r3_index_range[1]+N_BINS_R2)
    
    # Loop on each VFM mask in the file
    i_vfm = 0
    while i_vfm < nb_vfm:
        # 20.2 to 30.1 km
        for i in np.arange(N_PROFILES_VFM_R4):
            start = i_vfm*N_LASER_PULSES_PER_5km+i*N_333M_BINS_PER_BIN_R4
            vfm_unfolded[start:start+N_333M_BINS_PER_BIN_R4, r4_index_range[0]:r4_index_range[1]] = \
                vfm[i_vfm, i*N_BINS_R4:(i+1)*N_BINS_R4]
        # 8.2 to 20.2 km
        for i in np.arange(N_PROFILES_VFM_R3):
            start = i_vfm*N_LASER_PULSES_PER_5km+i*N_333M_BINS_PER_BIN_R3
            index_first_bin_r3 = N_PROFILES_VFM_R4*N_BINS_R4
            vfm_unfolded[start:start+N_333M_BINS_PER_BIN_R3, r3_index_range[0]:r3_index_range[1]] =\
                vfm[i_vfm, index_first_bin_r3+i*N_BINS_R3:index_first_bin_r3+(i+1)*N_BINS_R3]
        # -0.5 to 8.2 km
        for i in np.arange(N_PROFILES_VFM_R2):
            index_first_bin_r2 = N_PROFILES_VFM_R4*N_BINS_R4+N_PROFILES_VFM_R3*N_BINS_R3
            vfm_unfolded[i_vfm*N_LASER_PULSES_PER_5km+i, r2_index_range[0]:r2_index_range[1]] =\
                vfm[i_vfm, index_first_bin_r2+i*N_BINS_R2:index_first_bin_r2+(i+1)*N_BINS_R2]
        i_vfm += 1
    
    if put_in_all_alt_grid:
        vfm_unfolded_all = np.zeros((nb_vfm*N_LASER_PULSES_PER_5km, NUMBER_OF_VERTICAL_BINS),
                                    dtype='uint16')
        vfm_unfolded_all[:, LAYER_ALTITUDE_R4_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1]+1] = vfm_unfolded
        vfm_unfolded = vfm_unfolded_all
        
    return vfm_unfolded


def get_single_shot_index_from_5km_index(i_5km):
    """
    Return min and max single shot indexes of a 5 km profile index
    """
    ss_min = i_5km*N_LASER_PULSES_PER_5km
    ss_max = (i_5km+1)*N_LASER_PULSES_PER_5km - 1

    return ss_min, ss_max


def split_granule_date(granule_date):
    
    granule_date_dict = {}
    granule_date_dict['year'] = int(granule_date[:4])
    granule_date_dict['month'] = int(granule_date[5:7])
    granule_date_dict['day'] = int(granule_date[8:10])
    granule_date_dict['hour'] = int(granule_date[11:13])
    granule_date_dict['min'] = int(granule_date[14:16])
    granule_date_dict['sec'] = int(granule_date[17:19])
    daynight_flag = granule_date[19:21]
    if daynight_flag == 'ZD':
        granule_date_dict['daynigth'] = 'day'
    elif daynight_flag == 'ZN':
        granule_date_dict['daynigth'] = 'night'
    
    return granule_date_dict


def from_30mx333m_to_new_resolution(data, vgrid, hgrid, prof_min, print_first_ID=False):
    """
    Average CALIOP data to a new resolution.
    Attention: the data resolution change but the grid stays 30mx333m.
    """
    
    # Look for first profile of a 5 km chunk
    if prof_min:
        profID_first_in_chunk = get_first_profileID_of_chunk(prof_min)
    else:
        profID_first_in_chunk = 0
    if print_first_ID:
        print("First profile of first 5 km chunk: %d" % profID_first_in_chunk)
    
    if vgrid == '30m':
            pass # already in 30 m resolution
    elif vgrid == '60m':
        # Average 30 m to 60 m
        data_copy = np.ma.copy(data)
        for i in range(0, data.shape[1], 2):
            for j in range(i, i+2):
                data[:, j] = np.ma.mean(data_copy[:, i:i+2], axis=1)
    elif vgrid == '180m':
        # Average 30 m to 180 m
        data_copy = np.ma.copy(data)
        # for i in np.arange(data_copy.shape[1]):
        #     print(f"{i:3d} {data_copy[20, i]:15.10f}")
        # stop
        for i in range(0, data.shape[1], 6)[:-1]:
            for j in range(i, i+6):
                data[:, j] = np.ma.mean(data_copy[:, i:i+6], axis=1)
    else:
        raise Exception('Error: vgrid unknown')
    
    if hgrid == '333m':
        pass # already at 333 m resolution
    elif hgrid == '1km':
        data_copy = np.ma.copy(data) # data already averaged vertically
        for i in range(profID_first_in_chunk, data.shape[0]-N_LASER_PULSES_PER_1km+1, N_LASER_PULSES_PER_1km):
            for j in range(i, i+N_LASER_PULSES_PER_1km):
                data[j, :] = np.ma.mean(data_copy[i:i+N_LASER_PULSES_PER_1km, :], axis=0)
    elif hgrid == '5km':
        data_copy = np.ma.copy(data) # data already averaged vertically
        for i in range(profID_first_in_chunk, data.shape[0]-N_LASER_PULSES_PER_5km+1, N_LASER_PULSES_PER_5km):
            for j in range(i, i+N_LASER_PULSES_PER_5km):
                data[j, :] = np.ma.mean(data_copy[i:i+N_LASER_PULSES_PER_5km, :], axis=0)
    else:
        raise Exception('Error: hgrid unknown')
    
    return data


def get_first_profileID_of_chunk(profID):
    """
    Look for first profile of a 5 km chunk
    :param profID: profile ID in a chunk
    :return: first profile ID of the chunk
    """
    
    modulo_prof = profID % N_LASER_PULSES_PER_5km
    profID_first_in_chunk = 15 - modulo_prof
    if profID_first_in_chunk == 15:
        profID_first_in_chunk = 0
    
    return profID_first_in_chunk
