"""
This file contains constants that are defined in the CALIPSO products.
"""

from pathlib import Path
import numpy as np
import os
import pickle
import sys

WAVELENGTH_532 = 532
WAVELENGTH_1064 = 1064

FILL_VALUE_FLOAT = -9999.0
FILL_VALUE_SHORT = -127
FILL_VALUE_FLOAT_EXTINCTION = -333.0
FILL_VALUE_EXTINCTION_QC = 32768
FILL_VALUE_ODCOD_QC = 4294967295
FILL_VALUE_OPACITY_FLAG = 99

IGBP_FLAG_OPEN_SHRUBLAND = 7
IGBP_FLAG_BARREN_DESERT = 16
IGBP_FLAG_WATER = 17

NUMBER_OF_VERTICAL_BINS = 583
NUMBER_OF_VERTICAL_BINS_MET = 33
NUMBER_OF_VERTICAL_BINS_05KMPRO = 399

REGION_5_ALTITUDE_BOUNDARIES = (40.0, 30.1)
REGION_4_ALTITUDE_BOUNDARIES = (30.1, 20.2)
REGION_3_ALTITUDE_BOUNDARIES = (20.2, 8.2)
REGION_2_ALTITUDE_BOUNDARIES = (8.2, -0.5)
REGION_1_ALTITUDE_BOUNDARIES = (-0.5, -2.0)

LAYER_ALTITUDE_R5_INDEX_RANGE = (0, 32)
LAYER_ALTITUDE_R4_INDEX_RANGE = (33, 87)
LAYER_ALTITUDE_R3_INDEX_RANGE = (88, 287)
LAYER_ALTITUDE_R2_INDEX_RANGE = (288, 577)
LAYER_ALTITUDE_R1_INDEX_RANGE = (578, 582)

N_BINS_R5 = LAYER_ALTITUDE_R5_INDEX_RANGE[1] - LAYER_ALTITUDE_R5_INDEX_RANGE[0] + 1
N_BINS_R4 = LAYER_ALTITUDE_R4_INDEX_RANGE[1] - LAYER_ALTITUDE_R4_INDEX_RANGE[0] + 1
N_BINS_R3 = LAYER_ALTITUDE_R3_INDEX_RANGE[1] - LAYER_ALTITUDE_R3_INDEX_RANGE[0] + 1
N_BINS_R2 = LAYER_ALTITUDE_R2_INDEX_RANGE[1] - LAYER_ALTITUDE_R2_INDEX_RANGE[0] + 1
N_BINS_R1 = LAYER_ALTITUDE_R1_INDEX_RANGE[1] - LAYER_ALTITUDE_R1_INDEX_RANGE[0] + 1

N_15M_BINS_PER_BIN_R5 = 20
N_15M_BINS_PER_BIN_R4 = 12
N_15M_BINS_PER_BIN_R3 = 4
N_15M_BINS_PER_BIN_R2 = 2
N_15M_BINS_PER_BIN_R1 = 20

N_30M_BINS_PER_BIN_R5 = int(N_15M_BINS_PER_BIN_R5 / 2)
N_30M_BINS_PER_BIN_R4 = int(N_15M_BINS_PER_BIN_R4 / 2)
N_30M_BINS_PER_BIN_R3 = int(N_15M_BINS_PER_BIN_R3 / 2)
N_30M_BINS_PER_BIN_R2 = int(N_15M_BINS_PER_BIN_R2 / 2)
N_30M_BINS_PER_BIN_R1 = int(N_15M_BINS_PER_BIN_R1 / 2)

SIZE_30M_BIN_KM = 0.03

N_LASER_PULSES_PER_333m = 1
N_LASER_PULSES_PER_1km = 3
N_LASER_PULSES_PER_1667m = 5
N_LASER_PULSES_PER_5km = 15

N_333M_BINS_PER_BIN_R5 = 15
N_333M_BINS_PER_BIN_R4 = 5
N_333M_BINS_PER_BIN_R3 = 3
N_333M_BINS_PER_BIN_R2 = 1
N_333M_BINS_PER_BIN_R1 = 1

NUMBER_OF_VFM_ELEMENTS = 5515
NUMBER_OF_VFM_VERTICAL_BINS = N_BINS_R2 + N_BINS_R3 + N_BINS_R4
N_PROFILES_VFM_R4 = 3
N_PROFILES_VFM_R3 = 5
N_PROFILES_VFM_R2 = 15

N_LAYERS_05KM_LAYER_PRODUCT = 15
N_LAYERS_01KM_LAYER_PRODUCT = 10
N_LAYERS_333M_LAYER_PRODUCT = 5

LIDAR_ALTITUDES_FILE = (
    Path(__file__).resolve().parent / "lidar_data_altitudes.pkl"
)

with LIDAR_ALTITUDES_FILE.open("rb") as file:
    LIDAR_DATA_ALTITUDES = pickle.load(file)["Lidar_Data_Altitudes"]

CALIPSO_STRFTIME_FMT = "%Y-%m-%dT%H-%M-%S"

CAL_LID_FILENAME_FMT = "CAL_LID_%s-%s-%s.%s.hdf" # product (ex: 'L2_VFM'), type (ex: Standard),
                                                 # version (ex: 'V4-10'),
                                                 # granule date (ex: '2010-06-01T01-33-28ZN')

CAL_IIR_FILENAME_FMT = "CAL_IIR_%s-%s-%s.%s.hdf" # product (ex: 'L2_Track'), type (ex: Standard),
                                                 # version (ex: 'V4-10'),
                                                 # granule date (ex: '2010-06-01T01-33-28ZN')

PRODUCT_H_RESOLUTION = {'333m': ['L1',
                                 'L2_333mMLay'],
                        '1km': ['L2_01kmCLay',],
                        '5km': ['L2_05kmALay',
                                'L2_05kmCLay',
                                'L2_05kmMLay',
                                'L2_05kmAPro',
                                'L2_05kmCPro']}

def get_caliop_correction_function(wl):
    """Input: wl = wavelength of the lidar channel
       Output: fcorr = correction function at each altitude level from
                       Table 2 of Liu (2011), updated values sent by M.
                       Vaughan"""

    fcorr = np.ones((NUMBER_OF_VERTICAL_BINS, 11)) # fcorr[bin index range, Nshift]
    fcorr[LAYER_ALTITUDE_R5_INDEX_RANGE[0]:LAYER_ALTITUDE_R5_INDEX_RANGE[1] + 1] = \
        [1.596, 1.448, 1.322, 1.224, 1.161, 1.140, 1.161, 1.224, 1.322, 1.448, 1.596]
    fcorr[LAYER_ALTITUDE_R4_INDEX_RANGE[0]:LAYER_ALTITUDE_R4_INDEX_RANGE[1] + 1] = \
        [1.573, 1.345, 1.188, 1.131, 1.188, 1.345, 1.573, 1.345, 1.188, 1.130, 1.188]
    fcorr[LAYER_ALTITUDE_R3_INDEX_RANGE[0]:LAYER_ALTITUDE_R3_INDEX_RANGE[1] + 1] = \
        [1.451, 1.080, 1.451, 1.080, 1.451, 1.080, 1.451, 1.080, 1.451, 1.080, 1.451]
    if wl==532:
        fcorr[LAYER_ALTITUDE_R2_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1] + 1] = \
            [1.269, 1.269, 1.269, 1.269, 1.269, 1.269, 1.269, 1.269, 1.269, 1.269, 1.269]
    elif wl==1064:
        fcorr[LAYER_ALTITUDE_R2_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1] + 1] = \
            [1.451, 1.451, 1.451, 1.451, 1.451, 1.451, 1.451, 1.451, 1.451, 1.451, 1.451]
    else:
        sys.exit(f'Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n')
    fcorr[LAYER_ALTITUDE_R1_INDEX_RANGE[0]:LAYER_ALTITUDE_R1_INDEX_RANGE[1] + 1] = \
        [1.596, 1.448, 1.322, 1.224, 1.161, 1.140, 1.161, 1.224, 1.322, 1.448, 1.596]

    return fcorr


def get_nb_pixels(wl):
    """Input: wl = wavelength of the lidar channel
       Output: nb_pixels = number of original resolution lidar "pixels"
                           (bins) averaged together at each altitude
                           level at the wavelength channel wl"""

    nb_pixels = np.ones(583)*-9999.

    if wl==532:
        nb_pixels[LAYER_ALTITUDE_R5_INDEX_RANGE[0]:LAYER_ALTITUDE_R5_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R5*N_15M_BINS_PER_BIN_R5
        nb_pixels[LAYER_ALTITUDE_R4_INDEX_RANGE[0]:LAYER_ALTITUDE_R4_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R4*N_15M_BINS_PER_BIN_R4
        nb_pixels[LAYER_ALTITUDE_R3_INDEX_RANGE[0]:LAYER_ALTITUDE_R3_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R3*N_15M_BINS_PER_BIN_R3
        nb_pixels[LAYER_ALTITUDE_R2_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R2*N_15M_BINS_PER_BIN_R2
        nb_pixels[LAYER_ALTITUDE_R1_INDEX_RANGE[0]:LAYER_ALTITUDE_R1_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R1*N_15M_BINS_PER_BIN_R1

    elif wl==1064:
        nb_pixels[LAYER_ALTITUDE_R5_INDEX_RANGE[0]:LAYER_ALTITUDE_R5_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R5*N_15M_BINS_PER_BIN_R5
        nb_pixels[LAYER_ALTITUDE_R4_INDEX_RANGE[0]:LAYER_ALTITUDE_R4_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R4*N_15M_BINS_PER_BIN_R4
        nb_pixels[LAYER_ALTITUDE_R3_INDEX_RANGE[0]:LAYER_ALTITUDE_R3_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R3*N_15M_BINS_PER_BIN_R3
        nb_pixels[LAYER_ALTITUDE_R2_INDEX_RANGE[0]:LAYER_ALTITUDE_R2_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R2*N_15M_BINS_PER_BIN_R2*2 # average at 60 m instead of 30 m in original data
        nb_pixels[LAYER_ALTITUDE_R1_INDEX_RANGE[0]:LAYER_ALTITUDE_R1_INDEX_RANGE[1] + 1] = \
            N_333M_BINS_PER_BIN_R1*N_15M_BINS_PER_BIN_R1
    
    else:
        sys.exit('Unrecognized wavelength: %d; use 532 or 1064 instead\n\n' %
                 wl)

    return nb_pixels
