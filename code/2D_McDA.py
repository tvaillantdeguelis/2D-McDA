#!/usr/bin/env python
# coding: utf8

"""Main program of 2D-McDA. Takes granule to process as input."""

__author__     = "Thibault Vaillant de Guélis"
__version__    = "1.0.2"
__email__      = "thibault.vaillantdeguelis@outlook.com"
__status__     = "Prototype"

import sys
import os

import numpy as np

from standard_outputs import print_time, print_elapsed_time
from readers.calipso_reader import CALIOPRegularGridReader, split_granule_date
from config import NB_PROF_SLICE, NB_PROF_OVERLAP, NB_PROF_EDGE
from surface_detection import detect_surface
from feature_detection import detect_features
from merged_3channels_feature_mask import merged_feature_masks
from calipso_constants import *
from writers.hdf_writer import SDSData, write_hdf
from datetime import datetime


def get_start_end_indexes(prof_min, prof_max, nb_prof_slice, nb_prof_overlap):
    """Compute start and end indexes of each slice of profiles to process
        Outputs: index_start_slice_array = array of slice start profile indexes
                 index_end_slice_array = array of slice end profile indexes"""
    
    nb_prof = prof_max - prof_min + 1
    
    # If total number of profiles to process less than slice size
    if nb_prof <= nb_prof_slice:
        # Put prof_min as start index
        index_start_slice_array = np.array((prof_min,))
        # Put prof_max as end index
        index_end_slice_array = np.array((prof_max,))

    else:
        # Create array of start indexes
        # Note: if last slice size < nb_prof_slice/2, then it is merge 
        # with previous
        index_start_slice_array = np.arange(prof_min, 
                                            prof_max - int(nb_prof_slice/2.) + 2, 
                                            nb_prof_slice-nb_prof_overlap)
        # Create array of end indexes
        index_end_slice_array = index_start_slice_array + nb_prof_slice - 1
        index_end_slice_array[-1] = prof_max

    return index_start_slice_array, index_end_slice_array


def rm_prof(array, nb_prof_to_remove, side):
    """Remove first profiles from previous file added to allow window 
       image technique at the edges"""
    
    if side == 'start':
        if array.ndim == 1:
            array = array[nb_prof_to_remove:]
        elif array.ndim == 2:
            array = array[nb_prof_to_remove:, :]
        elif array.ndim == 3:
            array = array[:, nb_prof_to_remove:, :]
        else:
            sys.exit('Error: ndim array unknown')
    elif side == 'end':
        if array.ndim == 1:
            array = array[:-nb_prof_to_remove]
        elif array.ndim == 2:
            array = array[:-nb_prof_to_remove, :]
        elif array.ndim == 3:
            array = array[:, :-nb_prof_to_remove, :]
        else:
            sys.exit('Error: ndim array unknown')
    else:
        sys.exit('Error: side unknown')

    return array


if __name__ == '__main__':
    tic_main_program = print_time()
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    if len(sys.argv) > 1:
        GRANULE_DATE = sys.argv[1]
        VERSION_CAL_LID_L1 = sys.argv[2]
        TYPE_CAL_LID_L1 = sys.argv[3]
        PREVIOUS_GRANULE = None if sys.argv[4] == 'None' else sys.argv[4]
        NEXT_GRANULE = None if sys.argv[5] == 'None' else sys.argv[5]
        SLICE_START_END_TYPE = sys.argv[6] # 'profindex' or 'longitude'
        SLICE_START = None if sys.argv[7] == 'None' else float(sys.argv[7])
        SLICE_END = None if sys.argv[8] == 'None' else float(sys.argv[8])
        SAVE_DEVELOPMENT_DATA = sys.argv[9] == 'True'
        VERSION_2D_McDA = sys.argv[10]
        TYPE_2D_McDA = sys.argv[11]
        OUT_FOLDER = sys.argv[12]
    else:
        GRANULE_DATE = "2006-06-25T06-58-31ZN"
        VERSION_CAL_LID_L1 = "V4.51"
        TYPE_CAL_LID_L1 = "Standard"
        PREVIOUS_GRANULE = None
        NEXT_GRANULE = None
        SLICE_START_END_TYPE = "profindex" # "profindex" or "longitude"
        SLICE_START = 1000 # profindex or longitude, use "profindex" with None
        SLICE_END = 1300 # profindex or longitude, use "profindex" with None
        SAVE_DEVELOPMENT_DATA = False # if "True" save step by step data
        VERSION_2D_McDA = "V1.0.2"
        TYPE_2D_McDA = "Prototype"
        OUT_FOLDER="/work_users/vaillant/data/2D_CALIOP/2D_McDA/"
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

        
    # ********************************
    # *** Configuration parameters ***
    print("\n*****Configuration parameters...*****")
    
    print("\tGRANULE_DATE =", GRANULE_DATE)
    print("\tVERSION_CAL_LID_L1 =", VERSION_CAL_LID_L1)
    print("\tTYPE_CAL_LID_L1 =", TYPE_CAL_LID_L1)
    print("\tPREVIOUS_GRANULE =", PREVIOUS_GRANULE)
    print("\tNEXT_GRANULE =", NEXT_GRANULE)
    print("\tSLICE_START_END_TYPE =", SLICE_START_END_TYPE)
    print("\tSLICE_START =", SLICE_START)
    print("\tSLICE_END =", SLICE_END)
    print("\tSAVE_DEVELOPMENT_DATA =", SAVE_DEVELOPMENT_DATA)
    print("\tVERSION_2D_McDA =", VERSION_2D_McDA)
    print("\tTYPE_2D_McDA =", TYPE_2D_McDA)
    print("\tOUT_FOLDER =", OUT_FOLDER)


    # **********************
    # *** CALIOP L1 data ***
    print("\n*****CALIOP L1 data...*****")
    
    cal_l1 = CALIOPRegularGridReader(product='L1',
                                     version=VERSION_CAL_LID_L1,
                                     data_type=TYPE_CAL_LID_L1,
                                     granule_date=GRANULE_DATE,
                                     grid='333mx30m',
                                     slice_start=SLICE_START,
                                     slice_end=SLICE_END,
                                     slice_start_end_type=SLICE_START_END_TYPE)

    # Print filepaths of loading files
    print(f"\tGranule path: {cal_l1.filepath}")

    # Print lat/lon of min and max prof indices
    print(f"\tFrom min profile index {cal_l1.prof_min:d} "
          f"(lat = {cal_l1.lat_min:.2f} / lon = {cal_l1.lon_min:.2f}) "
          f"to max profile index {cal_l1.prof_max:d} "
          f"(lat = {cal_l1.lat_max:.2f} / lon = {cal_l1.lon_max:.2f})")

    # Load L1 parameters
    cal_l1_keys = [
        "Latitude",
        "Longitude",
        "Lidar_Data_Altitudes",
        "Profile_ID",
        "Profile_Time",
        "Profile_UTC_Time",
        "Perpendicular_Attenuated_Backscatter_532",
        "Parallel_Attenuated_Backscatter_532",
        "Attenuated_Backscatter_1064",
        "Parallel_RMS_Baseline_532",
        "Perpendicular_RMS_Baseline_532",
        "RMS_Baseline_1064",
        "Spacecraft_Altitude",
        "Calibration_Constant_532",
        "Calibration_Constant_1064",
        "Depolarization_Gain_Ratio_532",
        "Laser_Energy_532",
        "Laser_Energy_1064",
        "Parallel_Amplifier_Gain_532",
        "Perpendicular_Amplifier_Gain_532",
        "Amplifier_Gain_1064",
        "Molecular_Parallel_Attenuated_Backscatter_532",
        "Molecular_Perpendicular_Attenuated_Backscatter_532",
        "Molecular_Attenuated_Backscatter_1064",
        "Molecular_Parallel_Backscatter_532",
        "Molecular_Perpendicular_Backscatter_532",
        "Molecular_Backscatter_1064",
        "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel",
        "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular",
        "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064",
        "Temperature",
        "IGBP_Surface_Type",
        "Surface_Elevation",
        "Off_Nadir_Angle"
    ]

    # Load previous granule
    if PREVIOUS_GRANULE:
        # TODO: no need to load the whole granule
        cal_l1_prev = CALIOPRegularGridReader(product='L1',
                                              version=VERSION_CAL_LID_L1,
                                              data_type=TYPE_CAL_LID_L1,
                                              granule_date=PREVIOUS_GRANULE,
                                              grid='333mx30m',
                                              slice_start=-NB_PROF_OVERLAP,
                                              slice_end=None,
                                              slice_start_end_type='profindex')
    
        # Print filepaths of loading files
        print(f"\tPrevious granule path: {cal_l1_prev.filepath}")
    
        data_dict_cal_lid_l1_prev = {}
        for key in cal_l1_keys:
            data_dict_cal_lid_l1_prev[key] = cal_l1_prev.get_data(key)
        
    # Load next granule
    if NEXT_GRANULE:
        # TODO: no need to load the whole granule
        # Load data with various regular grid sizes
        cal_l1_next = CALIOPRegularGridReader(product='L1',
                                              version=VERSION_CAL_LID_L1,
                                              data_type=TYPE_CAL_LID_L1,
                                              granule_date=NEXT_GRANULE,
                                              grid='333mx30m',
                                              slice_start=None,
                                              slice_end=NB_PROF_OVERLAP,
                                              slice_start_end_type='profindex')
    
        # Print filepaths of loading files
        print(f"\tNext granule path: {cal_l1_next.filepath}")
    
        data_dict_cal_lid_l1_next = {}
        for key in cal_l1_keys:
            data_dict_cal_lid_l1_next[key] = cal_l1_next.get_data(key)
    
    # Infer number of profiles to process
    nb_prof = cal_l1.prof_max - cal_l1.prof_min + 1
    
    # Get number of altitude level
    nb_alt = cal_l1.get_data("Lidar_Data_Altitudes").size

    # Print number of profiles in the granule
    print(f"\tNumber of profiles to process: {nb_prof}")
    
    # Initialization
    data_dict_2d_mcda = {}
    data_dict_2d_mcda_dev = {}
    data_dict_2d_mcda_all_slices = {"Profile_ID": np.ones(nb_prof, dtype=np.int32) * FILL_VALUE_FLOAT,
                                    "Profile_Time": np.ones(nb_prof, dtype=np.float64) * FILL_VALUE_FLOAT,
                                    "Profile_UTC_Time": np.ones(nb_prof, dtype=np.float64) * FILL_VALUE_FLOAT,
                                    "Latitude": np.ma.ones(nb_prof, dtype=np.float32) * FILL_VALUE_FLOAT,
                                    "Longitude": np.ma.ones(nb_prof, dtype=np.float32) * FILL_VALUE_FLOAT,
                                    "Parallel_Detection_Flags_532": np.zeros((nb_prof, nb_alt), dtype=np.uint8),
                                    "Perpendicular_Detection_Flags_532": np.zeros((nb_prof, nb_alt), dtype=np.uint8),
                                    "Detection_Flags_1064": np.zeros((nb_prof, nb_alt), dtype=np.uint8),
                                    "Composite_Detection_Flags": np.zeros((nb_prof, nb_alt), dtype=np.uint8)}

    
    # ********************************
    # *** Apply algorithm by slice ***
    print("\n*****Apply algorithm by slice...*****")
    
    # Get start and end slice indexes
    index_start_slice_array, index_end_slice_array =\
        get_start_end_indexes(cal_l1.prof_min, cal_l1.prof_max, NB_PROF_SLICE, NB_PROF_OVERLAP)
    
    # Process each slice
    for i_slice in range(index_start_slice_array.size):
        
        # Initialization
        previous_file_data_used = False
        next_file_data_used = False
    
        # Min and max slice profile indexes
        prof_slice_min = index_start_slice_array[i_slice]
        prof_slice_max = index_end_slice_array[i_slice]
    
        # Infer number of profiles in slice
        nb_prof_i_slice = prof_slice_max - prof_slice_min + 1
    
        # Print profile indexes of slice processed
        print("\n\n############################################################\n"
              f"Processing slice with profile indexes from {prof_slice_min} to {prof_slice_max}...")
        
        tic_slice = print_time()
        
        # ***********************
        # *** Load slice data ***
        print("\n\n*****Load slice data...*****")
        
        tic = datetime.now()
        
        # Load data for the slice
        cal_l1_slice = CALIOPRegularGridReader(product='L1',
                                               version=VERSION_CAL_LID_L1,
                                               data_type=TYPE_CAL_LID_L1,
                                               granule_date=GRANULE_DATE,
                                               grid='333mx30m',
                                               slice_start=prof_slice_min,
                                               slice_end=prof_slice_max)
        data_dict_cal_lid_l1_slice = {}
        for key in cal_l1_keys:
            data_dict_cal_lid_l1_slice[key] = cal_l1_slice.get_data(key)
        
        # If beginning of file and previous file given
        if (prof_slice_min == 0) & (PREVIOUS_GRANULE is not None):
            # If less than 1 second between last profile of previous file and
            # first profile of current file (no missing granule)
            time_between_profiles = np.abs(data_dict_cal_lid_l1_slice["Profile_Time"][0] -
                                           data_dict_cal_lid_l1_prev["Profile_Time"][-1])
            print(f"\tTime between last profile of previous file and first profile of current file"
                  f" = {time_between_profiles:.2f} s")
            if time_between_profiles < 1:
                print("\tAppend previous granule")
                previous_file_data_used = True
                # Append last profiles of previous file to data
                nb_prof_i_slice = nb_prof_i_slice + NB_PROF_OVERLAP
                for key in cal_l1_keys:
                    if key != "Lidar_Data_Altitudes":
                        data_dict_cal_lid_l1_slice[key] = np.append(data_dict_cal_lid_l1_prev[key],
                                                                    data_dict_cal_lid_l1_slice[key], axis=0)
            else:
                print("\tPrevious granule does not seem consecutive. First profiles not processed.")
    
        elif (prof_slice_min == 0) & (PREVIOUS_GRANULE is None):
            print("\tNo previous file to load. First profiles not processed.")
    
        elif (prof_slice_max == cal_l1.prof_max) & (NEXT_GRANULE is not None):
            # If less than 1 second between first profile of next file and
            # last profile of current file (no missing granule)
            time_between_profiles = np.abs(data_dict_cal_lid_l1_next["Profile_Time"][0] -
                                           data_dict_cal_lid_l1_slice["Profile_Time"][-1])
            print(f"\tTime between last profile of previous file and first profile of current file"
                  f" = {time_between_profiles:.2f} s")
            if time_between_profiles < 1:
                print("\tAppend next granule")
                next_file_data_used = True
                # Append first profiles of next file to data
                nb_prof_i_slice = nb_prof_i_slice + NB_PROF_OVERLAP
                for key in cal_l1_keys:
                    if key != "Lidar_Data_Altitudes":
                        data_dict_cal_lid_l1_slice[key] = np.append(data_dict_cal_lid_l1_slice[key],
                                                                    data_dict_cal_lid_l1_next[key], axis=0)
            else:
                print("\tNext granule does not seem consecutive. Last profiles not processed.")
    
        elif (prof_slice_max == cal_l1.prof_max) & (NEXT_GRANULE is None):
            print("\tNo next file to load. Last profiles not processed.")
        
        tic = print_elapsed_time(tic, '\t')
        
    
        # *************************
        # *** Surface detection ***
        print("\n\n*****Surface detection...*****")

        # Surface detection at 532 nm parallel
        surf_alt_index_532_par = detect_surface(data_dict_cal_lid_l1_slice["Parallel_Attenuated_Backscatter_532"],
                                                data_dict_cal_lid_l1_slice["IGBP_Surface_Type"],
                                                data_dict_cal_lid_l1_slice["Surface_Elevation"],
                                                data_dict_cal_lid_l1_slice["Spacecraft_Altitude"],
                                                data_dict_cal_lid_l1_slice["Lidar_Data_Altitudes"],
                                                data_dict_cal_lid_l1_slice["Parallel_RMS_Baseline_532"],
                                                data_dict_cal_lid_l1_slice["Laser_Energy_532"],
                                                data_dict_cal_lid_l1_slice["Calibration_Constant_532"],
                                                1,
                                                data_dict_cal_lid_l1_slice["Parallel_Amplifier_Gain_532"],
                                                data_dict_cal_lid_l1_slice["Off_Nadir_Angle"],
                                                '532_par')
        
        # Surface detection at 532 nm perpendicular
        surf_alt_index_532_per = detect_surface(data_dict_cal_lid_l1_slice["Perpendicular_Attenuated_Backscatter_532"],
                                                data_dict_cal_lid_l1_slice["IGBP_Surface_Type"],
                                                data_dict_cal_lid_l1_slice["Surface_Elevation"],
                                                data_dict_cal_lid_l1_slice["Spacecraft_Altitude"],
                                                data_dict_cal_lid_l1_slice["Lidar_Data_Altitudes"],
                                                data_dict_cal_lid_l1_slice["Perpendicular_RMS_Baseline_532"],
                                                data_dict_cal_lid_l1_slice["Laser_Energy_532"],
                                                data_dict_cal_lid_l1_slice["Calibration_Constant_532"],
                                                data_dict_cal_lid_l1_slice["Depolarization_Gain_Ratio_532"],
                                                data_dict_cal_lid_l1_slice["Perpendicular_Amplifier_Gain_532"],
                                                data_dict_cal_lid_l1_slice["Off_Nadir_Angle"],
                                                '532_per')
    
        # Surface detection at 1064 nm
        surf_alt_index_1064 = detect_surface(data_dict_cal_lid_l1_slice["Attenuated_Backscatter_1064"],
                                             data_dict_cal_lid_l1_slice["IGBP_Surface_Type"],
                                             data_dict_cal_lid_l1_slice["Surface_Elevation"],
                                             data_dict_cal_lid_l1_slice["Spacecraft_Altitude"],
                                             data_dict_cal_lid_l1_slice["Lidar_Data_Altitudes"],
                                             data_dict_cal_lid_l1_slice["RMS_Baseline_1064"],
                                             data_dict_cal_lid_l1_slice["Laser_Energy_1064"],
                                             data_dict_cal_lid_l1_slice["Calibration_Constant_1064"],
                                             1,
                                             data_dict_cal_lid_l1_slice["Amplifier_Gain_1064"],
                                             data_dict_cal_lid_l1_slice["Off_Nadir_Angle"],
                                             '1064')
    
    
        # *************************
        # *** Feature detection ***
        print("\n\n*****Feature detection...*****")
    
        # Feature detection at 532 nm parallel
        data_dict_2d_mcda["Parallel_Detection_Flags_532"], \
        data_dict_2d_mcda_dev["Parallel_Detection_Flags_532_steps"], \
        data_dict_2d_mcda_dev["Parallel_Attenuated_Scattering_Ratio_532_steps"], \
        data_dict_2d_mcda_dev["Parallel_CumulativeTwoWayTransmittance_532"] =\
            detect_features(data_dict_cal_lid_l1_slice["Parallel_Attenuated_Backscatter_532"]/\
                                data_dict_cal_lid_l1_slice["Molecular_Parallel_Attenuated_Backscatter_532"],
                            data_dict_cal_lid_l1_slice["Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel"],
                            data_dict_cal_lid_l1_slice["Molecular_Parallel_Backscatter_532"],
                            data_dict_cal_lid_l1_slice["Temperature"],
                            surf_alt_index_532_par, '532_par')
    
        # Feature detection at 532 nm perpendicular
        # Note: use surface detection at 532 nm parallel
        data_dict_2d_mcda["Perpendicular_Detection_Flags_532"], \
        data_dict_2d_mcda_dev["Perpendicular_Detection_Flags_532_steps"], \
        data_dict_2d_mcda_dev["Perpendicular_Attenuated_Scattering_Ratio_532_steps"], \
        data_dict_2d_mcda_dev["Perpendicular_CumulativeTwoWayTransmittance_532"] =\
            detect_features(data_dict_cal_lid_l1_slice["Perpendicular_Attenuated_Backscatter_532"]/\
                                data_dict_cal_lid_l1_slice["Molecular_Perpendicular_Attenuated_Backscatter_532"],
                            data_dict_cal_lid_l1_slice["Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular"],
                            data_dict_cal_lid_l1_slice["Molecular_Perpendicular_Backscatter_532"],
                            data_dict_cal_lid_l1_slice["Temperature"],
                            surf_alt_index_532_per, '532_per')
    
        # Feature detection at 1064 nm
        data_dict_2d_mcda["Detection_Flags_1064"], \
        data_dict_2d_mcda_dev["Detection_Flags_1064_steps"], \
        data_dict_2d_mcda_dev["Attenuated_Scattering_Ratio_1064_steps"], \
        data_dict_2d_mcda_dev["CumulativeTwoWayTransmittance_1064"] =\
            detect_features(data_dict_cal_lid_l1_slice["Attenuated_Backscatter_1064"]/\
                                data_dict_cal_lid_l1_slice["Molecular_Attenuated_Backscatter_1064"],
                            data_dict_cal_lid_l1_slice["Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064"],
                            data_dict_cal_lid_l1_slice["Molecular_Backscatter_1064"],
                            data_dict_cal_lid_l1_slice["Temperature"],
                            surf_alt_index_1064, '1064')
    
    
        # If beginning of file and previous file given
        if previous_file_data_used:
            # ******************************************
            # *** Remove profiles from previous file ***
            print("\n\n*****Remove profiles from previous file...*****")
            nb_prof_i_slice = nb_prof_i_slice - NB_PROF_OVERLAP
            for key in cal_l1_keys:
                data_dict_cal_lid_l1_slice[key] = rm_prof(data_dict_cal_lid_l1_slice[key], NB_PROF_OVERLAP, 'start')
            for key in data_dict_2d_mcda:
                data_dict_2d_mcda[key] = rm_prof(data_dict_2d_mcda[key], NB_PROF_OVERLAP, 'start')
            for key in data_dict_2d_mcda_dev:
                data_dict_2d_mcda_dev[key] = rm_prof(data_dict_2d_mcda_dev[key], NB_PROF_OVERLAP, 'start')

    
        # If end of file and next file given
        elif next_file_data_used:
            # **************************************
            # *** Remove profiles from next file ***
            print("\n\n*****Remove profiles from next file...*****")
            nb_prof_i_slice = nb_prof_i_slice - NB_PROF_OVERLAP
            for key in cal_l1_keys:
                data_dict_cal_lid_l1_slice[key] = rm_prof(data_dict_cal_lid_l1_slice[key], NB_PROF_OVERLAP, 'end')
            for key in data_dict_2d_mcda:
                data_dict_2d_mcda[key] = rm_prof(data_dict_2d_mcda[key], NB_PROF_OVERLAP, 'end')
            for key in data_dict_2d_mcda_dev:
                data_dict_2d_mcda_dev[key] = rm_prof(data_dict_2d_mcda_dev[key], NB_PROF_OVERLAP, 'end')

    
        # *******************************************
        # *** Merged 3 channels feature detection ***
        print("\n\n*****Merged 3 channels feature detection...*****")
    
        data_dict_2d_mcda["Composite_Detection_Flags"] = \
            merged_feature_masks(data_dict_2d_mcda["Parallel_Detection_Flags_532"],
                                 data_dict_2d_mcda["Perpendicular_Detection_Flags_532"],
                                 data_dict_2d_mcda["Detection_Flags_1064"])
    
    
        # ************************************************
        # *** Copy slice data to the whole data arrays ***
        print("\n\n*****Copy slice data to the whole data arrays...*****")
    
        prof_store_min = prof_slice_min - cal_l1.prof_min
        prof_store_max = prof_slice_max - cal_l1.prof_min
        
        for key in ("Profile_ID", "Profile_Time", "Profile_UTC_Time", "Latitude", "Longitude"):
            data_dict_2d_mcda_all_slices[key][prof_store_min:prof_store_max+1] = np.copy(data_dict_cal_lid_l1_slice[key])
        
        # If beginning of file and previous file given
        if previous_file_data_used:
            # Don't overwrite the NB_PROF_EDGE last profiles
            for key in ("Parallel_Detection_Flags_532", "Perpendicular_Detection_Flags_532",
                        "Detection_Flags_1064", "Composite_Detection_Flags"):
                data_dict_2d_mcda_all_slices[key][prof_store_min:prof_store_max-NB_PROF_EDGE+1, :] = \
                    data_dict_2d_mcda[key][:-NB_PROF_EDGE, :]
        elif next_file_data_used:
            # Don't overwrite the NB_PROF_EDGE first profiles
            for key in ("Parallel_Detection_Flags_532", "Perpendicular_Detection_Flags_532",
                        "Detection_Flags_1064", "Composite_Detection_Flags"):
                data_dict_2d_mcda_all_slices[key][prof_store_min+NB_PROF_EDGE:prof_store_max+1, :] = \
                    data_dict_2d_mcda[key][NB_PROF_EDGE:, :]
        else:
            # Don't overwrite the NB_PROF_EDGE first and last profiles
            for key in ("Parallel_Detection_Flags_532", "Perpendicular_Detection_Flags_532",
                        "Detection_Flags_1064", "Composite_Detection_Flags"):
                data_dict_2d_mcda_all_slices[key][prof_store_min+NB_PROF_EDGE:prof_store_max-NB_PROF_EDGE+1, :] = \
                    data_dict_2d_mcda[key][NB_PROF_EDGE:-NB_PROF_EDGE, :]
        
        print_elapsed_time(tic_slice)
    
    
    # *****************************
    # *** Save data in HDF file ***
    print("\n\n############################################################\n"\
          "*****Save data in HDF file...*****")
    
    # Create a dictionary of parameters to save in HDF file
    params = {}
    
    # Add prof_ID to params
    params['prof_ID'] = SDSData('Profile_ID', data_dict_2d_mcda_all_slices["Profile_ID"])
    params['prof_ID'].description = "Profile number from start of file"
    # params['prof_ID'].valid_range = (1, 228630)
    params['prof_ID'].dim_labels = ['Profile_ID']

    # Add prof_time to params
    params['prof_time'] = SDSData('Profile_Time', data_dict_2d_mcda_all_slices["Profile_Time"])
    params['prof_time'].units = "seconds...TAI"
    # params['prof_time'].valid_range = (4.204e8, 1.072e9)
    params['prof_time'].dim_labels = ['Profile_ID']

    # Add prof_UTC_time to params
    params['prof_UTC_time'] = SDSData('Profile_UTC_Time', data_dict_2d_mcda_all_slices["Profile_UTC_Time"])
    params['prof_UTC_time'].units = "UTC - yymmdd.ffffffff"
    # params['prof_UTC_time'].valid_range = (60426.0, 261231.0)
    params['prof_UTC_time'].dim_labels = ['Profile_ID']

    # Add lat to params
    params['lat'] = SDSData('Latitude', data_dict_2d_mcda_all_slices["Latitude"], FILL_VALUE_FLOAT)
    params['lat'].units = "degrees"
    # params['lat'].valid_range = (-90.0, 90.0)
    params['lat'].dim_labels = ['Profile_ID']

    # Add lon to params
    params['lon'] = SDSData('Longitude', data_dict_2d_mcda_all_slices["Longitude"], FILL_VALUE_FLOAT)
    params['lon'].units = "degrees"
    # params['lon'].valid_range = (-90.0, 90.0)
    params['lon'].dim_labels = ['Profile_ID']
    
    # Add alt to params
    params['alt'] = SDSData('Altitude', cal_l1.get_data("Lidar_Data_Altitudes"))
    params['alt'].units = "kilometer"
    # params['alt'].valid_range = (-0.5, 30.1)
    params['alt'].dim_labels = ['Altitude']
    
    # Add feature_mask_532_par to params
    params['feature_mask_532_par'] = SDSData('Parallel_Detection_Flags_532',
                                             data_dict_2d_mcda_all_slices["Parallel_Detection_Flags_532"])
    params['feature_mask_532_par'].valid_range = (0, 255)
    params['feature_mask_532_par'].dim_labels = ['Profile_ID', 'Altitude']

    # Add feature_mask_532_per to params
    params['feature_mask_532_per'] = SDSData('Perpendicular_Detection_Flags_532',
                                             data_dict_2d_mcda_all_slices["Perpendicular_Detection_Flags_532"])
    params['feature_mask_532_per'].valid_range = (0, 255)
    params['feature_mask_532_per'].dim_labels = ['Profile_ID', 'Altitude']

    # Add feature_mask_1064 to params
    params['feature_mask_1064'] = SDSData('Detection_Flags_1064',
                                          data_dict_2d_mcda_all_slices["Detection_Flags_1064"])
    params['feature_mask_1064'].valid_range = (0, 255)
    params['feature_mask_1064'].dim_labels = ['Profile_ID', 'Altitude']

    # Add feature_mask_merged to params
    params['feature_mask_merged'] = SDSData('Composite_Detection_Flags',
                                            data_dict_2d_mcda_all_slices["Composite_Detection_Flags"])
    params['feature_mask_merged'].valid_range = (0, 255)
    params['feature_mask_merged'].dim_labels = ['Profile_ID', 'Altitude']
    
    # Parameters saved for development
    if SAVE_DEVELOPMENT_DATA:
    
        # Add twoway_transmittance_532_par to params
        params['twoway_transmittance_532_par'] =\
            SDSData('Parallel_CumulativeTwoWayTransmittance_532',
                    data_dict_2d_mcda_dev["Parallel_CumulativeTwoWayTransmittance_532"], FILL_VALUE_FLOAT)
        params['twoway_transmittance_532_par'].valid_range = (0., 1.)
        params['twoway_transmittance_532_par'].dim_labels = ['Profile_ID', 'Altitude']
    
        # Add twoway_transmittance_532_per to params
        params['twoway_transmittance_532_per'] =\
            SDSData('Perpendicular_CumulativeTwoWayTransmittance_532',
                    data_dict_2d_mcda_dev["Perpendicular_CumulativeTwoWayTransmittance_532"], FILL_VALUE_FLOAT)
        params['twoway_transmittance_532_per'].valid_range = (0., 1.)
        params['twoway_transmittance_532_per'].dim_labels = ['Profile_ID', 'Altitude']
    
        # Add twoway_transmittance_1064 to params
        params['twoway_transmittance_1064'] =\
            SDSData('CumulativeTwoWayTransmittance_1064',
                    data_dict_2d_mcda_dev["CumulativeTwoWayTransmittance_1064"], FILL_VALUE_FLOAT)
        params['twoway_transmittance_1064'].valid_range = (0., 1.)
        params['twoway_transmittance_1064'].dim_labels = ['Profile_ID', 'Altitude']
    
        # Add feature_mask_532_par_steps to params
        params['feature_mask_532_par_steps'] =\
            SDSData('Parallel_Detection_Flags_532_steps',
                    data_dict_2d_mcda_dev["Parallel_Detection_Flags_532_steps"])
        params['feature_mask_532_par_steps'].valid_range = (0, 255)
        params['feature_mask_532_par_steps'].dim_labels = ['Step_532_par', 'Profile_ID', 'Altitude']
    
        # Add feature_mask_532_per_steps to params
        params['feature_mask_532_per_steps'] =\
            SDSData('Perpendicular_Detection_Flags_532_steps',
                    data_dict_2d_mcda_dev["Perpendicular_Detection_Flags_532_steps"])
        params['feature_mask_532_per_steps'].valid_range = (0, 255)
        params['feature_mask_532_per_steps'].dim_labels = ['Step_532_per', 'Profile_ID', 'Altitude']
    
        # Add feature_mask_1064_steps to params
        params['feature_mask_1064_steps'] =\
            SDSData('Detection_Flags_1064_steps', data_dict_2d_mcda_dev["Detection_Flags_1064_steps"])
        params['feature_mask_1064_steps'].valid_range = (0, 255)
        params['feature_mask_1064_steps'].dim_labels = ['Step_1064', 'Profile_ID', 'Altitude']
    
        # Add atsr_532_par_steps to params
        params['atsr_532_par_steps'] =\
            SDSData('Parallel_Attenuated_Scattering_Ratio_532_steps',
                    data_dict_2d_mcda_dev["Parallel_Attenuated_Scattering_Ratio_532_steps"], FILL_VALUE_FLOAT)
        params['atsr_532_par_steps'].dim_labels = ['Step_532_par', 'Profile_ID', 'Altitude']
    
        # Add atsr_532_per_steps to params
        params['atsr_532_per_steps'] =\
            SDSData('Perpendicular_Attenuated_Scattering_Ratio_532_steps',
                    data_dict_2d_mcda_dev["Perpendicular_Attenuated_Scattering_Ratio_532_steps"], FILL_VALUE_FLOAT)
        params['atsr_532_per_steps'].dim_labels = ['Step_532_per', 'Profile_ID', 'Altitude']
    
        # Add atsr_1064_steps to params
        params['atsr_1064_steps'] =\
            SDSData('Attenuated_Scattering_Ratio_1064_steps',
                    data_dict_2d_mcda_dev["Attenuated_Scattering_Ratio_1064_steps"], FILL_VALUE_FLOAT)
        params['atsr_1064_steps'].dim_labels = ['Step_1064', 'Profile_ID', 'Altitude']
    
    # Create folder to store output data
    granule_date_dict = split_granule_date(GRANULE_DATE)
    outdata_folder = os.path.join(OUT_FOLDER, f"2D_McDA.{VERSION_2D_McDA.replace('V', 'v')}",
                                  str(granule_date_dict['year']), f"{granule_date_dict['year']}_"
                                                                  f"{granule_date_dict['month']:02d}_"
                                                                  f"{granule_date_dict['day']:02d}")
    os.makedirs(outdata_folder, exist_ok=True)
    
    # Write in HDF file
    if (SLICE_START == 0) and (SLICE_END == None) and SLICE_START_END_TYPE == 'profindex':
        filename_end = '' # nothing, it is the whole file
    else:
        filename_end = f"_lon_{cal_l1.lon_min:.2f}_{cal_l1.lon_max:.2f}"
        
    filename = f"CAL_LID_L2_2D_McDA-{TYPE_2D_McDA}-{VERSION_2D_McDA.replace('.', '-')}." \
               f"{GRANULE_DATE}{filename_end}.hdf"
    write_hdf(outdata_folder+"/"+filename, params)
    
    
    print_time(tic_main_program)
    