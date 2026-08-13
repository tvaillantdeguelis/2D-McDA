"""Orchestrate the successive 2D-McDA feature-detection levels."""

import numpy as np

from twod_mcda.algorithm.attenuation import transmission_correction
from twod_mcda.algorithm.filtering import (
    FLAG_WEAK_SIGNAL,
    apply_surface_detection,
    apply_threshold,
    apply_window,
    average_below_8_2,
    fill_fully_attenuated,
    fill_likely_artifact,
    fill_small_strips,
    gaussian_2d_window,
    remove_detect_from_sr,
    replace_maybe,
    reput_low_confidence_flags,
)
from twod_mcda.algorithm.parameters import (
    FeatureDetectionParameters,
    get_feature_detection_coef,
)
from twod_mcda.caliop.constants import FILL_VALUE_FLOAT
from twod_mcda.utils.timing import timer


def detect_features(sr, sr_sigma, b_mol, temperature, surf_alt_index, channel):
    """Detect features in ATSR signal of lidar channel"""

    # Print channel
    print(f"\n\t***{channel}***")

    # Get feature detection parameters
    params = FeatureDetectionParameters(channel)

    # Initialization
    feature_dict = {}
    sr_dict = {}
    step = 0
    sr_dict[step] = np.ma.copy(sr)
    last_sr = step
    feature_dict[step] = np.ma.zeros(sr.shape, dtype=np.uint8)
    twoway_transmittance_array = np.ma.ones(sr.shape)*FILL_VALUE_FLOAT
    last_feature = step
    FLAG_DETECTION_LEVEL = 1 # incremented after each detection level: 1, 2,...
    ############################################
    #### Put 'Surface' flag on feature mask ####
    print("\t=> Put 'Surface' flag on feature mask...")
    with timer("Put 'Surface' flag on feature mask"):
        step += 1 
        feature_dict[step] = apply_surface_detection(feature_dict[last_feature], surf_alt_index)
        last_feature = step


    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...")
    with timer("Remove detected pixel from ATSR"):
        step += 1
        sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
        last_sr = step


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 1 ####
    print("\t=> Detection level 1:")
    with timer("Detection level 1"):
        # If 532 nm par or 532 nm per channel
        if (channel == '532_par') | (channel == '532_per'):

            # Get detection coefficients
            k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

            #####################################################################
            #### Apply threshold to get very high echo (likely PMT artifact) ####
            print("\t\t- Apply threshold to get very high echo (likely PMT artifact)...")
            with timer("Apply threshold to get very high echo (likely PMT artifact)"):
                step += 1
                feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr],
                                                    sr_sigma)
                last_feature = step

            #######################################
            #### Replace 'maybe' by 'detected' ####
            print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels meet neighbors number "
                "limit condition...")
            with timer("Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels meet neighbors number limit condition"):
                step += 1
                feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
                last_feature = step


            ################################
            #### Flag 'Likely artifact' ####
            print("\t\t- Flag 'Likely Artifact' below those high signal to some extent...")
            with timer("Flag 'Likely Artifact' below those high signal to some extent"):
                step += 1
                feature_dict[step] = fill_likely_artifact(params, feature_dict[last_feature],
                                                        FLAG_DETECTION_LEVEL)
                last_feature = step

        # If 1064 nm, not used
        elif channel == '1064':
            print("\t\tNot used")
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 2 ####
    print("\t=> Detection level 2:")
    with timer("Detection level 2"):

        # Increase FLAG_DETECTION_LEVEL
        FLAG_DETECTION_LEVEL += 1

        # Get detection coefficients
        k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

        ###################################################
        #### Apply threshold to get the 'maybe' pixels ####
        print("\t\t- Apply threshold...")
        with timer("Apply threshold"):
            step += 1
            feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr], sr_sigma)
            last_feature = step


        #######################################
        #### Replace 'maybe' by 'detected' ####
        print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels "\
            "meet neighbors number limit condition...")
        with timer("Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels meet neighbors number limit condition"):
            step += 1
            feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
            last_feature = step

    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 3 ####
    print("\t=> Detection level 3:")
    with timer("Detection level 3"):

        # Increase FLAG_DETECTION_LEVEL
        FLAG_DETECTION_LEVEL += 1

        # Get detection coefficients
        k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

        ###################################################
        #### Apply threshold to get the 'maybe' pixels ####
        print("\t\t- Apply threshold...")
        with timer("Apply threshold"):
            step += 1
            feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr], sr_sigma)
            last_feature = step


        #########################################
        #### Windowing on the 'maybe' pixels ####
        print("\t\t- Windowing on the 'maybe' pixels...")
        with timer("Windowing on the 'maybe' pixels"):
            step += 1
            feature_dict[step] = apply_window(s[0], s[1], feature_dict[last_feature], FLAG_DETECTION_LEVEL)
            last_feature = step


        #######################################
        #### Replace 'maybe' by 'detected' ####
        print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels "\
            "meet neighbors number limit condition...")
        with timer("Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels meet neighbors number limit condition"):
            step += 1
            feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
            last_feature = step

    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 4 ####
    print("\t=> Detection level 4:")
    with timer("Detection level 4"):
        # Increase FLAG_DETECTION_LEVEL
        FLAG_DETECTION_LEVEL += 1

        # Get detection coefficients
        k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

        ###################################################
        #### Apply threshold to get the 'maybe' pixels ####
        print("\t\t- Apply threshold...")
        with timer("Apply threshold"):
            step += 1
            feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr], sr_sigma)
            last_feature = step


        #########################################
        #### Windowing on the 'maybe' pixels ####
        print("\t\t- Windowing on the 'maybe' pixels...")
        with timer("Windowing on the 'maybe' pixels"):
            step += 1
            feature_dict[step] = apply_window(s[0], s[1], feature_dict[last_feature], FLAG_DETECTION_LEVEL)
            last_feature = step


        #######################################
        #### Replace 'maybe' by 'detected' ####
        print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels "\
            "meet neighbors number limit condition...")
        with timer("Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels meet neighbors number limit condition"):
            step += 1
            feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
            last_feature = step

    #--------------------------------------------------------------------------
    

    #####################################################################
    #### Fill fully attenuated from lowest altitude to first feature ####
    print("\t=> Flag 'Fully Attenuated' from lowest altitude to first "\
          "feature...")
    with timer("Flag 'Fully Attenuated' from lowest altitude to first feature"):
        step += 1
        feature_dict[step] = fill_fully_attenuated(feature_dict[last_feature])
        last_feature = step



    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...")
    with timer("Remove detected pixel from ATSR"):
        step += 1
        sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
        last_sr = step


    ##############################
    #### Average below 8.2 km ####
    print("\t=> Average below 8.2 km as between 8.2 km and 20.2 km "\
          "(60 m × 1 km)...")
    with timer("Average below 8.2 km as between 8.2 km and 20.2 km (60 m × 1 km)"):
        step += 1
        # Note: sr_sigma needs to be modified below 8.2 km
        sr_dict[step], sr_sigma = average_below_8_2(sr_dict[last_sr], sr_sigma)
        last_sr = step


    ########################################################
    #### Flag almost FA where lidar signal is very weak ####
    print("\t=> Flag 'almost FA' where lidar signal is very weak...")
    with timer("Flag 'almost FA' where lidar signal is very weak"):
        step += 1
        feature_dict[step] = FLAG_WEAK_SIGNAL(params, feature_dict[last_feature], sr_dict[last_sr],
                                            sr_sigma)
        last_feature = step


    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...")
    with timer("Remove detected pixel from ATSR"):
        step += 1
        sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
        last_sr = step


    ############################################################
    #### Correct sr signal below feature from transmittance ####
    print("\t=> Correct sr signal below feature from transmittance using "\
        f"fixed lidar ratio above and below {params.temp_ice_liquid} °C...")
    with timer("Correct sr signal below feature from transmittance using fixed lidar ratio above and below {params.temp_ice_liquid} °C"):
        step += 1
        sr_dict[step], twoway_transmittance_array[:, :] =\
            transmission_correction(sr_dict[last_sr], sr, b_mol, feature_dict[last_feature],
                                    temperature, params)
        last_sr = step



    ########################################################
    #### Fill small strip between FA where strip < nb_prof_min prof ####
    print("\t=> Fill small strip between FA where strip < nb_prof_min prof...", 
          end='')
    with timer("Fill small strip between FA where strip < nb_prof_min prof"):
        step += 1
        feature_dict[step] = fill_small_strips(params, feature_dict[last_feature])
        last_feature = step
        last_feature_before_averaging = step


    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...")
    with timer("Remove detected pixel from ATSR"):
        step += 1
        sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
        last_sr = step


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 5 ####
    print("\t=> Detection level 5:")
    with timer("Detection level 5"):

        # Increase FLAG_DETECTION_LEVEL
        FLAG_DETECTION_LEVEL += 1

        # Get detection coefficients
        k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

        ###########################################################
        #### Apply a gaussian horizontal line window averaging ####
        print("\t\t- Apply a gaussian horizontal line window averaging...")
        with timer("Apply a gaussian horizontal line window averaging"):
            step += 1
            # sr_dict[step], sr_sigma = gaussian_line_window(a[0], a[1], sr_dict[last_sr],
            #                                                feature_dict[last_feature], sr_sigma)
            sr_dict[step], sr_sigma = gaussian_2d_window(a[0], a[1], sr_dict[last_sr],
                                                        feature_dict[last_feature], sr_sigma)
            last_sr = step


        ###################################################
        #### Apply threshold to get the 'maybe' pixels ####
        print("\t\t- Apply threshold...")
        with timer("Apply threshold"):
            step += 1
            feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr], sr_sigma)
            last_feature = step


        #########################################
        #### Windowing on the 'maybe' pixels ####
        print("\t\t- Windowing on the 'maybe' pixels...")
        with timer("Windowing on the 'maybe' pixels"):
            step += 1
            feature_dict[step] = apply_window(s[0], s[1], feature_dict[last_feature], FLAG_DETECTION_LEVEL)
            last_feature = step



        #######################################
        #### Replace 'maybe' by 'detected' ####
        print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels "\
            "meet neighbors number limit condition...")
        with timer("Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels meet neighbors number limit condition"):
            step += 1
            feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
            last_feature = step

    #--------------------------------------------------------------------------


    ##########################################################################
    #### Reput all not confident flags where overwritten during averaging ####
    print("\t=> Reput all not confident flags where overwritten during "\
          "averaging...")
    with timer("Reput all not confident flags where overwritten during averaging"):
        step += 1
        feature_dict[step] = reput_low_confidence_flags(feature_dict[last_feature],
                                                        feature_dict[last_feature_before_averaging])
        last_feature = step



    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...")
    with timer("Remove detected pixel from ATSR"):
        step += 1
        sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
        last_sr = step

    ############################################################
    #### Transform feature and sr dictionaries to 3D arrays ####
    print("\t=> Transform feature and sr dictionaries to 3D arrays...")
    with timer("Transform feature and sr dictionaries to 3D arrays"):
        # Initialization
        feature_array_steps = np.ma.zeros((step+1, sr.shape[0], sr.shape[1]), dtype=np.uint8)
        sr_array_steps = np.ma.ones((step+1, sr.shape[0], sr.shape[1]))*FILL_VALUE_FLOAT

        # Transform dictionaries to arrays
        for i_step in np.arange(step+1):
            # Test if feature_dict has a i_step
            try:
                feature_dict[i_step]
            # If not go directly to next step
            except:
                continue
            feature_array_steps[i_step, :, :] = feature_dict[i_step]

        for i_step in np.arange(step+1):
            # Test if sr_dict has a i_step
            try:
                sr_dict[i_step]
            # If not go directly to next step
            except:
                continue   
            sr_array_steps[i_step, :, :] = sr_dict[i_step]


    return feature_dict[last_feature], feature_array_steps, sr_array_steps,\
           twoway_transmittance_array
