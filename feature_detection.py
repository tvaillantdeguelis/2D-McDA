#!/usr/bin/env python
# coding: utf8

from datetime import datetime
import numpy as np
import os
import sys
from numba import jit
import matplotlib.pyplot as plt

from my_modules.standard_outputs import print_elapsed_time
from my_modules.calipso_constants import *

from config import *

def apply_surface_detection(feature, surf_alt_index):
    """Put FLAG_SURFACE where and below the surface was detected"""

    # Initialization
    nb_prof = feature.shape[0]
    new_feature = np.ma.copy(feature)

    # Loop on profiles
    for i in np.arange(nb_prof):
        # If surface detected
        if surf_alt_index[i] != 999:
            # Put flag from lowest bin to surface altitude
            new_feature[i, :surf_alt_index[i]+1] = FLAG_SURFACE

    return new_feature


def apply_threshold(k, feature, sr, sr_sigma, where_FA=False):
    """Put FLAG_MAYBE where signal above threshold"""

    # Initialization
    new_feature = np.ma.copy(feature)

    # Define threshold
    sr_maybe = 1 + k*sr_sigma

    if where_FA:
        # Put flag where ATSR > threshold and where ATSR is not masked
        new_feature[np.ma.where(sr>sr_maybe)] = FLAG_MAYBE
    else:
        # Put flag where ATSR > threshold and where feature is still "nothing"
        new_feature[np.ma.where((sr>sr_maybe) &\
                            (new_feature==FLAG_NOTHING))] = FLAG_MAYBE

    return new_feature


@jit(nopython=True)
def apply_window_jit(w_side, h_side, feature, nb_pixels_window, min_percent, detected_pixels,
                     FLAG_DETECTION_LEVEL):
    """Part extracted from apply_window function for faster processing with
    @jit"""

    for i in np.arange(w_side, feature.shape[0]-w_side):
        for j in np.arange(h_side, feature.shape[1]-h_side):
            if (feature[i, j]==FLAG_NOTHING) | (feature[i, j]==FLAG_MAYBE):
                # Tuple with indexes of the window
                window = (slice(i-w_side, i+w_side+1), 
                          slice(j-h_side, j+h_side+1))

                # Count nb of "maybe"
                nb_maybe = list(feature[window].flatten()).count(FLAG_MAYBE)

                # Count nb of "previous detection level" (d-1)
                nb_detected_1 = 0
                if FLAG_DETECTION_LEVEL > 1: # if previous detection exists
                    prev_FLAG_DETECTION_LEVEL = FLAG_DETECTION_LEVEL-1
                    nb_detected_1 = list(feature[window].flatten()).count(prev_FLAG_DETECTION_LEVEL)
                
                # Total detected at n and n-1
                nb_tot = nb_maybe + nb_detected_1

                # Count nb of special (and detection <= d-2) and remove 
                # from nb_pixels_window
                nb_surface = list(feature[window].flatten()).count(FLAG_SURFACE)
                nb_likely_artifact = list(feature[window].flatten()).count(FLAG_LIKELY_ARTIFACT)
                nb_FA = list(feature[window].flatten()).count(FLAG_FA)
                nb_AFA = list(feature[window].flatten()).count(FLAG_AFA)
                nb_small_strips = list(feature[window].flatten()).count(FLAG_SMALL_STRIPS)
                nb_detected_2_and_before = 0
                if FLAG_DETECTION_LEVEL > 1: # if previous detection exists
                    for prev_FLAG_DETECTION_LEVEL in np.arange(1, FLAG_DETECTION_LEVEL-1):
                        nb_detected_2_and_before += list(feature[window].flatten()).\
                                      count(prev_FLAG_DETECTION_LEVEL)
                nb_pixels_window_2 = nb_pixels_window - nb_FA - nb_AFA - nb_surface - \
                                     nb_likely_artifact - nb_small_strips - nb_detected_2_and_before

                # Flag detected if amount above limit
                nb_min_tot = nb_pixels_window_2*min_percent
                if (nb_tot >= nb_min_tot):
                    detected_pixels[i, j] = 1
    
    return feature


def apply_window(height_window, width_window, feature, FLAG_DETECTION_LEVEL, min_percent=0.5):
    # min_percent: min pourcentage of total counted pixels in the window to flag the center as "detected"

    # Initialization
    new_feature = np.ma.copy(feature)

    # height_window and width_window should be odd numbers
    if (height_window%2 != 1) | (width_window%2 != 1):
        sys.exit(f"height_window (= {height_window}) and width_window "\
                 f"(= {width_window}) should be odd numbers")

    # Initialization
    detected_pixels = np.zeros(new_feature.shape, dtype=bool)
    nb_pixels_window = width_window * height_window
    h_side = int(height_window/2) # nb of pixel each side of the center
    w_side = int(width_window/2) # nb of pixel each side of the center

    # Apply moving window
    new_feature = apply_window_jit(w_side, h_side, new_feature, nb_pixels_window, min_percent,
                                   detected_pixels, FLAG_DETECTION_LEVEL)

    # Remove previous "maybe" pixels (or not if keep_all==True)
    new_feature[new_feature==FLAG_MAYBE] = FLAG_NOTHING

    # Replace by those which result from the windowing
    new_feature[detected_pixels==1] = FLAG_MAYBE

    return new_feature


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


@jit(nopython=True)
def replace_maybe_jit(nb_lim, feature, seen_pixels, FLAG_DETECTION_LEVEL,
                      prev_detect, prevprev_detect):
    """Part extracted from replace_maybe function for faster processing with @jit"""

    for i in np.arange(feature.shape[0]):
        for j in np.arange(feature.shape[1]):
            if not seen_pixels[i, j]:
                if feature[i, j] == FLAG_MAYBE:
                    connected_to_detected_pattern = False                      
                    # Count neighbors
                    accessible_pixels = [(i, j)]
                    pattern_pixels = np.zeros(feature.shape)
                    pattern_pixels[i, j] = True
                    while (len(accessible_pixels) != 0):
                        p = accessible_pixels[0] # 1st pixel of the list
                        accessible_pixels = accessible_pixels[1:] # Remove 1st
                        #-----------------------------------------------
                        # if FA/low confidence on the left, check on the 
                        # left of FA/low confidence if pattern connect 
                        # to an already detected pattern
                        i_FA_left = 1
                        while (i - i_FA_left >= 1) &\
                              (feature[p[0]-i_FA_left, p[1]] == FLAG_FA) |\
                              (feature[p[0]-i_FA_left, p[1]] == FLAG_AFA) |\
                              (feature[p[0]-i_FA_left, p[1]] == FLAG_SMALL_STRIPS):
                            i_FA_left +=1
                        if (i - i_FA_left >= 0) &\
                           (feature[p[0]-i_FA_left, p[1]] == FLAG_DETECTION_LEVEL):
                            connected_to_detected_pattern = True
                        #-----------------------------------------------
                        if not seen_pixels[p]:
                            seen_pixels[p] = True # We note that we see this 
                                                  # pixel
                            v = neighbors(feature.shape, p) # Get pixel 
                                                            # neighbors 
                            # Look for neighbors
                            for voisin in v:
                                c1 = seen_pixels[voisin]
                                c2 = feature[voisin] == FLAG_MAYBE # level n
                                c3 = False
                                if FLAG_DETECTION_LEVEL > 1: # if previous
                                                           # detection exists
                                    if prev_detect:
                                        c3 = feature[voisin]==FLAG_DETECTION_LEVEL-1
                                                                  # level n - 1
                                    else:
                                        c3 == False
                                    if prevprev_detect:
                                        c4 = feature[voisin]==FLAG_DETECTION_LEVEL-2
                                                                  # level n - 2
                                    else:
                                        c4== False
                                if (not c1) & (c2 | c3 | c4):
                                    accessible_pixels.append(voisin)
                                    pattern_pixels[voisin] = 1
                    
                    count = np.int(np.round(np.sum(pattern_pixels)))
                    # remove the too small "maybe pattern"
                    px, py = np.where(pattern_pixels==1)
                    for pix_i in np.arange(px.size):
                        # unless already classify
                        if feature[px[pix_i], py[pix_i]] == FLAG_MAYBE:
                            if (count < nb_lim) &\
                               (not connected_to_detected_pattern):
                                # replace "maybe" by "nothing"
                                feature[px[pix_i], py[pix_i]] = FLAG_NOTHING
                            else:
                                # replace "maybe" by the next level of detection
                                feature[px[pix_i], py[pix_i]] = FLAG_DETECTION_LEVEL

    return feature


def replace_maybe(n, feature, FLAG_DETECTION_LEVEL, prev_detect=True,
                  prevprev_detect=False):
    """Put flag 'FLAG_DETECTION_LEVEL' where patterns of connected 'FLAG_MAYBE'
    pixels consist of more than n pixels
    if prev_detect=True means that we also count detection pixels n-1
    if prevprev_detect=True means that we also count detection pixels n-2"""

    # Initialization
    new_feature = np.ma.copy(feature)

    # Initialization
    seen_pixels = np.zeros(new_feature.shape, dtype=bool)
    
    # Look for a "maybe" pixel and decide if it's really part of a pattern
    # based on nb of neighbors (neighbors in "level n" + "level n-1")
    if n == 1: # keep all
        new_feature[new_feature==FLAG_MAYBE] = FLAG_DETECTION_LEVEL
    else:
        new_feature = replace_maybe_jit(n, new_feature, seen_pixels, FLAG_DETECTION_LEVEL,
                                        prev_detect, prevprev_detect)

    return new_feature


def fill_likely_artifact(params, feature, FLAG_VERY_HIGH_ECHO):
    """Put flag "Likely artifact" below high signal points"""

    # Initialization
    nb_alt = feature.shape[1]
    nb_prof = feature.shape[0]
    new_feature = np.ma.copy(feature)

    # Loop on profiles
    for i in range(nb_prof):
        # From bottom to top
        for j in np.arange(nb_alt):
            # Look for FLAG_VERY_HIGH_ECHO
            if new_feature[i, j] == FLAG_VERY_HIGH_ECHO:
                # If FLAG_VERY_HIGH_ECHO at the very bottom
                if j==0: 
                    # Nothing to flag below
                    continue
                # If bin below is FLAG_VERY_HIGH_ECHO
                elif new_feature[i, j-1] == FLAG_VERY_HIGH_ECHO:
                    # Same layer, already done
                    continue
                # Else, flag below on the params.nb_bins_PMT_artifact extent
                else:
                    # Go down
                    j2 = j - 1
                    while (j2 >= 0) & (j - j2 <= params.nb_bins_PMT_artifact) &\
                          (new_feature[i, j2] == FLAG_NOTHING):
                        new_feature[i, j2] = FLAG_LIKELY_ARTIFACT
                        j2 -= 1 

    return new_feature


def reput_low_confidence_flags(feature, feature_before_av):
    """Reput all not confident flags where overwritten during averaging"""

    # Initialization
    new_feature = np.copy(feature)

    # Reput FA and AFA where they were
    new_feature[feature_before_av == FLAG_FA] = FLAG_FA
    new_feature[feature_before_av == FLAG_AFA] = FLAG_AFA
    new_feature[feature_before_av == FLAG_SMALL_STRIPS] = FLAG_SMALL_STRIPS
    new_feature[feature_before_av == FLAG_LIKELY_ARTIFACT] = FLAG_LIKELY_ARTIFACT

    return new_feature


def fill_fully_attenuated(feature):
    """Fill with flag 'Fully Attenuated' from lowest altitude to first feature"""

    # Initialization
    nb_prof = feature.shape[0]
    nb_alt = feature.shape[1]
    new_feature = np.ma.copy(feature)

    # Loop on profiles
    for i in np.arange(nb_prof):
        # If surface detected
        if new_feature[i, 0] == FLAG_SURFACE:
            # No 'Fully Attenuated' here
            continue
        # If surface not detected
        else:
            # From lowest altitude go up until reaching a layer
            j = 0
            # While layer not reached
            while ((new_feature[i, j] == FLAG_NOTHING) |\
                   (new_feature[i, j] == FLAG_LIKELY_ARTIFACT)) & (j < nb_alt):
                # Flag 'Fully Attenuated'
                new_feature[i, j] = FLAG_FA
                j += 1
                # If reach top (30.1 km)
                if j >= nb_alt:
                    # Remove all FA in the profile
                    new_feature[i, :] = FLAG_NOTHING
                    # And stop
                    break

    return new_feature


def FLAG_WEAK_SIGNAL(params, feature, sr, sr_sigma):
    """Flag where, between detected layers, more than ratio_nb are below 
    sr_thresold"""

    # Initialization
    sr_thresold = sr_sigma*params.weak_signal_ratio_threshold
    nb_prof = feature.shape[0]
    nb_alt = feature.shape[1]
    new_feature = np.ma.copy(feature)

    # Loop on profiles
    for i in np.arange(nb_prof):
        # From lowest altitude go up
        nb_below = 0 # nb ranges below threshold
        nb_tot = 0 # total nb ranges in the region between 2 layers
        j = 0
        # While top not reached
        while j < nb_alt: 
            if new_feature[i, j] != FLAG_NOTHING:
                j += 1 # not yet in region with no detection
                continue
            cs_min_index = j # min index of the "CS" region
            while new_feature[i, j] == FLAG_NOTHING:
                nb_tot +=1
                if sr[i,j] < sr_thresold[i,j]:
                    nb_below += 1
                j += 1
                if (j >= nb_alt): # if reach top of column
                    break
            cs_max_index = j-1 # max index of the "CS" region
            # If fraction_nb_below_threshold below limit put flag in this region
            if nb_below/nb_tot > params.weak_signal_ratio:
                if cs_max_index < nb_alt - 1: # not if no layer above
                    new_feature[i, cs_min_index:cs_max_index+1] = FLAG_AFA
            nb_below = 0
            nb_tot = 0

    return new_feature


@jit(nopython=True)
def fill_small_strips_jit(feature, nb_prof_min):
    """Part extracted from fill_small_strips function for faster processing 
    with @jit"""

    # Initialization
    nb_prof = feature.shape[0]
    nb_alt = feature.shape[1]

    # Loop on profiles
    for i in np.arange(nb_prof):
        # From bottom go up
        for j in np.arange(nb_alt):
            # If not the right end of the image
            if i < nb_prof - 3: 
                # If (A)FA, Likely artifact at prof i but not at prof i+1
                if ( (feature[i,j] == FLAG_FA) |\
                     (feature[i,j] == FLAG_AFA) |\
                     (feature[i,j] == FLAG_LIKELY_ARTIFACT) ) &\
                   ( (feature[i+1,j] != FLAG_FA) &\
                     (feature[i+1,j] != FLAG_AFA) &\
                     (feature[i+1,j] != FLAG_LIKELY_ARTIFACT) ):
                    # Look right if there is (A)FA or Likely artifact at 
                    # less than nb_prof_min
                    i2 = i + 1
                    nb_prof_strip = 1
                    less_than_nb_prof_min = False
                    while (nb_prof_strip <= nb_prof_min) & (i2 <= nb_prof-1):
                        if (feature[i2,j] == FLAG_FA) |\
                           (feature[i2,j] == FLAG_AFA) |\
                           (feature[i2,j] == FLAG_LIKELY_ARTIFACT):
                            less_than_nb_prof_min = True
                            break
                        i2 += 1
                        nb_prof_strip += 1
                    # If there is (A)FA at less than nb_prof_min far
                    if less_than_nb_prof_min:
                        # Put "Low confidence small strips" flag between
                        feature[i+1:i2, j][feature[i+1:i2, j] == FLAG_NOTHING] =\
                            FLAG_SMALL_STRIPS

    return feature


def fill_small_strips(params, feature):
    """Flag 'Low confidence small strips' where strips of signal between 
    (A)FA or 'Likely Artiafct' are horizontally less than 'nb_prof_min' 
    profiles"""

    # Initialization
    new_feature = np.ma.copy(feature)  

    new_feature = fill_small_strips_jit(new_feature, params.nb_prof_min_small_strips)

    return new_feature


def remove_detect_from_sr(sr, feature):
    """Remove detected pixel from the ATSR signal"""

    # Mask where not "nothing"
    new_sr = np.ma.masked_where(feature != FLAG_NOTHING, sr)

    return new_sr


def average_below_8_2(sr, sr_sigma):
    """Average below 8.2 km as between 8.2 km and 20.2 km (60 m × 1 km)"""

    # Initialization
    new_sr = np.ma.copy(sr)  
    nb_prof = sr.shape[0] 
    nb_bins_below_8_2km = N_30M_BINS_PER_BIN_R1*N_BINS_R1+N_30M_BINS_PER_BIN_R2*N_BINS_R2

    # Look for horizontal offset if 1st profile not the start of a 1-km profile
    index_vertical_bin = 100 # random bin in the R2 region 
    if sr[0, index_vertical_bin] == sr[1, index_vertical_bin]:
        if sr[1, index_vertical_bin] == sr[2, index_vertical_bin]:
            offset_h = 0
        else:
            offset_h = 2
    else:
        offset_h = 1

    # Average 60 m × 1 km (2 verticals × 3 horizontals)
    i_array = np.arange(offset_h, nb_prof-2, 3) # 3 horizontals
    j_array = np.arange(0, nb_bins_below_8_2km, 2) # 2 verticals
    i_progress = 0
    for i in i_array:
        for j in j_array:
            new_sr[i:i+3, j:j+2] = np.ma.mean(sr[i:i+3, j:j+2])
    
    # Remask where was already masked
    new_sr.mask = np.copy(sr.mask)

    # Adapt SR threshold below 8.2 km
    sr_sigma[:nb_bins_below_8_2km] = sr_sigma[:nb_bins_below_8_2km]/np.sqrt(6)

    return new_sr, sr_sigma


def transmission_correction(sr, sr_init, b_mol, feature, temperature, params):
    """Correct sr signal below feature from transmittance"""

    # Initialization
    new_sr = np.ma.copy(sr)
    twoway_transmittance_array = np.ma.ones(feature.shape)*FILL_VALUE_FLOAT
    nb_alt = new_sr.shape[1]
    nb_prof = new_sr.shape[0]  

    # Loop on profiles
    for i in range(nb_prof):

        # Initialization
        int_atb = 0
        twoway_transmittance = 1
        twoway_transmittance_current_layer = 1
        reenter_nothing = True

        # From highest altitude go down
        for j in np.arange(nb_alt-1, -1, -1):

            if feature[i, j] == FLAG_NOTHING:

                # Add (multiply) transmittance of the layer to those already detected
                if reenter_nothing:
                    twoway_transmittance *= twoway_transmittance_current_layer
                    reenter_nothing = False

                # Reinitialize for next layer
                int_atb = 0

                # Save in an array for plot
                twoway_transmittance_array[i, j] = twoway_transmittance

                # Don't use twoway_transmittance below limit
                twoway_transmittance = max(twoway_transmittance, params.twoway_transmittance_limit)

                # Correct from transmission of layers above
                new_sr[i, j] = new_sr[i, j]/twoway_transmittance # masked value not affected

            elif (feature[i, j] == FLAG_SURFACE) | (feature[i, j] == FLAG_FA):
                break

            else:
                int_atb += (sr_init[i, j] -1)*b_mol[i, j]*0.030 # integrate (R' -1)*beta_m
                if temperature[i, j] < params.temp_ice_liquid: # at cloud base
                    S = params.S_ice
                else:
                    S = params.S_liquid
                twoway_transmittance_current_layer = 1-2*params.mult_scatt*int_atb*S
                reenter_nothing = True

    # Mask where feature
    twoway_transmittance_array =\
        np.ma.masked_where(twoway_transmittance_array==FILL_VALUE_FLOAT, twoway_transmittance_array)

    return new_sr, twoway_transmittance_array


@jit(nopython=True)
def gaussian_line_window_jit(nb_prof, nb_alt, width_window, gauss_sigma, sr, 
                             new_sr, fill_value, feature):
    """Part extracted from gaussian_line_window function for faster processing 
    with @jit"""

    # Loop on profiles
    for i in np.arange(nb_prof):
        # From bottom go up
        for j in np.arange(nb_alt):
            # Apply n-elements line gaussian sliding window
            nside = np.int((width_window-1)/2)
            x = np.arange(width_window) - nside
            gaussian = np.exp(-x**2/(2*gauss_sigma**2))
            nb_prof_averaged = np.sum(gaussian)
            if sr[i, j] != fill_value:
                # Horizontal averaging (if not left/right edge)
                if not (i < nside) | (i > nb_prof - (nside+1)):
                    line = sr[i-nside:i+(nside+1), j]
                    gaussian = gaussian[line!=fill_value] # first
                    line = line[line!=fill_value] # then (in this order!)
                    new_sr[i, j] = np.sum(line*gaussian)/np.sum(gaussian)
            # Also apply window where special flags (FA, AFA, likely artifact, 
            # and no confidence)
            elif (feature[i,j] == FLAG_FA) | (feature[i,j] == FLAG_AFA) |\
                 (feature[i,j] == FLAG_LIKELY_ARTIFACT) |\
                 (feature[i,j] == FLAG_SMALL_STRIPS):
                if not (i < nside) | (i > nb_prof - (nside+1)):
                    line = sr[i-nside:i+(nside+1), j]
                    no_special_flag = line!=fill_value
                    nb_no_special_flag = np.sum(no_special_flag)
                    if nb_no_special_flag != 0: # if not all FA or AFA
                        gaussian = gaussian[line!=fill_value] # first
                        line = line[line!=fill_value] # then (in this order!)
                        new_sr[i, j] = np.sum(line*gaussian)/np.sum(gaussian)

    return new_sr, nb_prof_averaged


def gaussian_line_window(width_window, gauss_sigma, sr, feature, sr_sigma):
    """Apply a horizontal gaussian averaging window to the ATSR signal"""

    # Initialization
    nb_prof = sr.shape[0]
    nb_alt = sr.shape[1]
    sr2 = np.ma.copy(sr)
    sr2 = sr2.filled(FILL_VALUE_FLOAT) # fill mask value to use jit
    new_sr = np.ma.ones(sr.shape)*FILL_VALUE_FLOAT

    # width_window should be odd numbers
    if width_window%2 != 1:
        sys.exit(f"width_window (= {width_window}) should be odd numbers")

    # Apply gaussian line averaging
    new_sr, nb_prof_averaged = gaussian_line_window_jit(nb_prof, nb_alt, width_window, gauss_sigma,
                                                        sr2, new_sr, FILL_VALUE_FLOAT, feature)
    
    # Mask where FILL_VALUE_FLOAT
    new_sr[new_sr==FILL_VALUE_FLOAT] = np.ma.masked

    # Adapt SR threshold
    sr_sigma = sr_sigma/np.sqrt(nb_prof_averaged)

    return new_sr, sr_sigma


# **********************************************************************
# MAIN FUNCTION
# **********************************************************************
def detect_features(sr, sr_sigma, b_mol, temperature, surf_alt_index, channel):
    """Detect features in ATSR signal of lidar channel"""

    tic_function = datetime.now()

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
    last_strong_level    = 4 # for strong/weak figure
    last_weak_level      = 5 # for strong/weak figure
    if channel == '1064':
        last_strong_level -= 1
        last_weak_level -= 1


    ############################################
    #### Put 'Surface' flag on feature mask ####
    print("\t=> Put 'Surface' flag on feature mask...", end='')
    step += 1 
    feature_dict[step] = apply_surface_detection(feature_dict[last_feature], surf_alt_index)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic_function)


    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...", end='')
    step += 1
    sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
    last_sr = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 1 ####
    print("\t=> Detection level 1:")

    # If 532 nm par or 532 nm per channel
    if (channel == '532_par') | (channel == '532_per'):

        # Get detection coefficients
        k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

        #####################################################################
        #### Apply threshold to get very high echo (likely PMT artifact) ####
        print("\t\t- Apply threshold to get very high echo (likely PMT artifact)...", end='')
        step += 1
        feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr],
                                             sr_sigma)
        last_feature = step

        # Show elapsed time
        tic = print_elapsed_time(tic)


        #######################################
        #### Replace 'maybe' by 'detected' ####
        print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels meet neighbors number "
              "limit condition...", end='')
        step += 1
        feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
        last_feature = step

        # Show elapsed time
        tic = print_elapsed_time(tic)


        ################################
        #### Flag 'Likely artifact' ####
        print("\t\t- Flag 'Likely Artifact' below those high signal to some extent...", end='')
        step += 1
        feature_dict[step] = fill_likely_artifact(params, feature_dict[last_feature],
                                                  FLAG_DETECTION_LEVEL)
        last_feature = step

        # Show elapsed time
        tic = print_elapsed_time(tic)

    # If 1064 nm, not used
    elif channel == '1064':
        print("\t\tNot used")
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 2 ####
    print("\t=> Detection level 2:")

    # Increase FLAG_DETECTION_LEVEL
    FLAG_DETECTION_LEVEL += 1

    # Get detection coefficients
    k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

    ###################################################
    #### Apply threshold to get the 'maybe' pixels ####
    print("\t\t- Apply threshold...", end='')
    step += 1
    feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr], sr_sigma)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #######################################
    #### Replace 'maybe' by 'detected' ####
    print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels "\
          "meet neighbors number limit condition...", end='')
    step += 1
    feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 3 ####
    print("\t=> Detection level 3:")

    # Increase FLAG_DETECTION_LEVEL
    FLAG_DETECTION_LEVEL += 1

    # Get detection coefficients
    k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

    ###################################################
    #### Apply threshold to get the 'maybe' pixels ####
    print("\t\t- Apply threshold...", end='')
    step += 1
    feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr], sr_sigma)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #########################################
    #### Windowing on the 'maybe' pixels ####
    print("\t\t- Windowing on the 'maybe' pixels...", end='')
    step += 1
    feature_dict[step] = apply_window(s[0], s[1], feature_dict[last_feature], FLAG_DETECTION_LEVEL)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #######################################
    #### Replace 'maybe' by 'detected' ####
    print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels "\
          "meet neighbors number limit condition...", end='')
    step += 1
    feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 4 ####
    print("\t=> Detection level 4:")

    # Increase FLAG_DETECTION_LEVEL
    FLAG_DETECTION_LEVEL += 1

    # Get detection coefficients
    k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

    ###################################################
    #### Apply threshold to get the 'maybe' pixels ####
    print("\t\t- Apply threshold...", end='')
    step += 1
    feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr], sr_sigma)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #########################################
    #### Windowing on the 'maybe' pixels ####
    print("\t\t- Windowing on the 'maybe' pixels...", end='')
    step += 1
    feature_dict[step] = apply_window(s[0], s[1], feature_dict[last_feature], FLAG_DETECTION_LEVEL)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #######################################
    #### Replace 'maybe' by 'detected' ####
    print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels "\
          "meet neighbors number limit condition...", end='')
    step += 1
    feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)
    #--------------------------------------------------------------------------
    

    #####################################################################
    #### Fill fully attenuated from lowest altitude to first feature ####
    print("\t=> Flag 'Fully Attenuated' from lowest altitude to first "\
          "feature...", end='')
    step += 1
    feature_dict[step] = fill_fully_attenuated(feature_dict[last_feature])
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...", end='')
    step += 1
    sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
    last_sr = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    ##############################
    #### Average below 8.2 km ####
    print("\t=> Average below 8.2 km as between 8.2 km and 20.2 km "\
          "(60 m × 1 km)...", end='')
    step += 1
    # Note: sr_sigma needs to be modified below 8.2 km
    sr_dict[step], sr_sigma = average_below_8_2(sr_dict[last_sr], sr_sigma)
    last_sr = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    ########################################################
    #### Flag almost FA where lidar signal is very weak ####
    print("\t=> Flag 'almost FA' where lidar signal is very weak...", end='')
    step += 1
    feature_dict[step] = FLAG_WEAK_SIGNAL(params, feature_dict[last_feature], sr_dict[last_sr],
                                          sr_sigma)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...", end='')
    step += 1
    sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
    last_sr = step

    # Show elapsed time
    tic = print_elapsed_time(tic)

    if True:
        ############################################################
        #### Correct sr signal below feature from transmittance ####
        print("\t=> Correct sr signal below feature from transmittance using "\
            f"fixed lidar ratio above and below {params.temp_ice_liquid} °C...", end='')
        step += 1
        sr_dict[step], twoway_transmittance_array[:, :] =\
            transmission_correction(sr_dict[last_sr], sr, b_mol, feature_dict[last_feature],
                                    temperature, params)
        last_sr = step

        # Show elapsed time
        tic = print_elapsed_time(tic)


    ########################################################
    #### Fill small strip between FA where strip < nb_prof_min prof ####
    print("\t=> Fill small strip between FA where strip < nb_prof_min prof...", 
          end='')
    step += 1
    feature_dict[step] = fill_small_strips(params, feature_dict[last_feature])
    last_feature = step
    last_feature_before_averaging = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...", end='')
    step += 1
    sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
    last_sr = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #--------------------------------------------------------------------------
    ###########################
    #### Detection level 5 ####
    print("\t=> Detection level 5:")

    # Increase FLAG_DETECTION_LEVEL
    FLAG_DETECTION_LEVEL += 1

    # Get detection coefficients
    k, n, s, a = get_feature_detection_coef(channel, FLAG_DETECTION_LEVEL - 1)

    ###########################################################
    #### Apply a gaussian horizontal line window averaging ####
    print("\t\t- Apply a gaussian horizontal line window averaging...", end='')
    step += 1
    sr_dict[step], sr_sigma = gaussian_line_window(a[0], a[1], sr_dict[last_sr],
                                                   feature_dict[last_feature], sr_sigma)
    last_sr = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    ###################################################
    #### Apply threshold to get the 'maybe' pixels ####
    print("\t\t- Apply threshold...", end='')
    step += 1
    feature_dict[step] = apply_threshold(k, feature_dict[last_feature], sr_dict[last_sr], sr_sigma)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #########################################
    #### Windowing on the 'maybe' pixels ####
    print("\t\t- Windowing on the 'maybe' pixels...", end='')
    step += 1
    feature_dict[step] = apply_window(s[0], s[1], feature_dict[last_feature], FLAG_DETECTION_LEVEL)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #######################################
    #### Replace 'maybe' by 'detected' ####
    print("\t\t- Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels "\
          "meet neighbors number limit condition...", end='')
    step += 1
    feature_dict[step] = replace_maybe(n, feature_dict[last_feature], FLAG_DETECTION_LEVEL)
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)
    #--------------------------------------------------------------------------


    ##########################################################################
    #### Reput all not confident flags where overwritten during averaging ####
    print("\t=> Reput all not confident flags where overwritten during "\
          "averaging...", end='')
    step += 1
    feature_dict[step] = reput_low_confidence_flags(feature_dict[last_feature],
                                                    feature_dict[last_feature_before_averaging])
    last_feature = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    #########################################
    #### Remove detected pixel from ATSR ####
    print("\t=> Remove detected pixel from ATSR...", end='')
    step += 1
    sr_dict[step] = remove_detect_from_sr(sr_dict[last_sr], feature_dict[last_feature])
    last_sr = step

    # Show elapsed time
    tic = print_elapsed_time(tic)


    ############################################################
    #### Transform feature and sr dictionaries to 3D arrays ####
    print("\t=> Transform feature and sr dictionaries to 3D arrays...", end='')
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

    # Show elapsed time
    tic = print_elapsed_time(tic)


    print(f'\t(Elapsed time: {datetime.now() - tic_function})')

    return feature_dict[last_feature], feature_array_steps, sr_array_steps,\
           twoway_transmittance_array
