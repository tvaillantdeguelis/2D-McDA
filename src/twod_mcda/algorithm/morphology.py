"""Neighborhood operations on the detection mask.

These decide whether a pixel belongs to a real structure by looking at the
pixels around it: windowing, connected patterns of candidate pixels, and small
horizontal strips. Each one pairs a masked-array wrapper with a Numba kernel,
which is why the mask has to be converted through ``feature_for_numba`` first.
"""

import numpy as np
from numba import jit

from twod_mcda.parameters import (
    FLAG_AFA,
    FLAG_FA,
    FLAG_LIKELY_ARTIFACT,
    FLAG_MAYBE,
    FLAG_NOTHING,
    FLAG_SMALL_STRIPS,
    FLAG_SURFACE,
)


def feature_for_numba(feature):
    mask = np.ma.getmaskarray(feature)
    data = np.asarray(np.ma.getdata(feature)).copy()

    # Empêche Numba de traiter les pixels masqués.
    data[mask] = FLAG_SURFACE

    return data, mask

@jit(nopython=True)
def apply_window_jit(
    w_side,
    h_side,
    feature,
    nb_pixels_window,
    min_percent,
    detected_pixels,
    FLAG_DETECTION_LEVEL,
):
    """Part extracted from apply_window function for faster processing with
    @jit"""

    for i in np.arange(w_side, feature.shape[0] - w_side):
        for j in np.arange(h_side, feature.shape[1] - h_side):
            if (feature[i, j] == FLAG_NOTHING) | (feature[i, j] == FLAG_MAYBE):
                # Tuple with indexes of the window
                window = (
                    slice(i - w_side, i + w_side + 1),
                    slice(j - h_side, j + h_side + 1),
                )

                # Count nb of "maybe"
                nb_maybe = list(feature[window].flatten()).count(FLAG_MAYBE)

                # Count nb of "previous detection level" (d-1)
                nb_detected_1 = 0
                if FLAG_DETECTION_LEVEL > 1:  # if previous detection exists
                    prev_FLAG_DETECTION_LEVEL = FLAG_DETECTION_LEVEL - 1
                    nb_detected_1 = list(feature[window].flatten()).count(
                        prev_FLAG_DETECTION_LEVEL
                    )

                # Total detected at n and n-1
                nb_tot = nb_maybe + nb_detected_1

                # Count nb of special (and detection <= d-2) and remove
                # from nb_pixels_window
                nb_surface = list(feature[window].flatten()).count(FLAG_SURFACE)
                nb_likely_artifact = list(feature[window].flatten()).count(
                    FLAG_LIKELY_ARTIFACT
                )
                nb_FA = list(feature[window].flatten()).count(FLAG_FA)
                nb_AFA = list(feature[window].flatten()).count(FLAG_AFA)
                nb_small_strips = list(feature[window].flatten()).count(
                    FLAG_SMALL_STRIPS
                )
                nb_detected_2_and_before = 0
                if FLAG_DETECTION_LEVEL > 1:  # if previous detection exists
                    for prev_FLAG_DETECTION_LEVEL in np.arange(
                        1, FLAG_DETECTION_LEVEL - 1
                    ):
                        nb_detected_2_and_before += list(
                            feature[window].flatten()
                        ).count(prev_FLAG_DETECTION_LEVEL)
                nb_pixels_window_2 = (
                    nb_pixels_window
                    - nb_FA
                    - nb_AFA
                    - nb_surface
                    - nb_likely_artifact
                    - nb_small_strips
                    - nb_detected_2_and_before
                )

                # Flag detected if amount above limit
                nb_min_tot = nb_pixels_window_2 * min_percent
                if nb_tot >= nb_min_tot:
                    detected_pixels[i, j] = 1

    return feature

def apply_window(
    height_window, width_window, feature, FLAG_DETECTION_LEVEL, min_percent=0.5
):
    # min_percent: min pourcentage of total counted pixels in the window to flag the center as "detected"

    # Initialization
    new_feature, feature_mask = feature_for_numba(feature)

    # height_window and width_window should be odd numbers
    if (height_window % 2 != 1) | (width_window % 2 != 1):
        raise ValueError(
            f"height_window (= {height_window}) and width_window "
            f"(= {width_window}) should be odd numbers"
        )

    # Initialization
    detected_pixels = np.zeros(new_feature.shape, dtype=bool)
    nb_pixels_window = width_window * height_window
    h_side = int(height_window / 2)  # nb of pixel each side of the center
    w_side = int(width_window / 2)  # nb of pixel each side of the center

    # Apply moving window
    new_feature = apply_window_jit(
        w_side,
        h_side,
        new_feature,
        nb_pixels_window,
        min_percent,
        detected_pixels,
        FLAG_DETECTION_LEVEL,
    )

    # Remove previous "maybe" pixels (or not if keep_all==True)
    new_feature[new_feature == FLAG_MAYBE] = FLAG_NOTHING

    # Replace by those which result from the windowing
    new_feature[detected_pixels == 1] = FLAG_MAYBE

    return np.ma.array(new_feature, mask=feature_mask)

@jit(nopython=True)
def replace_maybe_jit(
    nb_lim, feature, seen_pixels, FLAG_DETECTION_LEVEL, prev_detect, prevprev_detect
):
    """Classify connected components containing ``FLAG_MAYBE`` pixels.

    The queue is allocated once and reused for every component.  Besides
    avoiding a full-size temporary image per component, the head/tail indices
    make removing a queue item O(1).
    """

    nb_rows, nb_cols = feature.shape
    pixel_queue = np.empty(feature.size, dtype=np.int64)

    for i in range(nb_rows):
        for j in range(nb_cols):
            if seen_pixels[i, j] or feature[i, j] != FLAG_MAYBE:
                continue

            # Store flat indices so that a single array is enough for both the
            # breadth-first queue and the list of pixels in this component.
            head = 0
            tail = 1
            pixel_queue[0] = i * nb_cols + j
            seen_pixels[i, j] = True
            connected_to_detected_pattern = False

            while head < tail:
                flat_index = pixel_queue[head]
                head += 1
                row = flat_index // nb_cols
                col = flat_index - row * nb_cols

                # If FA/low-confidence pixels are immediately to the left,
                # look beyond them for an already detected pattern.  Use the
                # current component pixel for the boundary check as intended.
                left = row - 1
                while left >= 0 and (
                    feature[left, col] == FLAG_FA
                    or feature[left, col] == FLAG_AFA
                    or feature[left, col] == FLAG_SMALL_STRIPS
                ):
                    left -= 1
                if left >= 0 and feature[left, col] == FLAG_DETECTION_LEVEL:
                    connected_to_detected_pattern = True

                # Test the four directly adjacent pixels without allocating a
                # temporary neighbors list.
                for direction in range(4):
                    neighbor_row = row
                    neighbor_col = col
                    if direction == 0:
                        neighbor_row += 1
                    elif direction == 1:
                        neighbor_row -= 1
                    elif direction == 2:
                        neighbor_col += 1
                    else:
                        neighbor_col -= 1

                    if (
                        neighbor_row < 0
                        or neighbor_row >= nb_rows
                        or neighbor_col < 0
                        or neighbor_col >= nb_cols
                        or seen_pixels[neighbor_row, neighbor_col]
                    ):
                        continue

                    neighbor_value = feature[neighbor_row, neighbor_col]
                    is_accessible = neighbor_value == FLAG_MAYBE
                    if FLAG_DETECTION_LEVEL > 1:
                        if prev_detect and neighbor_value == FLAG_DETECTION_LEVEL - 1:
                            is_accessible = True
                        if (
                            prevprev_detect
                            and neighbor_value == FLAG_DETECTION_LEVEL - 2
                        ):
                            is_accessible = True

                    if is_accessible:
                        # Mark on insertion so a pixel cannot be queued by two
                        # different neighbors.
                        seen_pixels[neighbor_row, neighbor_col] = True
                        pixel_queue[tail] = neighbor_row * nb_cols + neighbor_col
                        tail += 1

            replacement = FLAG_DETECTION_LEVEL
            if tail < nb_lim and not connected_to_detected_pattern:
                replacement = FLAG_NOTHING

            for queue_index in range(tail):
                flat_index = pixel_queue[queue_index]
                row = flat_index // nb_cols
                col = flat_index - row * nb_cols
                if feature[row, col] == FLAG_MAYBE:
                    feature[row, col] = replacement

    return feature

def replace_maybe(
    n, feature, FLAG_DETECTION_LEVEL, prev_detect=True, prevprev_detect=False
):
    """Put flag 'FLAG_DETECTION_LEVEL' where patterns of connected 'FLAG_MAYBE'
    pixels consist of at least n pixels
    if prev_detect=True means that we also count detection pixels n-1
    if prevprev_detect=True means that we also count detection pixels n-2"""

    # Initialization
    new_feature, feature_mask = feature_for_numba(feature)

    # Initialization
    seen_pixels = np.zeros(new_feature.shape, dtype=bool)

    # Look for a "maybe" pixel and decide if it's really part of a pattern
    # based on nb of neighbors (neighbors in "level n" + "level n-1")
    if n == 1:  # keep all
        new_feature[new_feature == FLAG_MAYBE] = FLAG_DETECTION_LEVEL
    else:
        new_feature = replace_maybe_jit(
            n,
            new_feature,
            seen_pixels,
            FLAG_DETECTION_LEVEL,
            prev_detect,
            prevprev_detect,
        )

    return np.ma.array(new_feature, mask=feature_mask)

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
                if (
                    (feature[i, j] == FLAG_FA)
                    | (feature[i, j] == FLAG_AFA)
                    | (feature[i, j] == FLAG_LIKELY_ARTIFACT)
                ) & (
                    (feature[i + 1, j] != FLAG_FA)
                    & (feature[i + 1, j] != FLAG_AFA)
                    & (feature[i + 1, j] != FLAG_LIKELY_ARTIFACT)
                ):
                    # Look right if there is (A)FA or Likely artifact at
                    # less than nb_prof_min
                    i2 = i + 1
                    nb_prof_strip = 1
                    less_than_nb_prof_min = False
                    while (nb_prof_strip <= nb_prof_min) & (i2 <= nb_prof - 1):
                        if (
                            (feature[i2, j] == FLAG_FA)
                            | (feature[i2, j] == FLAG_AFA)
                            | (feature[i2, j] == FLAG_LIKELY_ARTIFACT)
                        ):
                            less_than_nb_prof_min = True
                            break
                        i2 += 1
                        nb_prof_strip += 1
                    # If there is (A)FA at less than nb_prof_min far
                    if less_than_nb_prof_min:
                        # Put "Low confidence small strips" flag between
                        feature[i + 1 : i2, j][
                            feature[i + 1 : i2, j] == FLAG_NOTHING
                        ] = FLAG_SMALL_STRIPS

    return feature

def fill_small_strips(params, feature):
    """Flag short horizontal strips between low-confidence regions."""

    feature_mask = np.ma.getmaskarray(feature)

    new_feature = np.asarray(np.ma.getdata(feature)).copy()

    # Valeur exclue des traitements dans la fonction JIT.
    new_feature[feature_mask] = FLAG_SURFACE

    new_feature = fill_small_strips_jit(
        new_feature,
        params.nb_prof_min_small_strips,
    )

    return np.ma.array(
        new_feature,
        mask=feature_mask,
    )
