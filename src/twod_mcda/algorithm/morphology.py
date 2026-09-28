"""Neighborhood operations on the detection mask.

These decide whether a pixel belongs to a real structure by looking at the
pixels around it: windowing, connected patterns of candidate pixels, and small
horizontal strips. The windowing counts pixels with summed-area tables; the
two others pair a wrapper with a Numba kernel, which works in place on a copy
of the mask.
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


def _window_sums(pixels, width_window, height_window):
    """Count the ``pixels`` in the window centered on each interior pixel.

    ``pixels`` is a boolean image indexed (profile, altitude); the window spans
    ``width_window`` profiles × ``height_window`` altitudes. A summed-area table
    gives every count in four lookups, so the cost does not depend on the window
    size. Returns the counts of the pixels whose window lies entirely inside the
    image, i.e. an array smaller than ``pixels`` by the window size minus one.
    """

    table = np.zeros((pixels.shape[0] + 1, pixels.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(pixels, axis=0, dtype=np.int64), axis=1, out=table[1:, 1:])
    return (
        table[width_window:, height_window:]
        - table[:-width_window, height_window:]
        - table[width_window:, :-height_window]
        + table[:-width_window, :-height_window]
    )

def apply_window(
    height_window, width_window, feature, FLAG_DETECTION_LEVEL, min_percent=0.5
):
    """Keep as 'maybe' the pixels whose window is mostly made of detections.

    A 'nothing' or 'maybe' pixel becomes 'maybe' when the 'maybe' pixels and the
    previous level detections fill at least ``min_percent`` of its window, not
    counting the pixels already flagged otherwise (surface, (almost) fully
    attenuated, likely artifact, small strips, and detections from two levels
    back or more). Pixels closer to the image edge than half a window are never
    flagged.
    """

    # Initialization
    new_feature = feature.copy()

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

    # Pixels counted as detected: "maybe" and previous detection level (d-1)
    counted = feature == FLAG_MAYBE
    # Pixels removed from the window: special flags and detection <= d-2
    excluded = np.isin(
        feature,
        (FLAG_SURFACE, FLAG_LIKELY_ARTIFACT, FLAG_FA, FLAG_AFA, FLAG_SMALL_STRIPS),
    )
    if FLAG_DETECTION_LEVEL > 1:  # if previous detection exists
        counted |= feature == FLAG_DETECTION_LEVEL - 1
        excluded |= (feature >= 1) & (feature <= FLAG_DETECTION_LEVEL - 2)

    # Apply moving window, if the image is at least as large as the window
    if feature.shape[0] >= width_window and feature.shape[1] >= height_window:
        interior = (
            slice(w_side, feature.shape[0] - w_side),
            slice(h_side, feature.shape[1] - h_side),
        )
        nb_tot = _window_sums(counted, width_window, height_window)
        nb_pixels_window_2 = nb_pixels_window - _window_sums(
            excluded, width_window, height_window
        )

        # Flag detected if amount above limit
        detected_pixels[interior] = (
            (feature[interior] == FLAG_NOTHING) | (feature[interior] == FLAG_MAYBE)
        ) & (nb_tot >= nb_pixels_window_2 * min_percent)

    # Remove previous "maybe" pixels
    new_feature[new_feature == FLAG_MAYBE] = FLAG_NOTHING

    # Replace by those which result from the windowing
    new_feature[detected_pixels] = FLAG_MAYBE

    return new_feature

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
    new_feature = feature.copy()
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

    return fill_small_strips_jit(
        feature.copy(),
        params.nb_prof_min_small_strips,
    )
