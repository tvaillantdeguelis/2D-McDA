"""Tunable parameters for the detection algorithm and pipeline execution."""


# --- Pixel flags used in channel-specific detection masks ---

FLAG_NOTHING = 0
FLAG_DETECTION_LEVEL = 1
FLAG_MAYBE = 255
FLAG_SURFACE = 254
FLAG_LIKELY_ARTIFACT = 253
FLAG_FA = 252
FLAG_AFA = 251
FLAG_SMALL_STRIPS = 250


# --- Slicing / execution ---

NB_PROF_SLICE = 3000
NB_PROF_CONTEXT = 250


# --- Surface detection ---

class SurfaceDetectionParameters:
    """Surface-detection parameters for one lidar channel."""

    def __init__(self, channel):
        self.offset_dem_water = 3
        self.offset_dem_perm_snow = 17
        self.offset_dem_other = 5
        self.offset_dem_false_positive = 1
        self.coef_nb_std = 5

        if channel in ("532_par", "532_per"):
            self.N = 2
        elif channel == "1064":
            self.N = 4
        else:
            raise ValueError(f"Unrecognized channel: {channel}")


# --- Feature detection ---

class FeatureDetectionParameters:
    """Feature-detection parameters for one lidar channel."""

    def __init__(self, channel):
        self.S_liquid = 10
        self.S_ice = 18
        self.temp_ice_liquid = -38
        self.twoway_transmittance_limit = 0.1
        self.mult_scatt = 0.7
        self.nb_bins_PMT_artifact = 20

        if channel == "532_par":
            self.weak_signal_ratio = 0.3
            self.weak_signal_ratio_threshold = 0.1
        elif channel == "532_per":
            self.weak_signal_ratio = 0.9
            self.weak_signal_ratio_threshold = 1
        elif channel == "1064":
            self.weak_signal_ratio = 0.85
            self.weak_signal_ratio_threshold = 1
        else:
            raise ValueError(f"Unrecognized channel: {channel}")

        self.nb_prof_min_small_strips = 15


def get_feature_detection_coef(channel, level):
    """Return threshold, neighbor and smoothing parameters for a level.

    Each coefficient is a list of five entries, one per detection level, and
    ``None`` means the corresponding step is skipped at that level (``k`` set to
    ``None`` skips the whole level, as level 1 does for the 1064 nm channel).

    ``k``
        Number of noise standard deviations above the molecular signal that a
        pixel must exceed to become a 'maybe' candidate, i.e. the threshold is
        ``ATSR > 1 + k * sigma``. The lower the value, the more sensitive the
        level.
    ``n``
        Minimum number of connected 'maybe' pixels (counting the previous
        level's detections) a pattern must hold to be turned into a detection;
        smaller patterns are discarded. ``n = 1`` keeps every candidate. This is
        the extent of the pattern, counted on the uniform 30 m × 333 m grid, so
        it means the same area at every altitude.
    ``m``
        Minimum number of downlinked CALIOP measurements the same pattern must
        hold, which is how much evidence supports it. The pixels of the regular
        grid are copies of one another wherever CALIOP downlinks a coarser
        resolution, so one measurement covers 1 pixel below 8.2 km at 532 nm
        (2 at 1064 nm), 6 between 8.2 km and 20.2 km, 30 between 20.2 km and
        30.1 km, and 150 above. These are the measurements as downlinked, at
        every level and whatever averaging that level applies to them. Without
        this condition a single noisy measurement above 20.2 km fills enough
        pixels to pass ``n`` on its own. ``m`` starts discarding the patterns of
        a region once it exceeds ``n`` divided by the pixels one measurement
        covers there, which at level 3 is 60 in region 2, 10 in region 3, 2 in
        region 4 and 0.4 in region 5. ``m = 1`` imposes nothing.
    ``s``
        Morphological window used to grow the candidates, as
        ``(height_window, width_window)`` in pixels, both odd. A pixel becomes
        'maybe' when detections fill at least half of that window.
    ``a``
        Gaussian averaging window applied to the signal before thresholding, as
        ``(height_window, vertical_gauss_sigma, width_window,
         horizontal_gauss_sigma)``: the window spans ``height_window``
        altitude bins (odd) by ``width_window`` profiles (odd), and the two
        sigmas, in pixels, set how fast the Gaussian weights fall off
        vertically and horizontally. ``height_window = 1`` gives a purely
        horizontal averaging, and ``vertical_gauss_sigma`` is then unused. It
        is undefined at every level for now, since the signal is brought to
        5 km × 180 m before level 5 runs.
    """

    if channel == "532_par":
        k = [100, 20, 2, 1, 1]
        n = [2, 2, 60, 200, 10000]
        m = [2, 2, 10, 30, 30]
        s = [None, None, (11, 11), (3, 21), (9, 51)]
        a = [None, None, None, None, None]
    elif channel == "532_per":
        k = [500, 100, 2, 1, 1]
        n = [2, 2, 60, 200, 1000]
        m = [2, 2, 10, 30, 30]
        s = [None, None, (11, 11), (3, 21), (9, 51)]
        a = [None, None, None, None, None]
    elif channel == "1064":
        k = [None, 20, 2, 1, 2]
        n = [None, 1, 60, 200, 10000]
        m = [None, 1, 10, 30, 30]
        s = [None, None, (11, 11), (3, 21), (9, 51)]
        a = [None, None, None, None, None]
    else:
        raise ValueError(f"Unrecognized channel: {channel}")

    return k[level], n[level], m[level], s[level], a[level]
