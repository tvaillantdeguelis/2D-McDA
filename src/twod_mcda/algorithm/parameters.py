"""Scientific parameters for surface and feature detection."""


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
    """Return threshold, neighbor and smoothing parameters for a level."""

    if channel == "532_par":
        k = [100, 20, 2, 1, 1]
        n = [1, 1, 60, 200, 100000]
        s = [None, None, (11, 11), (3, 21), (9, 51)]
        a = [None, None, None, None, (15, 5)]
    elif channel == "532_per":
        k = [500, 100, 2, 1, 1]
        n = [1, 1, 60, 200, 1000]
        s = [None, None, (11, 11), (3, 21), (9, 51)]
        a = [None, None, None, None, (15, 5)]
    elif channel == "1064":
        k = [None, 20, 2, 1, 1]
        n = [None, 1, 60, 200, 10000]
        s = [None, None, (11, 11), (3, 21), (9, 51)]
        a = [None, None, None, None, (15, 5)]
    else:
        raise ValueError(f"Unrecognized channel: {channel}")

    return k[level], n[level], s[level], a[level]
