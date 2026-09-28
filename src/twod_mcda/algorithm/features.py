"""Orchestrate the successive 2D-McDA feature-detection levels."""

import numpy as np
import xarray as xr

from twod_mcda.algorithm.attenuation import transmission_correction
from twod_mcda.algorithm.flagging import (
    FLAG_WEAK_SIGNAL,
    apply_surface_detection,
    apply_threshold,
    fill_fully_attenuated,
    fill_likely_artifact,
    reput_low_confidence_flags,
)
from twod_mcda.algorithm.morphology import (
    apply_window,
    fill_small_strips,
    replace_maybe,
)
from twod_mcda.algorithm.smoothing import (
    average_below_8_2,
    gaussian_2d_window,
    remove_detect_from_sr,
)
from twod_mcda.caliop.constants import FILL_VALUE_FLOAT
from twod_mcda.parameters import (
    FeatureDetectionParameters,
    get_feature_detection_coef,
)
from twod_mcda.utils.arrays import mask_invalid


class _DetectionHistory:
    """The current mask and signal, together with every intermediate state.

    ``detect_features`` replaces ``feature`` and ``sr`` step after step. Each
    assignment is recorded under its own step number, so ``to_arrays`` can stack
    the whole run into the two 3-D arrays of the development product. Assigning
    is therefore not free: every ``history.feature = ...`` consumes a step.

    With ``record_steps`` false, only the current mask and signal are kept, so
    that the intermediate states are freed as soon as they are replaced.
    """

    def __init__(self, sr, feature, record_steps):
        self._sr = sr
        self._feature = feature
        self._record_steps = record_steps
        self._signals = {0: sr} if record_steps else {}
        self._features = {0: feature} if record_steps else {}
        self._step = 0

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, values):
        self._step += 1
        self._feature = values
        if self._record_steps:
            self._features[self._step] = values

    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, values):
        self._step += 1
        self._sr = values
        if self._record_steps:
            self._signals[self._step] = values

    def to_arrays(self, shape):
        """Stack every recorded state, leaving untouched steps at their default."""

        features = np.zeros((self._step + 1, *shape), dtype=np.uint8)
        signals = np.full((self._step + 1, *shape), FILL_VALUE_FLOAT)
        for step, values in self._features.items():
            features[step, :, :] = values
        for step, values in self._signals.items():
            signals[step, :, :] = values
        return features, signals


def _apply_detection_level(history, level, channel, sr_sigma, params):
    """Run one detection level on the current mask.

    The coefficients decide which steps apply, so the five levels differ only by
    their parameters: a level whose gaussian window ``a`` is undefined does not
    smooth the signal beforehand, one whose window ``s`` is undefined does not
    window the candidate pixels, and one whose threshold ``k`` is undefined does
    not apply to this channel at all.

    Returns the noise threshold, which gaussian averaging lowers.
    """

    k, n, s, a = get_feature_detection_coef(channel, level - 1)

    if k is None:
        # Level 1 only applies to the two 532 nm channels.
        return sr_sigma

    if a is not None:
        # Apply a gaussian horizontal line window averaging
        history.sr, sr_sigma = gaussian_2d_window(
            a[0], a[1], history.sr, history.feature, sr_sigma
        )

    # Apply threshold (at level 1, to get very high echo, likely PMT artifact)
    history.feature = apply_threshold(k, history.feature, history.sr, sr_sigma)

    if s is not None:
        # Windowing on the 'maybe' pixels
        history.feature = apply_window(s[0], s[1], history.feature, level)

    # Flag 'Detected' where patterns of 'FLAG_MAYBE' pixels meet neighbors
    # number limit condition
    history.feature = replace_maybe(n, history.feature, level)

    if level == 1:
        # Flag 'Likely Artifact' below those high signal to some extent
        history.feature = fill_likely_artifact(params, history.feature, level)

    return sr_sigma


def detect_features(
    sr,
    sr_sigma,
    b_mol,
    temperature,
    surf_alt_index,
    channel,
    first_profile,
    save_steps,
):
    """Detect features in ATSR signal of lidar channel.

    Five detection levels run in turn, each one lowering its threshold and so
    picking up weaker features than the previous one. Levels 1 to 4 run back to
    back; level 5 runs last, after the signal has been averaged and the fully
    attenuated columns have been flagged.

    Every input is a plain array, with NaN where missing; the feature mask is a
    uint8 array of flags. ``first_profile`` is the index, in its granule, of the
    first profile. The intermediate masks and signals are only stacked into 3-D
    arrays when ``save_steps`` is true; otherwise both are returned as None.
    """

    # Get feature detection parameters
    params = FeatureDetectionParameters(channel)

    history = _DetectionHistory(
        sr.copy(),
        np.zeros(sr.shape, dtype=np.uint8),
        save_steps,
    )

    # Put 'Surface' flag on feature mask
    history.feature = apply_surface_detection(history.feature, surf_alt_index)
    history.sr = remove_detect_from_sr(history.sr, history.feature)

    for level in (1, 2, 3, 4):
        sr_sigma = _apply_detection_level(history, level, channel, sr_sigma, params)

    # Flag 'Fully Attenuated' from lowest altitude to first feature
    history.feature = fill_fully_attenuated(history.feature)
    history.sr = remove_detect_from_sr(history.sr, history.feature)

    # Average below 8.2 km as between 8.2 km and 20.2 km (60 m × 1 km)
    # Note: sr_sigma needs to be modified below 8.2 km
    history.sr, sr_sigma = average_below_8_2(history.sr, sr_sigma, first_profile)

    # Flag 'almost FA' where lidar signal is very weak
    history.feature = FLAG_WEAK_SIGNAL(params, history.feature, history.sr, sr_sigma)
    history.sr = remove_detect_from_sr(history.sr, history.feature)

    # Correct sr signal below feature from transmittance using fixed lidar ratio
    # above and below params.temp_ice_liquid
    history.sr, twoway_transmittance_array = transmission_correction(
        history.sr, sr, b_mol, history.feature, temperature, params
    )

    # Fill small strip between FA where strip < nb_prof_min prof
    history.feature = fill_small_strips(params, history.feature)
    feature_before_averaging = history.feature
    history.sr = remove_detect_from_sr(history.sr, history.feature)

    sr_sigma = _apply_detection_level(history, 5, channel, sr_sigma, params)

    # Reput all not confident flags where overwritten during averaging
    history.feature = reput_low_confidence_flags(
        history.feature, feature_before_averaging
    )
    history.sr = remove_detect_from_sr(history.sr, history.feature)

    # Stack every recorded detection step into 3D arrays
    feature_array_steps, sr_array_steps = (
        history.to_arrays(sr.shape) if save_steps else (None, None)
    )

    return (
        history.feature,
        feature_array_steps,
        sr_array_steps,
        twoway_transmittance_array,
    )


FEATURE_INPUTS_BY_CHANNEL = {
    "532_par": (
        "Parallel_Detection_Flags_532",
        "Parallel_Attenuated_Backscatter_532",
        "Molecular_Parallel_Attenuated_Backscatter_532",
        "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel",
        "Molecular_Parallel_Backscatter_532",
        "Parallel_Detection_Flags_532_steps",
        "Parallel_Attenuated_Scattering_Ratio_532_steps",
        "Parallel_CumulativeTwoWayTransmittance_532",
    ),
    "532_per": (
        "Perpendicular_Detection_Flags_532",
        "Perpendicular_Attenuated_Backscatter_532",
        "Molecular_Perpendicular_Attenuated_Backscatter_532",
        "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular",
        "Molecular_Perpendicular_Backscatter_532",
        "Perpendicular_Detection_Flags_532_steps",
        "Perpendicular_Attenuated_Scattering_Ratio_532_steps",
        "Perpendicular_CumulativeTwoWayTransmittance_532",
    ),
    "1064": (
        "Detection_Flags_1064",
        "Attenuated_Backscatter_1064",
        "Molecular_Attenuated_Backscatter_1064",
        "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064",
        "Molecular_Backscatter_1064",
        "Detection_Flags_1064_steps",
        "Attenuated_Scattering_Ratio_1064_steps",
        "CumulativeTwoWayTransmittance_1064",
    ),
}


def detect_features_in_channel(data, surface_indexes, channel, save_development_data):
    """Run feature detection for one lidar channel.

    The development dataset is empty unless ``save_development_data`` is true.
    """

    try:
        (
            mask_name,
            attenuated_name,
            molecular_attenuated_name,
            uncertainty_name,
            molecular_name,
            steps_name,
            ratio_steps_name,
            transmittance_name,
        ) = FEATURE_INPUTS_BY_CHANNEL[channel]
    except KeyError:
        raise ValueError(
            f"Unknown lidar channel {channel!r}. Expected '532_par', "
            "'532_per', or '1064'."
        ) from None

    template = data[attenuated_name]
    mask, steps, ratio_steps, transmittance = detect_features(
        mask_invalid(template / data[molecular_attenuated_name]).values,
        mask_invalid(data[uncertainty_name]).values,
        mask_invalid(data[molecular_name]).values,
        mask_invalid(data["Temperature"]).values,
        surface_indexes.values,
        channel,
        int(template.coords["profile"][0]),
        save_development_data,
    )
    base_coords = {
        "profile": template.coords["profile"],
        "altitude": template.coords["altitude"],
    }
    detection_mask = xr.DataArray(
        mask,
        dims=("profile", "altitude"),
        coords=base_coords,
        name=mask_name,
    )
    if not save_development_data:
        return detection_mask, xr.Dataset()

    step_dimension = f"step_{channel}"
    step_coords = {
        step_dimension: np.arange(steps.shape[0]),
        **base_coords,
    }
    development = xr.Dataset(
        {
            steps_name: xr.DataArray(
                steps,
                dims=(step_dimension, "profile", "altitude"),
                coords=step_coords,
            ),
            ratio_steps_name: xr.DataArray(
                ratio_steps,
                dims=(step_dimension, "profile", "altitude"),
                coords=step_coords,
            ),
            transmittance_name: xr.DataArray(
                transmittance,
                dims=("profile", "altitude"),
                coords=base_coords,
            ),
        }
    )
    return detection_mask, development


def detect_features_in_3_channels(data, surface_indexes, save_development_data):
    """Run feature detection and return masks and development arrays.

    The development arrays are only built when ``save_development_data`` is
    true; otherwise the returned development dataset is empty.
    """

    masks = []
    development = []
    for channel in ("532_par", "532_per", "1064"):
        mask, channel_development = detect_features_in_channel(
            data,
            surface_indexes[channel],
            channel,
            save_development_data,
        )
        masks.append(mask)
        development.append(channel_development)

    return xr.merge(masks), xr.merge(development)
