"""Run the unchanged scientific detection functions for each channel."""

from twod_mcda.algorithm.features import detect_features
from twod_mcda.algorithm.surface import detect_surface


def detect_surfaces(data):
    """Return surface indexes for the three lidar channels."""

    common = (
        data["IGBP_Surface_Type"],
        data["Surface_Elevation"],
        data["Spacecraft_Altitude"],
        data["Lidar_Data_Altitudes"],
    )

    parallel = detect_surface(
        data["Parallel_Attenuated_Backscatter_532"],
        *common,
        data["Parallel_RMS_Baseline_532"],
        data["Laser_Energy_532"],
        data["Calibration_Constant_532"],
        1,
        data["Parallel_Amplifier_Gain_532"],
        data["Off_Nadir_Angle"],
        "532_par",
    )
    perpendicular = detect_surface(
        data["Perpendicular_Attenuated_Backscatter_532"],
        *common,
        data["Perpendicular_RMS_Baseline_532"],
        data["Laser_Energy_532"],
        data["Calibration_Constant_532"],
        data["Depolarization_Gain_Ratio_532"],
        data["Perpendicular_Amplifier_Gain_532"],
        data["Off_Nadir_Angle"],
        "532_per",
    )
    infrared = detect_surface(
        data["Attenuated_Backscatter_1064"],
        *common,
        data["RMS_Baseline_1064"],
        data["Laser_Energy_1064"],
        data["Calibration_Constant_1064"],
        1,
        data["Amplifier_Gain_1064"],
        data["Off_Nadir_Angle"],
        "1064",
    )

    return {
        "532_par": parallel,
        "532_per": perpendicular,
        "1064": infrared,
    }


def detect_channel_features(data, surface_indexes):
    """Run feature detection and return masks and development arrays."""

    masks = {}
    development = {}

    channel_inputs = (
        (
            "532_par",
            "Parallel_Detection_Flags_532",
            "Parallel_Attenuated_Backscatter_532",
            "Molecular_Parallel_Attenuated_Backscatter_532",
            "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel",
            "Molecular_Parallel_Backscatter_532",
            "Parallel_Detection_Flags_532_steps",
            "Parallel_Attenuated_Scattering_Ratio_532_steps",
            "Parallel_CumulativeTwoWayTransmittance_532",
        ),
        (
            "532_per",
            "Perpendicular_Detection_Flags_532",
            "Perpendicular_Attenuated_Backscatter_532",
            "Molecular_Perpendicular_Attenuated_Backscatter_532",
            "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular",
            "Molecular_Perpendicular_Backscatter_532",
            "Perpendicular_Detection_Flags_532_steps",
            "Perpendicular_Attenuated_Scattering_Ratio_532_steps",
            "Perpendicular_CumulativeTwoWayTransmittance_532",
        ),
        (
            "1064",
            "Detection_Flags_1064",
            "Attenuated_Backscatter_1064",
            "Molecular_Attenuated_Backscatter_1064",
            "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064",
            "Molecular_Backscatter_1064",
            "Detection_Flags_1064_steps",
            "Attenuated_Scattering_Ratio_1064_steps",
            "CumulativeTwoWayTransmittance_1064",
        ),
    )

    for (
        channel,
        mask_name,
        attenuated_name,
        molecular_attenuated_name,
        uncertainty_name,
        molecular_name,
        steps_name,
        ratio_steps_name,
        transmittance_name,
    ) in channel_inputs:
        mask, steps, ratio_steps, transmittance = detect_features(
            data[attenuated_name] / data[molecular_attenuated_name],
            data[uncertainty_name],
            data[molecular_name],
            data["Temperature"],
            surface_indexes[channel],
            channel,
        )
        masks[mask_name] = mask
        development[steps_name] = steps
        development[ratio_steps_name] = ratio_steps
        development[transmittance_name] = transmittance

    return masks, development
