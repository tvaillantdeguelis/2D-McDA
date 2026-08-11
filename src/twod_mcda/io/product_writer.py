"""Build and write the HDF4 product produced by 2D-McDA."""

from twod_mcda.calipso_constants import FILL_VALUE_FLOAT
from twod_mcda.io.hdf_writer import SDSData, write_hdf


def _variable(name, data, *, fill_value=None, units=None, valid_range=None, dims=None):
    variable = SDSData(name, data, fill_value)
    variable.units = units
    variable.valid_range = valid_range
    variable.dim_labels = list(dims) if dims is not None else None
    return variable


def build_output_variables(result, save_development_data):
    """Translate processing arrays to the legacy HDF4 product schema."""

    data = result.data
    variables = {
        "prof_ID": _variable("Profile_ID", data["Profile_ID"], dims=("Profile_ID",)),
        "prof_time": _variable(
            "Profile_Time",
            data["Profile_Time"],
            units="seconds...TAI",
            dims=("Profile_ID",),
        ),
        "prof_UTC_time": _variable(
            "Profile_UTC_Time",
            data["Profile_UTC_Time"],
            units="UTC - yymmdd.ffffffff",
            dims=("Profile_ID",),
        ),
        "lat": _variable(
            "Latitude",
            data["Latitude"],
            fill_value=FILL_VALUE_FLOAT,
            units="degrees",
            dims=("Profile_ID",),
        ),
        "lon": _variable(
            "Longitude",
            data["Longitude"],
            fill_value=FILL_VALUE_FLOAT,
            units="degrees",
            dims=("Profile_ID",),
        ),
        "alt": _variable(
            "Altitude",
            result.altitude,
            units="kilometer",
            dims=("Altitude",),
        ),
    }
    variables["prof_ID"].description = "Profile number from start of file"

    masks = (
        ("feature_mask_532_par", "Parallel_Detection_Flags_532"),
        ("feature_mask_532_per", "Perpendicular_Detection_Flags_532"),
        ("feature_mask_1064", "Detection_Flags_1064"),
        ("feature_mask_merged", "Composite_Detection_Flags"),
    )
    for key, name in masks:
        variables[key] = _variable(
            name,
            data[name],
            valid_range=(0, 255),
            dims=("Profile_ID", "Altitude"),
        )

    if not save_development_data:
        return variables

    development = result.development
    transmittances = (
        ("twoway_transmittance_532_par", "Parallel_CumulativeTwoWayTransmittance_532"),
        ("twoway_transmittance_532_per", "Perpendicular_CumulativeTwoWayTransmittance_532"),
        ("twoway_transmittance_1064", "CumulativeTwoWayTransmittance_1064"),
    )
    for key, name in transmittances:
        variables[key] = _variable(
            name,
            development[name],
            fill_value=FILL_VALUE_FLOAT,
            valid_range=(0.0, 1.0),
            dims=("Profile_ID", "Altitude"),
        )

    step_variables = (
        ("feature_mask_532_par_steps", "Parallel_Detection_Flags_532_steps", "Step_532_par", None),
        ("feature_mask_532_per_steps", "Perpendicular_Detection_Flags_532_steps", "Step_532_per", None),
        ("feature_mask_1064_steps", "Detection_Flags_1064_steps", "Step_1064", None),
        ("atsr_532_par_steps", "Parallel_Attenuated_Scattering_Ratio_532_steps", "Step_532_par", FILL_VALUE_FLOAT),
        ("atsr_532_per_steps", "Perpendicular_Attenuated_Scattering_Ratio_532_steps", "Step_532_per", FILL_VALUE_FLOAT),
        ("atsr_1064_steps", "Attenuated_Scattering_Ratio_1064_steps", "Step_1064", FILL_VALUE_FLOAT),
    )
    for key, name, step_dimension, fill_value in step_variables:
        variables[key] = _variable(
            name,
            development[name],
            fill_value=fill_value,
            valid_range=(0, 255) if fill_value is None else None,
            dims=(step_dimension, "Profile_ID", "Altitude"),
        )

    return variables


def output_filename(request, result):
    """Return the filename used by the legacy product."""

    whole_file = (
        request.subset_start == 0
        and request.subset_end is None
        and request.subset_mode == "profindex"
    )
    suffix = "" if whole_file else (
        f"_lon_{result.longitude_min:.2f}_{result.longitude_max:.2f}"
    )
    version = request.output_version.replace(".", "-")
    return (
        f"CAL_LID_L2_2D_McDA-{request.output_product_type}-{version}."
        f"{request.granule_date}{suffix}.hdf"
    )


def write_product(request, result):
    """Write one result and return its output path."""

    output_path = request.output_directory / output_filename(request, result)
    variables = build_output_variables(
        result,
        request.save_development_data,
    )
    write_hdf(str(output_path), variables)
    return output_path
