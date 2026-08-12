"""Build and write the CF-oriented netCDF-4 product produced by 2D-McDA."""

from datetime import datetime, timedelta, timezone

import numpy as np

from twod_mcda.calipso_constants import FILL_VALUE_FLOAT
from twod_mcda.io.netcdf_writer import NetCDFVariable, write_netcdf


PROFILE_DIMENSION = "Profile_ID"
ALTITUDE_DIMENSION = "Altitude"
PROFILE_COORDINATES = "Profile_Time Latitude Longitude"
CHANNEL_FLAG_VALUES = (0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255)
CHANNEL_FLAG_MEANINGS = (
    "no_detection detection_level_1 detection_level_2 detection_level_3 "
    "detection_level_4 detection_level_5 low_confidence_small_strip "
    "almost_fully_attenuated fully_attenuated likely_artifact "
    "surface_or_subsurface candidate_detection"
)
COMPOSITE_FLAG_MASKS = (7, 7, 7, 7, 7, 7, 8, 16, 32)
COMPOSITE_FLAG_VALUES = (0, 1, 2, 3, 5, 7, 8, 16, 32)
COMPOSITE_FLAG_MEANINGS = (
    "invalid clear_air atmospheric_feature low_confidence "
    "surface_or_subsurface fully_attenuated "
    "parallel_532_detection perpendicular_532_detection 1064_detection"
)


def _variable(
    name,
    data,
    *,
    dimensions=(),
    fill_value=None,
    compress=True,
    **attributes,
):
    return NetCDFVariable(
        name=name,
        data=data,
        dimensions=tuple(dimensions),
        fill_value=fill_value,
        attributes=attributes,
        compress=compress,
    )


def build_output_variables(result, save_development_data):
    """Translate processing arrays to a CF-oriented netCDF schema."""

    data = result.data
    variables = {
        "trajectory_id": _variable(
            "Trajectory_ID",
            np.asarray(1, dtype=np.int32),
            compress=False,
            cf_role="trajectory_id",
            long_name="CALIOP granule trajectory identifier",
        ),
        "prof_ID": _variable(
            "Profile_ID",
            data["Profile_ID"],
            dimensions=(PROFILE_DIMENSION,),
            compress=False,
            long_name="profile number from the start of the source granule",
        ),
        "prof_time": _variable(
            "Profile_Time",
            data["Profile_Time"],
            dimensions=(PROFILE_DIMENSION,),
            compress=False,
            standard_name="time",
            long_name="profile observation time",
            units="seconds since 1993-01-01 00:00:00",
            calendar="tai",
            axis="T",
        ),
        "prof_UTC_time": _variable(
            "Profile_UTC_Time",
            data["Profile_UTC_Time"],
            dimensions=(PROFILE_DIMENSION,),
            fill_value=FILL_VALUE_FLOAT,
            compress=False,
            long_name="CALIOP encoded UTC profile time",
            comment=(
                "Original CALIOP UTC representation: yymmdd.ffffffff, "
                "where ffffffff is the fractional part of the UTC day."
            ),
        ),
        "lat": _variable(
            "Latitude",
            data["Latitude"],
            dimensions=(PROFILE_DIMENSION,),
            fill_value=FILL_VALUE_FLOAT,
            compress=False,
            standard_name="latitude",
            long_name="profile latitude",
            units="degrees_north",
            valid_range=(-90.0, 90.0),
            axis="Y",
        ),
        "lon": _variable(
            "Longitude",
            data["Longitude"],
            dimensions=(PROFILE_DIMENSION,),
            fill_value=FILL_VALUE_FLOAT,
            compress=False,
            standard_name="longitude",
            long_name="profile longitude",
            units="degrees_east",
            valid_range=(-180.0, 180.0),
            axis="X",
        ),
        "alt": _variable(
            "Altitude",
            result.altitude,
            dimensions=(ALTITUDE_DIMENSION,),
            compress=False,
            standard_name="altitude",
            long_name="altitude above mean sea level",
            units="km",
            positive="up",
            axis="Z",
        ),
    }

    masks = (
        ("feature_mask_532_par", "Parallel_Detection_Flags_532", "532 nm parallel-channel detection flags"),
        ("feature_mask_532_per", "Perpendicular_Detection_Flags_532", "532 nm perpendicular-channel detection flags"),
        ("feature_mask_1064", "Detection_Flags_1064", "1064 nm detection flags"),
        ("feature_mask_merged", "Composite_Detection_Flags", "composite three-channel detection flags"),
    )
    for key, name, long_name in masks:
        if name == "Composite_Detection_Flags":
            flag_attributes = {
                "flag_masks": COMPOSITE_FLAG_MASKS,
                "flag_values": COMPOSITE_FLAG_VALUES,
                "flag_meanings": COMPOSITE_FLAG_MEANINGS,
            }
            valid_range = (0, 63)
        else:
            flag_attributes = {
                "flag_values": CHANNEL_FLAG_VALUES,
                "flag_meanings": CHANNEL_FLAG_MEANINGS,
            }
            valid_range = (0, 255)
        variables[key] = _variable(
            name,
            data[name],
            dimensions=(PROFILE_DIMENSION, ALTITUDE_DIMENSION),
            long_name=long_name,
            valid_range=valid_range,
            coordinates=PROFILE_COORDINATES,
            **flag_attributes,
        )

    if not save_development_data:
        return variables

    development = result.development
    transmittances = (
        ("twoway_transmittance_532_par", "Parallel_CumulativeTwoWayTransmittance_532", "532 nm parallel cumulative two-way transmittance"),
        ("twoway_transmittance_532_per", "Perpendicular_CumulativeTwoWayTransmittance_532", "532 nm perpendicular cumulative two-way transmittance"),
        ("twoway_transmittance_1064", "CumulativeTwoWayTransmittance_1064", "1064 nm cumulative two-way transmittance"),
    )
    for key, name, long_name in transmittances:
        variables[key] = _variable(
            name,
            development[name],
            dimensions=(PROFILE_DIMENSION, ALTITUDE_DIMENSION),
            fill_value=FILL_VALUE_FLOAT,
            long_name=long_name,
            units="1",
            valid_range=(0.0, 1.0),
            coordinates=PROFILE_COORDINATES,
        )

    step_variables = (
        ("feature_mask_532_par_steps", "Parallel_Detection_Flags_532_steps", "Step_532_par", None, "intermediate 532 nm parallel-channel detection flags"),
        ("feature_mask_532_per_steps", "Perpendicular_Detection_Flags_532_steps", "Step_532_per", None, "intermediate 532 nm perpendicular-channel detection flags"),
        ("feature_mask_1064_steps", "Detection_Flags_1064_steps", "Step_1064", None, "intermediate 1064 nm detection flags"),
        ("atsr_532_par_steps", "Parallel_Attenuated_Scattering_Ratio_532_steps", "Step_532_par", FILL_VALUE_FLOAT, "intermediate 532 nm parallel attenuated scattering ratio"),
        ("atsr_532_per_steps", "Perpendicular_Attenuated_Scattering_Ratio_532_steps", "Step_532_per", FILL_VALUE_FLOAT, "intermediate 532 nm perpendicular attenuated scattering ratio"),
        ("atsr_1064_steps", "Attenuated_Scattering_Ratio_1064_steps", "Step_1064", FILL_VALUE_FLOAT, "intermediate 1064 nm attenuated scattering ratio"),
    )
    for key, name, step_dimension, fill_value, long_name in step_variables:
        flag_attributes = {}
        if fill_value is None:
            flag_attributes = {
                "flag_values": CHANNEL_FLAG_VALUES,
                "flag_meanings": CHANNEL_FLAG_MEANINGS,
            }
        variables[key] = _variable(
            name,
            development[name],
            dimensions=(step_dimension, PROFILE_DIMENSION, ALTITUDE_DIMENSION),
            fill_value=fill_value,
            long_name=long_name,
            units="1" if fill_value is not None else None,
            valid_range=(0, 255) if fill_value is None else None,
            coordinates=PROFILE_COORDINATES,
            **flag_attributes,
        )

    return variables


def output_filename(request, result):
    """Return the netCDF product filename."""

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
        f"{request.granule_date}{suffix}.nc"
    )


def _valid_values(values):
    values = np.ma.asarray(values)
    values = np.ma.masked_invalid(values)
    values = np.ma.masked_equal(values, FILL_VALUE_FLOAT)
    return values.compressed()


def _utc_datetime(value):
    """Decode the native CALIOP ``yymmdd.fraction`` UTC representation."""

    encoded_date = int(value)
    year = 2000 + encoded_date // 10000
    month = encoded_date // 100 % 100
    day = encoded_date % 100
    fraction = float(value) - encoded_date
    return datetime(year, month, day, tzinfo=timezone.utc) + timedelta(
        days=fraction
    )


def _global_attributes(request, result, filename):
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    attributes = {
        "Conventions": "CF-1.13",
        "title": "2D-McDA CALIOP feature detection product",
        "summary": (
            "Two-dimensional, multi-channel feature detection masks derived "
            "from CALIOP Level 1 attenuated-backscatter profiles."
        ),
        "id": filename,
        "naming_authority": "org.github.tvaillantdeguelis",
        "source": (
            f"CALIOP Level 1 {request.caliop_product_type} "
            f"{request.caliop_version}"
        ),
        "processing_level": "L2",
        "product_version": request.output_version,
        "featureType": "trajectoryProfile",
        "date_created": created,
        "history": f"{created}: generated by 2D-McDA {request.output_version}",
        "references": "https://github.com/tvaillantdeguelis/2D-McDA",
    }

    latitude = _valid_values(result.data["Latitude"])
    longitude = _valid_values(result.data["Longitude"])
    altitude = _valid_values(result.altitude)
    if latitude.size:
        attributes.update(
            geospatial_lat_min=float(latitude.min()),
            geospatial_lat_max=float(latitude.max()),
            geospatial_lat_units="degrees_north",
        )
    if longitude.size:
        attributes.update(
            geospatial_lon_min=float(longitude.min()),
            geospatial_lon_max=float(longitude.max()),
            geospatial_lon_units="degrees_east",
        )
    if altitude.size:
        attributes.update(
            geospatial_vertical_min=float(altitude.min()),
            geospatial_vertical_max=float(altitude.max()),
            geospatial_vertical_units="km",
            geospatial_vertical_positive="up",
        )

    utc_time = _valid_values(result.data["Profile_UTC_Time"])
    if utc_time.size:
        try:
            start = _utc_datetime(utc_time.min())
            end = _utc_datetime(utc_time.max())
        except (OverflowError, ValueError):
            pass
        else:
            attributes["time_coverage_start"] = start.isoformat().replace(
                "+00:00", "Z"
            )
            attributes["time_coverage_end"] = end.isoformat().replace(
                "+00:00", "Z"
            )

    return attributes


def write_product(request, result):
    """Write one netCDF-4 result and return its output path."""

    filename = output_filename(request, result)
    output_path = request.output_directory / filename
    variables = build_output_variables(
        result,
        request.save_development_data,
    )
    write_netcdf(
        output_path,
        variables.values(),
        _global_attributes(request, result, filename),
    )
    return output_path
