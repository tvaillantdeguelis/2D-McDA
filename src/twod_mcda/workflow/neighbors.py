"""Adjoin profiles from neighboring CALIOP granules."""

import xarray as xr

from twod_mcda.caliop.input import open_granule, read_slice


def read_adjacent_profiles(request, granule_date, directory, profile_start, profile_end):
    """Load context profiles from one adjacent granule, then close its file."""

    with open_granule(
        request,
        granule_date,
        directory,
        profile_start,
        profile_end,
    ) as adjacent_granule:
        adjacent_profiles = read_slice(
            adjacent_granule,
            adjacent_granule.prof_min,
            adjacent_granule.prof_max,
        )
        adjacent_granule_path = adjacent_granule.filepath

    return adjacent_profiles, adjacent_granule_path


def append_adjacent_profiles(current, adjacent, side):
    """Append all profile-dependent variables from an adjacent granule."""

    if side not in {"start", "end"}:
        raise ValueError(f"Invalid side: {side!r}")

    arrays = (
        (adjacent, current)
        if side == "start"
        else (current, adjacent)
    )
    return xr.concat(
        arrays,
        dim="profile",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="override",
    )


def profiles_are_consecutive(first_time, second_time):
    """Return whether two boundary profiles are less than one second apart."""

    difference = abs(second_time - first_time)
    return bool(difference.item() < 1)
