"""Adjoin profiles from neighboring CALIOP granules."""

import xarray as xr


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
