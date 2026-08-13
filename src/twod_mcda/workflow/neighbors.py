"""Adjoin profiles from neighboring CALIOP granules."""

import numpy as np


def append_adjacent_profiles(current, adjacent, side):
    """Append all profile-dependent variables from an adjacent granule."""

    if side not in {"start", "end"}:
        raise ValueError(f"Invalid side: {side!r}")

    result = dict(current)
    for name, current_values in current.items():
        if name == "Lidar_Data_Altitudes":
            continue
        adjacent_values = adjacent[name]
        arrays = (
            (adjacent_values, current_values)
            if side == "start"
            else (current_values, adjacent_values)
        )
        result[name] = np.append(*arrays, axis=0)

    return result


def profiles_are_consecutive(first_time, second_time):
    """Return whether two boundary profiles are less than one second apart."""

    return np.abs(second_time - first_time) < 1
