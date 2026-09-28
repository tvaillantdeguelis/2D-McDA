"""Physical calculations derived from native CALIOP measurements."""

from typing import NamedTuple

import numpy as np
import xarray as xr

from twod_mcda.caliop.constants import (
    FILL_VALUE_FLOAT,
    LIDAR_ALTITUDE_DIMENSION,
    LAYER_ALTITUDE_R1_INDEX_RANGE,
    LAYER_ALTITUDE_R2_INDEX_RANGE,
    LAYER_ALTITUDE_R3_INDEX_RANGE,
    LAYER_ALTITUDE_R4_INDEX_RANGE,
    LAYER_ALTITUDE_R5_INDEX_RANGE,
    LIDAR_DATA_ALTITUDES,
    NUMBER_OF_VERTICAL_BINS,
    N_15M_BINS_PER_BIN_R1,
    N_15M_BINS_PER_BIN_R2,
    N_15M_BINS_PER_BIN_R3,
    N_15M_BINS_PER_BIN_R4,
    N_15M_BINS_PER_BIN_R5,
    N_333M_BINS_PER_BIN_R1,
    N_333M_BINS_PER_BIN_R2,
    N_333M_BINS_PER_BIN_R3,
    N_333M_BINS_PER_BIN_R4,
    N_333M_BINS_PER_BIN_R5,
)


def compute_par_ab532(tot_ab532, per_ab532, dim=LIDAR_ALTITUDE_DIMENSION):
    """
    Compute parallel attenuated backscatter at 532 nm as the difference between total
    attenuated backscatter at 532 nm and perpendicular attenuated backscatter at 532 nm

    :param tot_ab532: total attenuated backscatter at 532 nm, as a DataArray with NaN
                      where missing
    :param per_ab532: perpendicular attenuated backscatter at 532 nm, same layout
    :param dim: vertical dimension along which a profile is summed
    """

    # Missing perpendicular values count as 0, so that they do not mask the
    # parallel channel
    par_ab532 = tot_ab532 - per_ab532.fillna(0)

    # Mask par_ab532 where 0 (all is due to per => fill value was in par)
    # check if the whole profile is 0 in order not to mask isolated pixel with
    # value exactly equal to 0 by chance
    return par_ab532.where(par_ab532.sum(dim) != 0)


# Molecular and ozone cross sections for each wavelength, see Table 4.2 in
# Hostetler et al. (2006; ATBD)
MOL_BACKSCATTER_CROSS_SECT = {532: 5.982e-32, 1064: 3.620e-33}  # (m^2 / sr^-1)
MOL_EXT_CROSS_SECT = {532: 5.167e-31, 1064: 3.127e-32}  # (m^2)
O3_EXT_CROSS_SECT_532 = 2.72846e-25  # (m^2), negligible at 1064 nm
DEPOLAR_532 = 0.00366  # depolarization ratio (b_per/b_par) for Cabannes scattering


class MolecularModel(NamedTuple):
    """Unpolarized molecular model of one wavelength.

    Every array is indexed (profile, lidar altitude), NaN for the profiles
    without a valid model. ``T2_O3`` is None at 1064 nm, where the ozone
    two-way transmittance is 1.
    """

    beta_mol: xr.DataArray
    T2_mol: xr.DataArray
    T2_O3: xr.DataArray | None


def compute_molecular_model(mol_nd, O3_nd, alt, met_alt, wl):
    """
    Compute the molecular backscatter and two-way transmittances of every profile from
    the molecular and ozone number densities.

    :param mol_nd: molecular number density (m^-3), DataArray (profile, met altitude)
                   with NaN where missing, max altitude at index 0
    :param O3_nd: ozone number density (m^-3), same layout
    :param alt: lidar data altitudes (km), DataArray (lidar altitude)
    :param met_alt: meteorological data altitudes (km), DataArray (met altitude)
    :param wl: wavelength (nm), 532 or 1064
    :return: MolecularModel, NaN for the profiles without a valid molecular model
    """

    if wl not in (532, 1064):
        raise Exception(
            f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n"
        )

    # Handle fill values; a profile without any valid value has no model
    mol_ND_met, mol_invalid = replace_fillvalue_with_lowest_valid(
        mol_nd.values, positive_only=True
    )
    O3_ND_met, O3_invalid = replace_fillvalue_with_lowest_valid(O3_nd.values)
    invalid_profiles = mol_invalid | O3_invalid
    if np.any(invalid_profiles):
        print(
            f"\t{np.count_nonzero(invalid_profiles)} profiles without valid "
            "molecular or ozone values: no molecular model there"
        )

    # Missing lidar altitudes are replaced with the reference CALIOP grid
    Z_met = np.asarray(met_alt, dtype=float)
    Z_data = np.asarray(alt, dtype=float)
    Z_data_mask = ~np.isfinite(Z_data)
    if np.any(Z_data_mask):
        reference_altitudes = np.asarray(LIDAR_DATA_ALTITUDES, dtype=float)
        if reference_altitudes.shape != Z_data.shape:
            raise ValueError("LIDAR_DATA_ALTITUDES has an unexpected shape")
        Z_data = Z_data.copy()
        Z_data[Z_data_mask] = reference_altitudes[Z_data_mask]

    # Interpolate (using log) to get density values for all lidar data alt
    interpolate = _linear_interpolator(Z_met, Z_data)
    mol_ND_data = np.exp(interpolate(np.log(mol_ND_met)))

    # Convert number density to molecular backscatter and extinction coefficients
    beta_mol = 1000.0 * MOL_BACKSCATTER_CROSS_SECT[wl] * mol_ND_data  # (km^-1 sr^-1)
    ext_mol = 1000.0 * MOL_EXT_CROSS_SECT[wl] * mol_ND_data  # (km^-1)

    # Derive molecular two-way transmittance values from the extinction
    # coefficient
    T2_mol = extinction2two_way_transmittance(ext_mol, Z_data)

    T2_O3 = None
    if wl == 532:
        # Same for the ozone, interpolated without log
        ext_O3 = 1000.0 * O3_EXT_CROSS_SECT_532 * interpolate(O3_ND_met)  # (km^-1)
        T2_O3 = extinction2two_way_transmittance(ext_O3, Z_data)

    def labelled(values):
        values[invalid_profiles, :] = np.nan
        return xr.DataArray(
            values,
            dims=(mol_nd.dims[0], *alt.dims),
            coords={mol_nd.dims[0]: mol_nd.coords[mol_nd.dims[0]]},
        )

    return MolecularModel(
        labelled(beta_mol),
        labelled(T2_mol),
        None if T2_O3 is None else labelled(T2_O3),
    )


def compute_ab_mol_and_b_mol(model, polar):
    """
    Return the molecular attenuated backscatter and backscatter of one channel.

    :param model: MolecularModel of the channel's wavelength
    :param polar: "par" or "per" for the 532 nm polarized channels, "" otherwise
    :return: molecular attenuated backscatter and backscatter, DataArrays (profile,
             lidar altitude)
    """

    b_mol = model.beta_mol
    if polar == "par":
        b_mol = b_mol / (1 + DEPOLAR_532)
    elif polar == "per":
        b_mol = b_mol * DEPOLAR_532 / (1 + DEPOLAR_532)

    ab_mol = b_mol * model.T2_mol
    if model.T2_O3 is not None:
        ab_mol = ab_mol * model.T2_O3
    return ab_mol, b_mol


def _linear_interpolator(x, x_new):
    """Return a function interpolating rows of values given at ``x`` to ``x_new``.

    The interpolation weights only depend on the altitudes, shared by every
    profile, so they are computed once. Same formula as ``scipy.interpolate.
    interp1d(x, y, fill_value="extrapolate")``: the first and last segments are
    extended beyond the range of ``x``.
    """

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    hi = np.searchsorted(x, x_new).clip(1, len(x) - 1)
    lo = hi - 1
    w_hi = (x_new - x[lo]) / (x[hi] - x[lo])
    w_lo = (x[hi] - x_new) / (x[hi] - x[lo])

    def interpolate(values):
        values = values[:, order]
        return w_hi * values[:, hi] + w_lo * values[:, lo]

    return interpolate


def replace_fillvalue_with_lowest_valid(
    ND_met,
    fill_value=FILL_VALUE_FLOAT,
    positive_only=False,
):
    """Replace missing (NaN) or fill values with the lowest valid value.

    ``ND_met`` is indexed (profile, met altitude), max altitude at index 0, so
    the lowest valid value of a profile is its last valid one. Returns the
    completed values and whether each profile has no valid value at all.
    """

    values = np.asarray(ND_met, dtype=float)

    valid = np.isfinite(values) & (values != fill_value)

    if positive_only:
        valid &= values > 0

    no_valid = ~np.any(valid, axis=1)
    last_valid = values.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    lowest_valid = values[np.arange(values.shape[0]), last_valid]
    # A profile without any valid value becomes missing (NaN) everywhere
    lowest_valid[no_valid] = np.nan

    return np.where(valid, values, lowest_valid[:, np.newaxis]), no_valid


def extinction2two_way_transmittance(sigma, Z):
    """Use trapezoid integration to convert extinction coefficients to
    optical depths and derive two-way transmittances

    Args:
        sigma: extinction coefficients, indexed (profile, altitude)
        Z: corresponding altitudes, from top to bottom

    Returns:
        two-way transmittance values, same shape as sigma
    """
    dz = -np.diff(Z, prepend=Z[0])  # Prepend avoids mismatch in size
    optical_depth = np.cumsum((sigma + np.roll(sigma, 1, axis=1)) * dz / 2, axis=1)
    optical_depth[:, 0] = 0  # Ensure first value is 0

    return np.exp(-2 * optical_depth)


def nsf_from_V_domain_to_betap_domain(
    nsf, r_alt, laser_energy, calib, pgr=np.array((1,))
):
    return nsf * np.sqrt(r_alt**2 / (laser_energy * calib * pgr))


def rms_from_P_domain_to_betap_domain(
    rms, r_alt, laser_energy, gain, calib, pgr=np.array((1,))
):
    return rms * r_alt**2 / (laser_energy * gain * calib * pgr)


def range_from_altitude(spacecraft_alt, data_alt, caliop_lidar_tilt):
    """Return the range between the spacecraft and a lidar altitude bin."""
    # CALIOP stores the tilt in float32: take its cosine in double precision
    tilt = caliop_lidar_tilt.astype(np.float64) * np.pi / 180.0
    return (spacecraft_alt - data_alt) / np.cos(tilt)


def _correction_values(fcorr, bin_shifts):
    """Select correction values from integer-valued CALIOP bin shifts.

    :param fcorr: correction table, DataArray (lidar altitude, bin_shift)
    :param bin_shifts: absolute bin shift of each profile, DataArray (profile) with NaN
                       where missing
    :return: DataArray (profile, lidar altitude), NaN where the bin shift is missing
    """

    valid = np.isfinite(bin_shifts)
    values = bin_shifts.values[valid.values]
    if values.size and not np.allclose(values, np.rint(values)):
        raise ValueError("Number_Bins_Shift contains non-integer values")

    indices = bin_shifts.where(valid, 0).astype(np.intp)
    if indices.size and (
        indices.min() < 0 or indices.max() >= fcorr.sizes["bin_shift"]
    ):
        raise IndexError("Number_Bins_Shift is outside the correction table")

    correction = fcorr.isel(bin_shift=indices).where(valid)
    return correction.transpose(*bin_shifts.dims, ...)


def compute_shotnoise(fcorr, nb_bins_shift_abs, nb_pixels, nsf, mol_ab):
    correction = _correction_values(fcorr, nb_bins_shift_abs)
    return correction * 1 / np.sqrt(nb_pixels) * nsf * 1 / np.sqrt(mol_ab)


def compute_backgroundnoise(fcorr, nb_bins_shift_abs, nb_pixels, rms, mol_ab):
    correction = _correction_values(fcorr, nb_bins_shift_abs)
    return correction * 1 / np.sqrt(nb_pixels) * 1 / mol_ab * rms


def get_caliop_correction_function(wl):
    """Input: wl = wavelength of the lidar channel
    Output: fcorr = correction function at each altitude level from
                    Table 2 of Liu (2011), updated values sent by M.
                    Vaughan"""

    fcorr = np.ones((NUMBER_OF_VERTICAL_BINS, 11))  # fcorr[bin index range, Nshift]
    fcorr[LAYER_ALTITUDE_R5_INDEX_RANGE[0] : LAYER_ALTITUDE_R5_INDEX_RANGE[1] + 1] = [
        1.596,
        1.448,
        1.322,
        1.224,
        1.161,
        1.140,
        1.161,
        1.224,
        1.322,
        1.448,
        1.596,
    ]
    fcorr[LAYER_ALTITUDE_R4_INDEX_RANGE[0] : LAYER_ALTITUDE_R4_INDEX_RANGE[1] + 1] = [
        1.573,
        1.345,
        1.188,
        1.131,
        1.188,
        1.345,
        1.573,
        1.345,
        1.188,
        1.130,
        1.188,
    ]
    fcorr[LAYER_ALTITUDE_R3_INDEX_RANGE[0] : LAYER_ALTITUDE_R3_INDEX_RANGE[1] + 1] = [
        1.451,
        1.080,
        1.451,
        1.080,
        1.451,
        1.080,
        1.451,
        1.080,
        1.451,
        1.080,
        1.451,
    ]
    if wl == 532:
        fcorr[
            LAYER_ALTITUDE_R2_INDEX_RANGE[0] : LAYER_ALTITUDE_R2_INDEX_RANGE[1] + 1
        ] = [
            1.269,
            1.269,
            1.269,
            1.269,
            1.269,
            1.269,
            1.269,
            1.269,
            1.269,
            1.269,
            1.269,
        ]
    elif wl == 1064:
        fcorr[
            LAYER_ALTITUDE_R2_INDEX_RANGE[0] : LAYER_ALTITUDE_R2_INDEX_RANGE[1] + 1
        ] = [
            1.451,
            1.451,
            1.451,
            1.451,
            1.451,
            1.451,
            1.451,
            1.451,
            1.451,
            1.451,
            1.451,
        ]
    else:
        raise ValueError(f"Unrecognized wavelength: {wl}; use 532 or 1064 instead")
    fcorr[LAYER_ALTITUDE_R1_INDEX_RANGE[0] : LAYER_ALTITUDE_R1_INDEX_RANGE[1] + 1] = [
        1.596,
        1.448,
        1.322,
        1.224,
        1.161,
        1.140,
        1.161,
        1.224,
        1.322,
        1.448,
        1.596,
    ]

    return xr.DataArray(fcorr, dims=(LIDAR_ALTITUDE_DIMENSION, "bin_shift"))


def get_nb_pixels(wl):
    """Input: wl = wavelength of the lidar channel
    Output: nb_pixels = number of original resolution lidar "pixels"
                        (bins) averaged together at each altitude
                        level at the wavelength channel wl"""

    nb_pixels = np.ones(583) * -9999.0

    if wl == 532:
        nb_pixels[
            LAYER_ALTITUDE_R5_INDEX_RANGE[0] : LAYER_ALTITUDE_R5_INDEX_RANGE[1] + 1
        ] = (N_333M_BINS_PER_BIN_R5 * N_15M_BINS_PER_BIN_R5)
        nb_pixels[
            LAYER_ALTITUDE_R4_INDEX_RANGE[0] : LAYER_ALTITUDE_R4_INDEX_RANGE[1] + 1
        ] = (N_333M_BINS_PER_BIN_R4 * N_15M_BINS_PER_BIN_R4)
        nb_pixels[
            LAYER_ALTITUDE_R3_INDEX_RANGE[0] : LAYER_ALTITUDE_R3_INDEX_RANGE[1] + 1
        ] = (N_333M_BINS_PER_BIN_R3 * N_15M_BINS_PER_BIN_R3)
        nb_pixels[
            LAYER_ALTITUDE_R2_INDEX_RANGE[0] : LAYER_ALTITUDE_R2_INDEX_RANGE[1] + 1
        ] = (N_333M_BINS_PER_BIN_R2 * N_15M_BINS_PER_BIN_R2)
        nb_pixels[
            LAYER_ALTITUDE_R1_INDEX_RANGE[0] : LAYER_ALTITUDE_R1_INDEX_RANGE[1] + 1
        ] = (N_333M_BINS_PER_BIN_R1 * N_15M_BINS_PER_BIN_R1)

    elif wl == 1064:
        nb_pixels[
            LAYER_ALTITUDE_R5_INDEX_RANGE[0] : LAYER_ALTITUDE_R5_INDEX_RANGE[1] + 1
        ] = (N_333M_BINS_PER_BIN_R5 * N_15M_BINS_PER_BIN_R5)
        nb_pixels[
            LAYER_ALTITUDE_R4_INDEX_RANGE[0] : LAYER_ALTITUDE_R4_INDEX_RANGE[1] + 1
        ] = (N_333M_BINS_PER_BIN_R4 * N_15M_BINS_PER_BIN_R4)
        nb_pixels[
            LAYER_ALTITUDE_R3_INDEX_RANGE[0] : LAYER_ALTITUDE_R3_INDEX_RANGE[1] + 1
        ] = (N_333M_BINS_PER_BIN_R3 * N_15M_BINS_PER_BIN_R3)
        nb_pixels[
            LAYER_ALTITUDE_R2_INDEX_RANGE[0] : LAYER_ALTITUDE_R2_INDEX_RANGE[1] + 1
        ] = (
            N_333M_BINS_PER_BIN_R2 * N_15M_BINS_PER_BIN_R2 * 2
        )  # average at 60 m instead of 30 m in original data
        nb_pixels[
            LAYER_ALTITUDE_R1_INDEX_RANGE[0] : LAYER_ALTITUDE_R1_INDEX_RANGE[1] + 1
        ] = (N_333M_BINS_PER_BIN_R1 * N_15M_BINS_PER_BIN_R1)

    else:
        raise ValueError(f"Unrecognized wavelength: {wl}; use 532 or 1064 instead")

    return xr.DataArray(nb_pixels, dims=(LIDAR_ALTITUDE_DIMENSION,))
