"""Physical calculations derived from native CALIOP measurements."""

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

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


class NoValidMolecularProfile(Exception):
    pass


def compute_ab_mol_and_b_mol(mol_nd, O3_nd, alt, met_alt, wl, polar=None):
    """
    Compute molecular attenuated backscatter and backscatter from molecular and ozone number
    density.

    :param mol_nd: molecular number density, DataArray (profile, met altitude) with NaN
                   where missing
    :param O3_nd: ozone number density, same layout
    :param alt: lidar data altitudes, DataArray (lidar altitude)
    :param met_alt: meteorological data altitudes, DataArray (met altitude)
    :return: molecular attenuated backscatter and backscatter, DataArrays (profile, lidar
             altitude), NaN for the profiles without a valid molecular model
    """

    # Initialization
    profile_dim = mol_nd.dims[0]
    nb_prof = mol_nd.sizes[profile_dim]
    ab_mol = np.full((nb_prof, alt.size), np.nan)
    b_mol = np.full((nb_prof, alt.size), np.nan)

    # Loop on profiles: the model is a 1D kernel on plain arrays
    mol_nd_values = mol_nd.values
    O3_nd_values = O3_nd.values
    for i in range(nb_prof):
        try:
            _, b_mol_i, T2_mol, T2_O3 = make_molecular_model(
                mol_nd_values[i, :],
                O3_nd_values[i, :],
                met_alt.values,
                alt.values,
                wl,
                polar,
            )
        except NoValidMolecularProfile as e:
            print(f"Profile {i}: {e}")
            continue
        b_mol[i, :] = b_mol_i
        ab_mol[i, :] = b_mol_i * T2_mol * T2_O3

    dims = (profile_dim, *alt.dims)
    coords = {profile_dim: mol_nd.coords[profile_dim]}
    return (
        xr.DataArray(ab_mol, dims=dims, coords=coords),
        xr.DataArray(b_mol, dims=dims, coords=coords),
    )


def make_molecular_model(mol_ND_met, O3_ND_met, Z_met, Z_data, wl, polar=None):
    # mol_ND_met (molecular number density) and O3_ND_met (ozone number density)
    # are 1-D arrays, with the max altitude at index 0 (I'm assuming these will be
    # 33-element met data arrays)
    #
    # Z_met is the altitude array corresponding to the mol_ND_met
    # and O3_ND_met array; Z_data is (intended to be) the standard
    # CALIPSO altitude array
    #
    # wl is an integer -- either 532 or 1064 -- specifying the wavelength
    # for the model
    #
    # the return value is an array of molecular attenuated backscatter
    # coefficients, with dimensions equal to the dimensions of Z_data

    # mol_ND_met (m^-3)
    # O3_ND_met (m^-3)
    # Z_met (km)
    # Z_data (km)
    # wl (nm)

    wavelengthOK = (wl == 532) | (wl == 1064)
    if not wavelengthOK:
        raise Exception(
            f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n"
        )

    # Define molecular and ozone cross section for each wavelength
    # see Table 4.2 in Hostetler et al. (2006; ATDB)
    if wl == 532:
        mol_backscatter_cross_sect = 5.982e-32  # (m^2 / sr^-1)
        mol_ext_cross_sect = 5.167e-31  # (m^2)
        O3_ext_cross_sect = 2.72846e-25  # (m^2)
        depolar = 0.00366  # depolarization ratio (b_per/b_par) for Cabannes
        # scattering
    else:  #  wl = 1064
        mol_backscatter_cross_sect = 3.620e-33  # (m^2 / sr^-1)
        mol_ext_cross_sect = 3.127e-32  # (m^2)

    # Handle fill values
    mol_ND_met = replace_fillvalue_with_lowest_valid(
        mol_ND_met,
        positive_only=True,
    )
    O3_ND_met = replace_fillvalue_with_lowest_valid(O3_ND_met)

    if mol_ND_met is None or O3_ND_met is None:
        raise NoValidMolecularProfile("All molecular or ozone values are fill_value")

    # Interpolate (using log) to get density values for all lidar data alt
    Z_met = np.asarray(Z_met, dtype=float)

    # Missing lidar altitudes are replaced with the reference CALIOP grid
    Z_data_values = np.asarray(Z_data, dtype=float)
    Z_data_mask = ~np.isfinite(Z_data_values)

    if np.any(Z_data_mask):
        reference_altitudes = np.asarray(
            LIDAR_DATA_ALTITUDES,
            dtype=float,
        )

        if reference_altitudes.shape != Z_data_values.shape:
            raise ValueError("LIDAR_DATA_ALTITUDES has an unexpected shape")

        Z_data_values = Z_data_values.copy()
        Z_data_values[Z_data_mask] = reference_altitudes[Z_data_mask]

    Z_data = Z_data_values

    interp_log_mol = interp1d(
        Z_met, np.log(mol_ND_met), bounds_error=False, fill_value="extrapolate"
    )
    mol_ND_data = np.exp(interp_log_mol(Z_data))
    if False:
        ax = plt.subplot(111)
        plt.plot(mol_ND_met, Z_met, marker="o", c="r", label="met", zorder=-1)
        plt.scatter(mol_ND_data, Z_data, s=2, label="data")
        # ax.set_xscale('log')
        plt.legend()
        plt.title("Molecular number density")
        plt.show()

    # Convert number density to molecular backscatter coefficients
    beta_mol = 1000.0 * mol_backscatter_cross_sect * mol_ND_data  # (km^-1 / sr^-1)

    if polar == "par":
        beta_mol = beta_mol / (1 + depolar)
    elif polar == "per":
        beta_mol = beta_mol * depolar / (1 + depolar)

    # Convert number density to molecular extinction coefficients
    ext_mol = 1000.0 * mol_ext_cross_sect * mol_ND_data  # (km^-1)

    # Derive molecular two-way transmittance values from the extinction
    # coefficient
    T2_mol = extinction2two_way_transmittance(ext_mol, Z_data)

    if wl == 532:

        # Interpolate to get density values for all lidar data alt
        # f = interp1d(Z_met, O3_ND_met)
        # O3_ND_data = f(Z_data)
        interp_O3 = interp1d(
            Z_met, O3_ND_met, bounds_error=False, fill_value="extrapolate"
        )
        O3_ND_data = interp_O3(Z_data)
        if False:
            plt.plot(O3_ND_met, Z_met, marker="o", c="r", label="met", zorder=-1)
            plt.scatter(O3_ND_data, Z_data, s=2, label="data")
            plt.legend()
            plt.title("Molecular number density")
            plt.show()

        # Convert number density to molecular extinction coefficients
        ext_O3 = 1000.0 * O3_ext_cross_sect * O3_ND_data  # (km^-1)

        # Derive O3 two-way transmittance values from the extinction
        # coefficient
        T2_O3 = extinction2two_way_transmittance(ext_O3, Z_data)

    else:  # 1064 nm
        # O3 two-way transmittance = 1
        T2_O3 = np.ones(T2_mol.size)

    return mol_ND_data, beta_mol, T2_mol, T2_O3


def replace_fillvalue_with_lowest_valid(
    ND_met,
    fill_value=FILL_VALUE_FLOAT,
    positive_only=False,
):
    """Replace missing (NaN) or fill values with the lowest valid value."""

    values = np.asarray(ND_met, dtype=float)

    valid = np.isfinite(values) & (values != fill_value)

    if positive_only:
        valid &= values > 0

    if not np.any(valid):
        return None

    lowest_valid = values[np.flatnonzero(valid)[-1]]

    values = values.copy()
    values[~valid] = lowest_valid

    return values


def extinction2two_way_transmittance(sigma, Z):
    """Use trapezoid integration to convert extinction coefficients to
    optical depths and derive two-way transmittances

    Args:
        sigma (_type_): extinction coefficients
        Z (_type_): corresponding altitudes

    Returns:
        _type_: two-way transmittance values
    """
    dz = -np.diff(Z, prepend=Z[0])  # Prepend avoids mismatch in size
    optical_depth = np.cumsum((sigma + np.roll(sigma, 1)) * dz / 2)
    optical_depth[0] = 0  # Ensure first value is 0

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
