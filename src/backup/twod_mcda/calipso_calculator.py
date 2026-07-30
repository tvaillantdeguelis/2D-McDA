import sys

import numpy as np
from scipy.interpolate import interp1d

from .calipso_constants import *

def compute_par_ab532(tot_ab532, per_ab532):
    """
    Compute parallel attenuated backscatter at 532 nm as the difference between total
    attenuated backscatter at 532 nm and perpendicular attenuated backscatter at 532 nm
    """

    par_ab532 = tot_ab532 - per_ab532.filled(0) # filled mask value of per with 0 in order not
                                                # to get filled value here

    # Mask par_ab532 where 0 (all is due to per => fill value was in par)
    # check if the whole column is 0 in order not to mask isolated pixel with
    # value exactly equal to 0 by chance
    par_ab532 = np.ma.masked_where(np.repeat(np.sum(par_ab532, axis=1),
                                   par_ab532.shape[1]).reshape(par_ab532.shape)==0, par_ab532)

    return par_ab532

class NoValidMolecularProfile(Exception):
    pass


def compute_ab_mol_and_b_mol(mol_nd, O3_nd, alt, met_alt, wl, polar=None):
    """
    Compute molecular attenuated backscatter and backscatter from molecular and ozone number
    density.
    """

    # Initialization
    nb_prof = mol_nd.shape[0]
    ab_mol = np.ones((nb_prof, alt.size))*FILL_VALUE_FLOAT
    b_mol = np.ones((nb_prof, alt.size))*FILL_VALUE_FLOAT

    # Loop on profiles
    for i in range(nb_prof):
        try:
            _, b_mol_i, T2_mol, T2_O3 = make_molecular_model(mol_nd[i, :], O3_nd[i, :], met_alt,
                                                            alt, wl, polar)
        except NoValidMolecularProfile as e:
            print(f"Profile {i}: {e}")
            continue
        b_mol  [i, :] = b_mol_i
        ab_mol[i, :] = b_mol_i*T2_mol*T2_O3

    return ab_mol, b_mol


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
        raise Exception(f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n")
    
    # Define molecular and ozone cross section for each wavelength
    # see Table 4.2 in Hostetler et al. (2006; ATDB)
    if wl == 532:
        mol_backscatter_cross_sect = 5.982e-32 # (m^2 / sr^-1)
        mol_ext_cross_sect = 5.167e-31 # (m^2)
        O3_ext_cross_sect = 2.72846e-25 # (m^2)
        depolar = 0.00366 # depolarization ratio (b_per/b_par) for Cabannes
                          # scattering
    else: #  wl = 1064
        mol_backscatter_cross_sect = 3.620e-33 # (m^2 / sr^-1)
        mol_ext_cross_sect  = 3.127e-32 # (m^2)
    
    # Handle fill values
    mol_ND_met = replace_fillvalue_with_lowest_valid(mol_ND_met)
    O3_ND_met = replace_fillvalue_with_lowest_valid(O3_ND_met)
    
    if mol_ND_met is None or O3_ND_met is None:
        raise NoValidMolecularProfile("All molecular or ozone values are fill_value")

    # Interpolate (using log) to get density values for all lidar data alt
    # Z_data = np.ma.filled(Z_data, -9999.) # pass in ndarray because masked
    #                                       # arrays are not supported by interp
    # mol_ND_data = get_full_density_array(mol_ND_met, Z_met, Z_data)
    interp_log_mol = interp1d(Z_met, np.log(mol_ND_met), bounds_error=False, fill_value="extrapolate")
    mol_ND_data = np.exp(interp_log_mol(Z_data))
    if False:
        ax = plt.subplot(111)
        plt.plot(mol_ND_met, Z_met, marker='o', c='r', label='met', zorder=-1)
        plt.scatter(mol_ND_data, Z_data, s=2, label='data')
        # ax.set_xscale('log')
        plt.legend()
        plt.title('Molecular number density')
        plt.show()

    # Convert number density to molecular backscatter coefficients
    beta_mol = 1000. * mol_backscatter_cross_sect * mol_ND_data # (km^-1 / sr^-1)

    if polar=='par':
        beta_mol = beta_mol / (1 + depolar)
    elif polar=='per':
        beta_mol = beta_mol * depolar / (1 + depolar)

    # Convert number density to molecular extinction coefficients
    ext_mol = 1000. * mol_ext_cross_sect * mol_ND_data # (km^-1)


    # Derive molecular two-way transmittance values from the extinction
    # coefficient
    T2_mol = extinction2two_way_transmittance(ext_mol, Z_data)


    if wl == 532:

        # Interpolate to get density values for all lidar data alt
        # f = interp1d(Z_met, O3_ND_met)
        # O3_ND_data = f(Z_data)
        interp_O3 = interp1d(Z_met, O3_ND_met, bounds_error=False, fill_value="extrapolate")
        O3_ND_data = interp_O3(Z_data)
        if False:
            plt.plot(O3_ND_met, Z_met, marker='o', c='r', label='met',
                     zorder=-1)
            plt.scatter(O3_ND_data, Z_data, s=2, label='data')
            plt.legend()
            plt.title('Molecular number density')
            plt.show()

        # Convert number density to molecular extinction coefficients
        ext_O3 = 1000. * O3_ext_cross_sect * O3_ND_data # (km^-1)

        # Derive O3 two-way transmittance values from the extinction
        # coefficient
        T2_O3 = extinction2two_way_transmittance(ext_O3, Z_data)

    else: # 1064 nm
        # O3 two-way transmittance = 1
        T2_O3 = np.ones(T2_mol.size)


    return mol_ND_data, beta_mol, T2_mol, T2_O3


def replace_fillvalue_with_lowest_valid(ND_met, fill_value=-9999.):
    """Replace fillValue (used for negative altitudes) by lowest altitude
    where no fillValue in order to get correct values at altitude
    close to 0 km after interpolation"""
    valid = ND_met != fill_value
    if np.any(valid):
        lowest_valid = ND_met[valid][-1] 
        ND_met = np.where(valid, ND_met, lowest_valid)
        return ND_met
    else:
        return None
    

def get_full_density_array(metDensity, metAltitude, Z):
# metDensity and metAltitude are meteorological data from the CALIPSO
# level 1 files both are 1-D arrays, with the max altitude at index 0,
# and a max dimension of 33
#
# Z is a 1-D array of CALIPSO lidar data altitudes max altitude at index
# 0, max dimension = 583
#
# the return value, rho, is an array of interpolated "metDensity" values
# corresponding to each altitude in Z
    
    lnDensity = np.ma.log(metDensity)
    f = interp1d(metAltitude, lnDensity)
    rho = f(Z)
    rho = np.exp(rho)
    
    return rho


def extinction2two_way_transmittance(sigma, Z):
    """Use trapezoid integration to convert extinction coefficients to
    optical depths and derive two-way transmittances

    Args:
        sigma (_type_): extinction coefficients
        Z (_type_): corresponding altitudes

    Returns:
        _type_: two-way transmittance values
    """
    dz = -np.diff(Z, prepend=Z[0]) # Prepend avoids mismatch in size
    optical_depth = np.cumsum((sigma + np.roll(sigma, 1)) * dz / 2)
    optical_depth[0] = 0  # Ensure first value is 0

    return np.exp(-2*optical_depth)


def nsf_from_V_domain_to_betap_domain(nsf, r_alt, laser_energy, calib, pgr=np.array((1,))):
    return nsf*np.sqrt(r_alt**2/(laser_energy*calib*pgr))


def rms_from_P_domain_to_betap_domain(rms, r_alt, laser_energy, gain, calib, pgr=np.array((1,))):
    return rms*r_alt**2/(laser_energy*gain*calib*pgr)


def compute_shotnoise(fcorr, nb_bins_shift_abs, nb_pixels, nsf, mol_ab):
    return fcorr[:, nb_bins_shift_abs].T * 1/np.sqrt(nb_pixels) * nsf * 1/np.sqrt(mol_ab)

  
def compute_backgroundnoise(fcorr, nb_bins_shift_abs, nb_pixels, rms, mol_ab):
    return fcorr[:, nb_bins_shift_abs].T * 1/np.sqrt(nb_pixels) * 1/mol_ab * rms