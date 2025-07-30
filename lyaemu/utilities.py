"""
Separate some utility functions from the main code, make them decoupled.
"""
import numpy as np
import scipy.interpolate


def rebin_power_to_kms(kfkms, kfmpc, flux_powers, zbins, omega_m, omega_l=None):
    """
    Rebins a power spectrum to constant km/s bins.
    Bins larger than the box are discarded. The return type is thus a list,
    with each redshift bin having potentially different lengths.
    """
    if omega_l is None: omega_l = 1 - omega_m
    nz = np.size(zbins)
    assert np.size(flux_powers) == nz * np.size(kfmpc)
    # conversion factor from mpc to kms units
    velfac = 1./(1+zbins) * 100.0*np.sqrt(omega_m*(1 + zbins)**3 + omega_l)
    # original simulation output
    okmsbins = np.outer(kfmpc, 1./velfac).T
    flux_powers = flux_powers.reshape(okmsbins.shape)
    # interpolate simulation output for averaging
    rebinned = [scipy.interpolate.interpolate.interp1d(okmsbins[i], flux_powers[i]) for i in range(nz)]
    new_flux_powers = np.array([rebinned[i](kfkms) for i in range(nz)])
    # final flux power array
    assert np.min(new_flux_powers) > 0
    return np.repeat([kfkms], nz, axis=0), new_flux_powers
