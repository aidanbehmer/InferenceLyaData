"""Module for holding different mean flux models"""

import numpy as np


def obs_mean_tau(redshift, amp=0, slope=0, sdss_name="kodiaq"):
    """
    The mean flux from 0711.1862: is (0.0023±0.0007) (1+z)^(3.65±0.21)
    Note we constrain this much better from the SDSS data itself:
    this is a weak prior

    Parameters
    ----
    sdss_name (str): The name of the SDSS dataset to use. Default is "kodiaq".

    if sdss_name == "kodiaq":
        The mean flux from KODIAQ: (C + tau_0 ((1 + z) / (1 + z0))^beta) )

    C = 0.18
    tau_0 = 0.373
    z0 = 3.5
    beta = 5.13
    """
    if sdss_name == "kodiaq":
        # Use SQUAD instead of KODIAQ as default since it gives better consistency with the data
        # (see indivdual F(z) redshift bins from the paper).
        # return obs_mean_tau_kodiaq(redshift, amp, slope)
        return (2.3 + amp) * 1e-3 * (1.0 + redshift) ** (3.65 + slope)
    # elif sdss_name == "squad":
    # return obs_mean_tau_squad(redshift, amp, slope)
    else:
        return (2.3 + amp) * 1e-3 * (1.0 + redshift) ** (3.65 + slope)


def obs_mean_tau_becker(redshift, amp=0, slope=0, const=0):
    """
    The mean flux from KODIAQ: (C + tau_0 ((1 + z) / (1 + z0))^beta) )

    C = -0.132
    tau_0 = 0.751
    z0 = 3.5
    beta = 2.90

    dtau0 = 0.00049
    dbeta = 0.01336
    dC = 0.00049
    """
    return (-0.132 + const) + (0.751 + amp) * ((1 + redshift) / (1 + 3.5)) ** (
        2.90 + slope
    )


def obs_mean_tau_kodiaq(redshift, amp=0, slope=0):
    """
    The mean flux from KODIAQ: (C + tau_0 ((1 + z) / (1 + z0))^beta) )

    C = 0.18
    tau_0 = 0.373
    z0 = 3.5
    beta = 5.13
    """
    return 0.18 + (0.373 + amp) * ((1.0 + redshift) / (1 + 3.5)) ** (5.13 + slope)


def obs_mean_tau_squad(redshift, amp=0, slope=0):
    """
    The mean flux from SQUAD: (C + tau_0 ((1 + z) / (1 + z0))^beta) )

    C = 0.24
    tau_0 = 0.377
    z0 = 3.5
    beta = 5.54
    """
    return 0.24 + (0.377 + amp) * ((1 + redshift) / (1 + 3.5)) ** (5.54 + slope)


class ConstMeanFlux(object):
    """Object which implements different mean flux models. This model fixes the mean flux to a constant value."""

    def __init__(self, value=1.0):
        self.value = value
        self.dense_param_names = {}

    def get_t0(self, zzs, params=None):
        """Get mean optical depth."""
        if params is None:
            params = self.value
        if params is None:
            return np.array(
                [
                    None,
                ]
            )
        return np.array(
            [
                params * obs_mean_tau(zzs),
            ]
        )

    def get_mean_flux(self, zzs, params=None):
        """Get mean flux"""
        t0 = self.get_t0(zzs, params=params)
        if t0[0] is None:
            return t0
        return np.exp(-1 * t0)

    def get_params(self):
        """Returns a list of parameters where the mean flux is evaluated."""
        return None

    def get_limits(self):
        """Get limits on the dense parameters"""
        return None


class MeanFluxFactor(ConstMeanFlux):
    """Object which implements different mean flux models. This model parametrises
    uncertainty in the mean flux with a simple scaling factor.
    """

    def __init__(self, dense_samples=10, dense_limits=None, sdss_name="kodiaq"):
        # Limits on factors to multiply the thermal history by.
        # Mean flux is known to about 10% from SDSS, so we don't need a big range.
        if dense_limits is None:
            # 43,751 quasars from eBOSS DR14, KODIAQ-SQUAD-XQ100 has 538 quasars.
            # a rough estimate for the mean-flux measurement error as sigma/sqrt(N).
            # If eBOSS has [0.75, 1.25] range, then KODIAQ-SQUAD-XQ100 has 0.25/sqrt(538/43751) = 2.25
            slopehigh = np.max(
                mean_flux_slope_to_factor(
                    np.linspace(2.2, 4.6, 13), slope=0.25, sdss_name=sdss_name
                )
            )
            slopelow = np.min(
                mean_flux_slope_to_factor(
                    np.linspace(2.2, 4.6, 13), slope=-0.25, sdss_name=sdss_name
                )
            )
            self.dense_param_limits = np.array([[1 - 0.25, 1 + 0.25]]) * np.array(
                [slopelow, slopehigh]
            )
        else:
            self.dense_param_limits = dense_limits
        self.dense_samples = dense_samples
        self.dense_param_names = {
            "tau0": 0,
        }

        # Which mean flux model to use
        self.sdss_name = sdss_name

    def get_t0(self, zzs, params=None):
        """Get the mean optical depth as a function of redshift for all parameters."""
        if params is None:
            params = self.get_params()
        return np.array(
            [t0 * obs_mean_tau(zzs, sdss_name=self.sdss_name) for t0 in params]
        )

    def get_params(self):
        """Returns a list of parameters where the mean flux is evaluated."""
        # Number of dense parameters
        ndense = np.shape(self.dense_param_limits)[0]
        # This grid will hold the expanded grid of parameters: dense parameters are on the end.
        # Initially make it NaN as a poisoning technique.
        pvals = np.nan * np.zeros((self.dense_samples, ndense))
        for dd in range(ndense):
            # Build grid of mean fluxes
            dlim = self.dense_param_limits[dd]
            dense = np.linspace(dlim[0], dlim[1], self.dense_samples)
            pvals[:, dd] = dense
        assert not np.any(np.isnan(pvals))
        return pvals

    def get_limits(self):
        """Get limits on the dense parameters"""
        return self.dense_param_limits


def mean_flux_slope_to_factor(zzs, slope, sdss_name="kodiaq"):
    """Convert a mean flux slope into a list of mean flux amplitudes."""
    # tau_0_i[z] @dtau_0 / tau_0_i[z] @[dtau_0 = 0]
    taus = obs_mean_tau(zzs, amp=0, slope=slope, sdss_name=sdss_name) / obs_mean_tau(
        zzs, amp=0, slope=0, sdss_name=sdss_name
    )
    ii = np.argmin(np.abs(zzs - 3.0))
    # Divide by redshift 3 bin
    return taus / taus[ii]
