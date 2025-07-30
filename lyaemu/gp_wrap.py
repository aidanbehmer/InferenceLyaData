"""
A light wrapper for Gaussian Process emulator predictions in an understandable format
to me (without all dependencies of IC, data, etc. Only GP and input/output.)
"""

import os
from datetime import datetime
import numpy as np
import h5py
import json  # For loading the emulator.json file
from .utilities import rebin_power_to_kms
from . import mean_flux as mflux
from . import lyman_data
from . import gpemulator

# resolution correction
from .get_resolution_correction import ResCorr

# Get the resolution correction for KODIAQ emulator
# Get the directory of the current file
current_dir = os.path.dirname(__file__)
# Construct the relative path to the HDF5 file
file_path = os.path.join(current_dir, "res_corr", "resolution_correction.h5")
res_corr = ResCorr(file_path)


def DLAcorr(kf, z, alpha):
    """The correction for DLA contamination, from arXiv:1706.08532"""
    # fit values and pivot redshift directly from arXiv:1706.08532
    z_0 = 2
    # parameter order: LLS, Sub-DLA, Small-DLA, Large-DLA
    a_0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
    a_1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
    b_0 = np.array([36.449, 81.388, 162.95, 429.58])
    b_1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
    # compute the z-dependent correction terms
    a_z = a_0 * ((1 + z) / (1 + z_0)) ** a_1
    b_z = b_0 * ((1 + z) / (1 + z_0)) ** b_1
    factor = np.ones(kf.size)  # alpha_0 degenerate with mean flux, set to 1
    # a_lls and a_sub degenerate with each other (as are a_sdla, a_ldla), so only use two values
    # TOOD: is this really okay?
    factor += (
        alpha[0]
        * ((1 + z) / (1 + z_0)) ** -3.55
        * (
            (a_z[0] * np.exp(b_z[0] * kf) - 1) ** -2
            + (a_z[1] * np.exp(b_z[1] * kf) - 1) ** -2
        )
    )
    factor += (
        alpha[1]
        * ((1 + z) / (1 + z_0)) ** -3.55
        * (
            (a_z[2] * np.exp(b_z[2] * kf) - 1) ** -2
            + (a_z[3] * np.exp(b_z[3] * kf) - 1) ** -2
        )
    )
    return factor


def DLA4corr(kf, z, alpha):
    """
    The correction for DLA contamination, from arXiv:1706.08532

    Here we use four parameters to model the DLA correction:
    a_lls, a_sub, a_sdla, a_ldla

    Parameters
    ----------
    kf : np.ndarray
        The k values to interpolate onto
    z : float
        The redshift value
    alpha : np.ndarray
        The alpha values (prior) for the DLA correction
    """
    # fit values and pivot redshift directly from arXiv:1706.08532
    z_0 = 2
    # parameter order: LLS, Sub-DLA, Small-DLA, Large-DLA
    a_0 = np.array([2.2001, 1.5083, 1.1415, 0.8633])
    a_1 = np.array([0.0134, 0.0994, 0.0937, 0.2943])
    b_0 = np.array([36.449, 81.388, 162.95, 429.58])
    b_1 = np.array([-0.0674, -0.2287, 0.0126, -0.4964])
    # compute the z-dependent correction terms
    a_z = a_0 * ((1 + z) / (1 + z_0)) ** a_1
    b_z = b_0 * ((1 + z) / (1 + z_0)) ** b_1
    factor = np.ones(kf.size)  # alpha_0 degenerate with mean flux, set to 1
    # Here we use four values for the DLA correction
    # LLS
    factor += (
        alpha[0]
        * ((1 + z) / (1 + z_0)) ** -3.55
        * ((a_z[0] * np.exp(b_z[0] * kf) - 1) ** -2)
    )
    # Sub-DLA
    factor += (
        alpha[1]
        * ((1 + z) / (1 + z_0)) ** -3.55
        * ((a_z[1] * np.exp(b_z[1] * kf) - 1) ** -2)
    )
    # Small-DLA
    factor += (
        alpha[2]
        * ((1 + z) / (1 + z_0)) ** -3.55
        * ((a_z[2] * np.exp(b_z[2] * kf) - 1) ** -2)
    )
    # Large-DLA
    factor += (
        alpha[3]
        * ((1 + z) / (1 + z_0)) ** -3.55
        * ((a_z[3] * np.exp(b_z[3] * kf) - 1) ** -2)
    )
    return factor


def Rescorr(kf: np.ndarray, z: float, kfmin: float = 0.02) -> np.ndarray:
    """
    Get the resolution correction factor at a given redshift and
    interpolate onto the k values

    Parameters
    ----------
    kf : np.ndarray
        The k values to interpolate onto
    z : float
        The redshift value
    kfmin : float
        The minimum k value to consider, default is 0.02 s/km, as in the
        PRIYA paper Figure 6.

    Returns
    -------
    np.ndarray
        The resolution correction factor at the given redshift and k values
    """
    ind = kf > kfmin
    _kf = kf[ind]

    # no resolution correction for k < 0.02 s/km,
    # since the res corr function is not reliable for large k due to
    # the box size of the simulation (15 Mpc/h)
    _res_corr = np.ones(kf.size)

    res_corr_interp = res_corr.get_res_corr(z, _kf)

    _res_corr[ind] = res_corr_interp

    return _res_corr


class GPWrap:
    def __init__(
        self,
        basedir,
        emulator_json_file="emulator_params.json",
        param_names=None,
        param_limits=None,
        kf=None,
        mf=None,
        tau_thresh=None,
        mean_flux_mode="s",
        use_res_corr=True,
        data_corr4=True,
    ):

        # TODO: These should not be hardcoded
        if param_names is None:
            self.param_names = {
                "ns": 0,
                "Ap": 1,
                "herei": 2,
                "heref": 3,
                "alphaq": 4,
                "hub": 5,
                "omegamh2": 6,
                "hireionz": 7,
                "bhfeedback": 8,
            }
        else:
            self.param_names = param_names
        # Parameters:
        if param_limits is None:
            self.param_limits = np.array(
                [
                    [
                        0.8,
                        0.995,
                    ],  # ns: 0.8 - 0.995. Notice that this is not ns at the CMB scale!
                    [
                        1.2e-09,
                        2.6e-09,
                    ],  # Ap: amplitude of power spectrum at 8/2pi Mpc scales (see 1812.04654)!
                    [3.5, 4.1],  # herei: redshift at which helium reionization starts.
                    # 4.0 is default, we use a linear history with 3.5-4.5
                    [
                        2.6,
                        3.2,
                    ],  # heref: redshift at which helium reionization finishes. 2.8 is default.
                    # Thermal history suggests late, HeII Lyman alpha suggests earlier.
                    [
                        1.3,
                        2.5,
                    ],  # alphaq: quasar spectral index. 1 - 2.5 Controls IGM temperature.
                    [0.65, 0.75],  # hub: hubble constant (also changes omega_M)
                    [
                        0.14,
                        0.146,
                    ],  # omegam h^2: We fix omega_m h^2 = 0.143+-0.001 (Planck 2018 best-fit) and vary omega_m and h^2 to match it.
                    # h^2 itself has little effect on the forest.
                    [6.5, 8],  # Mid-point of HI reionization
                    [0.03, 0.07],  # BH feedback parameter
                    #   [3.2, 4.2] # Wind speed
                ]
            )
        else:
            self.param_limits = param_limits
        # k bins
        if kf is None:
            self.kf = lyman_data.KSData().get_kf()
        else:
            self.kf = kf

        # Mean flux model
        if mf is None:
            self.set_mf_param_limits(basedir, emulator_json_file=emulator_json_file)
        else:
            self.mf = mf

        # This is the Planck best-fit value. We do not have to change it because
        # it is a) very well measured and b) mostly degenerate with the mean flux.
        self.omegabh2 = 0.0224
        self.max_z = 5.4
        self.min_z = 2.0
        self.sample_params = []
        self.set_maxk()

        self.basedir = os.path.expanduser(basedir)
        self.tau_thresh = tau_thresh
        if not os.path.exists(basedir):
            os.mkdir(basedir)

        # Load the parameters, etc. associated with this emulator (overwrite defaults)
        print("[Info] Load the emulator.json parameter file to the emulator.")
        self.load(
            dumpfile=emulator_json_file,
        )

        self.mean_flux_mode = mean_flux_mode
        self.use_res_corr = use_res_corr
        self.data_corr4 = data_corr4

    def set_mf_param_limits(self, basedir, emulator_json_file="emulator_params.json"):
        """
        Set the parameter limits for the mean flux parameters.
        """
        # Mean flux model
        # Param limits on t0
        t0_factor = np.array([0.75, 1.25])
        # Add a slope to the parameter limits
        t0_slope = np.array([-0.40, 0.25])
        self.mf_slope = True
        # Get the min_z and max_z for the emulator, regardless of what is requested
        with open(basedir + "/" + emulator_json_file, "r") as emulator_json:
            loaded = json.load(emulator_json)
            nz = int(np.round((loaded["max_z"] - loaded["min_z"]) / 0.2, 1)) + 1
            z_mflux = np.linspace(loaded["min_z"], loaded["max_z"], nz)
        slopehigh = np.max(mflux.mean_flux_slope_to_factor(z_mflux, t0_slope[1]))
        slopelow = np.min(mflux.mean_flux_slope_to_factor(z_mflux, t0_slope[0]))
        dense_limits = np.array([np.array(t0_factor) * np.array([slopelow, slopehigh])])
        self.mf = mflux.MeanFluxFactor(dense_limits=dense_limits)

        # Redefine the parameter limits to include the dense parameters
        self._param_limits = self.get_param_limits(include_dense=True)
        # Add a slope to the parameter limits
        self._param_limits = np.vstack([t0_slope, self._param_limits])
        # Shrink param limits t0 so that they are within the emulator range
        self._param_limits[1, :] = t0_factor

        # check with sbird if this is relevant
        ndense = np.shape(dense_limits)[1]
        # this is to shift the parameter names to the right place, limits include tau but not names
        herei = self.param_names["herei"] + ndense
        heref = self.param_names["heref"] + ndense
        # Reset the extended parameter limits to their original values as the extended emulator was not accurate enough
        self._param_limits[herei][1] = 4.1
        self._param_limits[heref][0] = 2.6
        # pring the parameter limits with names
        print("[Info] Parameter limits for the emulator:")
        for i, (name, lim) in enumerate(
            zip(self.param_names.keys(), self._param_limits[ndense:])
        ):
            print(f"{name}: {lim[0]:.3f} - {lim[1]:.3f}")

    def set_data_corr(self):
        print("[Info] DLA corrections")
        self.ndim = np.shape(self._param_limits)[0]
        if self.data_corr4:
            self.dnames = [
                ("a_lls", r"\alpha_{lls}"),
                ("a_sub", r"\alpha_{sub}"),
                ("a_sdla", r"\alpha_{sdla}"),
                ("a_ldla", r"\alpha_{ldla}"),
            ]
            alpha_limits = np.array([[-3, 8], [0, 2], [0, 0.5], [0, 0.3]])
        else:
            self.dnames = [("a_lls", r"\alpha_{lls}"), ("a_dla", r"\alpha_{dla}")]
            alpha_limits = np.array([[0, 1], [0, 0.3]])
        self.data_params = {k[0]: i + self.ndim for i, k in enumerate(self.dnames)}
        self._param_limits = np.vstack([self._param_limits, alpha_limits])
        self.ndim = self._param_limits.shape[0]
        self.param_limits = self._param_limits

    def set_emulator(
        self,
        HRbasedir=None,
        max_z=4.6,
        min_z=2.2,
        traindir=None,
        savefile="emulator_flux_vectors.hdf5",
    ):
        """
        Generate the emulator for the desired k_F and our simulations.
        kf gives the desired k bins in s/km.
        """
        print("Beginning to generate emulator at " + str(datetime.now()), flush=True)
        if HRbasedir is None:
            gpemu = self.get_emulator(
                max_z=max_z, min_z=min_z, traindir=traindir, savefile=savefile
            )
        else:
            gpemu = self.get_MFemulator(
                HRbasedir,
                max_z=max_z,
                min_z=min_z,
                traindir=traindir,
            )
        print("Finished generating emulator at", str(datetime.now()), flush=True)
        self.gpemu = gpemu

    def set_maxk(self):
        """Get the maximum k in Mpc/h that we will need."""
        # Maximal velfactor: the h dependence cancels but there is an omegam
        minhub = 0.65  # self.param_limits[self.param_names['hub'],0] # TODO: These are prior specific, might need to be changed in the future
        omgah2 = 0.146  # self.param_limits[self.param_names['omegamh2'],1]
        velfac = (
            lambda a: a
            * 100.0
            * np.sqrt(omgah2 / minhub**2 / a**3 + (1 - omgah2 / minhub))
        )
        # Maximum k value to use in comoving Mpc/h.
        # Comes out to k ~ 5, which is a bit larger than strictly necessary.
        self.maxk = np.max(self.kf) * velfac(1 / (1 + 4.4)) * 2

    def load(self, dumpfile="emulator_params.json"):
        """Load parameters from a textfile."""
        kf = self.kf
        mf = self.mf
        tau_thresh = self.tau_thresh
        real_basedir = self.basedir
        with open(os.path.join(real_basedir, dumpfile), "r", encoding="UTF-8") as jsin:
            indict = json.load(jsin)
        # self.__dict__ = indict
        self.__dict__.update(indict)
        self._fromarray()
        self.kf = kf
        self.mf = mf
        self.tau_thresh = tau_thresh
        self.basedir = real_basedir
        self.set_maxk()

    def _fromarray(self):
        """Convert the data stored as lists back to arrays."""
        for arr in self.really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self.really_arrays = []

    def get_emulator(
        self, max_z=4.6, min_z=2.0, traindir=None, savefile="emulator_flux_vectors.hdf5"
    ):
        """Build an emulator for the desired k_F and our simulations.
        kf gives the desired k bins in s/km.
        Mean flux rescaling is handled (if mean_flux=True) as follows:
        1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
        2. Each flux power spectrum in the set is rescaled to the same mean flux.
        3.
        """
        aparams, kf, flux_vectors = self.get_flux_vectors(
            max_z=max_z, min_z=min_z, kfunits="mpc", savefile=savefile
        )
        plimits = self.get_param_limits(include_dense=True)
        nz = int(flux_vectors.shape[1] / kf.size)

        # JBCat: no need for parallelize the GP optimization
        gp = gpemulator.MultiBinGP(
            params=aparams,
            kf=kf,
            powers=flux_vectors,
            param_limits=plimits,
            zout=np.linspace(max_z, min_z, nz),
            traindir=traindir,
        )

        return gp

    def get_MFemulator(
        self,
        HRbasedir,
        dumpfile="emulator_params.json",
        max_z=4.6,
        min_z=2.2,
        traindir=None,
        savefile="emulator_flux_vectors.hdf5",
    ):
        """Build a multi-fidelity emulator for the flux power spectrum."""
        # get lower resolution parameters & temperatures
        self.load(dumpfile=dumpfile)  # TODO: double check
        # Parallelize the get_flux_vectors for Low-fidelity
        LRparams, kf, LRfps = self.get_flux_vectors(
            max_z=max_z,
            min_z=min_z,
            kfunits="mpc",
            savefile=savefile,
        )
        nz = int(LRfps.shape[1] / kf.size)

        # get higher resolution parameters & temperatures
        HRemu = GPWrap(
            HRbasedir,
            emulator_json_file=dumpfile,
            mf=self.mf,
            kf=self.kf,
            tau_thresh=self.tau_thresh,
        )
        HRemu.load(dumpfile=dumpfile)
        # Not parallelize the get_flux_vectors for High-fidelity
        HRparams, HRkf, HRfps = HRemu.get_flux_vectors(
            max_z=max_z,
            min_z=min_z,
            kfunits="mpc",
            savefile=savefile,
        )
        # check parameter limits, k-bins, number of redshifts, and get/train the multi-fidelity GP
        assert np.all(
            self.get_param_limits(include_dense=True)
            == HRemu.get_param_limits(include_dense=True)
        )
        assert np.all(kf - HRkf < 1e-3)
        assert nz == int(HRfps.shape[1] / HRkf.size)

        gp = gpemulator.MultiBinGP(
            params=LRparams,
            HRdat=[HRparams, HRfps],
            powers=LRfps,
            param_limits=self.get_param_limits(include_dense=True),
            kf=kf,
            zout=np.linspace(max_z, min_z, nz),
            traindir=traindir,
        )

        return gp

    def get_param_limits(self, include_dense=True):
        """Get the reprocessed limits on the parameters for the likelihood."""
        if not include_dense:
            return self.param_limits
        dlim = self.mf.get_limits()
        if dlim is not None:
            # Dense parameters go first as they are 'slow'
            plimits = np.vstack([dlim, self.param_limits])
            assert np.shape(plimits)[1] == 2
            return plimits
        return self.param_limits

    def get_parameters(self):
        """Get the list of parameter vectors in this emulator."""
        return self.sample_params

    def load_flux_vectors(
        self, aparams, mfc="mf", savefile="emulator_flux_vectors.hdf5"
    ):
        """Save the flux vectors and parameters to a file, which is the only thing read on reload."""
        if self.tau_thresh is not None:
            savefile = (
                savefile[:-5] + "_tau" + str(int(self.tau_thresh)) + savefile[-5:]
            )
        finalpath = os.path.join(self.basedir, mfc + "_" + savefile)
        print("Loading flux powers from: ", finalpath)
        load = h5py.File(finalpath, "r")
        inparams = np.array(load["params"])
        flux_vectors = np.array(load["flux_vectors"])
        kfkms = np.array(load["kfkms"])
        kfmpc = np.array(load["kfmpc"])
        zout = np.array(load["zout"])

        # TODO: Don't think we need to set this
        # self.myspec.zout = zout
        self.zout = zout

        # name = str(load.attrs["classname"])
        load.close()

        # assert (
        #     name.rsplit(".", maxsplit=1)[-1]
        #     == str(self.__class__).rsplit(".", maxsplit=1)[-1]
        # )

        assert np.shape(inparams) == np.shape(aparams)
        assert np.all(inparams - aparams < 1e-3)

        return kfmpc, kfkms, flux_vectors

    def get_flux_vectors(
        self, max_z=4.6, min_z=2.0, kfunits="kms", savefile="emulator_flux_vectors.hdf5"
    ):
        """Get the desired flux vectors and their parameters"""
        pvals = self.get_parameters()
        nparams = np.shape(pvals)[1]
        nsims = np.shape(pvals)[0]
        assert nparams == len(self.param_names)
        aparams = pvals
        # Note this gets tau_0 as a linear scale factor from the observed power law
        dpvals = self.mf.get_params()
        # Savefile prefix
        mfc = "cc"
        if dpvals is not None:
            newdp = dpvals[0] + (dpvals - dpvals[0]) / (np.size(dpvals) + 1) * np.size(
                dpvals
            )
            # Make sure we don't overflow the parameter limits
            dpvals = newdp
            aparams = np.array(
                [np.concatenate([dp, pvals[i]]) for dp in dpvals for i in range(nsims)]
            )
            mfc = "mf"

        # Load the flux vectors - assuming you have those saved
        kfmpc, kfkms, flux_vectors = self.load_flux_vectors(
            aparams, mfc=mfc, savefile=savefile
        )

        assert np.shape(flux_vectors)[0] == np.shape(aparams)[0]

        if kfunits == "kms":
            kf = kfkms
        else:
            kf = kfmpc

        # Cut out redshifts that we don't want this time
        assert np.round(self.zout[-1], 1) <= min_z
        maxbin = np.where(np.round(self.zout, 1) >= min_z)[0].max() + 1
        assert np.round(self.zout[0], 1) >= max_z
        minbin = np.where(np.round(self.zout, 1) <= max_z)[0].min()
        kflen = np.shape(kf)[-1]
        newflux = flux_vectors[:, minbin * kflen : maxbin * kflen]

        return aparams, kf, newflux

    def get_data_correction(self, okf, params, redshift):
        if self.data_corr4:
            return DLA4corr(
                okf,
                redshift,
                params[self.data_params["a_lls"] : self.data_params["a_ldla"] + 1],
            )
        else:
            return DLAcorr(
                okf,
                redshift,
                params[self.data_params["a_lls"] : self.data_params["a_dla"] + 1],
            )

    def get_predicted(self, params):
        nparams = params
        if self.mean_flux_mode == "s":
            tau0_fac = mflux.mean_flux_slope_to_factor(self.zout, params[0])
            nparams = params[1:]
        elif self.mean_flux_mode == "tau0_only":
            tau0_fac = np.ones_like(self.zout)
            nparams = params[1:]
        elif self.mean_flux_mode == "per_z":
            tau0_fac = params[: self.zout.size]
            nparams = params[self.zout.size + 1 :]
            nparams[0] = 1.0
        else:
            tau0_fac = None

        # .predict should take [{list of parameters: t0; cosmo.; thermal},]
        # Here: emulating @ cosmo.; thermal; sampled t0 * [tau0_fac from above]
        predicted_nat, std_nat = self.gpemu.predict(
            np.array(nparams).reshape(1, -1), tau0_factors=tau0_fac
        )

        ndense = len(self.mf.dense_param_names)
        hindex = ndense + self.param_names["hub"]
        omegamh2_index = ndense + self.param_names["omegamh2"]
        assert 0.5 < nparams[hindex] < 1
        omega_m = nparams[omegamh2_index] / nparams[hindex] ** 2

        okf, predicted = rebin_power_to_kms(
            kfkms=self.kf,
            kfmpc=self.gpemu.kf,
            flux_powers=predicted_nat[0],
            zbins=self.zout,
            omega_m=omega_m,
        )
        _, std = rebin_power_to_kms(
            kfkms=self.kf,
            kfmpc=self.gpemu.kf,
            flux_powers=std_nat[0],
            zbins=self.zout,
            omega_m=omega_m,
        )

        if self.use_res_corr:
            for i in range(len(self.zout)):
                predicted[i] = predicted[i] * res_corr.get_res_corr(
                    self.zout[i], okf[i]
                )

        return okf, predicted, std

    def chi2_zbin(
        self,
        params,
        bb,
        okf,
        predicted,
        std,
        data_power,
        include_emu=True,
        per_dof=False,
    ):
        """Get the likelihood for a single bin of the flux power spectrum."""
        idp = np.where(self.kf >= okf[bb][0])
        if len(self.data_params) != 0:
            # Get and apply the DLA and SiIII corrections to the prediction
            predicted[bb] = predicted[bb] * self.get_data_correction(
                okf[bb], params, self.zout[bb]
            )
        diff_bin = predicted[bb] - data_power[bb][idp]
        std_bin = std[bb]
        bindx = np.min(idp)
        covar_bin = self.get_BOSS_error(bb)[bindx:, bindx:]
        assert np.shape(np.outer(std_bin, std_bin)) == np.shape(covar_bin)
        if include_emu:
            # Assume completely correlated emulator errors within this bin
            covar_emu = np.outer(std_bin, std_bin)
            covar_bin += covar_emu
            icov_bin = np.linalg.inv(covar_bin)
            cdet = np.linalg.slogdet(covar_bin)[1]
        else:
            icov_bin = self.icov_bin[bb]
            cdet = self.cdet[bb]
        dcd = (
            -np.dot(
                diff_bin,
                np.dot(icov_bin, diff_bin),
            )
            / 2.0
        )
        assert 0 > dcd > -(2**31), dcd
        if per_dof:
            chi2 = dcd / (np.size(data_power[bb][idp]) - np.size(params))
        else:
            chi2 = dcd - 0.5 * cdet
        assert not np.isnan(chi2)
        return chi2

    def likelihood_per_zbin(
        self, zbin, params, include_emu=True, data_power=None, per_dof=False
    ):
        """Get the likelihood (raw chi2 without priors) for a single redshift bin."""
        # Default data to use is BOSS data
        if data_power is None:
            data_power = np.copy(self.BOSS_flux_power)
        # Set parameter limits as the hull of the original emulator.
        if np.any(params >= self.param_limits[:, 1]) or np.any(
            params <= self.param_limits[:, 0]
        ):
            return -np.inf

        okf, predicted, std = self.get_predicted(
            params[: self.ndim - len(self.data_params)]
        )
        nkf = int(np.size(self.kf))
        nz = np.shape(predicted)[0]
        assert nz == int(np.size(data_power) / nkf)
        # Make sure the data power is correctly shaped
        assert np.shape(data_power)[0] == nz
        assert np.shape(data_power)[1] == nkf
        # Likelihood using full covariance matrix
        bb = np.argmin(np.abs(zbin - self.zout))
        assert np.abs(zbin - self.zout[bb]) < 0.1
        chi2 = self.chi2_zbin(
            params,
            bb,
            okf,
            predicted,
            std,
            data_power,
            include_emu=include_emu,
            per_dof=per_dof,
        )
        return chi2
