"""
PRIYAEmulatorExplorer

A convenience interface for exploring and plotting Lyman-alpha forest flux power spectrum
predictions from the PRIYA Gaussian Process emulator, with support for 1-parameter slices
and observational data overlay from KODIAQ.

Author: Ming-Feng Ho
License: MIT
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import qmc
import json

from . import gp_wrap as lk
from . import lyman_data as ld


class PRIYAEmulatorExplorer:
    """
    A high-level wrapper to use the PRIYA GP emulator and visualize model predictions
    compared with KODIAQ flux power spectrum data.

    Parameters
    ----------
    basedir : str
        Path to the base directory where emulator data and configuration reside.
    hires_subdir : str, optional
        Subdirectory for high-resolution simulation output. Default is "hires".
    tau_thresh : float, optional
        Optical depth threshold for emulator setup. Default is 1e6.
    """

    def __init__(self, basedir, hires_subdir="hires", tau_thresh=1e6, kf=None):
        self.basedir = basedir
        if hires_subdir is None:
            self.hires_basedir = None
        else:
            self.hires_basedir = os.path.join(basedir, hires_subdir)

        # Initialize emulator
        self.emulator_wrap = lk.GPWrap(
            basedir=self.basedir,
            emulator_json_file="emulator_params.json",
            tau_thresh=tau_thresh,
            kf=kf,
        )
        self.emulator_wrap.set_emulator(
            HRbasedir=self.hires_basedir,
            max_z=4.6,
            min_z=2.2,
            traindir=os.path.join(self.basedir, "trained_mf"),
        )
        self.emulator_wrap.set_mf_param_limits(basedir=self.basedir)
        self.emulator_wrap.set_data_corr()

        self.zz = np.round(self.emulator_wrap.zout, 1)
        self.mf_slope = self.emulator_wrap.mf_slope
        self._set_best_fit_params()

        # Load observational data from KODIAQ
        self.boss = ld.KSData(scale_covar=1.0, conservative=False)
        self._init_boss_data()

    def _set_best_fit_params(self):
        """
        Define internal best-fit parameters for emulator prediction.
        Used as a baseline when sweeping a single parameter.
        """
        if self.mf_slope:
            print("Using best-fit parameters for MF emulator (with slope).")
            self.best_par = np.array(
                [
                    -0.009,
                    1.090,
                    0.983,
                    1.46e-09,
                    4.0,
                    2.765,
                    1.74,
                    0.688,
                    0.1439,
                    7.24,
                    0.050,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            self.param_names = {
                "dtau0": 0,
                "tau0": 1,
                "ns": 2,
                "Ap": 3,
                "herei": 4,
                "heref": 5,
                "alphaq": 6,
                "hub": 7,
                "omegamh2": 8,
                "hireionz": 9,
                "bhfeedback": 10,
                "a_lls": 11,
                "a_sub": 12,
                "a_sdla": 13,
                "a_ldla": 14,
            }
        else:
            print("Using default best-fit parameters for non-MF emulator (no slope).")
            self.best_par = np.array(
                [1.0, 0.95, 2.59e-9, 3.84, 2.76, 1.41, 0.69, 0.146, 6.5, 0.05, 0.0, 0.0]
            )
            self.param_names = {
                "tau0": 0,
                "ns": 1,
                "Ap": 2,
                "herei": 3,
                "heref": 4,
                "alphaq": 5,
                "hub": 6,
                "omegamh2": 7,
                "hireionz": 8,
                "bhfeedback": 9,
                "a_lls": 10,
                "a_dla": 11,
            }

    def _init_boss_data(self):
        """
        Load and process the KODIAQ data for comparison.
        """
        _z = self.boss.get_redshifts()
        self.bosskf = self.boss.kf.reshape(len(_z), -1)
        self.bosspf = (
            self.boss.get_pf().reshape(len(_z), -1)[::-1] * self.bosskf / np.pi
        )
        self.boss_err = (
            self.boss.covar_diag.reshape(len(_z), -1)[::-1] * self.bosskf**2 / np.pi**2
        )

    def sample_1P_predictions(
        self, param_name, npoints=10, data_corr=False, return_prior=False
    ):
        """
        Generate emulator predictions by varying one parameter within its prior.

        Parameters
        ----------
        param_name : str
            Name of the parameter to vary.
        npoints : int, optional
            Number of steps to sample within the prior range. Default is 10.
        data_corr : bool, optional
            Whether to include DLA correction in prediction. Default is True.

        Returns
        -------
        okf : list of np.ndarray
            List of k-bin arrays for each sampled prediction.
        pred : list of np.ndarray
            Flux power spectrum predictions.
        std : list of np.ndarray
            Emulator standard deviations.
        """
        param_num = self.param_names[param_name]
        prior_range = np.linspace(
            self.emulator_wrap._param_limits[param_num][0],
            self.emulator_wrap._param_limits[param_num][1],
            npoints,
        )

        okf, pred, std = [], [], []
        params_1p = []

        for val in prior_range:
            self.best_par[param_num] = val
            clipped_par = np.clip(
                self.best_par,
                self.emulator_wrap._param_limits[:, 0] * 1.0001,
                self.emulator_wrap._param_limits[:, 1] * 0.9999,
            )

            okfi, predi, stdi = self.emulator_wrap.get_predicted(
                clipped_par[
                    : self.emulator_wrap.ndim - len(self.emulator_wrap.data_params)
                ]
            )

            if data_corr:
                predi = [
                    p * self.emulator_wrap.get_data_correction(k, clipped_par, z)
                    for p, k, z in zip(predi, okfi, self.emulator_wrap.zout)
                ]

            predi = [k * p / np.pi for p, k in zip(predi, okfi)]
            stdi = [k * s / np.pi for s, k in zip(stdi, okfi)]

            okf.append(okfi)
            pred.append(predi)
            std.append(stdi)
            params_1p.append(
                clipped_par[
                    : self.emulator_wrap.ndim - len(self.emulator_wrap.data_params)
                ]
            )

        if return_prior:
            return okf, pred, std, params_1p

        return okf, pred, std

    def plot_flux_prediction(self, okf, pred, param_name):
        """
        Plot the flux power spectrum prediction compared with KODIAQ data.

        Parameters
        ----------
        okf : list of np.ndarray
            List of k-bin arrays for each sampled prediction.
        pred : list of np.ndarray
            List of emulator predictions for each redshift and k-bin.
        param_name : str
            The parameter that was varied (used in plot title).
        """
        nrows, ncols = 7, 2
        colors = matplotlib.cm.viridis(np.linspace(0, 1, len(pred)))
        fig, axes = plt.subplots(
            figsize=(21.25, 38.5), nrows=nrows, ncols=ncols, sharex=True
        )
        axes = axes.flatten()

        for mm, ax in enumerate(axes):
            if mm >= len(self.zz):
                break

            for ii in range(len(pred)):
                ax.errorbar(okf[ii][mm], pred[ii][mm], color=colors[ii], lw=4, ls="--")

            ax.plot(self.bosskf[mm], self.bosspf[mm], "-o", color="C0", lw=2, zorder=0)
            ax.fill_between(
                x=self.bosskf[mm],
                y1=self.bosspf[mm] - np.sqrt(self.boss_err[mm]),
                y2=self.bosspf[mm] + np.sqrt(self.boss_err[mm]),
                color="C0",
                alpha=0.35,
                zorder=0,
            )
            ax.text(
                0.014, 1.5 * np.min(self.bosspf[mm]), f"z: {self.zz[mm]}", fontsize=30
            )
            ax.set_xlim(0.001, 0.1)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        plt.grid(False)
        plt.ylabel(r"$k P_F(k) / \pi$", size=30, labelpad=25)
        plt.xlabel("k [s/km]", size=30)
        plt.title(param_name, fontsize=30)
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.show()

    def plot_flux_ratio(self, okf, pred, param_name, ref_index=5):
        """
        Plot the ratio of emulator predictions to a reference prediction.

        Parameters
        ----------
        okf : list of np.ndarray
            List of k-bin arrays for each sampled prediction.
        pred : list of np.ndarray
            List of emulator predictions for each redshift and k-bin.
        param_name : str
            Name of the parameter that was varied (used in plot title).
        ref_index : int, optional
            Index of the reference prediction in `pred` used for ratio. Default is 5.
        """
        nrows, ncols = 7, 2
        colors = matplotlib.cm.viridis(np.linspace(0, 1, len(pred)))
        fig, axes = plt.subplots(
            figsize=(21.25, 38.5), nrows=nrows, ncols=ncols, sharex=True
        )
        axes = axes.flatten()

        for mm, ax in enumerate(axes):
            if mm >= len(self.zz):
                break

            for ii in range(len(pred)):
                ratio = pred[ii][mm] / pred[ref_index][mm]
                ax.plot(okf[ii][mm], ratio, color=colors[ii], lw=4)

            ax.axhline(y=1, color="k", linestyle="--", lw=2)
            ax.text(
                0.05, 0.93, f"z: {self.zz[mm]}", fontsize=30, transform=ax.transAxes
            )

            ax.set_xlim(0.001, 0.1)
            # ax.set_ylim(0.95, 1.05)

            if mm % 2 == 0:
                ax.tick_params(
                    which="both",
                    direction="inout",
                    right=False,
                    labelright=False,
                    labelleft=True,
                    length=12,
                )
            else:
                ax.tick_params(
                    which="both",
                    direction="inout",
                    right=True,
                    left=False,
                    labelright=True,
                    labelleft=False,
                    length=12,
                )

        # Add common axis labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        plt.grid(False)
        plt.ylabel(r"$P(k)/P_{ref}(k)$", size=30, labelpad=25)
        plt.xlabel("k [s/km]", size=30)
        plt.title(f"Ratio Plot — {param_name}", fontsize=30)
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.show()

    def sample_sobol_predictions(
        self,
        nsamples=256,
        filename="sobol_widget.json",
        data_corr=True,
        param_subset=None,
        z_indices=None,
    ):
        """
        Sample the emulator using a Sobol sequence over selected parameters,
        truncate to k <= 0.06, and save results to JSON for widget use.

        Parameters
        ----------
        nsamples : int
            Number of Sobol samples to generate.
        filename : str
            Output JSON filename for widget use.
        data_corr : bool
            Whether to apply DLA corrections to predictions.
        param_subset : list of str or None
            List of parameter names to vary. If None, use 13 default.
        z_indices : list of int or None
            List of redshift bin indices to include. If None, include all.
        """
        # Default parameter subset
        if param_subset is None:
            param_subset = [
                "dtau0",
                "tau0",
                "ns",
                "Ap",
                "herei",
                "heref",
                "alphaq",
                "hub",
                "omegamh2",
                "hireionz",
                "bhfeedback",
                "a_lls",
                "a_sub",
            ]
        param_indices = [self.param_names[p] for p in param_subset]

        # Default z bins
        if z_indices is None:
            z_indices = list(range(len(self.emulator_wrap.zout)))

        # Generate Sobol sequence
        sampler = qmc.Sobol(d=len(param_indices), scramble=True)
        unit_samples = sampler.random(n=nsamples)

        # Scale to parameter bounds
        bounds = np.array([self.emulator_wrap._param_limits[i] for i in param_indices])
        scaled_samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

        # Initialize output structure
        output = {
            "params": [],
            "z": [float(self.emulator_wrap.zout[i]) for i in z_indices],
            "k": None,  # filled on first loop
            "pk": [],
        }

        for i, point in enumerate(scaled_samples):
            # Construct full param vector
            x = np.copy(self.best_par)
            x[param_indices] = point

            # Clip carefully to avoid boundary issues
            for idx in param_indices:
                lower, upper = self.emulator_wrap._param_limits[idx]
                eps = 1e-6 * (upper - lower)
                x[idx] = np.clip(x[idx], lower + eps, upper - eps)

            # Get prediction
            okf, predi, _ = self.emulator_wrap.get_predicted(
                x[: self.emulator_wrap.ndim - len(self.emulator_wrap.data_params)]
            )

            if data_corr:
                predi = [
                    p * self.emulator_wrap.get_data_correction(k, x, z)
                    for p, k, z in zip(predi, okf, self.emulator_wrap.zout)
                ]

            predi = [k * p / np.pi for p, k in zip(predi, okf)]

            pk_cut = []
            if i == 0:
                output["k"] = []

            for j in z_indices:
                mask = okf[j] <= 0.06
                pk_cut.append(list(predi[j][mask]))
                if i == 0:
                    output["k"].append(list(okf[j][mask]))

            output["params"].append(dict(zip(param_subset, map(float, point))))
            output["pk"].append(pk_cut)

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[✓] Saved Sobol samples ({nsamples}) to {filename}")

    def sample_1P_sweep(
        self,
        param_names=None,
        nsteps=100,
        z_indices=[2, 7, 10],
        filename="sweep_1p_widget.json",
        data_corr=True,
    ):
        """
        Generate 1-parameter-at-a-time sweeps for a widget.

        Parameters
        ----------
        param_names : list of str or None
            Parameters to vary. If None, uses all primary 13.
        nsteps : int
            Number of steps per parameter.
        z_indices : list of int or None
            Which redshift bins to include. If None, include all.
        filename : str
            Output JSON file path.
        data_corr : bool
            Whether to apply DLA corrections.
        """
        # Default to 13D
        if param_names is None:
            param_names = [
                "dtau0",
                "tau0",
                "ns",
                "Ap",
                "herei",
                "heref",
                "alphaq",
                "hub",
                "omegamh2",
                "hireionz",
                "bhfeedback",
                "a_lls",
                "a_sub",
            ]

        param_indices = [self.param_names[p] for p in param_names]
        bounds_all = self.emulator_wrap._param_limits
        midpoint = 0.5 * (bounds_all[:, 0] + bounds_all[:, 1])

        # Redshift selection
        if z_indices is None:
            z_indices = list(range(len(self.emulator_wrap.zout)))
        z_out = [float(self.emulator_wrap.zout[i]) for i in z_indices]

        output = {
            "z": z_out,
            "k": None,
            "sweeps": {},
        }

        for p, pidx in zip(param_names, param_indices):
            low, high = bounds_all[pidx]
            sweep_values = np.linspace(low, high, nsteps)
            curves = []

            for val in sweep_values:
                x = np.copy(midpoint)
                x[pidx] = val

                # Safe clipping to stay inside bounds
                eps = 1e-6 * (high - low)
                x[pidx] = np.clip(x[pidx], low + eps, high - eps)

                okf, predi, _ = self.emulator_wrap.get_predicted(
                    x[: self.emulator_wrap.ndim - len(self.emulator_wrap.data_params)]
                )

                if data_corr:
                    predi = [
                        p * self.emulator_wrap.get_data_correction(k, x, z)
                        for p, k, z in zip(predi, okf, self.emulator_wrap.zout)
                    ]

                predi = [k * p / np.pi for p, k in zip(predi, okf)]

                # Store only selected z bins, and truncate to k <= 0.064
                pkz = []
                for zi in z_indices:
                    mask = okf[zi] <= 0.064
                    pkz.append(list(predi[zi][mask]))

                if output["k"] is None:
                    output["k"] = [list(okf[zi][okf[zi] <= 0.064]) for zi in z_indices]

                curves.append({"val": float(val), "pk": pkz})

            output["sweeps"][p] = curves

        # Save to JSON
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[✓] 1P sweep saved to {filename} for {len(param_names)} parameters")
