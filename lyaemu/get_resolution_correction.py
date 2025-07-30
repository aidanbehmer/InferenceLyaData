"""
Get the file for the resolution correction from Fig 6 of PRIYA paper
"""

import os

import numpy as np
from scipy.interpolate import interpolate
import h5py

# A class to store flux vectors and k values
class FluxPower:
    def __init__(self, flux_vectors: np.ndarray, kfkms: np.ndarray) -> None:

        self.kfkms = kfkms
        self.flux_vectors = flux_vectors.reshape(kfkms.shape)

def generate_resolution_correction(output_file: str = "res_corr_512_384.hdf5") -> None:
    """
    Generate the resolution correction file
    """
    with h5py.File("../../lya_suite_paper/data/fluxpower_converge_2.hdf5", "r") as f:
        print(f.keys())
        # <KeysViewHDF5 ['flux_vectors', 'kfkms', 'params', 'zout']>
        flux_vectors = f["flux_vectors"]
        kfkms = f["kfkms"]
        print(kfkms.keys())
        # params = f["params"][:]
        zout = f["zout"][:]

        print(flux_vectors.keys())
        # <KeysViewHDF5 ['L15n192', 'L15n256', 'L15n384', 'L15n512']>
        # Use class to store the flux power data and k values
        L15n384 = FluxPower(flux_vectors["L15n384mf"][:], kfkms["L15n384"][:])
        L15n512 = FluxPower(flux_vectors["L15n512mf"][:], kfkms["L15n512"][:])

        print(L15n384.flux_vectors.shape, L15n384.kfkms.shape)
        print(L15n512.flux_vectors.shape, L15n512.kfkms.shape)
    
    # resolution correction factor: L15n512 / L15n384
    res_corr = L15n512.flux_vectors / L15n384.flux_vectors

    assert np.all(L15n384.kfkms == L15n512.kfkms)

    # Save into h5 file
    with h5py.File(output_file, "w") as f:
        f.create_dataset("res_corr", data=res_corr)
        f.create_dataset("kfkms", data=L15n384.kfkms)
        f.create_dataset("zout", data=zout)

# a class to load ResCorr data
class ResCorr:

    def __init__(self, filename: str) -> None:
        """
        Load the resolution correction data
        """
        with h5py.File(filename, "r") as f:
            self.res_corr = f["res_corr"][:]
            self.kfkms = f["kfkms"][:]
            self.zout = f["zout"][:]

        # get the inpoterlant of res_corr
        self.list_res_corr_interps = []
        for i,res_corr_z in enumerate(self.res_corr):
            self.list_res_corr_interps.append(interpolate.interp1d(self.kfkms[i], res_corr_z))

    def get_res_corr(self, z: float, kfkms: np.ndarray) -> np.ndarray:
        """
        Get the resolution correction factor at a given redshift and 
        interpolate onto the k values
        """
        zind = np.argmin(np.abs(self.zout - z))
        # make sure the z values not too far off
        assert np.abs(self.zout[zind] - z) < 0.1

        res_corr_interp = self.list_res_corr_interps[zind]

        return res_corr_interp(kfkms)
