import time

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import pysr
import h5py
import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

# Configure plot defaults
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.color"] = "#666666"

#lf_herei_npoints50_datacorrFalse.hdf5
with h5py.File(
    "../InferenceMultiFidelity/1pvar/lf_herei_npoints50_datacorrFalse.hdf5", "r"
) as file:
    print(file.keys())

    flux_vectors = file["flux_vectors"][:]
    kfkms = file["kfkms"][:]
    # kfmpc = file["kfmpc"][:]
    zout = file["zout"][:]

    params = file["params"][:]