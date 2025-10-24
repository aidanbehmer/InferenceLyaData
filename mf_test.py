import numpy as np


class PySREmu:


  def equation_(self, dtau0, Ap, x1, x2):
    """
    x1: normalized k
    x2: resoltuion (LF: 0.4, HF: 0.8)

    return: normalized P1D
    """
    return ((((dtau0 - 1.1355175) / 0.37476897) -
      ((dtau0 / (x1 - 2.0014355)) +
      (x2 * 2.295514))) + 2.2722373 *
      (Ap + (x2 * -1.6689187)) +
      (np.log(Ap + 0.60608655) * 2.6401424))

  def predict(self, X):
    """
    X: (number of points, number of parameters) -> e.g., (1750, 4)

    0: dtau0
    1: Ap
    2: x1
    3: x2
    """
    y_pred = []

    for _x in X:
      # _x is (4, )
      dtau0, Ap, x1, x2 = _x
      this_y_pred = self.equation_(dtau0, Ap, x1, x2)
      y_pred.append(this_y_pred)

    return np.array(y_pred)



#### NOW STARTS THE REST OF THE CODE ####


import time

import matplotlib.pyplot as plt
from typing import List
import pysr
import h5py
import sympy
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



# Configure plot defaults
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.color"] = "#666666"

####### Set Input Arguments ########
# This is where you set the args to your function.
# Parameter name
param_name = "dtau0"  
# take z = 3.6
z = 3.6

# Plotting
quantile_low = 0.16  # quantile for parameter value to fix
quantile_high = 0.84  # quantile for parameter value to fix
####################################

param_dict = {
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
}
param_idx = param_dict[param_name]  # index of the parameter in the params array

# TODO: Probably also be careful about the filepath~
with h5py.File(
    "../2pvar/lf_sobol2p_n['dtau0', 'Ap'].hdf5", "r"
) as file:
    print(file.keys())

    flux_vectors_low = file["flux_vectors"][:]
    kfkms_low = file["kfkms"][:]
    # kfmpc = file["kfmpc"][:]
    zout = file["zout"][:]
    resolution_low=np.full((1750,1),0.4)

    params_low = file["params"][:]
#kfkms.shape, flux_vectors.shape, zout.shape, params.shape
"""
with h5py.File(
    "../InferenceMultiFidelity/1pvar/hf_{}_npoints50_datacorrFalse.hdf5".format(param_name), "r"
) as file:
    print(file.keys())

    flux_vectors_hi = file["flux_vectors"][:]
    kfkms_hi = file["kfkms"][:]
    # kfmpc = file["kfmpc"][:]
    zout_hi = file["zout"][:]
    resolution_hi=np.full((1750,1),0.8)
    params_hi = file["params"][:]
"""
zindex = np.where(zout == z)[0][0]  # index of z = 5

# take z=3.6, and flatten the flux vectors, such that the dim=1 is p1d values per k and parameter
flux_vectors_z_low = flux_vectors_low[:, zindex, :]
# TODO: Check this later: I want the normalized to mean as function of k
mean_flux_low = np.mean(flux_vectors_z_low, axis=0)
std_flux_low = np.std(flux_vectors_z_low, axis=0)
flux_vectors_z_low = (flux_vectors_z_low - mean_flux_low) / std_flux_low  # normalize to mean
#use the mean and std variables later when reverting back to original scale
#make this a function instead of in here
########################################################################
flux_vectors_z_low = flux_vectors_z_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

#flux_vectors_z_hi = flux_vectors_hi[:, zindex, :]
# TODO: Check this later: I want the normalized to mean as function of k
# mean_flux_hi = np.mean(flux_vectors_z_hi, axis=0)
#flux_vectors_z_hi = (flux_vectors_z_hi - mean_flux_low) / std_flux_low  # normalize to mean of low fidelity
########################################################################
#flux_vectors_z_hi = flux_vectors_z_hi.flatten()[:, np.newaxis]  # add a new axis to make it 2D

# do the same for kfkms
kfkms_z_low = kfkms_low[:, zindex, :]
kfkms_z_low = kfkms_z_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

params_values_low = params_low[:, param_idx]
# repeat this for the number of kfkms
params_values_low = np.repeat(params_values_low[:, np.newaxis], kfkms_low.shape[2], axis=1)
params_values_low = params_values_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

# Shapes: (1750, 1)
X_param = params_values_low
X_k = kfkms_z_low
y = flux_vectors_z_low

assert(y.shape == (1750, 1))
# Concatenate inputs to form design matrix


X_max=(np.max(X_param,axis=0))
X_min=(np.min(X_param,axis=0))
X_param_normalized=(X_param-X_min)/(X_max-X_min)
#save the max and min for use in reverting back to original scale
#make this a function as well

X_k_max=np.max(X_k,axis=0)
X_k_min=np.min(X_k,axis=0)
X_k_normalized=(X_k-X_k_min)/(X_k_max-X_k_min)

X = np.hstack([X_param_normalized, X_k_normalized])  # shape: (1750, 2)
X_1 = np.hstack([X_param_normalized, X_k_normalized,resolution_low])  # shape: (1750, 3)
assert(X.shape== (1750, 2))

#params_values_hi = params_low[:, param_idx]
# repeat this for the number of kfkms
#params_values_hi = np.repeat(params_values_hi[:, np.newaxis], kfkms_low.shape[2], axis=1)
#params_values_hi = params_values_hi.flatten()[:, np.newaxis]  # add a new axis to make it 2D

# Shapes: (1750, 1)
#X_param_hi = params_values_hi

#y_hi = flux_vectors_z_hi

# #normalization of y
# y_low_mean=np.mean(y, axis=0)
# y_low_std=np.std(y, axis=0)
# y_low_normalized=(y-y_low_mean)/y_low_std

# y_hi_mean=np.mean(y_hi, axis=0)
# y_hi_std=np.std(y_hi, axis=0)
# y_hi_normalized=(y_hi-y_low_mean)/y_low_std

#stacking
#X_hi_max=np.max(X_param_hi,axis=0)
#X_hi_min=np.min(X_param_hi,axis=0)
#X_param_hi_normalized=(X_param_hi-X_hi_min)/(X_hi_max-X_hi_min)

#X2=np.hstack([X_param_hi_normalized, X_k_normalized]) #non resolution
#X_2=np.hstack([X_param_hi_normalized, X_k_normalized,resolution_hi])  # shape: (1750, 3)

#normalization of x
#X_1_normalized=X_1/(np.max(X_1,axis=0)-np.min(X_1,axis=0))
#X_2_normalized=X_2/(np.max(X_2,axis=0)-np.min(X_2,axis=0))
#THROWS ERROR, I BELIEVE BECAUSE OF DIVISION BY 0

#end stacking
X_act=np.vstack([X_1])  # shape: (3500, 3)
Y_act=np.vstack([y])  # shape: (3500, 1)

assert(X_act.shape== (3500, 3))
assert(Y_act.shape== (3500, 1))



with h5py.File(
    "../2pvar/lf_sobol2p_n['dtau0', 'Ap']].hdf5", "r"
) as file:
    flux_vectors_low_test = file["flux_vectors"][:]
    kfkms_low_test = file["kfkms"][:]
    zout = file["zout"][:]
    params_low_test = file["params"][:]



zindex = np.where(zout == z)[0][0]
flux_vectors_z_test = flux_vectors_low_test[:, zindex, :]
kfkms_z_test = kfkms_low_test[:, zindex, :]

# --- Normalize with training stats ---
flux_vectors_z_test = (flux_vectors_z_test - mean_flux_low) / std_flux_low
kfkms_z_test = (kfkms_z_test - X_k_min) / (X_k_max - X_k_min)

param_dict = {"dtau0":0, "tau0":1, "ns":2, "Ap":3, "herei":4, "heref":5,
              "alphaq":6, "hub":7, "omegamh2":8, "hireionz":9, "bhfeedback":10}
param_idx_test = param_dict[param_test]

X_param_test = params_low_test[:, param_idx_test]
X_param_test = (X_param_test - X_param_test.min()) / (X_param_test.max() - X_param_test.min())

# Flatten to match training shape
X_param_test = np.repeat(X_param_test[:, np.newaxis], kfkms_z_test.shape[1], axis=1).flatten()[:, np.newaxis]
X_k_test = kfkms_z_test.flatten()[:, np.newaxis]

# Resolution: same shape, just constant
resolution_test = np.full_like(X_k_test, 0.4)

X_test = np.hstack([X_param_test, X_k_test, resolution_test])
y_true = flux_vectors_z_test.flatten()[:, np.newaxis]

# --- Predict using your trained model ---
model = PySREmu()
y_pred = model.predict(X_test)


n_sims, n_k = flux_vectors_low_test.shape[0], flux_vectors_low_test.shape[2]

mean_flux_expand = np.repeat(mean_flux_low[np.newaxis, :], n_sims, axis=0)
std_flux_expand = np.repeat(std_flux_low[np.newaxis, :], n_sims, axis=0)

# Flatten to align with y_pred
mean_flux_flat = mean_flux_expand.flatten()
std_flux_flat = std_flux_expand.flatten()

# Denormalize
y_pred_denorm = y_pred.flatten() * std_flux_flat + mean_flux_flat
