import numpy as np


class PySREmu:


  def equation_(self, dtau0, Ap, x1, x2):
    """
    x1: normalized k
    x2: resoltuion (LF: 0.4, HF: 0.8)

    return: normalized P1D
    """
    return (np.sqrt((dtau0 / 0.43236703)**np.exp(x1)) - np.exp(np.cos(-0.34073448 / x2)) 
            + ((((0.63997537 - np.sin(0.09439085 + Ap)) * (x1 -2.6272435)) + Ap)) +0.28042912) # right after Ap): - x2

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
# param_idx = param_dict[param_name]  # index of the parameter in the params array
param_subset=["dtau0","Ap"]
param_subset_name = "-".join(param_subset) # make list into string
outdir = "2pvar"

import os
print(os.path.abspath("lf_sobol2p_n['dtau0', 'ns']"))

# TODO: Probably also be careful about the filepath~
with h5py.File(
    f"{outdir}/lf_sobol2p_n{param_subset_name}.hdf5", "r"
) as file:
    print(file.keys())
    
    flux_vectors_low = file["flux_vectors"][:]
    kfkms_low = file["kfkms"][:]
    # kfmpc = file["kfmpc"][:]
    zout = file["zout"][:]
    
    nnparam, nzz, nkk = kfkms_low.shape

    # this is a flatten array of param and k
    resolution_low=np.full((nnparam * nkk, 1),0.4)

    print(kfkms_low.shape)
    params_low = file["params"][:]
    print(zout)
    print(zout==z)
    # closest index z to zout
    zindex = np.argmin(np.abs(zout - z))
    print("Closest index to z={} is at index {}, zout={}".format(z, zindex, zout[zindex]))
    # difference should be small such that |z- zout| < 0.1
    assert np.abs(zout[zindex] - z) < 0.1
    print(kfkms_low[:, zindex, :])
    
    
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
# zindex = np.where(zout == z)[0]#[0]  # index of z = 5

# take z=3.6, and flatten the flux vectors, such that the dim=1 is p1d values per k and parameter
flux_vectors_z_low = flux_vectors_low[:, zindex, :]
mean_flux_low = np.mean(flux_vectors_z_low, axis=0)
std_flux_low = np.std(flux_vectors_z_low, axis=0)
flux_vectors_z_low = (flux_vectors_z_low - mean_flux_low) / std_flux_low  # normalize to mean


#use the mean and std variables later when reverting back to original scale
#make this a function instead of in here
########################################################################
flux_vectors_z_low = flux_vectors_z_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

# do the same for kfkms
kfkms_z_low = kfkms_low[:, zindex, :]
kfkms_z_low = kfkms_z_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

# loop over param_subset to get the values for each parameter
X_param = []
for param_test in param_subset:
    # get the index from the dict
    param_idx = param_dict[param_test]
    # get the values for this parameter
    params_values_low = params_low[:, param_idx]

    # repeat this for the number of kfkms
    params_values_low = np.repeat(params_values_low[:, np.newaxis], kfkms_low.shape[2], axis=1)
    params_values_low = params_values_low.flatten()[:, np.newaxis]  # add a new axis to make it 2D

    # append to the list
    X_param.append(params_values_low)

# Shapes: (1750, 1)
X_param = np.hstack(X_param)
print("X_param shape: "+str(X_param.shape))
X_k = kfkms_z_low
print("X_k shape: "+str(X_k.shape))
y = flux_vectors_z_low

assert(y.shape == (nnparam * nkk, 1))
# Concatenate inputs to form design matrix

# normalization of x
X_param_normalized = np.copy(X_param)
for i in range(X_param.shape[1]):
    X_param_normalized[:, i] = (X_param[:, i] - np.min(X_param[:, i])) / (np.max(X_param[:, i]) - np.min(X_param[:, i]))
    print(f"X_param column {i} normalized: min={np.min(X_param[:, i])}, max={np.max(X_param[:, i])}")

#save the max and min for use in reverting back to original scale
#make this a function as well

X_k_max=np.max(X_k,axis=0)
X_k_min=np.min(X_k,axis=0)
X_k_normalized=(X_k-X_k_min)/(X_k_max-X_k_min)

X = np.hstack([X_param_normalized, X_k_normalized])  # shape: (1750, 2)
X_1 = np.hstack([X_param_normalized, X_k_normalized,resolution_low])  # shape: (1750, 4)
assert(X.shape== (nnparam * nkk, 3))

# --- Preparing the input to the model ---
X_test = X_1  # only low-fidelity data for testing
y_true = y  # true values for comparison

# --- Predict using your trained model ---
model = PySREmu()
y_pred = model.predict(X_test)
print("y_pred shape: "+str(y_pred.shape))
# difference between true and predicted
y_diff = y_true.flatten() - y_pred.flatten()
# RMSE
rmse = np.sqrt(np.mean(y_diff**2))
print("RMSE: "+str(rmse))
# relative error
relative_error = np.mean(np.abs(y_diff / y_true.flatten())) * 100
print("Relative Error (%): "+str(relative_error))
# prediction plot
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], 'r--')
plt.xlabel("True P1D (normalized)")
plt.ylabel("Predicted P1D (normalized)")
plt.title("True vs Predicted P1D")
plt.grid()
plt.show()


# TODO: remove this later for clean
n_sims, n_k = nnparam, nkk

mean_flux_expand = np.repeat(mean_flux_low[np.newaxis, :], n_sims, axis=0)
std_flux_expand = np.repeat(std_flux_low[np.newaxis, :], n_sims, axis=0)

# Flatten to align with y_pred
mean_flux_flat = mean_flux_expand.flatten()
std_flux_flat = std_flux_expand.flatten()

# Denormalize
y_pred_denorm = y_pred.flatten() * std_flux_flat + mean_flux_flat

y_true_denorm = y_true.flatten() * std_flux_flat + mean_flux_flat

y_diff_denorm = y_true_denorm.flatten() - y_pred_denorm.flatten()
# RMSE
rmse_denorm = np.sqrt(np.mean(y_diff_denorm**2))
print("RMSE: "+str(rmse_denorm))
# relative error
relative_error_denorm = np.mean(np.abs(y_diff_denorm / y_true_denorm.flatten())) * 100
print("Relative Error (%): "+str(relative_error_denorm))


plt.figure(figsize=(8, 6))
plt.scatter(y_true_denorm, y_pred_denorm, alpha=0.5)
plt.plot([np.min(y_true_denorm), np.max(y_true_denorm)], [np.min(y_true_denorm), np.max(y_true_denorm)], 'r--')
plt.xlabel("True P1D")
plt.ylabel("Predicted P1D")
plt.title("True vs Predicted P1D")
plt.grid()
plt.show()



# comparing true vs predicted

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    y_true.flatten(), y_pred.flatten(),
    c=X_param[:, 0],  # color by dtau0
    cmap='copper', alpha=0.5
)
plt.plot([np.min(y_true), np.max(y_true)], [np.min(y_true), np.max(y_true)], 'r--')
plt.xlabel("True P1D (normalized)")
plt.ylabel("Predicted P1D (normalized)")
plt.title("True vs Predicted P1D (colored by dtau0)")
plt.colorbar(sc, label="dtau0 value")
plt.grid(True)
plt.show()