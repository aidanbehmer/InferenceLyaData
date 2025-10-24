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
#kfkms.shape, flux_vectors.shape, zout.shape, params.shape

# take z = 3.6
z = 3.6
zindex = np.where(zout == z)[0][0]  # index of z = 5
print(zindex)

# take z=3.6, and flatten the flux vectors, such that the dim=1 is p1d values per k and parameter
flux_vectors_z = flux_vectors[:, zindex, :]
flux_vectors_z = flux_vectors_z.flatten()[:, np.newaxis]  # add a new axis to make it 2D
print(flux_vectors_z.shape)

# do the same for kfkms
kfkms_z = kfkms[:, zindex, :]
kfkms_z = kfkms_z.flatten()[:, np.newaxis]  # add a new axis to make it 2D
print(kfkms_z.shape)

# do the same for the parameter input

#"dtau0": 0,
#"tau0": 1,
#"ns": 2,
#"Ap": 3,
#"herei": 4,
#"heref": 5,
#"alphaq": 6,
#"hub": 7,
#"omegamh2": 8,
#"hireionz": 9,
#"bhfeedback": 10,
params_values = params[:, 4]
# repeat this for the number of kfkms
params_values = np.repeat(params_values[:, np.newaxis], kfkms.shape[2], axis=1)
params_values = params_values.flatten()[:, np.newaxis]  # add a new axis to make it 2D
print(params_values.shape)

# Shapes: (1750, 1)
X_param = params_values
X_k = kfkms_z
y = flux_vectors_z

# Concatenate inputs to form design matrix
X = np.hstack([X_param, X_k])  # shape: (1750, 2)

from pysr import PySRRegressor

model = PySRRegressor(
    model_selection="best",
    niterations=20,  # increase for better expressions
    binary_operators=["+", "*", "-", "/"],
    unary_operators=[
        "sin", "cos", "exp", "log", "square", "sqrt", "inv(x) = 1/x"
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    loss="loss(x, y) = (x - y)^2",
    maxsize=20,
    maxdepth=10,
    verbosity=1,
    random_state=42,
)

model.fit(X, y)

# print(model)
model.equations_  # to see all candidate expressions

# Choose a parameter value to fix
param_fixed = 0.95306122  # can be any float within the param range

# Get unique k values from your data
k_values = np.unique(X[:, 1])

# Construct input (param_fixed, k) pairs
X_pred = np.column_stack([
    np.full_like(k_values, fill_value=param_fixed),
    k_values
])

# Get model prediction
flux_pred = model.predict(X_pred)

# To compare with ground truth, find the closest match in your original param list
# (Assumes you used params[:,2] to construct your X)
param_list = np.unique(X[:, 0])
idx_closest = np.argmin(np.abs(param_list - param_fixed))
param_closest = param_list[idx_closest]

# Now find all rows in X with that param value
mask = X[:, 0] == param_closest
k_true = X[mask, 1]
flux_true = y[mask, 0]  # shape: (N,)

# Sort both by k to ensure aligned plotting
sort_idx_true = np.argsort(k_true)
sort_idx_pred = np.argsort(k_values)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(k_true[sort_idx_true], flux_true[sort_idx_true], label="True Flux", lw=2, color="C1")
plt.plot(k_values[sort_idx_pred], flux_pred[sort_idx_pred], label="PySR Prediction", lw=2, linestyle="--", color="C1")

############### Another param ###############


#run this with herei to double check


# Choose a parameter value to fix
param_fixed = np.quantile(params[:, 2], 0.25)
...

############### Another param ###############
# Choose a parameter value to fix
param_fixed = np.quantile(params[:, 2], 0.75)
# Get unique k values from your data
k_values = np.unique(X[:, 1])

# Construct input (param_fixed, k) pairs
X_pred = np.column_stack([
    np.full_like(k_values, fill_value=param_fixed),
    k_values
])

# Get model prediction
flux_pred = model.predict(X_pred)

# To compare with ground truth, find the closest match in your original param list
# (Assumes you used params[:,2] to construct your X)
param_list = np.unique(X[:, 0])
idx_closest = np.argmin(np.abs(param_list - param_fixed))
param_closest = param_list[idx_closest]

# Now find all rows in X with that param value
mask = X[:, 0] == param_closest
k_true = X[mask, 1]
flux_true = y[mask, 0]  # shape: (N,)

# Sort both by k to ensure aligned plotting
sort_idx_true = np.argsort(k_true)
sort_idx_pred = np.argsort(k_values)

# Plot
plt.plot(k_true[sort_idx_true], flux_true[sort_idx_true], label="True Flux", lw=2, color="C2")
plt.plot(k_values[sort_idx_pred], flux_pred[sort_idx_pred], label="PySR Prediction", lw=2, linestyle="--", color="C2")


plt.xlabel("k [s/km]")
plt.ylabel("P1D(k)")
plt.title(f"z = 3.6, param ≈ {param_fixed:.2f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()