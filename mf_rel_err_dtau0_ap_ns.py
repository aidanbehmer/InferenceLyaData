import numpy as np
import h5py
import matplotlib.pyplot as plt

param_subset=["dtau0","Ap",'ns']
param_subset_name = "-".join(param_subset)
outdir = "3pvar"

relative_error_denorm_array_lf = np.loadtxt(f"{outdir}/relative_error_denorm_lf_{param_subset_name}.txt")
relative_error_denorm_array_hf = np.loadtxt(f"{outdir}/relative_error_denorm_hf_{param_subset_name}.txt")
kfkms_low = np.loadtxt(f"{outdir}/kfkms_low_{param_subset_name}.txt")
kfkms_high = np.loadtxt(f"{outdir}/kfkms_high_{param_subset_name}.txt")

plt.plot(kfkms_low,relative_error_denorm_array_lf, label="Low Fidelity")
plt.plot(kfkms_high,relative_error_denorm_array_hf, label="High Fidelity")
plt.xlabel("k (h/Mpc)")
plt.title("Relative Error as a Function of k for both Models of dtau0, Ap, and ns")
plt.legend()
plt.savefig(f"{outdir}/relative_error_denorm_{param_subset_name}.pdf")
plt.show()
