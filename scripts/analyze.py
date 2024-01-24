# Plot temperature histogram
import glob
import os
import re

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.ticker import AutoMinorLocator
from openmm.unit import MOLAR_GAS_CONSTANT_R
from pymbar import timeseries

# gather all kinetic energy files
files = glob.glob("*_ke.txt")
# perform a natural sort


def natural_sort(text):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(text, key=alphanum_key)


files = natural_sort(files)

# Load kinetic energy data
kinetic_energies = []
for file in files:
    kinetic_energies.append(np.array(np.loadtxt(file)))

# flatten kinetic energy list
# convert to pure list
kinetic_energies_2 = []
for i in range(len(kinetic_energies)):
    if i == 0:
        kinetic_energies_2.append(kinetic_energies[i])
    else:
        for j in range(len(kinetic_energies[i])):
            kinetic_energies_2.append(kinetic_energies[i][j])
kinetic_energies = np.array(kinetic_energies_2).flatten()
# drop first 100 values
kinetic_energies = kinetic_energies[100:]

# convert from kinetic energy to temperature
dof = 3 * (4800 - 1)

temp = 2.0 * kinetic_energies * 1000 / (dof * MOLAR_GAS_CONSTANT_R)
temp = temp._value

dt = 1
t = np.arange(0, (len(temp)) * dt, dt)


# Get some statistics
def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


correlation_time = timeseries.statistical_inefficiency(temp, fft=True)
temp_indices = timeseries.subsample_correlated_data(temp, g=correlation_time)

# subsample
temp_mean, temp_err = mean_confidence_interval(temp[temp_indices])
temp_ref = 0.824 * 120
print(f"Reference temperature: {temp_ref}")
print(f"Mean temperature: {temp_mean}")
print(f"Temperature error: {temp_err}")
print(f"Correlation time: {correlation_time}")
# save statistics to file
# delete temp.txt if it exists
if os.path.exists("temp_size.txt"):
    os.remove("temp_size.txt")
with open("temp_size.txt", "a") as f:
    f.write(f"{temp_mean:.8e} {temp_err:.8e} {correlation_time:.8e}\n")

# Plot temperature
fig = plt.figure(figsize=(5.0, 5.0))
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0, 0])


# Plot histogram of temperature
ax1.hist(temp, bins="auto", alpha=0.5, density=True)
hist, bin_edges = np.histogram(temp, bins="auto", density=True)
# Make line at reference temperature
ax1.axvline(temp_ref, color="k", linestyle="--", linewidth=1.0)
ax1.set_xlabel(r"$ T $", fontsize=12)
ax1.set_ylabel(r"$ P(T) $", fontsize=12)
ax1.set_xlim(90, 105)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
# set y axis to log scale
ax1.set_yscale("log")
ax1.tick_params(axis="both", which="major", labelsize=10)
ax1.tick_params(axis="both", which="minor", labelsize=10)
# Add text for reference temperature, mean temperature, and error
ax1.text(
    0.02,
    0.95,
    f"$T_{{\\text{{ref}}}} = {temp_ref:.3f}$",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
)
ax1.text(
    0.02,
    0.85,
    f"$T_{{\\text{{mean}}}} = {temp_mean:.3f} \pm {temp_err:.3f}$",
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
)

plt.tight_layout()
plt.tight_layout()
plt.tight_layout()
plt.tight_layout()
plt.savefig("temp_hist_size.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
