"""Script for plotting proportions of halving steps in the training data."""

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import torch

num_timesteps = 3

# Load the target data from file as Torch Tensors
target_data = []
for i in range(1, num_timesteps + 1):
    with netCDF4.Dataset(f"ncsteps_{i}.nc", "r") as nc_file:
        ncsteps = torch.Tensor(nc_file.variables["ncsteps"][:])
        target_data.append(torch.round(torch.log2(ncsteps)).to(dtype=torch.int))
target_data = torch.vstack(target_data)
min_nhsteps = int(target_data.min().item())
max_nhsteps = int(target_data.max().item())
print(f"{min_nhsteps=}")
print(f"{max_nhsteps=}")
N = len(target_data[0])
nc = max_nhsteps - min_nhsteps + 1
print(f"{nc=}")

# Plot the halving steps in the training data as a histogram
fig, axes = plt.subplots(figsize=(12, 6))
plot_data, _, handles = axes.hist(target_data, bins=nc, rwidth=0.9)
axes.set_xlabel("Number of halving steps")
axes.set_xticks(np.linspace(min_nhsteps, max_nhsteps, 2 * nc + 1)[1::2])
axes.set_xticklabels(list(range(min_nhsteps, max_nhsteps + 1)))
axes.set_ylabel("Number of grid boxes")
axes.set_yscale("log")
axes.set_ylim([1, 1e6])
axes.legend(handles, [f"Timestep {i}" for i in range(1, num_timesteps + 1)])

# Annotate the plot with the percentages of each value
for bars, counts in zip(handles, plot_data):
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height >= 1:
            label = f"{100.0 * count / N:.2f}%"
            width = bar.get_width()
            x, y = bar.get_xy()
            axes.text(x + 0.5 * width, y + 1.05 * height, label, ha="center")

plt.savefig("target_data_hist.png", bbox_inches="tight")
