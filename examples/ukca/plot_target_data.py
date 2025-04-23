"""Script for plotting proportions of halving steps in the training data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the training data from file
df = pd.read_csv("target_data.csv")
target_data = df["nhsteps"][706:]
# FIXME: Why are the first 705 values unset?
N = len(target_data)
nc = len(set(target_data))

# Plot the halving steps in the training data as a histogram
fig, axes = plt.subplots()
counts, _, container = axes.hist(target_data, bins=nc, rwidth=0.9)
axes.set_xlabel("Number of halving steps")
axes.set_xticks(np.linspace(1, 4, 2 * nc + 1)[1::2])
axes.set_xticklabels([1, 2, 3, 4])
axes.set_ylabel("Frequency")
axes.set_yscale("log")

# Annotate the plot with the percentages of each value
for bar, count in zip(container, counts):
    percent = 100.0 * count / N
    width = bar.get_width()
    height = bar.get_height()
    x, y = bar.get_xy()
    axes.text(x + 0.5 * width, y + 1.05 * height, f"{percent:.2f}%", ha="center")

plt.savefig("target_data_hist.png", bbox_inches="tight")
