"""Script for plotting the loss function curves from the experiment."""

import os

import matplotlib.pyplot as plt
import torch

# Check the data directory exists
data_dir = "data"
if not os.path.exists(data_dir):
    errmsg = f"Data directory {data_dir} does not exist."
    raise IOError(errmsg)

# Plot training and validation losses on the same axes
fig, axes = plt.subplots()
axes.grid()
axes.plot(torch.load(f"{data_dir}/train_losses.pt"), "--x", label="Training")
axes.plot(torch.load(f"{data_dir}/validation_losses.pt"), ":o", label="Validation")
axes.set_xlabel("Epochs")
axes.set_ylabel("Cross entropy loss")

# Create the plot directory (if it doesn't already exist) then save the plot
plot_dir = "plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
plt.savefig(f"{plot_dir}/losses.png", bbox_inches="tight")
