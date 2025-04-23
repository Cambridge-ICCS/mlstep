"""Script for plotting the loss function curves from the experiment."""

import matplotlib.pyplot as plt
import torch

fig, axes = plt.subplots()
axes.grid()
axes.plot(torch.load("train_losses.pt"), "--x", label="Training")
axes.plot(torch.load("validation_losses.pt"), ":o", label="Validation")
axes.set_xlabel("Epochs")
axes.set_ylabel("Cross entropy loss")
plt.savefig("losses.png", bbox_inches="tight")
