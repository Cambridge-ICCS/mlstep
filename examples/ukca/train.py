"""Script for training the ML model for the UKCA example."""

import random
from time import perf_counter

import pandas as pd
import torch
from sklearn import model_selection

from mlstep.net import FCNN
from mlstep.propagate import propagate

# FIXME: Why are the first 705 values unset?
istart = 705

# Set parameters
test_size = 0.3
batch_size = 500
test_batch_size = batch_size
num_epochs = 10
device = "cpu"
lr = 10.0

# Set random state
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the target data from file as Torch Tensors
df_target = pd.read_csv("target_data.csv")
target_data = torch.Tensor(df_target["nhsteps"][istart:].values)
max_nhsteps = int(target_data.max().item())
print(f"{max_nhsteps=}")

# Load the input data from file as Torch Tensors
df_input = pd.read_csv("feature_data.csv", dtype=float)
input_size = len(df_input.keys())
stratflag = torch.Tensor(df_input["stratflag"][istart:]).to(dtype=torch.float)
zp = torch.Tensor(df_input["zp"][istart:]).to(dtype=torch.float)
zt = torch.Tensor(df_input["zt"][istart:]).to(dtype=torch.float)
zq = torch.Tensor(df_input["zq"][istart:]).to(dtype=torch.float)
cldf = torch.Tensor(df_input["cldf"][istart:]).to(dtype=torch.float)
cldl = torch.Tensor(df_input["cldl"][istart:]).to(dtype=torch.float)
zprt = [
    torch.from_numpy(df_input[f"prt{j}"][istart:].values).to(dtype=torch.float)
    for j in range(1, 61)
]
zdryrt = [
    torch.from_numpy(df_input[f"dryrt{j}"][istart:].values).to(dtype=torch.float)
    for j in range(1, 43)
]
zwetrt = [
    torch.from_numpy(df_input[f"wetrt{j}"][istart:].values).to(dtype=torch.float)
    for j in range(1, 35)
]
zftr = [
    torch.from_numpy(df_input[f"ftr{j}"][istart:].values).to(dtype=torch.float)
    for j in range(1, 88)
]

# Stack the input data arrays then normalise
feature_data = torch.stack(
    [zp, zt, zq, cldf, cldl, stratflag, *zprt, *zdryrt, *zwetrt, *zftr], dim=1
)
feature_data -= feature_data.min(0, keepdim=True)[0]
feature_data /= feature_data.max(0, keepdim=True)[0]

# Prepare training and validation data
xtrain, xval, ytrain, yval = model_selection.train_test_split(
    feature_data, target_data, test_size=test_size, random_state=seed
)
train_data = torch.utils.data.TensorDataset(torch.Tensor(xtrain), torch.Tensor(ytrain))
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=0
)
validate_data = torch.utils.data.TensorDataset(torch.Tensor(xval), torch.Tensor(yval))
validate_loader = torch.utils.data.DataLoader(
    validate_data, batch_size=test_batch_size, shuffle=False, num_workers=0
)

# Setup model, optimiser, and loss function
nn = FCNN(input_size, max_nhsteps=max_nhsteps).to(device, dtype=torch.float)
# nn.train(True)
optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss(reduction="sum")

# Train
train_losses, validation_losses = [], []
for epoch in range(1, num_epochs + 1):
    # Training step
    start_time = perf_counter()
    train = propagate(train_loader, nn, criterion, optimizer, device=device)
    mid_time = perf_counter()
    train_time = mid_time - start_time

    # Validation step
    val = propagate(validate_loader, nn, criterion, device=device)
    validation_time = perf_counter() - mid_time

    # Stash progress
    print(
        f"Epoch {epoch:4d}/{num_epochs:d}"
        f"  avg loss: {train:.4e} / {val:.4e}"
        f"  wallclock: {train_time:.2f}s / {validation_time:.2f}s"
    )
    train_losses.append(train)
    validation_losses.append(val)
torch.save(torch.Tensor(train_losses), "train_losses.pt")
torch.save(torch.Tensor(validation_losses), "validation_losses.pt")
torch.save(nn.state_dict(), "model.pt")
