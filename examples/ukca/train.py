"""Script for training the ML model for the UKCA example."""

import os
import random
from time import perf_counter

import netCDF4
import torch
from sklearn import model_selection

from mlstep.net import FCNN
from mlstep.propagate import propagate

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

# TODO: Account for all timesteps

# Check the data directory exists
data_dir = "data"
if not os.path.exists(data_dir):
    errmsg = f"Data directory {data_dir} does not exist."
    raise IOError(errmsg)

# Load the target data from file as Torch Tensors
with netCDF4.Dataset(f"{data_dir}/ncsteps_1.nc", "r") as nc_file:
    ncsteps = torch.Tensor(nc_file.variables["ncsteps"][:])
    target_data = torch.round(torch.log2(ncsteps)).to(dtype=torch.int)
max_nhsteps = int(target_data.max().item())
print(f"{max_nhsteps=}")

# # Load the input data from file as Torch Tensors
with netCDF4.Dataset(f"{data_dir}/stratflag_1.nc", "r") as nc_file:
    stratflag = torch.Tensor(nc_file.variables["stratflag"][:]).to(dtype=torch.float)
with netCDF4.Dataset(f"{data_dir}/zp_1.nc", "r") as nc_file:
    zp = torch.Tensor(nc_file.variables["zp"][:]).to(dtype=torch.float)
with netCDF4.Dataset(f"{data_dir}/zt_1.nc", "r") as nc_file:
    zt = torch.Tensor(nc_file.variables["zt"][:]).to(dtype=torch.float)
with netCDF4.Dataset(f"{data_dir}/zq_1.nc", "r") as nc_file:
    zq = torch.Tensor(nc_file.variables["zq"][:]).to(dtype=torch.float)
with netCDF4.Dataset(f"{data_dir}/cldf_1.nc", "r") as nc_file:
    cldf = torch.Tensor(nc_file.variables["cldf"][:]).to(dtype=torch.float)
with netCDF4.Dataset(f"{data_dir}/cldl_1.nc", "r") as nc_file:
    cldl = torch.Tensor(nc_file.variables["cldl"][:]).to(dtype=torch.float)
with netCDF4.Dataset(f"{data_dir}/prt_1.nc", "r") as nc_file:
    prt = torch.Tensor(nc_file.variables["prt"][:][:]).to(dtype=torch.float)
with netCDF4.Dataset(f"{data_dir}/dryrt_1.nc", "r") as nc_file:
    dryrt = torch.Tensor(nc_file.variables["dryrt"][:][:]).to(dtype=torch.float)
with netCDF4.Dataset(f"{data_dir}/wetrt_1.nc", "r") as nc_file:
    wetrt = torch.Tensor(nc_file.variables["wetrt"][:][:]).to(dtype=torch.float)
with netCDF4.Dataset(f"{data_dir}/ftr_1.nc", "r") as nc_file:
    ftr = torch.Tensor(nc_file.variables["ftr"][:][:]).to(dtype=torch.float)

# Stack the input data arrays then normalise
feature_data = torch.stack(
    [stratflag, zp, zt, zq, cldf, cldl, *prt, *dryrt, *wetrt, *ftr], dim=1
)
tot_n_pnts = feature_data.shape[0]
print(f"{tot_n_pnts=}")
input_size = feature_data.shape[1]
print(f"{input_size=}")
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
torch.save(torch.Tensor(train_losses), f"{data_dir}/train_losses.pt")
torch.save(torch.Tensor(validation_losses), f"{data_dir}/validation_losses.pt")
torch.save(nn.state_dict(), "model.pt")
