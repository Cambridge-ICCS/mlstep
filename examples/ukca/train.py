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
batch_size = 50
test_batch_size = batch_size
hidden_size = 500
num_epochs = 100
device = "cpu"
lr = 1.0
num_timesteps = 3

# Set random state
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Check the data directory exists
data_dir = "data"
if not os.path.exists(data_dir):
    errmsg = f"Data directory {data_dir} does not exist."
    raise IOError(errmsg)

# Load the target data from file as Torch Tensors
nhsteps = []
for i in range(1, num_timesteps + 1):
    with netCDF4.Dataset(f"{data_dir}/ncsteps_{i}.nc", "r") as nc_file:
        ncsteps = torch.Tensor(nc_file.variables["ncsteps"][:])
        nhsteps.append(torch.round(torch.log2(ncsteps)).to(dtype=torch.int))
nhsteps = torch.hstack(nhsteps)
max_nhsteps = int(nhsteps.max().item())
print(f"{max_nhsteps=}")
target_data = torch.zeros((len(nhsteps), max_nhsteps + 1), dtype=torch.int)
for i, nhstep in enumerate(nhsteps):
    target_data[i, nhstep] = 1

# Take the indices with non-zero targets and then the same number again of zero targets
indices = [int(i) for i in nhsteps.nonzero()]
N = 2 * len(indices)
# assert len(target_data) > N
i = 0
while len(indices) < N:
    if i not in indices:
        indices.append(i)
    i = i + 1
indices.sort()
indices = torch.Tensor(indices).to(dtype=torch.int)
target_data = target_data[indices]


def load1d(variable, dtype=torch.float):
    """
    Load data corresponding to a 1D variable from a NetCDF file.

    :param variable: Variable name to load.
    :param dtype: Data type to use.
    """
    arr = []
    for i in range(1, num_timesteps + 1):
        with netCDF4.Dataset(f"{data_dir}/{variable}_{i}.nc", "r") as nc_file:
            arr.append(torch.Tensor(nc_file.variables[variable][:]).to(dtype=dtype))
    return torch.hstack(arr)[indices]


def load2d(variable, dtype=torch.float):
    """
    Load data corresponding to a 2D variable from a NetCDF file.

    :param variable: Variable name to load.
    :param dtype: Data type to use.
    """
    arr = []
    for i in range(1, num_timesteps + 1):
        with netCDF4.Dataset(f"{data_dir}/{variable}_{i}.nc", "r") as nc_file:
            arr.append(torch.Tensor(nc_file.variables[variable][:][:]).to(dtype=dtype))
    return torch.hstack(arr)[:, indices]


# Load the input data from file as Torch Tensors
stratflag = load1d("stratflag")
zp = load1d("zp")
zt = load1d("zt")
zq = load1d("zq")
cldf = load1d("cldf")
cldl = load1d("cldl")
prt = load2d("prt")
dryrt = load2d("dryrt")
wetrt = load2d("wetrt")
ftr = load2d("ftr")

# Stack the input data arrays then normalise
feature_data = torch.stack(
    [stratflag, zp, zt, zq, cldf, cldl, *prt, *dryrt, *wetrt, *ftr], dim=1
)
tot_n_pnts = feature_data.shape[0]
print(f"{tot_n_pnts=}")
input_size = feature_data.shape[1]
print(f"{input_size=}")
feature_data -= feature_data.min(0, keepdim=True)[0]
feature_data /= torch.where(feature_data > 0, feature_data.max(0, keepdim=True)[0], 1)

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
nn = FCNN(input_size, max_nhsteps=max_nhsteps, hidden_size=hidden_size)
nn = nn.to(device, dtype=torch.float)
optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

# Train
train_losses, validation_losses = [], []
for epoch in range(1, num_epochs + 1):
    # Training step
    start_time = perf_counter()
    train_loss = propagate(train_loader, nn, criterion, optimizer, device=device)
    mid_time = perf_counter()
    train_time = mid_time - start_time

    # Validation step
    validation_loss = propagate(validate_loader, nn, criterion, device=device)
    validation_time = perf_counter() - mid_time

    # Stash progress
    print(
        f"Epoch {epoch:4d}/{num_epochs:d}"
        f"  avg loss: {train_loss:.4e} / {validation_loss:.4e}"
        f"  wallclock: {train_time:.2f}s / {validation_time:.2f}s"
    )
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
torch.save(torch.Tensor(train_losses), f"{data_dir}/train_losses.pt")
torch.save(torch.Tensor(validation_losses), f"{data_dir}/validation_losses.pt")
torch.save(nn.state_dict(), "model.pt")
