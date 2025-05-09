"""Script for training the ML model for the UKCA example."""

import random
from time import perf_counter

import torch
from sklearn import model_selection

from mlstep.data_utils import NetCDFDataLoader
from mlstep.net import FCNN
from mlstep.propagate import propagate

# Set parameters
test_size = 0.3
batch_size = 50
test_batch_size = batch_size
hidden_size = 600
num_epochs = 1000
device = "cpu"
lr = 0.0001
num_timesteps = 10
zero_factor = 3  # NOTE: This is a key parameter

# Set random state
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the target and feature data from file
features_1d = ["stratflag", "zp", "zt", "zq", "cldf", "cldl"]
features_2d = ["prt", "dryrt", "wetrt", "ftr"]
data_dir = "data"
ncloader = NetCDFDataLoader(
    features_1d, features_2d, num_timesteps, zero_factor=zero_factor, data_dir=data_dir
)
target_data = ncloader.load_target_data()
max_nhsteps = ncloader.max_nhsteps
print(f"{max_nhsteps=}")
print(f"Number of data points: {target_data.shape[0]}")
feature_data = ncloader.load_feature_data()
assert target_data.shape[0] == feature_data.shape[0]
print(f"Number of scalar features: {feature_data.shape[1]}")

# Normalise the feature data
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
criterion = torch.nn.CrossEntropyLoss(reduction="sum")

# Train
train_losses, validation_losses = [], []
for epoch in range(1, num_epochs + 1):
    # Training step
    start_time = perf_counter()
    train_loss, train_frac = propagate(
        train_loader, nn, criterion, optimizer, device=device
    )
    mid_time = perf_counter()
    train_time = mid_time - start_time

    # Validation step
    validation_loss, validation_frac = propagate(
        validate_loader, nn, criterion, device=device
    )
    validation_time = perf_counter() - mid_time

    # Stash progress
    print(
        f"Epoch {epoch:4d}/{num_epochs:d}"
        f"  avg loss: {train_loss:.4e} ({validation_loss:.4e})"
        f"  incorrect: {train_frac:.2f}% ({validation_frac:.2f})%"
        f"  wallclock: {train_time:.2f}s ({validation_time:.2f}s)"
    )
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)

    # gradients = torch.tensor([p.grad.norm() for p in nn.parameters()])
    # if gradients.allclose(torch.zeros_like(gradients)):
    #     print("Terminating due to all gradients being zero.")
    #     break

    # Save the model and loss progress perodically
    if epoch % 100 == 0:
        torch.save(torch.Tensor(train_losses), f"{data_dir}/train_losses.pt")
        torch.save(torch.Tensor(validation_losses), f"{data_dir}/validation_losses.pt")
        torch.save(nn.state_dict(), "model.pt")
