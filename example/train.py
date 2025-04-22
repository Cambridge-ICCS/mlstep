"""Script for training the ML model."""

import pandas as pd
import torch
from sklearn import model_selection

# FIXME: Why are the first 705 values unset?
istart = 705

# Set parameters
test_size = 0.3
seed = 42

# Load the target data from file as Torch Tensors
df_target = pd.read_csv("target_data.csv")
target_data = torch.Tensor(df_target["nhsteps"][istart:].values)

# Load the input data from file as Torch Tensors
df_input = pd.read_csv("feature_data.csv")
zp = torch.Tensor(df_input["zp"][istart:].values)
zt = torch.Tensor(df_input["zt"][istart:].values)
zq = torch.Tensor(df_input["zq"][istart:].values)
cldf = torch.Tensor(df_input["cldf"][istart:].values)
cldl = torch.Tensor(df_input["cldl"][istart:].values)
stratflag = torch.Tensor([1 if c == "T" else 0 for c in df_input["stratflag"][istart:]])

# Stack the input data arrays
feature_data = torch.stack([zp, zt, zq, cldf, cldl, stratflag], dim=1)

# Normalise the data and split for training and validation
feature_data -= feature_data.min(0, keepdim=True)[0]
feature_data /= feature_data.max(0, keepdim=True)[0]
xtrain, xval, ytrain, yval = model_selection.train_test_split(
    feature_data, target_data, test_size=test_size, random_state=seed
)
