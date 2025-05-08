"""Utilities for handling NetCDF data files."""

import os

import netCDF4
import torch

__all__ = ["load_nhsteps_data"]


def load_nhsteps_data(num_timesteps, data_dir="data"):
    """
    Load halving steps data from netCDF files.

    :param num_timesteps: Number of NetCDF files to load.
    :param data_dir: Directory where the netCDF files are stored (defaults to "data").
    :returns: The number of halving steps for each grid-box and timestep.
    """
    if not os.path.exists(data_dir):
        errmsg = f"Data directory {data_dir} does not exist."
        raise IOError(errmsg)
    nhsteps = []
    for i in range(1, num_timesteps + 1):
        with netCDF4.Dataset(f"{data_dir}/ncsteps_{i}.nc", "r") as nc_file:
            ncsteps = torch.Tensor(nc_file.variables["ncsteps"][:])
            nhsteps.append(torch.round(torch.log2(ncsteps)).to(dtype=torch.int))
    nhsteps = torch.hstack(nhsteps)
    return nhsteps
