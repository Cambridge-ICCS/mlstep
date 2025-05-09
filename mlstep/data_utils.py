"""Utilities for handling NetCDF data files."""

import os

import netCDF4
import torch

__all__ = ["NetCDFDataLoader", "prepare_for_classification"]


class NetCDFDataLoader:
    """Class for handling loading data from NetCDF files."""

    def __init__(self, num_timesteps, data_dir="data"):
        """
        Initialise the NetCDFDataLoader.

        :param num_timesteps: Number of NetCDF files to load.
        :param data_dir: Directory where the NetCDF files are stored (defaults to
            "data").
        """
        self.num_timesteps = num_timesteps
        if not os.path.exists(data_dir):
            errmsg = f"Data directory {data_dir} does not exist."
            raise IOError(errmsg)
        self.data_dir = data_dir

    def load_feature_data_1d(self, variable, dtype=torch.float):
        """
        Load feature data corresponding to a 1D variable from a NetCDF file.

        :param variable: Variable name to load.
        :param dtype: Data type to use.
        """
        arr = []
        for i in range(1, self.num_timesteps + 1):
            with netCDF4.Dataset(f"{self.data_dir}/{variable}_{i}.nc", "r") as nc:
                arr.append(torch.Tensor(nc.variables[variable][:]).to(dtype=dtype))
        return torch.hstack(arr)

    def load_feature_data_2d(self, variable, dtype=torch.float):
        """
        Load feature data corresponding to a 2D variable from a NetCDF file.

        :param variable: Variable name to load.
        :param dtype: Data type to use.
        """
        arr = []
        for i in range(1, self.num_timesteps + 1):
            with netCDF4.Dataset(f"{self.data_dir}/{variable}_{i}.nc", "r") as nc:
                arr.append(torch.Tensor(nc.variables[variable][:][:]).to(dtype=dtype))
        return torch.hstack(arr)

    def load_nhsteps_data(self):
        """
        Load halving steps data from netCDF files.

        :param num_timesteps: Number of NetCDF files to load.
        :returns: The number of halving steps for each grid-box and timestep.
        """
        nhsteps = []
        for i in range(1, self.num_timesteps + 1):
            with netCDF4.Dataset(f"{self.data_dir}/ncsteps_{i}.nc", "r") as nc:
                ncsteps = torch.Tensor(nc.variables["ncsteps"][:])
                nhsteps.append(torch.round(torch.log2(ncsteps)).to(dtype=torch.int))
        return torch.hstack(nhsteps)


def prepare_for_classification(nhsteps):
    """
    Prepare halving steps data for use in a classification problem.

    This involves reformating as a binary matrix, where entry :math:`(i,j)` is one if
    entry i of nhsteps takes the value :math:`2^j` and zero otherwise.

    :param nhsteps: Tensor of halving steps data as rank-1 tensor.
    :returns: Halving steps data in binary matrix format (rank-2).
    """
    max_nhsteps = int(nhsteps.max().item())
    target_data = torch.zeros((len(nhsteps), max_nhsteps + 1), dtype=torch.int)
    for i, nhstep in enumerate(nhsteps):
        target_data[i, nhstep] = 1
    return target_data
