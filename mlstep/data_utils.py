"""Utilities for handling NetCDF data files."""

import os

import netCDF4
import torch

__all__ = ["NetCDFDataLoader"]


class NetCDFDataLoader:
    """Class for handling loading data from NetCDF files."""

    def __init__(
        self, features_1d, features_2d, num_timesteps, zero_factor=3, data_dir="data"
    ):
        """
        Initialise the NetCDFDataLoader.

        :param features_1d: List of 1D feature variable names to load.
        :param features_2d: List of 2D feature variable names to load.
        :param num_timesteps: Number of NetCDF files to load.
        :param zero_factor: Number of zero targets to include for each non-zero target.
        :param data_dir: Directory where the NetCDF files are stored (defaults to
            "data").
        """
        self.features_1d = features_1d
        self.features_2d = features_2d
        self.num_timesteps = num_timesteps
        self.zero_factor = zero_factor
        if not os.path.exists(data_dir):
            errmsg = f"Data directory {data_dir} does not exist."
            raise IOError(errmsg)
        self.data_dir = data_dir
        self._indices = None
        self._max_nhsteps = None

    def _subsample_indices(self, nhsteps):
        """
        Subsample the indices to reduce the number of data points.

        This is achieved by taking the non-zero targets plus the :attr:`zero_factor`
        times as many zero targets.

        The output is stored in the :attr:`_indices` attribute.

        :param nhsteps: Halving steps data as rank-1 tensor.
        """
        indices = [int(i) for i in nhsteps.nonzero()]
        N = (self.zero_factor + 1) * len(indices)
        if len(nhsteps) < N:
            errmsg = "Not enough data points to subsample."
            raise ValueError(errmsg)
        i = 0
        while len(indices) < N:
            if i not in indices:
                indices.append(i)
            i = i + 1
        indices.sort()
        self._indices = torch.Tensor(indices).to(dtype=torch.int)

    @property
    def indices(self):
        """
        Get the indices of the subsampled data.

        :returns: The indices of the subsampled data.
        """
        if self._indices is None:
            errmsg = "Indices have not been set. Call load_target_data first."
            raise RuntimeError(errmsg)
        return self._indices

    @property
    def max_nhsteps(self):
        """
        Get the maximum number of halving steps.

        :returns: The maximum number of halving steps.
        """
        if self._max_nhsteps is None:
            errmsg = (
                "Max halving steps have not been set. Call load_target_data() first."
            )
            raise RuntimeError(errmsg)
        return self._max_nhsteps

    def _prepare_for_classification(self, nhsteps):
        """
        Prepare halving steps data for use in a classification problem.

        This involves reformating as a binary matrix, where entry :math:`(i,j)` is one
        if entry i of nhsteps takes the value :math:`2^j` and zero otherwise.

        :param nhsteps: Tensor of halving steps data as rank-1 tensor.
        :returns: Halving steps data in binary matrix format (rank-2).
        """
        max_nhsteps = int(nhsteps.max().item())
        target_data = torch.zeros((len(nhsteps), max_nhsteps + 1), dtype=torch.int)
        for i, nhstep in enumerate(nhsteps):
            target_data[i, nhstep] = 1
        return target_data

    def load_target_data(self):
        """
        Load halving steps data from netCDF files.

        The halving steps data are converted to a binary matrix format, where entry
        :math:`(i,j)` is one if entry i of nhsteps takes the value :math:`2^j` and zero
        otherwise.

        :param num_timesteps: Number of NetCDF files to load.
        :returns: The number of halving steps for each grid-box and timestep as a binary
            matrix (rank-2 tensor).
        """
        nhsteps = []
        for i in range(1, self.num_timesteps + 1):
            with netCDF4.Dataset(f"{self.data_dir}/ncsteps_{i}.nc", "r") as nc:
                ncsteps = torch.Tensor(nc.variables["ncsteps"][:])
                nhsteps.append(torch.round(torch.log2(ncsteps)).to(dtype=torch.int))
        nhsteps = torch.hstack(nhsteps)
        self._subsample_indices(nhsteps)
        self._max_nhsteps = int(nhsteps.max().item())
        nhsteps = nhsteps[self.indices]
        return self._prepare_for_classification(nhsteps)

    def load_feature_data_1d(self, dtype=torch.float):
        """
        Load feature data corresponding to 1D variable from NetCDF files.

        :param dtype: Data type to use.
        """
        data = []
        for variable in self.features_1d:
            arr = []
            for i in range(1, self.num_timesteps + 1):
                with netCDF4.Dataset(f"{self.data_dir}/{variable}_{i}.nc", "r") as nc:
                    arr.append(torch.Tensor(nc.variables[variable][:]).to(dtype=dtype))
            data.append(torch.hstack(arr)[self.indices])
        return data

    def load_feature_data_2d(self, dtype=torch.float):
        """
        Load feature data corresponding to 2D variables from NetCDF files.

        :param dtype: Data type to use.
        """
        data = []
        for variable in self.features_2d:
            arr = []
            for i in range(1, self.num_timesteps + 1):
                with netCDF4.Dataset(f"{self.data_dir}/{variable}_{i}.nc", "r") as nc:
                    arr.append(
                        torch.Tensor(nc.variables[variable][:][:]).to(dtype=dtype)
                    )
            data.append(torch.hstack(arr)[:, self.indices])
        return data

    def load_feature_data(self, dtype=torch.float):
        """
        Load feature data from NetCDF files.

        :param dtype: Data type to use.
        :returns: Feature data as a rank-2 tensor.
        """
        feature_data = self.load_feature_data_1d(dtype=dtype)
        for features in self.load_feature_data_2d(dtype=dtype):
            feature_data += features
        return torch.stack(feature_data, dim=1)
