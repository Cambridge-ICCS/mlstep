"""Utilities for handling NetCDF data files."""

import os

import netCDF4
import torch

__all__ = ["NetCDFDataLoader"]


class NetCDFDataLoader:
    """Class for handling loading data from NetCDF files."""

    def __init__(self, features_1d, features_2d, num_timesteps, data_dir="data"):
        """
        Initialise the NetCDFDataLoader.

        :param features_1d: List of 1D feature variable names to load.
        :param features_2d: List of 2D feature variable names to load.
        :param num_timesteps: Number of NetCDF files to load.
        :param data_dir: Directory where the NetCDF files are stored (defaults to
            "data").
        """
        self._features_1d = features_1d
        self._features_2d = features_2d
        self._features = None
        self.num_timesteps = num_timesteps
        if not os.path.exists(data_dir):
            errmsg = f"Data directory '{data_dir}' does not exist."
            raise IOError(errmsg)
        self.data_dir = data_dir
        self._max_nhsteps = None

    @property
    def features(self):
        """:returns: List of feature variable names."""
        if self._features is None:
            errmsg = "Features have not been loaded. Call load_feature_data() first."
            raise RuntimeError(errmsg)
        return self._features

    @property
    def max_nhsteps(self):
        """:returns: The maximum number of halving steps."""
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

        This method has the side-effect of determining the `max_nhsteps` attribute for
        the maximum number of halving steps across all target data.

        :param num_timesteps: Number of NetCDF files to load.
        :returns: The number of halving steps for each grid-box and timestep as a binary
            matrix (rank-2 tensor).
        """
        nhsteps = torch.hstack(
            [
                torch.log2(self._read_nc("ncsteps", timestep, 1)).round().int()
                for timestep in range(1, self.num_timesteps + 1)
            ]
        )
        self._max_nhsteps = int(nhsteps.max().item())
        return self._prepare_for_classification(nhsteps)

    def _read_nc(self, variable, timestep, dim):
        """
        Read a feature from a feature data NetCDF file.

        :param dim: Number of dimensions in the feature data.
        :returns: Torch tensor containing the feature data
        """
        with netCDF4.Dataset(f"{self.data_dir}/{variable}_{timestep}.nc", "r") as nc:
            return torch.Tensor(
                nc.variables[variable][:] if dim == 1 else nc.variables[variable][:][:]
            )

    def _load_feature_data(self, dim, dtype=torch.float):
        """
        Load feature data from NetCDF files.

        :param dim: Number of dimensions in the feature data.
        :param dtype: Data type to use.
        :returns: List of tensors containing feature data.
        """
        assert dim in (1, 2), "Only feature data with 1 or 2 dimensions are supported."
        return [
            torch.hstack(
                [
                    self._read_nc(feature, timestep, dim).to(dtype=dtype)
                    for timestep in range(1, self.num_timesteps + 1)
                ]
            )
            for feature in (self._features_1d if dim == 1 else self._features_2d)
        ]

    def load_feature_data(self, dtype=torch.float):
        """
        Load feature data from NetCDF files.

        This method has the side-effect of determining the overall `features` attribute,
        which lists each individual component of the feature data (i.e., 1D feature
        arrays and each 1D slice of 2D feature arrays.

        :param dtype: Data type to use.
        :returns: Feature data as a rank-2 tensor.
        """
        # TODO: Refactor this method
        feature_data = self._load_feature_data(1, dtype=dtype)
        self._features = self._features_1d
        for i, features in enumerate(self._load_feature_data(2, dtype=dtype)):
            feature_name = self._features_2d[i]
            self._features += [f"{feature_name}_{j}" for j in range(len(features))]
            feature_data += features
        return torch.stack(feature_data, dim=1)
