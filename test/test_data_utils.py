"""Unit tests for the data_utils module."""

import os
import re

import pytest

from mlstep.data_utils import NetCDFDataLoader


@pytest.fixture(params=[1, 2, 3])
def num_timesteps(request):
    """Fixture to provide a range of timesteps for testing."""
    return request.param


@pytest.fixture
def data_dir():
    """Fixture to provide a valid data directory for testing."""
    return os.path.join(os.path.dirname(__file__), "data")


def constructor(data_dir, num_timesteps=3):
    """Create a NetCDFDataLoader instance."""
    return NetCDFDataLoader(
        features_1d=["test_feature1d"],
        features_2d=["test_feature2d"],
        num_timesteps=num_timesteps,
        data_dir=data_dir,
    )


def test_netcdf_files_exist(data_dir):
    """Test that the NetCDF files exist in the data directory."""
    for timestep in [1, 2, 3]:
        target_file = os.path.join(data_dir, f"ncsteps_{timestep}.nc")
        if not os.path.isfile(target_file):
            io_err = (
                f"NetCDF target data file '{target_file}' does not exist. Did you"
                " generate the NetCDF files for the tests?"
            )
            raise IOError(io_err)
        for dim in ["1d", "2d"]:
            feature_file = os.path.join(data_dir, f"test_feature{dim}_{timestep}.nc")
            if not os.path.isfile(feature_file):
                io_err = (
                    f"NetCDF feature data file '{feature_file}' does not exist. Did you"
                    " generate the NetCDF files for the tests?"
                )
                raise IOError(io_err)


def test_init_valid_data_dir(data_dir):
    """Test initialization with a valid data directory."""
    assert constructor(data_dir).data_dir == data_dir


def test_init_invalid_data_dir():
    """Test initialization with an invalid data directory."""
    with pytest.raises(IOError, match="Data directory 'invalid_dir' does not exist."):
        constructor("invalid_dir")


def test_load_target_data(data_dir, num_timesteps):
    """Test loading target data."""
    num_points = 10  # Grid size in all input files
    loader = constructor(data_dir, num_timesteps=num_timesteps)
    target_data = loader.load_target_data()
    num_data_points = num_timesteps * num_points
    assert target_data.shape == (num_data_points, loader.max_nhsteps + 1)


def test_features_error(data_dir):
    """Check an error is raised if the features property is accessed prematurely."""
    msg = "Features have not been loaded. Call load_feature_data() first."
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        _ = constructor(data_dir).features


def test_load_feature_data(data_dir, num_timesteps):
    """Test loading all feature data."""
    num_points = 10  # Grid size in all input files
    num_data_points = num_timesteps * num_points
    num_features = 1 + 3  # 1D feature + 3-component 2D feature
    loader = constructor(data_dir, num_timesteps=num_timesteps)
    feature_data = loader.load_feature_data()
    assert feature_data.shape == (num_data_points, num_features)


def test_max_nhsteps_error(data_dir):
    """Check an error is raised if the max_nhsteps property is accessed prematurely."""
    msg = "Max halving steps have not been set. Call load_target_data() first."
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        _ = constructor(data_dir).max_nhsteps


def test_max_nhsteps(data_dir, num_timesteps):
    """Test the maximum number of halving steps calculation."""
    loader = constructor(data_dir, num_timesteps=num_timesteps)
    loader.load_target_data()
    assert loader.max_nhsteps == num_timesteps + 2
