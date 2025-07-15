"""Unit tests for the data_utils module."""

import os

import pytest
import torch

from mlstep.data_utils import NetCDFDataLoader


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


def test_init_valid_data_dir(data_dir):
    """Test initialization with a valid data directory."""
    loader = NetCDFDataLoader(
        features_1d=["test_feature1d"],
        features_2d=["test_feature2d"],
        num_timesteps=3,
        data_dir=data_dir,
    )
    assert loader.data_dir == data_dir


def test_init_invalid_data_dir():
    """Test initialization with an invalid data directory."""
    with pytest.raises(IOError, match="Data directory .* does not exist."):
        NetCDFDataLoader(
            features_1d=["test_feature1d"],
            features_2d=["test_feature2d"],
            num_timesteps=3,
            data_dir="invalid_dir",
        )


def test_load_target_data(data_dir):
    """Test loading target data."""
    loader = NetCDFDataLoader(
        features_1d=["test_feature1d"],
        features_2d=["test_feature2d"],
        num_timesteps=3,
        data_dir=data_dir,
    )
    # Mock the NetCDF file reading
    torch.manual_seed(0)
    nhsteps = torch.randint(1, 4, (10,))
    loader._max_nhsteps = int(nhsteps.max().item())
    target_data = loader._prepare_for_classification(nhsteps)
    assert target_data.shape == (len(nhsteps), loader.max_nhsteps + 1)


def test_load_feature_data_1d(data_dir):
    """Test loading 1D feature data."""
    loader = NetCDFDataLoader(
        features_1d=["test_feature1d"],
        features_2d=["test_feature2d"],
        num_timesteps=3,
        data_dir=data_dir,
    )
    # Mock the NetCDF file reading
    torch.manual_seed(0)
    data = loader.load_feature_data_1d()
    assert len(data) == len(loader.features_1d)


def test_load_feature_data_2d(data_dir):
    """Test loading 2D feature data."""
    loader = NetCDFDataLoader(
        features_1d=["test_feature1d"],
        features_2d=["test_feature2d"],
        num_timesteps=3,
        data_dir=data_dir,
    )
    # Mock the NetCDF file reading
    torch.manual_seed(0)
    data = loader.load_feature_data_2d()
    assert len(data) == len(loader.features_2d)


def test_load_feature_data(data_dir):
    """Test loading all feature data."""
    loader = NetCDFDataLoader(
        features_1d=["test_feature1d"],
        features_2d=["test_feature2d"],
        num_timesteps=3,
        data_dir=data_dir,
    )
    # Mock the NetCDF file reading
    torch.manual_seed(0)
    feature_data = loader.load_feature_data()
    assert feature_data.ndim == 2
