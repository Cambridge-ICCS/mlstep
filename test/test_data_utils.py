"""Unit tests for the data_utils module."""

import os

import pytest

from mlstep.data_utils import NetCDFDataLoader


@pytest.fixture
def data_dir():
    """Fixture to provide a valid data directory for testing."""
    return os.path.join(os.path.dirname(__file__), "data")


def constructor(data_dir):
    """Create a NetCDFDataLoader instance."""
    return NetCDFDataLoader(
        features_1d=["test_feature1d"],
        features_2d=["test_feature2d"],
        num_timesteps=3,
        data_dir=data_dir,
    )


def test_init_valid_data_dir(data_dir):
    """Test initialization with a valid data directory."""
    assert constructor(data_dir).data_dir == data_dir


def test_init_invalid_data_dir():
    """Test initialization with an invalid data directory."""
    with pytest.raises(IOError, match="Data directory 'invalid_dir' does not exist."):
        constructor("invalid_dir")


def test_load_target_data(data_dir):
    """Test loading target data."""
    loader = constructor(data_dir)
    target_data = loader.load_target_data()
    num_data_points = 3 * 10  # 3 timesteps * 10 spatial points
    assert target_data.shape == (num_data_points, loader.max_nhsteps + 1)


def test_load_feature_data_1d(data_dir):
    """Test loading 1D feature data."""
    loader = constructor(data_dir)
    feature_data = loader.load_feature_data_1d()
    assert len(feature_data) == len(loader.features_1d)


def test_load_feature_data_2d(data_dir):
    """Test loading 2D feature data."""
    loader = constructor(data_dir)
    feature_data = loader.load_feature_data_2d()
    assert len(feature_data) == len(loader.features_2d)


def test_load_feature_data(data_dir):
    """Test loading all feature data."""
    loader = constructor(data_dir)
    feature_data = loader.load_feature_data()
    num_data_points = 3 * 10  # 3 timesteps * 10 spatial points
    num_features = 1 + 3  # 1D feature + 3-component 2D feature
    assert feature_data.shape == (num_data_points, num_features)
