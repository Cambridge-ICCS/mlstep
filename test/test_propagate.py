"""Unit tests for the propagate module."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mlstep.propagate import propagate


@pytest.fixture
def sample_data():
    """Create a simple dataset."""
    x = torch.randn(100, 10)  # 100 samples, 10 features each
    y = torch.randint(0, 2, (100,))  # Binary classification (0 or 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=10)


@pytest.fixture
def model():
    """Create a simple model."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 2)
    )


@pytest.fixture
def loss_fn():
    """Use CrossEntropyLoss for classification."""
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(model):
    """Use Adam optimizer."""
    return torch.optim.Adam(model.parameters())


def test_propagate_validation(sample_data, model, loss_fn):
    """Test propagate in validation mode."""
    loss = propagate(sample_data, model, loss_fn)
    assert loss >= 0, "Loss should be non-negative"


def test_propagate_training(sample_data, model, loss_fn, optimizer):
    """Test propagate in training mode."""
    loss = propagate(sample_data, model, loss_fn, optimizer=optimizer)
    assert loss >= 0, "Loss should be non-negative"
