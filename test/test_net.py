"""Unit tests for the net module."""

import pytest
import torch

from mlstep.net import FCNN


@pytest.fixture
def input_size():
    """Fixture to provide the input size for the model."""
    return 10


@pytest.fixture
def max_nhsteps():
    """Fixture to provide the maximum number of halving steps."""
    return 5


@pytest.fixture
def hidden_size():
    """Fixture to provide the hidden size for the model."""
    return 50


@pytest.fixture(params=[1, 5])
def batch_size(request):
    """Fixture to provide the batch size for the model."""
    return request.param


def test_initialization(input_size, max_nhsteps, hidden_size):
    """Test that the model initialises correctly."""
    fcnn_model = FCNN(input_size, max_nhsteps, hidden_size)
    assert hasattr(fcnn_model, "hidden")
    assert hasattr(fcnn_model, "output")
    assert fcnn_model.hidden.in_features == input_size
    assert fcnn_model.hidden.out_features == hidden_size
    assert fcnn_model.output.in_features == hidden_size
    assert fcnn_model.output.out_features == max_nhsteps + 1


def test_forward_output_shape(input_size, max_nhsteps, hidden_size, batch_size):
    """Test that the forward method produces the correct output shape with batching."""
    fcnn_model = FCNN(input_size, max_nhsteps, hidden_size)
    input_tensor = torch.randn(batch_size, 10)
    output = fcnn_model(input_tensor)
    assert output.shape == (batch_size, max_nhsteps + 1)


def test_forward_output_sum(input_size, max_nhsteps, hidden_size):
    """Test that the output of the forward method sums to 1 (softmax property)."""
    fcnn_model = FCNN(input_size, max_nhsteps, hidden_size)
    input_tensor = torch.randn(1, 10)
    output = fcnn_model(input_tensor)
    assert pytest.approx(output.sum().item(), rel=1e-5) == 1.0
