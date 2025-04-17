"""
Module containing neural network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FCNN"]


class FCNN(nn.Module):
    """
    Simple FCNN architecture designed to estimate the number of timestep length halvings
    required by a solver.

    The FCNN accepts a vector of real-valued input data and returns a single scalar
    natural number for the number of halving steps required. It has a single hidden
    layer and uses a ReLU activation function.

    The input size is set upon constructing the class, as is the output size, which
    determines the maximum permissible number of halving steps. The number of halving
    steps is treated as a categorical variable, with each category corresponding to a
    non-negative integer. Note that the output size is equal to the maximum number of
    halving steps plus one, to account for the fact that in many cases the initial
    choice of timestep is sufficient, i.e. zero halvings are required.

    The number of neurons in the hidden layer is also configurable upon constructing the
    class.
    """

    def __init__(self, input_size, max_nhsteps=5, hidden_size=50):
        """
        :param input_size: Size of the input vector.
        :param max_nhsteps: Maximum permissible number of halving steps (defaults to 5).
        :param hidden_size: Size of the hidden layer (defaults to 50).
        """
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, max_nhsteps + 1)

    def forward(self, x):
        """
        :param x: input vector for the model
        """
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return torch.argmax(F.softmax(x, dim=1), dim=1)
