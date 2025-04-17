"""
Module containing neural network architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNN(nn.Module):
    """
    FCNN with a single hidden layer and ReLU activation function that accepts a vector
    of real-valued input data and returns a single scalar natural number in the range
    [1, 5].
    """
    def __init__(self, input_size, hidden_size=50):
        """
        :param input_size: Size of the input vector.
        :param hidden_size: Size of the hidden layer (defaults to 50).
        """
        super(FCNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 5)  # 5 outputs for classes [1, 2, 3, 4, 5]

    def forward(self, x):
        """
        :param x: input vector for the model
        """
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return torch.argmax(F.softmax(x, dim=1), dim=1) + 1  # Shift to range [1, 5]
