"""Module containing utilities for propagating data through ML models."""

import contextlib

import torch

__all__ = ["propagate"]


def propagate(data_loader, model, loss_fn, optimizer=None, device="cpu"):
    """
    Propagate data through a model using a given loss function.

    :param data_loader: PyTorch DataLoader providing data to propagate.
    :param model: PyTorch Model to propagate data through.
    :param loss_fn: PyTorch Loss function for checking outputs.
    :param optimizer: Optional PyTorch Optimizer instance, indicating training if
        present and validation otherwise.
    :param device: String denoting the device to run the computation on.
    """
    num_batches = len(data_loader)
    cumulative_loss = 0.0
    is_training = optimizer is not None
    if is_training:
        model.train(True)
    else:
        model.eval()

    for x, y in data_loader:
        # Configure the model for training or evaluation, as appropriate
        if is_training:
            optimizer.zero_grad()

        # Compute prediction and loss
        with contextlib.nullcontext() if is_training else torch.no_grad():
            prediction = model(x.to(device))
            target = y.to(device, dtype=torch.float)
            loss = loss_fn(prediction, target)
            cumulative_loss += loss.item()

        # assert target.shape == prediction.shape

        # Backpropagation
        if is_training:
            loss.backward()
            optimizer.step()

    # Keep track of the number of wrong predictions
    num_wrong = (prediction - target).abs().sum() / 2
    return cumulative_loss / num_batches, 100 * num_wrong / num_batches
