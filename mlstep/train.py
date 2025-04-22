"""Module containing utilities for training ML models."""

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
    cumulative_loss = 0

    for x, y in data_loader:
        # Compute prediction and loss
        prediction = model(x.to(device))
        loss = loss_fn(prediction, y.to(device))
        cumulative_loss += loss.item()

        # Backpropagation
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return cumulative_loss / num_batches
