import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class Trained:
    """A network trained with `train_classifier`, at the epoch with the lowest validation loss."""
    model: nn.Sequential
    epoch: int
    validation_loss: float
    validation_accuracy: float


def _tensor(X):
    return torch.as_tensor(np.asarray(X), dtype=torch.float32)


def train_classifier(X_train, y_train, X_validation, y_validation, units: int, seed: int = 0,
                     learning_rate: float = 1e-2, epochs: int = 400) -> Trained:
    """Network with one hidden layer of `units` neurons (ReLU), trained with Adam on the whole training set at each step.
    The parameters of the epoch with the lowest validation loss are kept."""
    torch.manual_seed(seed)
    x_train, x_validation = _tensor(X_train), _tensor(X_validation)
    t_train, t_validation = torch.as_tensor(y_train), torch.as_tensor(y_validation)
    n_classes = int(t_train.max()) + 1
    model = nn.Sequential(nn.Linear(x_train.shape[1], units), nn.ReLU(), nn.Linear(units, n_classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best = (np.inf, None, 0)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss_fn(model(x_train), t_train).backward()
        optimizer.step()
        with torch.no_grad():
            loss = loss_fn(model(x_validation), t_validation).item()
        if loss < best[0]:
            best = (loss, copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    with torch.no_grad():
        accuracy = (model(x_validation).argmax(dim=1) == t_validation).float().mean().item()
    return Trained(model, best[2], best[0], accuracy)


def hidden_vectors(model: nn.Sequential, X) -> np.ndarray:
    """Activations of the hidden layer: the point of the hidden space where each row of X is mapped."""
    with torch.no_grad():
        return model[:2](_tensor(X)).numpy()


def ingredient_vectors(model: nn.Sequential) -> np.ndarray:
    """Representation of each input term: the hidden vector of a recipe that contains only that term."""
    return hidden_vectors(model, np.eye(model[0].in_features))


def class_distribution(model: nn.Sequential, X) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(model(_tensor(X)), dim=1).numpy()
