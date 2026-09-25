import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp.classification import LabelScheme, split


def softmax(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def cross_entropy(P, Y, eps=1e-12):
    return -np.mean(np.log(np.clip((P * Y).sum(axis=1), eps, 1)))


@dataclass
class Task:
    """A classification problem: the recipes with a label, the split and the TF-IDF vectors (fitted on the training set)."""
    scheme: LabelScheme
    indices: np.ndarray
    labels: np.ndarray
    classes: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    vectorizer: TfidfVectorizer
    X_train: object
    X_validation: object
    X_test: object

    @property
    def vocabulary(self):
        return self.vectorizer.get_feature_names_out()

    def y(self, positions):
        return self.labels[positions]

    def one_hot(self, positions):
        return np.eye(len(self.classes))[self.labels[positions]]


def prepare_task(recipes, terms, scheme: LabelScheme, min_df: int = 2, seed: int = 0) -> Task:
    """Selects the recipes with a label, splits them, and builds the TF-IDF vectors of the terms."""
    indices, labels, classes = scheme.apply(recipes)
    train, validation, test = split(labels, seed=seed)
    vectorizer = TfidfVectorizer(analyzer=lambda t: t, min_df=min_df)
    matrix = lambda positions: [terms[indices[i]] for i in positions]
    X_train = vectorizer.fit_transform(matrix(train))
    return Task(scheme, indices, labels, classes, train, validation, test, vectorizer,
                X_train, vectorizer.transform(matrix(validation)), vectorizer.transform(matrix(test)))


def fit_linear(task: Task, learning_rate: float, epochs: int = 3000):
    """Model without hidden layers, trained by gradient descent on the training set. Returns the parameters of the epoch
    with the lowest validation loss: (validation loss, (W, b), epoch)."""
    X, Y = task.X_train, task.one_hot(task.train)
    X_val, Y_val = task.X_validation, task.one_hot(task.validation)
    W, b = np.zeros((X.shape[1], Y.shape[1])), np.zeros(Y.shape[1])
    best = (np.inf, None, 0)
    for epoch in range(1, epochs + 1):
        error = softmax(X @ W + b) - Y
        W -= learning_rate * (X.T @ error) / len(Y)
        b -= learning_rate * error.mean(axis=0)
        loss = cross_entropy(softmax(X_val @ W + b), Y_val)
        if loss < best[0]:
            best = (loss, (W.copy(), b.copy()), epoch)
    return best


def _tensor(X):
    return torch.tensor(X.toarray(), dtype=torch.float32)


def fit_hidden(task: Task, learning_rate: float, seed: int = 0, epochs: int = 300, units: int = 64, batch_size: int = 64):
    """Network with one hidden layer, trained with Adam on mini-batches. Returns (validation loss, model, epoch)."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(task.X_train.shape[1], units), nn.ReLU(), nn.Linear(units, len(task.classes)))
    x_train, t_train = _tensor(task.X_train), torch.tensor(task.y(task.train))
    x_val, t_val = _tensor(task.X_validation), torch.tensor(task.y(task.validation))
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best = (np.inf, None, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss_fn(model(x_train[batch]), t_train[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = loss_fn(model(x_val), t_val).item()
        if loss < best[0]:
            best = (loss, copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    return best[0], model, best[2]


def distribution_hidden(model, X):
    with torch.no_grad():
        return torch.softmax(model(_tensor(X)), dim=1).numpy()


@dataclass
class Comparison:
    """The two models of a task, trained with the protocol of the notebook: the learning rate is chosen on the validation
    set, the network with the hidden layer is trained with several seeds."""
    grid: pd.DataFrame
    chosen: dict
    P_validation: dict
    P_test: dict            # "linear" and "hidden layer" (first seed)
    P_test_seeds: list      # test distributions of the networks trained with all the seeds


def compare_models(task: Task, linear_rates=(1, 3, 10, 30), hidden_rates=(1e-4, 3e-4, 1e-3), seeds: int = 5) -> Comparison:
    grid = {}
    for rate in linear_rates:
        loss, _, epoch = fit_linear(task, rate)
        grid[("linear", rate)] = {"best epoch": epoch, "validation loss": loss}
    for rate in hidden_rates:
        loss, _, epoch = fit_hidden(task, rate)
        grid[("hidden layer", rate)] = {"best epoch": epoch, "validation loss": loss}
    grid = pd.DataFrame(grid).T.rename_axis(["model", "learning rate"])
    chosen = {model: grid.loc[model]["validation loss"].idxmin() for model in ["linear", "hidden layer"]}

    _, (W, b), _ = fit_linear(task, chosen["linear"])
    networks = [fit_hidden(task, chosen["hidden layer"], seed=seed)[1] for seed in range(seeds)]
    P_validation = {"linear": softmax(task.X_validation @ W + b), "hidden layer": distribution_hidden(networks[0], task.X_validation)}
    P_test_seeds = [distribution_hidden(model, task.X_test) for model in networks]
    P_test = {"linear": softmax(task.X_test @ W + b), "hidden layer": P_test_seeds[0]}
    return Comparison(grid, chosen, P_validation, P_test, P_test_seeds)
