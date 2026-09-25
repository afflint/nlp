import numpy as np
from sklearn.model_selection import train_test_split

MISSING = (None, "None", "")


class LabelScheme:
    """Selects a labeling of the recipes: which field is the label, and how it is turned into classes.

    - `field`: the field of the recipe used as label (`cuisine`, `type`);
    - `positive`: if given, the problem is binary (one class against all the others): the label is 1
      for the recipes of the positive class and 0 for all the others;
    - `subsample`: fraction of the recipes to keep for some classes, e.g. {"Dessert": 0.1}, to make
      the classes unbalanced. It is applied to the original classes, before the binarization.

    The recipes with a missing label are always dropped.
    """

    def __init__(self, field: str, positive: str = None, subsample: dict = None, seed: int = 0):
        self.field = field
        self.positive = positive
        self.subsample = subsample or {}
        self.seed = seed

    def apply(self, recipes):
        """Returns the indices of the recipes that are kept, their labels (integers) and the class names."""
        rng = np.random.default_rng(self.seed)
        indices, values = [], []
        for i, recipe in enumerate(recipes):
            value = recipe[self.field]
            if value in MISSING:
                continue
            if rng.random() >= self.subsample.get(value, 1.0):
                continue
            indices.append(i)
            values.append(value)

        if self.positive is not None:
            classes = np.array([f"not {self.positive}", self.positive])
            labels = np.array([v == self.positive for v in values], dtype=int)
        else:
            classes, labels = np.unique(values, return_inverse=True)
        return np.array(indices), labels, classes

    def __repr__(self):
        return f"LabelScheme({self.field!r}, positive={self.positive!r}, subsample={self.subsample!r})"


def split(labels, sizes=(0.6, 0.2, 0.2), seed: int = 0):
    """Splits the positions 0..n-1 in train, validation and test sets, keeping the proportion of the classes."""
    train_size, validation_size, test_size = sizes
    if abs(train_size + validation_size + test_size - 1) > 1e-9:
        raise ValueError("sizes must sum to 1")
    positions = np.arange(len(labels))
    train, rest = train_test_split(positions, train_size=train_size, stratify=labels, random_state=seed)
    validation, test = train_test_split(rest, train_size=validation_size / (validation_size + test_size),
                                        stratify=labels[rest], random_state=seed)
    return train, validation, test


def confusion_matrix(y_true, y_pred, n_classes: int):
    """Rows are the true classes, columns are the predicted classes."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return matrix


def _ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=np.asarray(denominator) > 0)


def precision_recall_f1(matrix):
    """Precision, recall and F1 of every class, from a confusion matrix.

    When a measure is not defined (a class that is never predicted has no precision) it is set to 0.
    """
    true_positives = np.diag(matrix)
    precision = _ratio(true_positives, matrix.sum(axis=0))
    recall = _ratio(true_positives, matrix.sum(axis=1))
    f1 = _ratio(2 * precision * recall, precision + recall)
    return precision, recall, f1


def averages(matrix):
    """Micro, macro and weighted averages of precision, recall and F1, from a confusion matrix."""
    precision, recall, f1 = precision_recall_f1(matrix)
    support = matrix.sum(axis=1)
    micro = np.diag(matrix).sum() / matrix.sum()
    return {
        "micro": {"precision": micro, "recall": micro, "f1": micro},
        "macro": {"precision": precision.mean(), "recall": recall.mean(), "f1": f1.mean()},
        "weighted": {"precision": np.average(precision, weights=support), "recall": np.average(recall, weights=support),
                     "f1": np.average(f1, weights=support)},
    }


def summarize(P, y):
    """Accuracy, macro precision, macro recall, macro F1 and perplexity of the distributions P (one row for each recipe)."""
    matrix = confusion_matrix(y, P.argmax(axis=1), P.shape[1])
    precision, recall, f1 = precision_recall_f1(matrix)
    return {"accuracy": np.diag(matrix).sum() / matrix.sum(), "macro precision": precision.mean(),
            "macro recall": recall.mean(), "macro F1": f1.mean(),
            "perplexity": np.exp(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))}
