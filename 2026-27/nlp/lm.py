import json
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

RECIPES = "/Users/Flint/Data/recipes/foodcom/subsets/recipes_cuisine_tipo_10k.json"
PLOTS = "/Users/Flint/Data/movies/subsets/plots_english_10k.json"

UNK, START, END = "<unk>", "<s>", "</s>"


def sentences_of(text: str):
    """Splits a text in sentences, and each sentence in lowercase words (letters, with the apostrophes inside a word)."""
    parts = re.split(r"[.!?;:\n]+", text.lower())
    sentences = [re.findall(r"[a-z]+(?:'[a-z]+)?", part) for part in parts]
    return [s for s in sentences if s]


def load_recipes():
    """The instructions of the recipes of the 10k subset: a list of documents, each one a list of sentences."""
    with open(RECIPES) as f:
        return [sentences_of(r["instructions"]) for r in json.load(f)]


def load_plots(tokens: int = 1_040_000):
    """The plots of the films, in the order of the file (a random sample), until the corpus has about `tokens` words."""
    documents, total = [], 0
    with open(PLOTS) as f:
        for film in json.load(f):
            sentences = sentences_of(film["plot"])
            documents.append(sentences)
            total += sum(map(len, sentences))
            if total >= tokens:
                break
    return documents


def split_documents(documents, fractions=(0.8, 0.1, 0.1), seed: int = 0):
    """Random split by document (never by sentence: sentences of the same text must stay together)."""
    order = np.random.default_rng(seed).permutation(len(documents))
    cuts = np.cumsum([int(f * len(documents)) for f in fractions[:-1]])
    return [[documents[i] for i in part] for part in np.split(order, cuts)]


class Vocabulary:
    """The `size` most frequent words of a text, plus <unk> (all the others), <s> and </s>."""

    def __init__(self, documents, size: int = 4000):
        counts = Counter(w for document in documents for sentence in document for w in sentence)
        self.words = [UNK, START, END] + [w for w, _ in counts.most_common(size - 3)]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.unk, self.start, self.end = 0, 1, 2

    def __len__(self):
        return len(self.words)

    def encode(self, sentence):
        """A sentence as a list of ids between <s> and </s>."""
        return [self.start] + [self.index.get(w, self.unk) for w in sentence] + [self.end]

    def decode(self, ids):
        return [self.words[i] for i in ids]

    def coverage(self, documents):
        """Share of the words of the documents that are in the vocabulary."""
        words = [w for document in documents for sentence in document for w in sentence]
        return np.mean([w in self.index for w in words])


def next_word_pairs(documents, vocabulary: Vocabulary):
    """All the pairs (previous word, next word) of the sentences, with the boundaries, as two arrays of ids."""
    encoded = [vocabulary.encode(sentence) for document in documents for sentence in document]
    previous = np.array([a for ids in encoded for a in ids[:-1]], dtype=np.int64)
    following = np.array([b for ids in encoded for b in ids[1:]], dtype=np.int64)
    return previous, following


class Table(nn.Module):
    """Softmax regression on the one-hot of the previous word: a table of V x V scores, a row for each previous word."""

    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.scores = nn.Embedding(vocabulary_size, vocabulary_size)
        nn.init.zeros_(self.scores.weight)

    def forward(self, previous):
        return self.scores(previous)

    def distribution(self, previous):
        with torch.no_grad():
            return torch.softmax(self(torch.as_tensor(previous)), dim=-1).numpy()


def loss_of(model, previous, following, batch: int = 100_000):
    """Average cross-entropy (the log of the perplexity) of a model on a set of pairs."""
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(previous), batch):
            x, y = torch.as_tensor(previous[i:i + batch]), torch.as_tensor(following[i:i + batch])
            total += F.cross_entropy(model(x), y, reduction="sum").item()
    return total / len(previous)


def train_next_word(model, train, validation, epochs: int = 30, batch: int = 4096, learning_rate: float = 0.05,
                    weight_decay: float = 0.0, seed: int = 0, extra=None, patience: int = None):
    """Adam on minibatches. After each epoch it records the loss on the training and on the validation pairs (and the
    dictionary returned by `extra(model)`, if given) and keeps a copy of the weights with the lowest validation loss
    (early stopping). With `patience` the training stops when the validation loss has not improved for that many epochs.
    Returns the history as a list of dicts."""
    torch.manual_seed(seed)
    x, y = torch.as_tensor(train[0]), torch.as_tensor(train[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history, best = [], (float("inf"), None, 0)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(x))
        for i in range(0, len(x), batch):
            index = order[i:i + batch]
            loss = F.cross_entropy(model(x[index]), y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        record = {"epoch": epoch, "train": loss_of(model, *train), "validation": loss_of(model, *validation)}
        if extra is not None:
            record.update(extra(model))
        history.append(record)
        if record["validation"] < best[0]:
            best = (record["validation"], {k: v.clone() for k, v in model.state_dict().items()}, epoch)
        if patience is not None and epoch - best[2] >= patience:
            break
    model.best_epoch = best[2]
    model.best_state = best[1]
    return history


def top_k_accuracy(model, previous, following, ks=(1, 5, 10), batch: int = 20_000, exclude: int = None):
    """Share of the examples whose next word is among the k words with the highest score (the word `exclude` is never proposed)."""
    hits = {k: 0 for k in ks}
    with torch.no_grad():
        for i in range(0, len(previous), batch):
            scores = model(torch.as_tensor(previous[i:i + batch]))
            if exclude is not None:
                scores[:, exclude] = float("-inf")
            ranked = torch.topk(scores, max(ks), dim=-1).indices.numpy()
            right = ranked == following[i:i + batch, None]
            for k in ks:
                hits[k] += right[:, :k].any(axis=1).sum()
    return {k: hits[k] / len(previous) for k in ks}


def frequency_top_k(train_following, following, vocabulary_size: int, ks=(1, 5, 10), exclude: int = None):
    """The same measure for the model that ignores the previous word: always the k most frequent words."""
    counts = np.bincount(train_following, minlength=vocabulary_size)
    if exclude is not None:
        counts[exclude] = -1
    order = np.argsort(-counts)
    return {k: np.isin(following, order[:k]).mean() for k in ks}


def generate_text(model, vocabulary: Vocabulary, rng=None, temperature: float = 1.0, start=None, max_words: int = 30):
    """Starts from <s> (or from the words in `start`) and chooses the next word until </s>. Greedy if there is no random
    generator, otherwise it draws the word from softmax(scores / temperature)."""
    ids = [vocabulary.start] + [vocabulary.index[w] for w in (start or [])]
    with torch.no_grad():
        while len(ids) <= max_words:
            scores = model(torch.tensor([ids[-1]]))[0]
            if rng is None:
                word = int(scores.argmax())
            else:
                p = torch.softmax(scores / temperature, dim=-1).numpy().astype("float64")
                word = int(rng.choice(len(p), p=p / p.sum()))
            if word == vocabulary.end:
                break
            ids.append(word)
    return vocabulary.decode(ids[1:])


def frequency_perplexity(train_following, following, vocabulary_size: int):
    """Perplexity of the model that ignores the previous word and gives to each word its relative frequency (+1 on the counts)."""
    frequency = np.bincount(train_following, minlength=vocabulary_size) + 1
    frequency = frequency / frequency.sum()
    return float(np.exp(-np.log(frequency[following]).mean()))
