import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def sentences_of(text: str):
    """Splits a text in sentences, and each sentence in lowercase alphabetic tokens."""
    parts = re.split(r"[.!?;:\n]+", text.lower())
    sentences = [re.findall(r"[a-z]+", part) for part in parts]
    return [s for s in sentences if s]


class Vocabulary:
    """The words that occur at least `min_count` times, and the probability of keeping an occurrence of a word in the
    subsampling of the frequent words of the paper: sqrt(t / f), that is, 1 minus the probability of discarding it."""

    def __init__(self, sentences, min_count: int = 5, subsample: float = 1e-3):
        counts = Counter(word for sentence in sentences for word in sentence)
        self.words = [word for word, count in counts.most_common() if count >= min_count]
        self.index = {word: i for i, word in enumerate(self.words)}
        self.counts = np.array([counts[word] for word in self.words])
        self.frequency = self.counts / self.counts.sum()
        self.keep = np.minimum(1.0, np.sqrt(subsample / self.frequency)) if subsample else np.ones(len(self.words))

    def __len__(self):
        return len(self.words)

    def encode(self, sentences):
        """The text as a sequence of word ids (the words outside the vocabulary are removed) and the sentence of each position."""
        tokens, sentence_ids = [], []
        for k, sentence in enumerate(sentences):
            ids = [self.index[word] for word in sentence if word in self.index]
            tokens += ids
            sentence_ids += [k] * len(ids)
        return np.array(tokens, dtype=np.int64), np.array(sentence_ids, dtype=np.int64)


def subsample(tokens, sentence_ids, vocabulary: Vocabulary, rng):
    """Discards the occurrences of the frequent words at random. The window then reaches words that were farther."""
    kept = rng.random(len(tokens)) < vocabulary.keep[tokens]
    return tokens[kept], sentence_ids[kept]


def _radius(n, window, shrink, rng):
    """Window of each position. As in the original implementation it is drawn at random between 1 and `window`."""
    return rng.integers(1, window + 1, size=n) if shrink else np.full(n, window)


def skipgram_pairs(tokens, sentence_ids, window: int, rng, shrink: bool = True):
    """All the pairs (center, context) inside a sentence, as two arrays."""
    radius = _radius(len(tokens), window, shrink, rng)
    centers, contexts = [], []
    for j in range(1, window + 1):
        same = sentence_ids[:-j] == sentence_ids[j:]
        i = np.where(same & (radius[:-j] >= j))[0]          # center i, context i + j
        centers.append(tokens[i]); contexts.append(tokens[i + j])
        i = np.where(same & (radius[j:] >= j))[0]           # center i + j, context i
        centers.append(tokens[i + j]); contexts.append(tokens[i])
    return np.concatenate(centers), np.concatenate(contexts)


def cbow_examples(tokens, sentence_ids, window: int, padding: int, rng, shrink: bool = True):
    """For each position, the matrix of the words of its window (padded) and the center word."""
    n = len(tokens)
    radius = _radius(n, window, shrink, rng)
    context = np.full((n, 2 * window), padding, dtype=np.int64)
    positions = np.arange(n)
    for k, j in enumerate([*range(-window, 0), *range(1, window + 1)]):
        other = positions + j
        valid = (other >= 0) & (other < n) & (np.abs(j) <= radius)
        valid[valid] = sentence_ids[other[valid]] == sentence_ids[valid]
        context[valid, k] = tokens[other[valid]]
    has_context = (context != padding).any(axis=1)
    return context[has_context], tokens[has_context]


class SkipGram(nn.Module):
    """From the center word to a distribution over the words of its window. `input` holds the vectors of the words that we keep,
    `output` is the layer with a neuron for each word (its weights are the context vectors u_w of the previous section)."""

    def __init__(self, vocabulary_size: int, d: int):
        super().__init__()
        self.input = nn.Embedding(vocabulary_size, d)
        self.output = nn.Linear(d, vocabulary_size, bias=False)

    def hidden(self, source):
        return self.input(source)

    def logits(self, source):
        return self.output(self.hidden(source))

    def loss(self, source, target):
        return F.cross_entropy(self.logits(source), target)

    def distribution(self, source):
        with torch.no_grad():
            return torch.softmax(self.logits(torch.as_tensor(source)), dim=-1).numpy()

    def vectors(self):
        return self.input.weight.detach().numpy()[:self.output.out_features]


class CBOW(SkipGram):
    """From the average of the vectors of the words of the window to a distribution over the center word."""

    def __init__(self, vocabulary_size: int, d: int):
        super().__init__(vocabulary_size, d)
        self.padding = vocabulary_size
        self.input = nn.Embedding(vocabulary_size + 1, d, padding_idx=self.padding)

    def hidden(self, context):
        mask = (context != self.padding).float().unsqueeze(-1)
        return (self.input(context) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def examples(model, tokens, sentence_ids, window, rng, shrink=True):
    """The training examples of the model, as (source, target): (center, context) for the skip-gram, (window, center) for the CBOW."""
    if isinstance(model, CBOW):
        return cbow_examples(tokens, sentence_ids, window, model.padding, rng, shrink)
    return skipgram_pairs(tokens, sentence_ids, window, rng, shrink)


def train(model, vocabulary: Vocabulary, tokens, sentence_ids, window: int = 5, epochs: int = 5, batch_size: int = 2048,
          learning_rate: float = 0.01, seed: int = 0, verbose: bool = True):
    """Trains the model with the softmax on the whole vocabulary. At every epoch the frequent words are subsampled again and the
    examples are rebuilt. The learning rate decreases linearly to zero. Returns the average loss of each epoch."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, step, total_steps = [], 0, None
    for epoch in range(epochs):
        t, s = subsample(tokens, sentence_ids, vocabulary, rng)
        source, target = examples(model, t, s, window, rng)
        order = rng.permutation(len(target))
        if total_steps is None:
            total_steps = epochs * int(np.ceil(len(order) / batch_size))
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * max(1e-3, 1 - step / total_steps)
            loss = model.loss(torch.as_tensor(source[batch]), torch.as_tensor(target[batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
        history.append(float(np.mean(losses)))
        if verbose:
            print(f"epoch {epoch + 1}: {len(order)} examples, loss {history[-1]:.3f}")
    return history


def evaluate(model, source, target, batch_size: int = 4096, ks=(1, 5, 10)):
    """Average cross-entropy (the perplexity is its exponential) and top-k accuracy of the model on a set of examples."""
    total, hits = 0.0, {k: 0 for k in ks}
    with torch.no_grad():
        for start in range(0, len(target), batch_size):
            x, y = torch.as_tensor(source[start:start + batch_size]), torch.as_tensor(target[start:start + batch_size])
            logits = model.logits(x)
            total += F.cross_entropy(logits, y, reduction="sum").item()
            ranks = (logits > logits.gather(1, y.unsqueeze(1))).sum(dim=1)
            for k in ks:
                hits[k] += (ranks < k).sum().item()
    n = len(target)
    return {"cross-entropy": total / n, "perplexity": float(np.exp(total / n)), **{f"top-{k}": hits[k] / n for k in ks}}


class WordVectors:
    """The vectors of the words, with the operations that we use to look at them. The similarity is the cosine."""

    def __init__(self, words, vectors):
        self.words = list(words)
        self.index = {word: i for i, word in enumerate(self.words)}
        self.vectors = np.asarray(vectors, dtype=np.float64)
        self.unit = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def __getitem__(self, word):
        return self.vectors[self.index[word]]

    def __contains__(self, word):
        return word in self.index

    def nearest(self, vector, n: int = 10, exclude=()):
        """The words whose vectors have the highest cosine with `vector`."""
        vector = np.asarray(vector, dtype=np.float64)
        similarity = self.unit @ (vector / np.linalg.norm(vector))
        skip = {self.index[w] for w in exclude}
        order = [i for i in np.argsort(-similarity) if i not in skip][:n]
        return [(self.words[i], float(similarity[i])) for i in order]

    def similar(self, word, n: int = 10):
        return self.nearest(self[word], n, exclude=[word])

    def odd_one_out(self, words):
        """The word that is less similar to the average of the others."""
        unit = np.array([self.unit[self.index[w]] for w in words])
        scores = []
        for i in range(len(words)):
            others = np.delete(unit, i, axis=0).mean(axis=0)
            scores.append(float(unit[i] @ others / np.linalg.norm(others)))
        return words[int(np.argmin(scores))], dict(zip(words, scores))


def unigram_baseline(train_target, test_target, vocabulary_size: int, ks=(1, 5, 10)):
    """A model that ignores the input and always answers with the frequency of the words: its cross-entropy, perplexity and top-k accuracy."""
    frequency = np.bincount(train_target, minlength=vocabulary_size) + 1
    frequency = frequency / frequency.sum()
    cross_entropy = float(-np.log(frequency[test_target]).mean())
    ranking = np.argsort(-frequency)
    return {"cross-entropy": cross_entropy, "perplexity": float(np.exp(cross_entropy)),
            **{f"top-{k}": float(np.isin(test_target, ranking[:k]).mean()) for k in ks}}


def analogy_accuracy(vectors: "WordVectors", pairs, n: int = 600, seed: int = 1, k: int = 5):
    """Accuracy (top 1 and top k) on n analogies a : b :: c : d built from random couples of pairs (a, b), (c, d) that share a relation.
    The answer is the word closest to b - a + c, excluding a, b and c."""
    rng = np.random.default_rng(seed)
    top1 = topk = 0
    for _ in range(n):
        (a, b), (c, d) = [pairs[i] for i in rng.choice(len(pairs), 2, replace=False)]
        answers = [w for w, _ in vectors.nearest(vectors[b] - vectors[a] + vectors[c], k, exclude=[a, b, c])]
        top1 += answers[0] == d
        topk += d in answers
    return top1 / n, topk / n
