import numpy as np


class NGramModel:
    """An n-gram language model built by counting, on sentences made of word ids (<s> w1 ... wn </s>).

    The n-grams are stored as sorted arrays of integer keys (a context of j words is a number in base V), so that the model
    works for n <= 5 with a vocabulary of 4,000 words (V^5 fits in 64 bits). For every context length j = 0, ..., n - 1 it
    keeps the counts of the contexts and of the pairs (context, next word): they are all we need for the maximum likelihood
    estimate, for the add-k estimate and for the interpolation of the orders."""

    def __init__(self, n: int, vocabulary_size: int, start: int = 1):
        assert 1 <= n <= 5
        self.n, self.V, self.start = n, vocabulary_size, start

    def positions(self, sentences):
        """For each word to be predicted (the words after <s>, with </s>): the n - 1 previous words (the sentence is padded
        on the left with <s>) and the word. Returns a matrix (positions, n - 1) and a vector."""
        pad = [self.start] * (self.n - 1)
        rows, targets = [], []
        for s in sentences:
            padded = pad + list(s)
            for t in range(self.n, len(padded)):
                rows.append(padded[t - self.n + 1:t])
                targets.append(padded[t])
        return np.array(rows, dtype=np.int64).reshape(len(targets), self.n - 1), np.array(targets, dtype=np.int64)

    def _context_key(self, X, j):
        """Key of the last j previous words of each row (j = 0 is the empty context)."""
        key = np.zeros(len(X), dtype=np.int64)
        for i in range(j):
            key += X[:, self.n - 2 - i] * self.V ** i
        return key

    def fit(self, sentences):
        X, y = self.positions(sentences)
        self.total = len(y)
        self.contexts, self.pairs = [], []
        for j in range(self.n):
            key = self._context_key(X, j)
            self.contexts.append(np.unique(key, return_counts=True))
            self.pairs.append(np.unique(key * self.V + y, return_counts=True))
        return self

    @staticmethod
    def _lookup(table, keys):
        unique, counts = table
        index = np.minimum(np.searchsorted(unique, keys), len(unique) - 1)
        return np.where(unique[index] == keys, counts[index], 0)

    def counts(self, X, y, j):
        """Counts of the pair (last j previous words, word) and of the context, for each position."""
        key = self._context_key(X, j)
        return self._lookup(self.pairs[j], key * self.V + y), self._lookup(self.contexts[j], key)

    def components(self, X, y):
        """Matrix (positions, n): column j is the maximum likelihood estimate that uses a context of j words (0 if the context
        has never been seen). The column 0 is the relative frequency of the word, with 1 added to the counts."""
        columns = []
        for j in range(self.n):
            pair, context = self.counts(X, y, j)
            if j == 0:
                columns.append((pair + 1) / (context + self.V))
            else:
                columns.append(np.where(context > 0, pair / np.maximum(context, 1), 0.0))
        return np.stack(columns, axis=1)

    def probabilities(self, X, y, method: str = "mle", k: float = 1.0, weights=None):
        """Probability of each word given its context. `method`: 'mle' (relative frequency of the pair, 0 if the context was
        never seen), 'add_k' ((pair + k) / (context + k V)) or 'interpolation' (weighted sum of the estimates of orders 1, ..., n)."""
        if method == "interpolation":
            return self.components(X, y) @ np.asarray(weights)
        pair, context = self.counts(X, y, self.n - 1)
        if method == "mle":
            return np.where(context > 0, pair / np.maximum(context, 1), 0.0)
        return (pair + k) / (context + k * self.V)

    def log_loss(self, sentences, **kwargs):
        """Average of -log P(word | context) over the positions of the sentences (the log of the perplexity)."""
        X, y = self.positions(sentences)
        with np.errstate(divide="ignore"):
            return float(-np.log(self.probabilities(X, y, **kwargs)).mean())

    def perplexity(self, sentences, **kwargs):
        return float(np.exp(self.log_loss(sentences, **kwargs)))

    def fit_weights(self, sentences, iterations: int = 100):
        """Weights of the interpolation that maximize the likelihood of the sentences (expectation-maximization)."""
        P = self.components(*self.positions(sentences))
        weights = np.full(self.n, 1 / self.n)
        for _ in range(iterations):
            responsibility = P * weights
            responsibility /= responsibility.sum(axis=1, keepdims=True)
            weights = responsibility.mean(axis=0)
        return weights

    def _continuations(self, j, context):
        """Dense vector of the counts of the words that follow a context of j words, and the count of the context."""
        key = 0
        for i in range(j):
            key += context[len(context) - 1 - i] * self.V ** i
        unique, counts = self.pairs[j]
        low, high = np.searchsorted(unique, [key * self.V, key * self.V + self.V])
        vector = np.zeros(self.V)
        vector[unique[low:high] - key * self.V] = counts[low:high]
        return vector

    def distribution(self, context, method: str = "mle", k: float = 1.0, weights=None):
        """Distribution over the next word given the n - 1 previous words (a vector of V probabilities; all zeros with 'mle'
        if the context has never been seen)."""
        context = tuple(context)
        if method == "interpolation":
            total = np.zeros(self.V)
            for j in range(self.n):
                vector = self._continuations(j, context)
                if j == 0:
                    total += weights[0] * (vector + 1) / (vector.sum() + self.V)
                elif vector.sum() > 0:
                    total += weights[j] * vector / vector.sum()
            return total
        vector = self._continuations(self.n - 1, context)
        if method == "mle":
            return vector / vector.sum() if vector.sum() > 0 else vector
        return (vector + k) / (vector.sum() + k * self.V)

    def generate(self, rng, end: int = 2, max_words: int = 30, **kwargs):
        """Starts from <s> and draws the next word from the distribution of the model until </s> (or until the limit)."""
        text = [self.start] * (self.n - 1)
        for _ in range(max_words):
            p = self.distribution(text[len(text) - (self.n - 1):] if self.n > 1 else (), **kwargs)
            if p.sum() == 0:
                break
            word = int(rng.choice(self.V, p=p / p.sum()))
            if word == end:
                break
            text.append(word)
        return text[self.n - 1:]
