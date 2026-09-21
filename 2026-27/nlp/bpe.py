import re
from collections import Counter, defaultdict

PRETOKENIZER = re.compile(r"\w+|[^\w\s]")


class BPETokenizer:
    """Byte-Pair Encoding tokenizer trained on characters, with a fixed vocabulary size."""

    def __init__(self, vocab_size: int, lowercase: bool = True, unk_token: str = "[UNK]"):
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.unk_token = unk_token
        self.merges = []
        self.vocab = {}
        self._ranks = {}
        self._cache = {}

    def pretokenize(self, text: str) -> list:
        if self.lowercase:
            text = text.lower()
        return PRETOKENIZER.findall(text)

    @staticmethod
    def _merge(symbols: list, a: str, b: str) -> list:
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def fit(self, texts) -> "BPETokenizer":
        word_freq = Counter(w for text in texts for w in self.pretokenize(text))
        alphabet = sorted({c for w in word_freq for c in w})
        self.vocab = {t: i for i, t in enumerate([self.unk_token] + alphabet)}
        if self.vocab_size < len(self.vocab):
            raise ValueError(f"vocab_size must be at least {len(self.vocab)} (the size of the alphabet)")

        segmentation = {w: list(w) for w in word_freq}
        pair_freq = Counter()
        words_with = defaultdict(set)
        for w, symbols in segmentation.items():
            for pair in zip(symbols, symbols[1:]):
                pair_freq[pair] += word_freq[w]
                words_with[pair].add(w)

        self.merges = []
        while len(self.vocab) < self.vocab_size and pair_freq:
            a, b = max(pair_freq, key=pair_freq.get)
            self.merges.append((a, b))
            self.vocab.setdefault(a + b, len(self.vocab))
            for w in words_with.pop((a, b)):
                old = segmentation[w]
                new = self._merge(old, a, b)
                if new == old:
                    continue
                for pair in zip(old, old[1:]):
                    pair_freq[pair] -= word_freq[w]
                    if pair_freq[pair] <= 0:
                        del pair_freq[pair]
                for pair in zip(new, new[1:]):
                    pair_freq[pair] += word_freq[w]
                    words_with[pair].add(w)
                segmentation[w] = new

        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._cache = {}
        return self

    def _tokenize_word(self, word: str) -> list:
        if word not in self._cache:
            symbols = [c if c in self.vocab else self.unk_token for c in word]
            while len(symbols) > 1:
                pairs = list(zip(symbols, symbols[1:]))
                best = min(pairs, key=lambda p: self._ranks.get(p, len(self._ranks)))
                if best not in self._ranks:
                    break
                symbols = self._merge(symbols, *best)
            self._cache[word] = symbols
        return self._cache[word]

    def tokenize(self, text: str) -> list:
        return [t for w in self.pretokenize(text) for t in self._tokenize_word(w)]

    def tokenize_words(self, text: str) -> list:
        return [self._tokenize_word(w) for w in self.pretokenize(text)]

    def encode(self, text: str) -> list:
        return [self.vocab[t] for t in self.tokenize(text)]
