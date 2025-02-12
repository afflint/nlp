from collections import defaultdict
import nltk 
import numpy as np 
from tqdm import tqdm

class MarkovModel:
    def __init__(self, k: int = 2, alpha: float = 0.0001):
        self.k = k
        self.alpha = alpha
        self.index = defaultdict(lambda: defaultdict(lambda: 0))
        
    def read(self, doc: list[str]):
        for k_gram in nltk.ngrams(doc, n=self.k, pad_left=True, pad_right=True, left_pad_symbol='[PAD]', right_pad_symbol='[PAD]'):
            prefix, suffix = k_gram[:-1], k_gram[-1]
            self.index[prefix][suffix] += 1
            
    def read_multi(self, docs: list):
        for doc in tqdm(docs):
            self.read(doc)
    def p(self, w: str, prefix: tuple):
        n = self.index[prefix][w]
        d = sum(self.index[prefix].values())
        if n == 0 or d == 0:
            return self.alpha
        else:
            return n / d
        
    def eval_prob(self, doc: list):
        probs = []
        for k_gram in nltk.ngrams(doc, n=self.k, pad_left=True, pad_right=True, left_pad_symbol='[PAD]', right_pad_symbol='[PAD]'):
            p = self.p(k_gram[-1], prefix=k_gram[:-1])
            probs.append(np.log(p))
        return sum(probs)
        
    def generate(self, prefix: tuple = None, max_len: int = 1000):
        if prefix is None:
            prefix = tuple(['[PAD]']*(self.k-2) + ['#S'])
        document = [x for x in prefix]
        for i in range(max_len):
            candidates, probabilities = [], []
            for w in self.index[prefix].keys():
                p = self.p(w, prefix)
                candidates.append(w)
                probabilities.append(p)
            new_word = np.random.choice(candidates, p=probabilities)
            document.append(new_word)
            prefix = tuple(document[-(self.k - 1):])
            if new_word == '#E':
                break 
        return document