import nltk
import pandas as pd
from string import punctuation
from collections import defaultdict
from tqdm import tqdm


class WordPieceTokenizer:
    """Simple WordPiece implementation
    """
    def __init__(self, text: str, maximum_size: int):
        assert maximum_size > len(text.split())
        self.max_vocabulary_size = maximum_size
        self.special_token = "#"
        self.unk_token = "[UNK]"
        self.text = text
        self.corpus = []
        self.vocabulary = self._init_vocabulary()
    
    def _init_vocabulary(self) -> dict:
        """Initialize vocabulary by just splitting words
        and substitute corpus with the new version

        Returns:
            dict: vocabulary
        """
        vocabulary = defaultdict(lambda: 0)
        for token in self.text.split():
            new_token = []
            for i, c in enumerate(token.strip()):
                if i == 0:
                    new_token.append(c)
                    vocabulary[c] += 1
                else:
                    k = self.special_token + c
                    new_token.append(k)
                    vocabulary[k] += 1
            self.corpus.append(tuple(new_token)) 
        return vocabulary
    
    @property
    def fancy_vocabulary(self):
        return pd.Series(self.vocabulary)
    
    def score_pairs(self, normalize: bool = True) -> pd.Series:
        """Compute the score given the actual corpus

        Returns:
            pd.Series: Sorted scores per pair
        """
        score = defaultdict(lambda: 0)
        for token in self.corpus:
            for a, b in nltk.ngrams(token, n=2):
                score[(a, b)] += 1
        if normalize:
            for (a, b), freq in score.items():
                score[(a, b)] = freq / (self.vocabulary[a] * self.vocabulary[b])
        return pd.Series(score).sort_values(ascending=False)
    
    def update(self, normalize: bool = True) -> bool:
        """Updates the vocabulary by inserting a new token is the room is sufficient
        Updates the corpus by substituting the pair in the current corpus
        
        return False if vocabulary is full
        """
        scores = self.score_pairs(normalize=normalize)
        if len(scores) == 0:
            return False
        (a, b), score = list(scores.items())[0]
        if len(self.vocabulary) < self.max_vocabulary_size - 1:
            self.vocabulary[a + b.replace("#", "")] += score
            # update corpus
            new_corpus = []
            for token in self.corpus:
                new_token = []
                i = 0
                while i < len(token):
                    k = token[i]
                    if i + 1 < len(token) and k + token[i+1].replace("#", "") in self.vocabulary.keys():
                        new_token.append(k + token[i+1].replace("#", ""))
                        i = i + 2
                    else:
                        new_token.append(k)
                        i = i + 1
                new_corpus.append(tuple(new_token))
            self.corpus = new_corpus
        else:
            return False 
        return True
        
    def train(self, normalize: bool = True):
        iterations = list(range(self.max_vocabulary_size))
        for i in tqdm(iterations):
            check = self.update(normalize=normalize)
            if not check:
                break 
    
    def tokenize(self, word: str):
        """Tokenize a word using the WordPiece algorithm."""
        if word in self.vocabulary:
            return [word]
        
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            subword = None
            
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = self.special_token + sub
                if sub in self.vocabulary:
                    subword = sub
                    break
                end -= 1
            
            if subword is None:
                return [self.unk_token]
            
            tokens.append(subword)
            start = end
        
        return tokens
    
    def tokenize_text(self, text: str):
        tokens = []
        for word in text.split():
            tokens.extend(self.tokenize(word))
        return tokens