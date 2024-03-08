"""Collections of methods for gathering data"""
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
import spacy 



SKLEARN_FOLDER = "/Users/flint/Data/sklearn"


def load_20news(subset: str = 'all', categories: list[str] = None, remove: tuple = ('headers', 'footers', 'quotes')):
    """Ovverrides 20 News fetching

    Args:
        subset (str, optional): Select the dataset to load: ‘train’ for the training set, ‘test’ for the test set, 
        ‘all’ for both, with shuffled ordering. Defaults to 'all'.
        categories (list[str], optional): If None (default), load all the categories. 
        If not None, list of category names to load (other categories ignored). Defaults to None.
        remove (tuple, optional): May contain any subset of (‘headers’, ‘footers’, ‘quotes’). 
        Defaults to ('headers', 'footers', 'quotes').

    Returns:
        object: Data bunch as for sklearn
    """
    return fetch_20newsgroups(data_home=SKLEARN_FOLDER, subset=subset, categories=categories, remove=remove)


def frequency(corpus: list[str], tokenizer: callable):
    data = defaultdict(lambda: 0)
    for document in corpus:
        for token in tokenizer(document):
            data[token] += 1
    return pd.Series(data)

def spacy_lemma_tokenizer(text: str, nlp: spacy.language.Language, filter_tokens: list = None):
    if filter_tokens is None:
        return [x.lemma_.lower() for x in nlp(text)]
    else:
        return [x.lemma_.lower() for x in nlp(text) if x.pos_ not in filter_tokens]

