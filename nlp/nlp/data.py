"""Collections of methods for gathering data"""
from collections import defaultdict
import sys
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import spacy
from SPARQLWrapper import SPARQLWrapper, JSON


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


def frequency(corpus: list[str], tokenizer: callable) -> pd.Series:
    """Compute corpus frequency

    Args:
        corpus (list[str]): testual data
        tokenizer (callable): tokenizer function

    Returns:
        list: pandas series of frequency
    """
    data = defaultdict(lambda: 0)
    for document in corpus:
        for token in tokenizer(document):
            data[token] += 1
    return pd.Series(data)

def spacy_lemma_tokenizer(text: str, nlp: spacy.language.Language, filter_tokens: list = None) -> list:
    """Spacy tokenizer

    Args:
        text (str): Text
        nlp (spacy.language.Language): Spacy language
        filter_tokens (list, optional): POS filters. Defaults to None.

    Returns:
        list: Tokens
    """
    if filter_tokens is None:
        return [x.lemma_.lower() for x in nlp(text)]
    else:
        return [x.lemma_.lower() for x in nlp(text) if x.pos_ not in filter_tokens]


class EnglishNames:
    """Handle popular English names extracted from Wikidata using the following query
        SELECT ?cid ?firstname (COUNT(*) AS ?count)
        WHERE
        {
        ?pid wdt:P19 wd:Q60.
        ?pid wdt:P735 ?cid.
        OPTIONAL {
            ?cid rdfs:label ?firstname
            FILTER((LANG(?firstname)) = "en")
        }
        }
        GROUP BY ?cid ?firstname
        ORDER BY DESC(?count) ?firstname
        LIMIT 200
    """
    def __init__(self, city_wiki: str = 'Q60', limit: int = 200):
        self.endpoint_url = "https://query.wikidata.org/sparql"
        self.query = "SELECT ?cid ?firstname (COUNT(*) AS ?count) WHERE { "
        self.query += f"?pid wdt:P19 wd:{city_wiki}. "
        self.query += "?pid wdt:P735 ?cid. "
        self.query += "OPTIONAL { ?cid rdfs:label ?firstname FILTER((LANG(?firstname)) = 'en')}} "
        self.query += "GROUP BY ?cid ?firstname ORDER BY DESC(?count) ?firstname "
        self.query += f"LIMIT {limit}"
        self.frequency = pd.Series({})
    
    def _get_results(self):
        user_agent = 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'
        sparql = SPARQLWrapper(self.endpoint_url, agent=user_agent)
        sparql.setQuery(self.query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()
    
    def run(self):
        """Execute the query

        Returns:
            JSON: results
        """
        return self._get_results()
    
    def stats(self):
        """Executes the query and collect statistics as normalized frequencies
        """
        names = self.run()
        for result in names['results']['bindings']:
            firstname = result['firstname']['value']
            frequency = int(result['count']['value'])
            try:
                self.frequency[firstname] += frequency
            except KeyError:
                self.frequency[firstname] = frequency
        self.frequency = self.frequency / self.frequency.sum()
    
    def generate_names(self, n: int = 10) -> list:
        """Random pick up names according to their frequency

        Args:
            n (int, optional): number of names. Defaults to 10.

        Returns:
            list: names
        """
        names = list(self.frequency.keys())
        p = self.frequency.values
        generated = np.random.choice(names, size=n, p=p)
        return list(generated)

class SketchList:
    """Load a Sketch list from CSV
    """
    def __init__(self, file_path: str):
        words = pd.read_csv(file_path, header=2)
        self.w = list(words['Item'])
        self.f = (words['Frequency'] / words['Frequency'].sum()).values
    
    def generate_words(self, n: int = 10) -> list:
        """Random pick up words according to their frequency

        Args:
            n (int, optional): number of words. Defaults to 10.

        Returns:
            list: words
        """
        generated = np.random.choice(self.w, size=n, p=self.f)
        return list(generated)