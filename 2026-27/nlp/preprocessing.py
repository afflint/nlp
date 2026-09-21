class TextNormalizer:
    """Turns a spaCy Doc into a list of terms, applying the selected normalization steps."""

    def __init__(self, lowercase: bool = True, lemmatize: bool = False,
                 remove_stopwords: bool = False, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation

    def __call__(self, doc) -> list:
        terms = []
        for token in doc:
            if token.is_space or (self.remove_punctuation and token.is_punct):
                continue
            if self.remove_stopwords and token.is_stop:
                continue
            term = token.lemma_ if self.lemmatize else token.text
            terms.append(term.lower() if self.lowercase else term)
        return terms
