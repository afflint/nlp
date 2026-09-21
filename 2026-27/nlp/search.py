import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import CountVectorizer

WEIGHTINGS = ("count", "tf_length", "tf_max", "tfidf", "bm25")
SIMILARITIES = ("dot", "cosine", "euclidean")


class SearchEngine:
    """Vector space search engine: documents and queries are vectors of term weights.

    The query is a vector of counts. Documents are ranked by dot product, cosine similarity
    or (negative) euclidean distance from the query.
    """

    def __init__(self, preprocess, weighting: str = "count", similarity: str = "dot",
                 k1: float = 1.2, b: float = 0.75):
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        if similarity not in SIMILARITIES:
            raise ValueError(f"similarity must be one of {SIMILARITIES}")
        self.preprocess = preprocess
        self.weighting = weighting
        self.similarity = similarity
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(analyzer=lambda terms: terms)
        self.counts = None
        self.matrix = None

    def fit(self, term_lists) -> "SearchEngine":
        self.counts = self.vectorizer.fit_transform(term_lists).tocsr()
        n_docs = self.counts.shape[0]
        self.doc_freq = np.asarray((self.counts > 0).sum(axis=0)).ravel()
        self.idf = np.log(n_docs / self.doc_freq)
        self.matrix = self._weight(self.counts)
        self._norms = np.sqrt(np.asarray(self.matrix.multiply(self.matrix).sum(axis=1)).ravel())
        return self

    def _bm25(self, counts):
        n_docs = counts.shape[0]
        idf = np.log(1 + (n_docs - self.doc_freq + 0.5) / (self.doc_freq + 0.5))
        coo = counts.tocoo()
        length_ratio = self.lengths[coo.row] / self.lengths.mean()
        saturation = coo.data * (self.k1 + 1) / (coo.data + self.k1 * (1 - self.b + self.b * length_ratio))
        return csr_matrix((saturation * idf[coo.col], (coo.row, coo.col)), shape=counts.shape)

    def _weight(self, counts):
        if self.weighting == "count":
            return counts
        if self.weighting == "bm25":
            return self._bm25(counts)
        if self.weighting == "tf_length":
            tf = diags(1 / np.maximum(self.lengths, 1)) @ counts
        else:
            tf = diags(1 / np.maximum(counts.max(axis=1).toarray().ravel(), 1)) @ counts
        if self.weighting == "tfidf":
            tf = tf @ diags(self.idf)
        return tf.tocsr()

    @property
    def vocabulary(self) -> dict:
        return self.vectorizer.vocabulary_

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray(self.counts.sum(axis=1)).ravel()

    def vectorize(self, text: str):
        return self.vectorizer.transform([self.preprocess(text)])

    def scores(self, text: str) -> np.ndarray:
        query = self.vectorize(text)
        dot = (self.matrix @ query.T).toarray().ravel()
        if self.similarity == "dot":
            return dot
        query_norm = np.sqrt(query.multiply(query).sum())
        if self.similarity == "cosine":
            return dot / np.maximum(self._norms * query_norm, 1e-12)
        return -np.sqrt(np.maximum(self._norms ** 2 + query_norm ** 2 - 2 * dot, 0))

    def search(self, text: str, k: int = 10) -> list:
        scores = self.scores(text)
        top = np.argsort(-scores, kind="stable")[:k]
        if self.similarity == "euclidean":
            return [(int(i), float(scores[i])) for i in top]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
