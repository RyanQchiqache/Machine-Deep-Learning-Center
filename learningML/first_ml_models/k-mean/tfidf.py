import numpy as np
import math
from collections import Counter


class TFIDFVectorizer:
    def __init__(self):
        self.ifd_dict = {}
        self.vocab = []

    def fit(self, documents):
        """Compute IDF values"""
        tokenized_documents = [doc.lower().split() for doc in documents]
        self.vocab = sorted(set(word for doc in tokenized_documents for word in doc))

        num_docs = len(tokenized_documents)

        self.ifd_dict = {
            word: math.log(num_docs / (1 + sum(1 for doc in tokenized_documents if word in doc)))
            for word in self.vocab
        }

    def transform(self, documents):
        """Convert documents into a TF-IDF matrix"""
        tokenized_documents = [doc.lower().split() for doc in documents]
        tf_idf_matrix = []

        for doc in tokenized_documents:
            tf = Counter(doc)
            total_words = len(doc)
            tfidf_vector = [tf.get(word, 0) / total_words * self.ifd_dict.get(word, 0) for word in self.vocab]
            tf_idf_matrix.append(tfidf_vector)

        return np.array(tf_idf_matrix)

    def fit_transform(self, documents):
        """Fit IDF and return TF-IDF matrix"""
        self.fit(documents)
        return self.transform(documents)
