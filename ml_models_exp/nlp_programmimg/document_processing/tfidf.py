from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, List
import pickle
import numpy as np


class TFIDFProcessor:
    """Computes TF-IDF for a collection of documents"""

    def __init__(self, documents: List[str], max_features: int = 50000):
        self.documents: List[str] = documents
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(max_features=max_features)
        self.tfidf_matrix: Optional[np.ndarray] = None

    def compute_tfidf_matrix(self) -> np.ndarray:
        """Computes the TF-IDF matrix for a collection of documents"""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents).toarray()
        return self.tfidf_matrix

    def get_feature_names(self) -> List[str]:
        """Returns the list of words used in TF-IDF."""
        return self.vectorizer.get_feature_names_out().tolist()

    def transform_new_document(self, new_text: str) -> np.ndarray:
        """Transforms a new text document into TF-IDF representation."""
        return self.vectorizer.transform([new_text]).toarray()

    def save_tfidf_matrix(self, filename: str = "tfidf_matrix.pkl") -> None:
        """Saves TF-IDF matrix and vectorizer to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump((self.vectorizer, self.tfidf_matrix), f)
        print(f"Saved TF-IDF matrix to {filename}")

    def load_tfidf_matrix(self, filename: str = "tfidf_matrix.pkl") -> None:
        """Loads TF-IDF matrix and vectorizer from a pickle file."""
        with open(filename, 'rb') as f:
            self.vectorizer, self.tfidf_matrix = pickle.load(f)  # Fixed order
        print(f"Loaded TF-IDF matrix from {filename}")
