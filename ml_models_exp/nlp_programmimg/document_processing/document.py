import os
from typing import Tuple, List

import numpy as np

from preprocessor import TextPreprocessor
from sklearn.metrics.pairwise import cosine_similarity
from tfidf import TFIDFProcessor

class Document:
    def __init__(self, filepath):
        self.filepath = filepath
        self.content = self.load_document()

    def load_document(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading {self.filepath} : {e}")
            return None


class DocumentCollection:
    """Loads multiple documents, preprocesses them, and stores the cleaned text."""

    def __init__(self, folder_path: str):
        self.folder_path: str = folder_path
        self.documents: List[str] = self.load_documents()

    def load_documents(self) -> List[str]:
        """Loads and preprocesses all .txt documents in a given folder."""
        docs: List[str] = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                doc = Document(os.path.join(self.folder_path, filename))
                if doc.content:
                    preprocessor = TextPreprocessor()
                    cleaned_text: str = " ".join(preprocessor.preprocess(doc.content))
                    docs.append(cleaned_text)
        return docs


class DocumentSearcher:
    """Finds similar documents using cosine similarity on TF-IDF vectors."""

    def __init__(self, tfidf_matrix: np.ndarray, tfidf_processor: TFIDFProcessor):
        self.tfidf_matrix: np.ndarray = tfidf_matrix
        self.tfidf_processor: TFIDFProcessor = tfidf_processor

    def search(self, query: str, top_n: int = 6) -> Tuple[List[int], List[float]]:
        """Searches for the most similar documents to the query."""
        query_vector = self.tfidf_processor.transform_new_document(query)
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top N document indices
        top_indices = similarities.argsort()[-top_n:][::-1]
        return top_indices.tolist(), similarities[top_indices].tolist()