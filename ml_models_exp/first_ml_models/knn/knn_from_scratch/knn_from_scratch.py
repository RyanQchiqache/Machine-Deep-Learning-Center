from nltk import FreqDist, word_tokenize
from collections import defaultdict
import os
import math
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Helper functions
def calculate_dot_product(dict_a, dict_b):
    """Calculates the dot product of two vectors represented as dictionaries."""
    return sum([dict_a.get(tok, 0) * dict_b.get(tok, 0) for tok in dict_a])

def normalize_tokens(text):
    """Tokenizes and normalizes text to lowercase tokens."""
    return [token.lower() for token in word_tokenize(text)]

# TextDocument class
class TextDocument:
    """
    Represents a single text document with tokenized content and metadata.
    """
    def __init__(self, text, doc_id=None, category=None):
        self.text = text
        self.token_counts = FreqDist(normalize_tokens(text))
        self.doc_id = doc_id
        self.category = category

    @classmethod
    def from_file(cls, filename, category):
        """Loads a document from a file and assigns a category."""
        with open(filename, 'r', encoding="ISO-8859-1") as myfile:
            text = myfile.read().strip()
        return cls(text, filename, category)

# DocumentCollection class
class DocumentCollection:
    """
    Represents a collection of documents, facilitating operations like TF-IDF computation and similarity comparisons.
    """
    def __init__(self, term_to_df, term_to_docids, docid_to_doc, doc_to_category):
        self.term_to_df = term_to_df  # Term-to-document frequency mapping
        self.term_to_docids = term_to_docids  # Term-to-document ID mapping
        self.docid_to_doc = docid_to_doc  # Document ID to document mapping
        self.doc_to_category = doc_to_category  # Document to category mapping

    @classmethod
    def from_document_list(cls, docs):
        """Creates a DocumentCollection from a list of TextDocument objects."""
        term_to_df = defaultdict(int)
        term_to_docids = defaultdict(set)
        docid_to_doc = {}
        doc_to_category = {}
        for doc in docs:
            docid_to_doc[doc.doc_id] = doc
            doc_to_category[doc] = doc.category
            for token in doc.token_counts.keys():
                term_to_df[token] += 1
                term_to_docids[token].add(doc.doc_id)
        return cls(term_to_df, term_to_docids, docid_to_doc, doc_to_category)

    def compute_tfidf(self, counts):
        """Computes the TF-IDF values for a given document's term frequencies."""
        total_docs = len(self.docid_to_doc)
        return {
            token: tf * math.log(total_docs / self.term_to_df[token])
            for token, tf in counts.items() if token in self.term_to_df
        }

    def compute_cosine_similarity(self, weighted_a, weighted_b):
        """Calculates the cosine similarity between two weighted vectors."""
        dot_ab = calculate_dot_product(weighted_a, weighted_b)
        norm_a = math.sqrt(calculate_dot_product(weighted_a, weighted_a))
        norm_b = math.sqrt(calculate_dot_product(weighted_b, weighted_b))
        return 0 if norm_a == 0 or norm_b == 0 else dot_ab / (norm_a * norm_b)

# KNNClassifier class
class KNNClassifier:
    """
    Implements a K-Nearest Neighbors classifier for text classification.
    """
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.doc_collection = None
        self.train_vectors = None

    def train(self, doc_collection):
        """Fits the classifier with a collection of training documents."""
        self.doc_collection = doc_collection
        self.train_vectors = [
            (doc, self.doc_collection.compute_tfidf(doc.token_counts))
            for doc in self.doc_collection.docid_to_doc.values()
        ]

    def compute_similarities(self, test_vector):
        """Calculates cosine similarities between a test vector and training vectors."""
        return [
            (self.doc_collection.compute_cosine_similarity(train_vector, test_vector),
             self.doc_collection.doc_to_category[train_doc])
            for train_doc, train_vector in self.train_vectors
        ]

    def rank_similarities(self, similarities):
        """Ranks similarities from highest to lowest."""
        return sorted(similarities, key=lambda x: x[0], reverse=True)

    def get_k_closest_labels(self, ranked_similarities):
        """Retrieves the labels of the k-closest neighbors."""
        return [label for _, label in ranked_similarities[:self.n_neighbors]]

    def resolve_tie(self, labels):
        """Selects the most frequent label, reducing k if there's a tie."""
        while labels:
            label_counts = Counter(labels)
            most_common = label_counts.most_common()
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                return most_common[0][0]
            labels.pop()

    def classify(self, test_document):
        """Classifies a test document by predicting its category."""
        test_vector = self.doc_collection.compute_tfidf(test_document.token_counts)
        similarities = self.compute_similarities(test_vector)
        ranked_similarities = self.rank_similarities(similarities)
        k_closest_labels = self.get_k_closest_labels(ranked_similarities)
        return self.resolve_tie(k_closest_labels)

# Example using the 20 Newsgroups Dataset
if __name__ == "__main__":
    from nltk import download
    download('punkt')

    # Load the 20 Newsgroups dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # Prepare training documents
    training_docs = [
        TextDocument(text, doc_id=str(i), category=category)
        for i, (text, category) in enumerate(zip(newsgroups_train.data, newsgroups_train.target))
    ]

    # Build the document collection
    training_collection = DocumentCollection.from_document_list(training_docs)

    # Train the KNN classifier
    knn = KNNClassifier(n_neighbors=3)
    knn.train(training_collection)

    # Test the classifier and collect predictions
    y_true = []
    y_pred = []

    for i, (test_text, true_category) in enumerate(zip(newsgroups_test.data[:100], newsgroups_test.target[:100])):
        test_doc = TextDocument(test_text, doc_id=f"test_{i}", category="unknown")
        predicted_category = knn.classify(test_doc)
        y_true.append(true_category)
        y_pred.append(predicted_category)

    # Display confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(newsgroups_test.target))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=newsgroups_train.target_names)
    disp.plot(cmap="viridis", xticks_rotation=45)
    plt.title("Confusion Matrix of KNN Classifier")
    plt.show()
