import numpy as np
import nltk
import os, math, string
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from numpy.linalg import norm
from sklearn.datasets import fetch_20newsgroups


# Ensure required NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')


def dot(dict_a, dict_b):
    """Computes the dot product of two vectors represented as dictionaries."""
    return sum([dict_a.get(tok, 0) * dict_b.get(tok, 0) for tok in dict_a])


def normalized_tokens(text):
    """Tokenizes and normalizes text (lowercasing, removing stopwords and punctuation)."""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english') and t not in string.punctuation]
    return tokens


class TextDocument:
    """Represents a single text document with tokenized content and metadata."""

    def __init__(self, text, doc_id=None):
        self.text = text.replace('-\n', '').replace('\n', ' ')  # Remove line breaks
        self.token_counts = defaultdict(int)
        for token in normalized_tokens(self.text):
            self.token_counts[token] += 1
        self.id = doc_id


class DocumentCollection:
    """Represents a collection of documents with indexing for efficient retrieval."""

    def __init__(self, term_to_df, term_to_docids, docid_to_doc):
        self.term_to_df = term_to_df
        self.term_to_docids = term_to_docids
        self.docid_to_doc = docid_to_doc

    @classmethod
    def from_document_list(cls, docs):
        term_to_df = defaultdict(int)
        term_to_docids = defaultdict(set)
        docid_to_doc = {}
        for doc in docs:
            docid_to_doc[doc.id] = doc
            for token in doc.token_counts:
                term_to_df[token] += 1
                term_to_docids[token].add(doc.id)
        return cls(term_to_df, term_to_docids, docid_to_doc)

    def tfidf(self, counts):
        """Computes the TF-IDF values for a document's term frequencies."""
        N = len(self.docid_to_doc)
        return {tok: tf * math.log((N + 1) / (self.term_to_df[tok] + 1)) + 1 for tok, tf in counts.items() if
                tok in self.term_to_df}

    def cosine_similarity(self, doc_a, doc_b):
        """Computes the cosine similarity between two documents."""
        tfidf_doc_a = self.tfidf(doc_a.token_counts)
        tfidf_doc_b = self.tfidf(doc_b.token_counts)
        dot_docs = dot(tfidf_doc_a, tfidf_doc_b)
        norm_doc_a = math.sqrt(dot(tfidf_doc_a, tfidf_doc_a))
        norm_doc_b = math.sqrt(dot(tfidf_doc_b, tfidf_doc_b))
        return dot_docs / (norm_doc_a * norm_doc_b + 1e-10)  # Avoid division by zero

    def docs_with_some_tokens(self, tokens):
        """Retrieves documents containing at least one of the tokens."""
        docids_for_each_token = [self.term_to_docids[token] for token in tokens if token in self.term_to_docids]
        if not docids_for_each_token:
            return []
        docids = set.union(*docids_for_each_token)
        return [self.docid_to_doc[_id] for _id in docids]


class SearchEngine:
    """Implements a search engine using TF-IDF and cosine similarity."""

    def __init__(self, doc_collection):
        self.doc_collection = doc_collection

    def ranked_documents(self, query, top_k=5):
        """Retrieves the top-k ranked documents for a given query."""
        query_doc = TextDocument(query)
        query_tokens = query_doc.token_counts.keys()
        docs = self.doc_collection.docs_with_some_tokens(query_tokens)
        docs_sims = [(doc, self.doc_collection.cosine_similarity(query_doc, doc)) for doc in docs]
        return sorted(docs_sims, key=lambda x: -x[1])[:top_k]


def main():
    """Main function to load the dataset, build the search engine, and execute a sample query."""
    # Load the 20 Newsgroups dataset
    news_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # Create documents from dataset
    documents = [TextDocument(text, f"doc{i}") for i, text in
                 enumerate(news_data.data[:1000])]  # Using 1000 documents for efficiency

    # Build the document collection
    search_engine = SearchEngine(DocumentCollection.from_document_list(documents))

    # Example search query
    query = "the white house"
    results = search_engine.ranked_documents(query)

    print("Search Results:")
    for doc, score in results:
        print(f"{score:.4f} - {doc.text[:400]}...")


if __name__ == "__main__":
    main()
