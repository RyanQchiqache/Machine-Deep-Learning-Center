import unittest
from IR_searchEngine import TextDocument, DocumentCollection, SearchEngine


class TestSearchEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a small document collection for testing."""
        cls.documents = [
            TextDocument("Artificial Intelligence is the future of technology.", "doc1"),
            TextDocument("Machine learning and AI are transforming industries.", "doc2"),
            TextDocument("The impact of AI in robotics and automation is huge.", "doc3"),
            TextDocument("AI-powered solutions are changing the world.", "doc4"),
            TextDocument("Ethics and bias in AI must be addressed.", "doc5")
        ]
        cls.collection = DocumentCollection.from_document_list(cls.documents)
        cls.search_engine = SearchEngine(cls.collection)

    def test_tfidf_computation(self):
        """Test if TF-IDF values are computed correctly."""
        doc = self.documents[0]
        tfidf_values = self.collection.tfidf(doc.token_counts)
        self.assertGreater(len(tfidf_values), 0)
        self.assertIn("artificial", tfidf_values)

    def test_cosine_similarity(self):
        """Test cosine similarity between two documents."""
        doc_a = self.documents[0]
        doc_b = self.documents[1]
        similarity = self.collection.cosine_similarity(doc_a, doc_b)
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)

    def test_docs_with_some_tokens(self):
        """Test document retrieval based on some tokens."""
        tokens = ["AI", "technology"]
        retrieved_docs = self.collection.docs_with_some_tokens(tokens)
        self.assertGreater(len(retrieved_docs), 0)

    def test_search_ranking(self):
        """Test ranked document retrieval for a query."""
        query = "AI and robotics"
        results = self.search_engine.ranked_documents(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0][1], results[-1][1])  # Ensure ranking works


if __name__ == "__main__":
    unittest.main()