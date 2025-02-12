import unittest
from hypothesis import given, strategies as st
from IR_searchEngine import TextDocument, DocumentCollection, SearchEngine


class TestSearchEnginePBT(unittest.TestCase):

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

    @given(st.text(min_size=1, max_size=100))
    def test_tfidf_non_negative(self, random_text):
        """TF-IDF values should always be non-negative."""
        doc = TextDocument(random_text)
        tfidf_values = self.collection.tfidf(doc.token_counts)
        for value in tfidf_values.values():
            self.assertGreaterEqual(value, 0)

    @given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=100))
    def test_cosine_similarity_bounds(self, text_a, text_b):
        """Cosine similarity should be between 0 and 1."""
        doc_a = TextDocument(text_a)
        doc_b = TextDocument(text_b)
        similarity = self.collection.cosine_similarity(doc_a, doc_b)
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)

    @given(st.text(min_size=1, max_size=100))
    def test_idempotent_search(self, query):
        """Search should return the same results for the same query."""
        results1 = self.search_engine.ranked_documents(query)
        results2 = self.search_engine.ranked_documents(query)
        self.assertEqual(results1, results2)

    def test_self_similarity(self):
        """A document compared to itself should have cosine similarity of 1."""
        for doc in self.documents:
            similarity = self.collection.cosine_similarity(doc, doc)
            self.assertAlmostEqual(similarity, 1.0)

    def test_empty_query(self):
        """An empty query should return no results."""
        results = self.search_engine.ranked_documents("")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
