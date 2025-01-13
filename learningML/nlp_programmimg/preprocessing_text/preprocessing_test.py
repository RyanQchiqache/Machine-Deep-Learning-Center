import unittest
from preprocessing import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_remove_html(self):
        text = "<p>Hello, world!</p> This is a test."
        expected = ['hello', 'world', 'test']
        self.preprocessor.config["remove_html"] = True
        result = self.preprocessor.preprocess(text)
        self.assertEqual(result, expected)

    def test_remove_punctuation(self):
        text = "Hello, world! This is a test."
        expected = ['hello', 'world', 'test']
        self.preprocessor.config["remove_punctuation"] = True
        result = self.preprocessor.preprocess(text)
        self.assertEqual(result, expected)

    def test_lowercase(self):
        text = "Hello WORLD"
        expected = ['hello', 'world']
        self.preprocessor.config["lowercase"] = True
        result = self.preprocessor.preprocess(text)
        self.assertEqual(result, expected)

    def test_remove_numbers(self):
        text = "Test 123 example 456"
        expected = ['test', 'example']
        self.preprocessor.config["remove_numbers"] = True
        result = self.preprocessor.preprocess(text)
        self.assertEqual(result, expected)

    def test_remove_stopwords(self):
        text = "This is a test example."
        expected = ['test', 'example']
        self.preprocessor.config["remove_stopwords"] = True
        result = self.preprocessor.preprocess(text)
        self.assertEqual(result, expected)

    def test_lemmatization(self):
        text = "running tests on words"
        expected = ['run', 'test', 'word']
        self.preprocessor.config["lemmatize"] = True
        result = self.preprocessor.preprocess(text)
        self.assertEqual(result, expected)

    def test_stemming(self):
        text = "running tests on words"
        expected = ['run', 'test', 'word']
        self.preprocessor.config["stem"] = True
        result = self.preprocessor.preprocess(text)
        self.assertEqual(result, expected)

    def test_min_word_length(self):
        text = "It is a big test."
        expected = ['big', 'test']
        self.preprocessor.config["min_word_length"] = 3
        result = self.preprocessor.preprocess(text)
        self.assertEqual(result, expected)

    def test_pos_tagging(self):
        text = "Hello, world!"
        cleaned_tokens = self.preprocessor.preprocess(text)
        pos_tags = self.preprocessor.pos_tag(cleaned_tokens)
        self.assertIsInstance(pos_tags, list)
        self.assertTrue(all(isinstance(tag, tuple) for tag in pos_tags))


if __name__ == '__main__':
    unittest.main()
