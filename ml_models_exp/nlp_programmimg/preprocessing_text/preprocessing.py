import re
import string
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from bs4 import BeautifulSoup
from langdetect import detect
from textblob import TextBlob
import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """
    A class for preprocessing text data. This includes tasks like tokenization, stopword removal, lemmatization, stemming,
    punctuation removal, and more. The preprocessing steps can be customized using configuration parameters.

    Attributes:
        config (dict): A dictionary of preprocessing options.
        stop_words (set): A set of stopwords to filter out from the text.
        lemmatizer (WordNetLemmatizer): An instance of the WordNetLemmatizer for lemmatizing tokens.
        stemmer (PorterStemmer): An instance of the PorterStemmer for stemming tokens.
    """

    def __init__(
            self,
            remove_punctuation=True,
            lowercase=True,
            remove_numbers=True,
            remove_stopwords=True,
            lemmatize=True,
            stem=False,
            remove_special_chars=True,
            remove_html=True,
            correct_spelling=False,
            detect_language=False,
            min_word_length=2,
    ):
        """
        Initializes the TextPreprocessor with configurable options.

        Args:
            remove_punctuation (bool): Whether to remove punctuation from the text.
            lowercase (bool): Whether to convert text to lowercase.
            remove_numbers (bool): Whether to remove numeric tokens from the text.
            remove_stopwords (bool): Whether to remove stopwords from the text.
            lemmatize (bool): Whether to lemmatize tokens to their base form.
            stem (bool): Whether to stem tokens to their root form.
            remove_special_chars (bool): Whether to remove special characters from the text.
            remove_html (bool): Whether to strip HTML tags from the text.
            correct_spelling (bool): Whether to correct spelling errors in the text.
            detect_language (bool): Whether to detect and print the language of the text.
            min_word_length (int): Minimum length of tokens to keep.
        """
        self.config = {
            "remove_punctuation": remove_punctuation,
            "lowercase": lowercase,
            "remove_numbers": remove_numbers,
            "remove_stopwords": remove_stopwords,
            "lemmatize": lemmatize,
            "stem": stem,
            "remove_special_chars": remove_special_chars,
            "remove_html": remove_html,
            "correct_spelling": correct_spelling,
            "detect_language": detect_language,
            "min_word_length": min_word_length,
        }

        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def preprocess(self, text):
        """
        Preprocesses the input text according to the configured options.

        Args:
            text (str): The text to preprocess.

        Returns:
            list: A list of preprocessed tokens.
        """
        print(self.config)
        if self.config["detect_language"]:
            lang = detect(text)
            print(f"Detected language:, {lang}")
        if self.config["remove_html"]:
            text = BeautifulSoup(text, "html.parser").get_text()
        tokens = word_tokenize(text)

        removed_counts = {
            "punctuation": 0,
            "numbers": 0,
            "stopwords": 0,
            "short_tokens": 0
        }

        preprocessed_tokens = []
        for token in tokens:
            if self.config["remove_punctuation"] and token in string.punctuation:
                removed_counts["punctuation"] += 1
                continue
            if self.config["remove_numbers"] and token.isdigit():
                removed_counts["numbers"] += 1
                continue
            if self.config["remove_stopwords"] and token.lower() in self.stop_words:
                removed_counts["stopwords"] += 1
                continue
            if len(token) < self.config["min_word_length"]:
                removed_counts["short_tokens"] += 1
                continue

            token = self._process_token(token)
            preprocessed_tokens.append(token)

        print("Removed counts:", removed_counts)
        return preprocessed_tokens

    def _process_token(self, token):
        """
        Processes a single token by applying transformations like lowercasing, lemmatization, and stemming.

        Args:
            token (str): The token to process.

        Returns:
            str: The processed token.
        """
        if self.config["lowercase"]:
            token = token.lower()

        if self.config["remove_special_chars"]:
            token = re.sub(r"[^A-Za-z0-9]+", " ", token)

        if self.config["correct_spelling"]:
            token = str(TextBlob(token).correct())

        if self.config["lemmatize"]:
            token = self.lemmatizer.lemmatize(token)

        if self.config["stem"]:
            token = self.stemmer.stem(token)

        return token

    def _is_valid_token(self, token):
        """
        Validates a token based on configuration options.

        Args:
            token (str): The token to validate.

        Returns:
            bool: True if the token is valid, False otherwise.
        """
        if self.config["remove_punctuation"] and token in string.punctuation:
            return False

        if self.config["remove_numbers"] and token.isdigit():
            return False

        if self.config["remove_stopwords"] and token.lower() in self.stop_words:
            return False

        if len(token) < self.config["min_word_length"]:
            return False

        return True

    def pos_tag(self, tokens):
        """
        Performs part-of-speech tagging on a list of tokens.

        Args:
            tokens (list): The tokens to tag.

        Returns:
            list: A list of tuples, where each tuple contains a token and its part-of-speech tag.
        """
        return nltk.pos_tag(tokens)


if __name__ == '__main__':
    text = """
    <h1>Ada Lovelace: The Enchantress of Numbers</h1>
    <p>Ada, born in 1815, was a pioneer of computer science. Her work with Charles Babbage on the Analytical Engine laid the groundwork for modern programming.</p>
    <p>Her contributions are celebrated worldwide! Numbers like 42 and 3.14159 often appear in discussions about mathematics and computer science. 
    <a href='https://example.com'>Learn more</a>.</p>
    <p>Did you know? She wrote, 'That brain of mine is something more than merely mortal, as time will show.'</p>
    """

    print(f"The length of text before preprocessing : {len(text)}")
    preprocessor = TextPreprocessor(correct_spelling=True, detect_language=True)
    cleaned_tokens = preprocessor.preprocess(text)
    print(f"Cleaned tokens: {cleaned_tokens}")
    print(f"The length of tokens: {len(cleaned_tokens)}")

    pos_tag = preprocessor.pos_tag(cleaned_tokens)
    print(f"Pos tag: {pos_tag}")
