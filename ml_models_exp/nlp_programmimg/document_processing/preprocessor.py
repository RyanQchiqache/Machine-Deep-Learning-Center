import re
import string
import unicodedata
from typing import List

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from bs4 import BeautifulSoup
from textblob import TextBlob
from langdetect import detect

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TextPreprocessor:
    """Handles text preprocessing steps like cleaning, tokenization, stopword removal, stemming, etc."""

    def __init__(self, remove_punctuation=True, lowercase=True, remove_numbers=True,
                 remove_stopwords=True, lemmatize=True, stem=False,
                 remove_special_chars=True, remove_html=True, correct_spelling=False,
                 detect_language=False, min_word_length=2):

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

    def preprocess(self, text: List[str]) -> List[str]:
        """Cleans and tokenizes input text based on configuration."""
        if self.config["detect_language"]:
            lang = detect(text)
            print(f"Detected language: {lang}")

        if self.config["remove_html"]:
            text = BeautifulSoup(text, "html.parser").get_text()

        tokens = word_tokenize(text)

        cleaned_tokens = [
            self._process_token(token) for token in tokens
            if self._is_valid_token(token)
        ]

        return cleaned_tokens

    def _process_token(self, token: str) -> str:
        """Applies cleaning steps to a single token."""
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

    def _is_valid_token(self, token: str) -> bool:
        """Checks if a token should be included based on settings."""
        if self.config["remove_punctuation"] and token in string.punctuation:
            return False
        if self.config["remove_numbers"] and token.isdigit():
            return False
        if self.config["remove_stopwords"] and token.lower() in self.stop_words:
            return False
        if len(token) < self.config["min_word_length"]:
            return False
        return True

    def pos_tag(self, tokens: List[str]) -> List[str]:
        """Performs part-of-speech tagging."""
        return nltk.pos_tag(tokens)
