from nltk import *
from nltk.probability import FreqDist, ConditionalFreqDist


class SymbolischDebug():
    def __init__(self, path):
        with open(path, 'r') as file:
            self.text = [w for line in file for w in word_tokenize(line.strip())]
            self.token_counts = FreqDist(self.text)


    def numberOfTokens(self):
        """
        Returns the number of tokens in the text.
        :return:
        """
        return self.token_counts.N()

    def ProbabilityDistribution(self):
        """
        Probability Distribution
        :return:
        """
        totalToken = self.numberOfTokens()
        return {token:self.token_counts[token] / totalToken for token in self.token_counts}

    def TopFiveFrequentTokens(self):
        """
        Returns the top five most frequent tokens.
        :return:
        """

        sortedTokens = sorted(self.token_counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in sortedTokens[:5]]

    def LexicalDiversity(self):
        """
        Returns the Lexical Diversity score.
        :return:
        """
        return len(self.token_counts) / self.numberOfTokens()



