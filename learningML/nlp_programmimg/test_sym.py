import os
from unittest import TestCase
from learningML.nlp_programmimg import sym


class SymbolischDebugTest(TestCase):

    def setUp(self):
        # make sure that the path to the folder data corresponds to the correct path on your computer
        path = os.path.join(os.path.dirname(__file__), "ada_lovelace.txt")
        self.analyzer = sym(path)

    def test_01_numberOfTokens(self):
        # Check if the number of tokens matches an expected value
        self.assertEqual(self.analyzer.numberOfTokens(), 4506)

    def test_02_ProbabilityDistribution(self):
        # Check the size of the probability distribution and probabilities of specific tokens
        self.assertEqual(len(self.analyzer.ProbabilityDistribution()), 1390)
        self.assertAlmostEqual(self.analyzer.ProbabilityDistribution().get('Ada', 0), 0.0175322, places=7)
        self.assertAlmostEqual(self.analyzer.ProbabilityDistribution().get('Mary', 0), 0.0008877, places=7)

    def test_03_TopFiveFrequentTokens(self):
        # Check if the top five frequent tokens match the expected list
        self.assertEqual(self.analyzer.TopFiveFrequentTokens(), [',', '.', 'the', 'of', 'her'])

    def test_04_LexicalDiversity(self):
        # Check if the lexical diversity is correct when rounded to 4 decimal places
        self.assertEqual(round(self.analyzer.LexicalDiversity(), 4), 0.3085)
