import sys
import os
import unittest
import pandas as pd

# Add the path to the spacy_column_classifier directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from column_classifier import ColumnClassifier

class TestColumnClassifier(unittest.TestCase):
    def test_basic_classification(self):
        classifier = ColumnClassifier()
        result = classifier.classify_column(pd.Series(['New York', 'Paris', 'Tokyo']))
        self.assertEqual(result['classification'], 'LOCATION')

if __name__ == '__main__':
    unittest.main()