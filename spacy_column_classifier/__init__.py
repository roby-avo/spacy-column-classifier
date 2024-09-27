# __init__.py for the spacy_column_classifier package

# Import the ColumnClassifier class so it can be accessed directly
from .column_classifier import ColumnClassifier

# Specify what is exported when users do 'from spacy_column_classifier import *'
__all__ = ['ColumnClassifier']