Here is the entire README in a single markdown block for easy copying:

# spacy-column-classifier

A Python package that classifies DataFrame columns into Named Entity (NER) or Literal types using spaCy's powerful natural language processing models. This library is optimized for batch processing, making it efficient for working with large datasets.

## Features

- **Classification of Columns**: Classifies each column of a DataFrame as Named Entity (NER) or Literal (LIT) types, including LOCATION, ORGANIZATION, PERSON, NUMBER, DATE, and more.
- **Batch Processing**: Uses spaCy’s `nlp.pipe()` to efficiently process multiple columns across multiple tables in parallel, improving performance for large datasets.
- **Customizable**: Supports both transformer-based models (for high accuracy) and smaller models (for speed).
- **Handles Multiple DataFrames**: Allows you to classify columns across multiple DataFrames in one go.
- **Conflict Resolution**: Handles cases where multiple class types are detected for a single column and resolves conflicts based on customizable thresholds.

## Installation

You can install the package via pip:

```bash
pip install column-classifier
```

Make sure you have installed one of the compatible spaCy models:

For accuracy (slower but more precise):    
```bash
python -m spacy download en_core_web_trf
```
	
For speed (faster but less accurate):
```bash
python -m spacy download en_core_web_sm
```

Quick Start

Here’s how you can use spacy-column-classifier in your project with hardcoded example data:
```bash
import pandas as pd
from column_classifier import ColumnClassifier

# Hardcoded sample data
data1 = {
    'title': ['Inception', 'The Matrix', 'Interstellar'],
    'director': ['Christopher Nolan', 'The Wachowskis', 'Christopher Nolan'],
    'release year': [2010, 1999, 2014],
    'domestic distributor': ['Warner Bros.', 'Warner Bros.', 'Paramount'],
    'length in min': [148, 136, 169],
    'worldwide gross': [829895144, 466364845, 677471339]
}

data2 = {
    'company': ['Google', 'Microsoft', 'Apple'],
    'location': ['California', 'Washington', 'California'],
    'founded': [1998, 1975, 1976],
    'CEO': ['Sundar Pichai', 'Satya Nadella', 'Tim Cook'],
    'employees': [139995, 163000, 147000],
    'revenue': [182527, 168088, 274515]
}

# Create DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# List of DataFrames to classify
dataframes = [df1, df2]

# Create an instance of ColumnClassifier
classifier = ColumnClassifier(model_type='accurate')  # 'accurate' for transformer model

# Classify multiple DataFrames
results = classifier.classify_multiple_tables(dataframes)

# Display the results
for table_result in results:
    for table_name, classification in table_result.items():
        print(f"Results for {table_name}:")
        for col, types in classification.items():
            print(f"  Column '{col}': Classified as {types['classification']}")
        print()
```

API Reference

ColumnClassifier

The main class used to classify DataFrame columns.

Parameters:

	•	model_type: Choose between ‘accurate’ (transformer-based) or ‘fast’ (small model).
	•	sample_size: Number of samples to analyze per column.
	•	classification_threshold: Minimum threshold for confident classification.
	•	close_prob_threshold: Threshold for resolving conflicts between close probabilities.
	•	word_threshold: If the average word count in a column exceeds this, the column is classified as a DESCRIPTION.

Methods:

	•	classify_multiple_tables(tables: list) -> list: Classifies all columns across multiple DataFrames. Returns a list of dictionaries containing the classification results.
	•	classify_column(column_data: pd.Series) -> dict: Classifies a single column and returns a dictionary of classifications and probabilities.

Example Output

After classifying your DataFrames, the output will be structured like this:
```bash
[
  {
    "table_1": {
      "title": {
        "classification": "OTHER",
        "probabilities": {
          "OTHER": 1.0
        }
      },
      "director": {
        "classification": "PERSON",
        "probabilities": {
          "PERSON": 1.0
        }
      },
      "release year": {
        "classification": "NUMBER",
        "probabilities": {
          "NUMBER": 1.0,
          "DATE": 1.0
        }
      },
      "domestic distributor": {
        "classification": "ORGANIZATION",
        "probabilities": {
          "ORGANIZATION": 1.0
        }
      },
      "length in min": {
        "classification": "NUMBER",
        "probabilities": {
          "NUMBER": 1.0
        }
      },
      "worldwide gross": {
        "classification": "NUMBER",
        "probabilities": {
          "NUMBER": 1.0
        }
      }
    }
  },
  {
    "table_2": {
      "company": {
        "classification": "ORGANIZATION",
        "probabilities": {
          "ORGANIZATION": 1.0
        }
      },
      "location": {
        "classification": "LOCATION",
        "probabilities": {
          "LOCATION": 1.0
        }
      },
      "founded": {
        "classification": "NUMBER",
        "probabilities": {
          "NUMBER": 1.0,
          "DATE": 1.0
        }
      },
      "CEO": {
        "classification": "PERSON",
        "probabilities": {
          "PERSON": 1.0
        }
      },
      "employees": {
        "classification": "NUMBER",
        "probabilities": {
          "NUMBER": 1.0
        }
      },
      "revenue": {
        "classification": "NUMBER",
        "probabilities": {
          "NUMBER": 1.0
        }
      }
    }
  }
]
```

Each column is classified with a winning classification, and the probabilities show the likelihood of different class types detected in the column.

License

This project is licensed under the Apache License.

This version should be easier to copy and paste correctly without errors.