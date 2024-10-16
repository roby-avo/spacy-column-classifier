import warnings
import pandas as pd
import spacy
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")

class ColumnClassifier:
    def __init__(self, model_type='accurate', sample_size=50, classification_threshold=0.5, word_threshold=10):
        """
        Initialize the ColumnClassifier.
        Parameters:
        - model_type (str): 'accurate' for transformer-based (slow) model, 'fast' for small model.
        - sample_size (int): Number of samples to analyze per column.
        - classification_threshold (float): Minimum threshold for confident classification.
        - word_threshold (int): Threshold for average word count to classify as DESCRIPTION.
        """
        self.model_type = model_type
        self.sample_size = sample_size
        self.classification_threshold = classification_threshold
        self.word_threshold = word_threshold
        
        # Load appropriate SpaCy model
        try:
            if model_type == 'accurate':
                self.nlp = spacy.load('en_core_web_trf')  # Accurate but slow
            elif model_type == 'fast':
                self.nlp = spacy.load('en_core_web_sm')   # Fast but less accurate
            else:
                raise ValueError("Invalid model_type. Choose 'accurate' or 'fast'.")
        except OSError as e:
            raise OSError(f"Model not installed. Error: {str(e)}")
        
        # Map SpaCy fine-grained labels to high-level types
        self.label_map = {
            'DATE': 'DATE', 'TIME': 'DATE', 'MONEY': 'NUMBER', 'PERCENT': 'NUMBER',
            'QUANTITY': 'NUMBER', 'CARDINAL': 'NUMBER', 'ORDINAL': 'NUMBER',
            'GPE': 'LOCATION', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'PERSON': 'PERSON',
            'WORK_OF_ART': 'OTHER', 'EVENT': 'OTHER', 'FAC': 'OTHER', 'PRODUCT': 'OTHER',
            'LAW': 'OTHER', 'NORP': 'OTHER', 'LANGUAGE': 'OTHER'
        }

    def map_spacy_label(self, label: str) -> str:
        """ Map SpaCy's fine-grained labels to high-level classes. """
        return self.label_map.get(label, 'OTHER')

    def is_number(self, value: str) -> bool:
        """ Check if a value is a number by attempting to convert it to float. """
        try:
            float(value)
            return True
        except ValueError:
            return False

    def classify_text_batch(self, texts, sample_data_list):
        """ Classify a batch of texts into high-level classes using SpaCy's nlp.pipe(). """
        results = []
        for doc, sample_data in zip(self.nlp.pipe(texts, batch_size=1024), sample_data_list):
            num_rows = len(sample_data)
            entity_counts = {'LOCATION': 0, 'ORGANIZATION': 0, 'PERSON': 0, 'OTHER': 0}
            literal_counts = {'NUMBER': 0, 'DATE': 0, 'STRING': 0}
            
            # Count entities using SpaCy
            for ent in doc.ents:
                high_level_class = self.map_spacy_label(ent.label_)
                if high_level_class in literal_counts:
                    literal_counts[high_level_class] += 1
                else:
                    entity_counts[high_level_class] += 1

            # Detect numbers using the float method
            number_count = sum(sample_data.apply(self.is_number))
            literal_counts['NUMBER'] = min(number_count, num_rows)  # Cap count to avoid exceeding num_rows

            # Count non-entities as STRING
            literal_counts['STRING'] += max(0, num_rows - len(doc.ents) - number_count)

            # Normalize the counts and ensure they are capped at 1
            probabilities = {key: round(min(count / num_rows, 1.0), 2) 
                             for key, count in {**entity_counts, **literal_counts}.items() if count > 0}

            # Default to 'STRING' if no entities detected
            if not probabilities:
                probabilities = {'STRING': 1.0}

            results.append(probabilities)

        return results

    def classify_column(self, probabilities: dict) -> dict:
        """ Classify a column based on the computed probabilities, applying the prioritization logic. """
        lit_types = ['NUMBER', 'DATE', 'STRING']
        ne_types = ['PERSON', 'LOCATION', 'ORGANIZATION', 'OTHER']
        
        # Step 1: Prioritize NER types if any have a probability >= classification_threshold
        max_ne_type = max({key: probabilities[key] for key in ne_types if key in probabilities}, 
                          key=probabilities.get, default=None)
        if max_ne_type and probabilities[max_ne_type] >= self.classification_threshold:
            return {'classification': max_ne_type, 'probabilities': probabilities}

        # Step 2: Check if any literal type (LIT) has a probability >= classification_threshold
        max_lit_type = max({key: probabilities[key] for key in lit_types if key in probabilities}, 
                           key=probabilities.get, default=None)
        if max_lit_type and probabilities[max_lit_type] >= self.classification_threshold:
            return {'classification': max_lit_type, 'probabilities': probabilities}

        # Step 3: If no NER type has a probability >= 0.5 and multiple NER types are detected, assign OTHER
        total_ne_probability = sum(probabilities[key] for key in ne_types if key in probabilities)
        if total_ne_probability >= self.classification_threshold:
            return {'classification': 'OTHER', 'probabilities': probabilities}

        # Step 4: If none of the above applies, return the type with the highest probability
        dominant_class = max(probabilities, key=probabilities.get)
        return {'classification': dominant_class, 'probabilities': probabilities}

    def classify_multiple_tables(self, tables: list, separator=' | ') -> list:
        """ Classify all columns across multiple DataFrames and return probabilities using nlp.pipe(). """
        texts = []
        sample_data_list = []
        table_column_map = []

        # Step 1: Prepare all text data for batch processing
        for idx, df in enumerate(tqdm(tables, desc="Preparing tables")):
            for column in df.columns:
                non_na_data = df[column].dropna().astype(str)
                num_rows = len(non_na_data)
                if num_rows == 0:
                    continue  # Skip empty columns
                sample_size = min(self.sample_size, num_rows)
                sample_data = non_na_data.sample(n=sample_size, random_state=1)
                concatenated_text = separator.join(sample_data.tolist())

                texts.append(concatenated_text)
                sample_data_list.append(sample_data)
                table_column_map.append((f"table_{idx + 1}", column))  # Track table and column

        # Step 2: Process the texts in batches using nlp.pipe()
        classifications = self.classify_text_batch(texts, sample_data_list)

        # Step 3: Assign results back to their respective columns in tables
        table_classification_results = [{} for _ in range(len(tables))]

        for (table_name, column), probabilities in zip(table_column_map, classifications):
            classified_result = self.classify_column(probabilities)
            table_index = int(table_name.split('_')[1]) - 1
            table_classification_results[table_index][column] = classified_result

        return [{f"table_{idx + 1}": result} for idx, result in enumerate(table_classification_results)]