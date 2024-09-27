import warnings
import pandas as pd
import spacy

# Suppress all warnings
warnings.filterwarnings("ignore")

class ColumnClassifier:
    def __init__(self, sample_size=100, classification_threshold=0.5, close_prob_threshold=0.05, word_threshold=10):
        """
        Initialize the ColumnClassifier.

        Parameters:
        - sample_size (int): Number of samples to analyze per column.
        - classification_threshold (float): Minimum threshold for confident classification.
        - close_prob_threshold (float): Threshold for considering two probabilities as "close".
        - word_threshold (int): Threshold for average word count to classify as DESCRIPTION.
        """
        try:
            self.nlp = spacy.load('en_core_web_trf')  # Load SpaCy's transformer model
        except OSError:
            raise OSError("SpaCy model 'en_core_web_trf' not installed. Install with: python -m spacy download en_core_web_trf")

        self.sample_size = sample_size
        self.classification_threshold = classification_threshold
        self.close_prob_threshold = close_prob_threshold
        self.word_threshold = word_threshold

        # Map SpaCy fine-grained labels to high-level types
        self.label_map = {
            'DATE': 'DATE', 'TIME': 'DATE', 'MONEY': 'NUMBER', 'PERCENT': 'NUMBER',
            'QUANTITY': 'NUMBER', 'CARDINAL': 'NUMBER', 'ORDINAL': 'NUMBER',
            'GPE': 'LOCATION', 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'PERSON': 'PERSON',
            'WORK_OF_ART': 'OTHER', 'EVENT': 'OTHER', 'FAC': 'OTHER', 'PRODUCT': 'OTHER',
            'LAW': 'OTHER', 'NORP': 'OTHER', 'LANGUAGE': 'OTHER'
        }

    def map_spacy_label(self, label: str) -> str:
        """
        Map SpaCy's fine-grained labels to high-level classes.

        Parameters:
        - label (str): The fine-grained label from SpaCy.

        Returns:
        - str: The mapped high-level label, defaulting to 'OTHER' for unmapped types.
        """
        return self.label_map.get(label, 'OTHER')

    def is_number(self, value: str) -> bool:
        """
        Check if a value is a number by attempting to convert it to float.

        Parameters:
        - value (str): The value to check.

        Returns:
        - bool: True if it's a number, False otherwise.
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    def normalize_count(self, count: int, num_rows: int) -> int:
        """
        Ensure the count does not exceed the number of rows.

        Parameters:
        - count (int): The current count of a specific type (entity/literal).
        - num_rows (int): The total number of rows in the dataset.

        Returns:
        - int: The normalized count, ensuring it does not exceed num_rows.
        """
        return min(count, num_rows)

    def classify_text(self, text: str, num_rows: int, sample_data: pd.Series) -> dict:
        """
        Classify text into high-level classes using SpaCy and float-based number detection.

        Parameters:
        - text (str): The concatenated text from the column.
        - num_rows (int): Total number of rows in the column.
        - sample_data (pd.Series): Sample data from the column for number recognition.

        Returns:
        - dict: A dictionary with high-level classes and their normalized probabilities.
        """
        entity_counts = {'LOCATION': 0, 'ORGANIZATION': 0, 'PERSON': 0, 'OTHER': 0}
        literal_counts = {'NUMBER': 0, 'DATE': 0, 'STRING': 0}
        print("text", text)
        # Count entities using SpaCy
        doc = self.nlp(text)
        for ent in doc.ents:
            high_level_class = self.map_spacy_label(ent.label_)
            if high_level_class in literal_counts:
                literal_counts[high_level_class] += 1
            else:
                entity_counts[high_level_class] += 1

        # Detect numbers using the float method
        number_count = sum(sample_data.apply(self.is_number))
        
        # Average SpaCy's number prediction and float-based detection
        literal_counts['NUMBER'] = number_count

        # Count non-entities as STRING
        literal_counts['STRING'] += (num_rows - len(doc.ents) - number_count)

        # Normalize the counts and ensure they do not exceed sample size
        probabilities = {key: round(self.normalize_count(count, num_rows) / num_rows, 2) 
                         for key, count in {**entity_counts, **literal_counts}.items() if count > 0}

        # Default to 'STRING' if no entities detected
        if not probabilities:
            probabilities = {'STRING': 1.0}

        return probabilities

    def resolve_conflict(self, probabilities: dict) -> str:
        """
        Resolve conflicts between similar probabilities for NE and LIT types, prioritizing NE.

        Parameters:
        - probabilities (dict): A dictionary of detected types and their probabilities.

        Returns:
        - str: The resolved dominant class.
        """
        lit_types = ['NUMBER', 'DATE', 'STRING']
        ne_types = ['PERSON', 'LOCATION', 'ORGANIZATION', 'OTHER']

        dominant_class = max(probabilities, key=probabilities.get)

        if any(lit in probabilities for lit in lit_types) and any(ne in probabilities for ne in ne_types):
            max_ne_type = max({key: probabilities[key] for key in ne_types if key in probabilities}, key=probabilities.get, default=None)
            max_lit_type = max({key: probabilities[key] for key in lit_types if key in probabilities}, key=probabilities.get, default=None)

            # Prefer NE type if probabilities are close
            if max_ne_type and max_lit_type and abs(probabilities[max_ne_type] - probabilities[max_lit_type]) <= self.close_prob_threshold:
                dominant_class = max_ne_type
            elif probabilities[max_ne_type] >= probabilities[max_lit_type]:
                dominant_class = max_ne_type

        return dominant_class

    def classify_column(self, column_data: pd.Series) -> dict:
        """
        Classify a single DataFrame column and return probabilities for each class.

        Parameters:
        - column_data (pd.Series): The data of the column to classify.

        Returns:
        - dict: The classification and probabilities.
        """
        non_na_data = column_data.dropna().astype(str)
        num_rows = len(non_na_data)
        sample_size = min(self.sample_size, num_rows)
        
        if num_rows == 0:
            return {'classification': 'LIT (STRING)', 'probabilities': {'STRING': 1.0}}

        sample_data = non_na_data.sample(n=sample_size, random_state=1)

        # Check average word count to identify description-like columns
        avg_word_count = sample_data.apply(lambda x: len(x.split())).mean()
        if avg_word_count > self.word_threshold:
            return {'classification': 'DESCRIPTION', 'probabilities': {'STRING': 1.0}}

        concatenated_text = ' | '.join(sample_data.tolist())

        probabilities = self.classify_text(concatenated_text, sample_size, sample_data)
        print("probabilities", probabilities)
        # Apply classification threshold: use STRING if confidence is low
        max_prob = max(probabilities.values())
        if max_prob < self.classification_threshold:
            return {'classification': 'STRING', 'probabilities': {'STRING': 1.0}}

        dominant_class = self.resolve_conflict(probabilities)

        return {'classification': dominant_class, 'probabilities': probabilities}

    def classify_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Classify all columns in a DataFrame and return probabilities.

        Parameters:
        - df (pd.DataFrame): The DataFrame to classify.

        Returns:
        - dict: A dictionary with column names as keys and their classifications and probabilities as values.
        """
        classification_results = {}
        for column in df.columns:
            try:
                column_result = self.classify_column(df[column])
                classification_results[column] = column_result
            except Exception as e:
                classification_results[column] = {'classification': 'LIT (STRING)', 'probabilities': {'STRING': 1.0}}
                print(f'Error classifying column \"{column}\": {e}')
        return classification_results