import os.path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm

from src.models.cefr_classifier import CEFRClassifier


class NBClassifier(CEFRClassifier):
    """Multinomial Naive Bayes with bag-of-words features."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        vectorizer_type = config.get('method')
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=config.get('max_features', 5000),
                ngram_range=config.get('ngram_range', (1, 1)),
                stop_words=config.get('stop_words', None),
                lowercase=True
            )
        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=config.get('max_features', 5000),
                ngram_range=config.get('ngram_range', (1, 1)),
                stop_words=config.get('stop_words', None),
                lowercase=True
            )
        else:
            raise ValueError("Vectorizer type: method must be 'count' or 'tfidf'")
        self.config = config

    def prepare_features(self, X_train, X_in_test, X_out_test, y_train, y_in_test, y_out_test):
        """Convert text to bag-of-words features."""
        print(f"Method: {self.config['method']}")
        print(f"Max features: {self.config['max_features']}")
        print(f"N-gram range: {self.config['ngram_range']}")

        # Fit on training data and transform all sets
        X_train_features = self.vectorizer.fit_transform(tqdm(X_train, desc='Vectorizing train dataset'))
        X_in_test_features = self.vectorizer.transform(tqdm(X_in_test, desc='Vectorizing in-domain test dataset'))
        X_out_test_features = self.vectorizer.transform(tqdm(X_out_test, desc='Vectorizing out-of-domain test dataset'))

        print(f"\nFeature shapes:")
        print(f"  Training: {X_train_features.shape}")
        print(f"  In-Domain Test: {X_in_test_features.shape}")
        print(f"  Out-Domain Test: {X_out_test_features.shape}")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_):,}")

        return {
            'X_train_features': X_train_features,
            'X_in_test_features': X_in_test_features,
            'X_out_test_features': X_out_test_features,
            'vectorizer': self.vectorizer
        }

    def train_model(self, X_train: Any, y_train: pd.Series) -> MultinomialNB:
        """Train Multinomial Naive Bayes model."""
        print(f"Algorithm: Multinomial Naive Bayes")
        model = MultinomialNB(alpha=self.config.get('alpha', 1.0))
        model.fit(X_train, y_train)
        print(f"✓ Model trained with classes: {model.classes_}")
        self.model = model

        return self.model

    def save_model(self):
        save_path = self.config.get('output_dir')
        subdir = os.path.join(save_path, 'nb/')

        os.makedirs(subdir, exist_ok=True)
        name = self.config.get('experiment_name')
        # Save vectorizer
        filename_vectorizer = os.path.join(subdir, "vectorizer.pkl")
        joblib.dump(self.vectorizer, filename_vectorizer)

        filename_model = os.path.join(subdir, "model.pkl")
        # Save model
        joblib.dump(self.model, filename_model)