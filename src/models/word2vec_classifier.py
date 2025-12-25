from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import torch
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.cefr_classifier import CEFRClassifier
import gensim.downloader as api

from src.models.neural_network import Classifier, DeepClassifier


class Word2VecClassifier(CEFRClassifier):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vectorizer = None
        self.nn_model = None
        self.embedding_model = config.get('embedding_model', 'w2v')
        self.embedding_name = config.get('embedding_name',
                                         'glove-wiki-gigaword-300') if self.embedding_model == 'w2v' else 'Doc2Vec'
        self.aggregation = config.get('agg_method', 'mean')
        self.batch_size = config.get('batch_size', 64)
        self.stop_words = config.get('stop_words', None)
        self.doc2vec_epochs = config.get('doc2vec_epochs', 10)
        self.doc2vec_mincount = config.get('doc2vec_mincount', 2)

    def prepare_features(self, X_train, X_in_test, X_out_test, y_train, y_in_test, y_out_test) -> Dict[str, Any]:
        """
        Prepare features by loading/training embeddings and creating datasets.
        """
        print(f"\nEmbedding model: {self.embedding_model}")
        print(f"Embedding: {self.embedding_name}")
        print(f"Aggregation method: {self.aggregation}")
        print(f"Batch size: {self.batch_size}")
        print(f"Stop words: {self.stop_words}")

        # Step 1: Load or train embedding model
        if self.embedding_model == 'doc2vec':
            # Train Doc2Vec on remaining_samples.csv
            model_obj, embedding_dim = self._train_doc2vec_model()
            tfidf_vectorizer = None
            tfidf_train = None
            tfidf_val = None
            tfidf_test = None
        else:  # Word2Vec/GloVe
            # Load pre-trained Word2Vec embeddings
            model_obj, embedding_dim = self._load_w2v_model(self.embedding_name)

            # Compute TF-IDF weights if using TF-IDF weighted aggregation
            if self.aggregation == 'tfidf_weighted':
                tfidf_vectorizer, tfidf_train, tfidf_val, tfidf_test = self._compute_tfidf(
                    X_train, X_in_test, X_out_test
                )
            else:
                # No TF-IDF needed for simple mean aggregation
                tfidf_vectorizer = None
                tfidf_train = None
                tfidf_val = None
                tfidf_test = None

        # Step 2: Build datasets (convert text to embeddings)
        print("\nCreating embedded datasets...")

        train_dataset = self._build_dataset(
            X_train, y_train, "Train embeddings",
            model_obj, embedding_dim,
            tfidf_vectorizer, tfidf_train
        )

        val_dataset = self._build_dataset(
            X_in_test, y_in_test, "Val embeddings",
            model_obj, embedding_dim,
            tfidf_vectorizer, tfidf_val
        )

        test_dataset = self._build_dataset(
            X_out_test, y_out_test, "Test embeddings",
            model_obj, embedding_dim,
            tfidf_vectorizer, tfidf_test
        )

        # Step 3: Create DataLoaders
        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"\n✓ Dataloaders created")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        self.vectorizer = model_obj

        # Return dictionary matching CEFRClassifier expectations
        return {
            # Required keys for CEFRClassifier.run_experiment()
            'X_train_features': train_loader,
            'X_in_test_features': val_loader,
            'X_out_test_features': test_loader,

            # Optional: model-specific metadata
            'vectorizer': model_obj,  # Store embedding model as "vectorizer"
            'embedding_dim': embedding_dim,
            'agg_method': self.aggregation,
            'embedding_name': self.embedding_name,

            # Keep backward compatibility (if needed elsewhere)
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'embedding_model': model_obj,
        }

    def train_model(self, X_train_features: Any, y_train: pd.Series) -> Any:
        """
        Train Word2Vec-based CEFR classifier with configurable architecture.
        """

        # X_train_features is actually a DataLoader
        train_loader = X_train_features

        # Get embedding dimension from first batch
        first_batch = next(iter(train_loader))
        embedding_dim = first_batch[0].shape[1]

        # Get configuration parameters
        architecture = self.config.get('architecture', 'simple')  # 'simple', 'deep', 'residual', 'attention'
        hidden_dim = self.config.get('hidden_dim', 128)
        epochs = self.config.get('epochs', 10)
        learning_rate = self.config.get('learning_rate', 0.001)
        dropout_rate = self.config.get('dropout_rate', 0.3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\nDevice: {device}")
        print(f"Architecture: {architecture}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Output classes: 5")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")

        # Initialize model based on architecture choice
        if architecture == 'deep':
            model = DeepClassifier(
                embedding_dim=embedding_dim,
                hidden_dim1=hidden_dim*2,
                hidden_dim2=hidden_dim,
                hidden_dim3=hidden_dim/2,
                num_classes=5,
                dropout_rate=dropout_rate
            ).to(device)
        else:  # 'simple'
            model = Classifier(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_classes=5
            ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Setup training
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        print(f"\nTraining...")

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(X)
                loss = loss_function(outputs, y)

                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        print(f"✓ Training complete")

        # Store model for later use in predict methods
        self.model = model

        # Wrap the PyTorch model to provide sklearn-compatible interface
        wrapped_model = PyTorchModelWrapper(model, device)

        return wrapped_model

    def _load_w2v_model(self, embedding_name: str) -> Tuple[Any, int]:
        """
        Load a pre-trained Word2Vec model from gensim's API.
        """
        print(f"Loading pre-trained embeddings...")

        # Download and load pre-trained embeddings from gensim-data repository
        model_obj = api.load(embedding_name)
        embedding_dim = model_obj.vector_size

        print(f"✓ Word2Vec Embeddings loaded")
        print(f"  Vocabulary: {len(model_obj):,} words")
        print(f"  Dimensions: {embedding_dim}d")

        return model_obj, embedding_dim

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split on whitespace.
        """
        tokens = str(text).lower().split()

        # Optionally remove stop words (common words like 'the', 'is', 'and')
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]

        return tokens

    def _train_doc2vec_model(self, remaining_samples_path: str = None) -> Tuple[Doc2Vec, int]:
        """
        Train a Doc2Vec model from scratch on the remaining samples corpus.

        Doc2Vec learns document-level embeddings directly, unlike Word2Vec
        which requires aggregating word vectors.

        Training on remaining_samples.csv (instead of train_100k.csv) ensures:
        - More diverse vocabulary from larger corpus
        - Better generalization (trained on data NOT used for classifier training)
        """
        print(f"Training Doc2Vec model...")

        # Load remaining samples for Doc2Vec training
        if remaining_samples_path is None:
            remaining_samples_path = self.config.get(
                'remaining_samples_path',
                '../../dataset/splits/remaining_samples.csv'
            )

        print(f"Loading Doc2Vec training data from: {remaining_samples_path}")

        try:
            df_remaining = pd.read_csv(remaining_samples_path)
            X_remaining = df_remaining['answer']  # Text column

            print(f"✓ Loaded {len(X_remaining):,} documents for Doc2Vec training")
            print(f"  Level distribution:")

            # Show level distribution
            if 'level' in df_remaining.columns:
                level_counts = df_remaining['level'].value_counts().sort_index()
                total = len(df_remaining)
                for level, count in level_counts.items():
                    pct = (count / total) * 100
                    print(f"    {level}: {count:,} ({pct:.1f}%)")

        except FileNotFoundError:
            print(f"ERROR: remaining_samples.csv not found at {remaining_samples_path}")
            print(f"Falling back to using training data (not recommended - causes data leakage)")
            raise

        print("\nCreating tagged documents...")

        # Doc2Vec requires TaggedDocument format: words + unique document ID
        # Each document needs a unique tag for the model to learn document vectors
        tagged_data = [
            TaggedDocument(words=self._tokenize_text(text), tags=[str(i)])
            for i, text in tqdm(enumerate(X_remaining.values), total=len(X_remaining),
                                desc="Tagging Documents", leave=True)
        ]

        print(f"\nInitializing Doc2Vec model...")
        print(f"  Vector size: 300")
        print(f"  Min count: {self.doc2vec_mincount}")
        print(f"  Epochs: {self.doc2vec_epochs}")

        # Initialize Doc2Vec with 300-dimensional vectors
        # min_count=2: ignore words appearing less than 2 times
        # epochs: can be congirued
        model_obj = Doc2Vec(
            vector_size=300,
            min_count=self.doc2vec_mincount,
            epochs=self.doc2vec_epochs,
            workers=4  # Use multiple cores for faster training
        )

        # Build vocabulary from the tagged documents
        print(f"\nBuilding vocabulary...")
        model_obj.build_vocab(tagged_data)
        print(f"  Vocabulary size: {len(model_obj.wv):,} words")

        # Train the model on the corpus
        print(f"\nTraining Doc2Vec model...")
        epochs = model_obj.epochs
        for epoch in tqdm(range(epochs), desc="Training epochs", unit="epoch"):
            model_obj.train(
                tagged_data,
                total_examples=model_obj.corpus_count,
                epochs=1
            )

        print(f"\n✓ Doc2Vec model trained successfully")
        print(f"  Dimensions: 300d")
        print(f"  Vocabulary: {len(model_obj.wv):,} words")
        print(f"  Training documents: {len(X_remaining):,}")

        return model_obj, 300


    def _compute_tfidf(self, X_train: pd.Series, X_in_test: pd.Series, X_out_test: pd.Series):
        """
        Compute TF-IDF (Term Frequency-Inverse Document Frequency) weights.
        """
        print(f"\nComputing TF-IDF weights...")

        # Initialize TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english' if self.stop_words else None,
            max_features=5000
        )

        # Fit on training data to learn vocabulary and IDF weights
        tfidf_vectorizer.fit(X_train)

        # Transform all splits to TF-IDF sparse matrices, then convert to dense arrays
        # Shape: (num_documents, num_features) where each value is the TF-IDF score
        tfidf_train = tfidf_vectorizer.transform(X_train).toarray()
        tfidf_val = tfidf_vectorizer.transform(X_in_test).toarray()
        tfidf_test = tfidf_vectorizer.transform(X_out_test).toarray()

        print(f"✓ TF-IDF weights computed")

        return tfidf_vectorizer, tfidf_train, tfidf_val, tfidf_test

    def _get_word_embeddings_and_weights(self, tokens: List[str], idx: int,
                                         model_obj: Any, tfidf_vectorizer: TfidfVectorizer,
                                         tfidf_weights: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract word embeddings and their TF-IDF weights for a list of tokens.
        """
        embeddings_list = []
        weights_list = []

        # Iterate through words and collect their embeddings + weights
        for word in tokens:
            # Check if word exists in Word2Vec vocabulary
            if word in model_obj:
                # Get pre-trained word embedding
                word_emb = model_obj[word]
                embeddings_list.append(word_emb)

                # Get TF-IDF weight for this word in this document
                if tfidf_weights is not None and idx is not None:
                    try:
                        # Find the feature index for this word in TF-IDF vocabulary
                        tfidf_idx = tfidf_vectorizer.vocabulary_[word]
                        # Retrieve the TF-IDF score for this word in this document
                        weights_list.append(tfidf_weights[idx, tfidf_idx])
                    except (KeyError, IndexError):
                        # Word not in TF-IDF vocabulary, use neutral weight
                        weights_list.append(1.0)
                else:
                    # No TF-IDF weighting, use uniform weight
                    weights_list.append(1.0)

        return embeddings_list, weights_list

    def _aggregate_embeddings(self, embeddings_list: List[np.ndarray],
                              weights_list: List[float]) -> np.ndarray:
        """
        Aggregate word embeddings into a single document embedding.
        """
        # Convert list of embeddings to numpy array for aggregation
        # Shape: (num_words, embedding_dim)
        embeddings_array = np.array(embeddings_list)

        if self.aggregation == 'mean':
            # Simple average: treat all words equally
            return np.mean(embeddings_array, axis=0)
        else:  # tfidf_weighted
            # Weighted average: important words (high TF-IDF) contribute more
            weights_array = np.array(weights_list)
            # Normalize weights to sum to 1 (adding epsilon to avoid division by zero)
            weights_array = weights_array / (np.sum(weights_array) + 1e-10)
            # Compute weighted average of word embeddings
            return np.average(embeddings_array, axis=0, weights=weights_array)

    def _get_document_embedding(self, text: str, idx: int, model_obj: Any,
                                embedding_dim: int, tfidf_vectorizer: TfidfVectorizer,
                                tfidf_weights: np.ndarray) -> torch.Tensor:
        """
        Convert a text document into a fixed-length embedding vector.

        Process:
        1. For Doc2Vec: directly infer document vector
        2. For Word2Vec:
           a. Tokenize text
           b. Get word embeddings and weights for each word
           c. Aggregate them using mean or TF-IDF weighted average
        """
        # Tokenize the text
        tokens = self._tokenize_text(text)

        # Doc2Vec path: directly infer document embedding
        if self.embedding_model == 'doc2vec':
            # Doc2Vec.infer_vector generates embedding for unseen documents
            return torch.tensor(model_obj.infer_vector(tokens), dtype=torch.float)

        # Word2Vec path: aggregate word embeddings into document embedding
        # Step 1: Get word embeddings and their weights
        embeddings_list, weights_list = self._get_word_embeddings_and_weights(
            tokens, idx, model_obj, tfidf_vectorizer, tfidf_weights
        )

        # Handle case where no words have embeddings (OOV words only)
        if not embeddings_list:
            return torch.zeros(embedding_dim)

        # Step 2: Aggregate word embeddings into document embedding
        doc_embedding = self._aggregate_embeddings(embeddings_list, weights_list)

        # Convert to PyTorch tensor for neural network compatibility
        return torch.tensor(doc_embedding, dtype=torch.float)

    def _build_dataset(self, X: pd.Series, y: pd.Series, desc: str,
                       model_obj: Any, embedding_dim: int,
                       tfidf_vectorizer: TfidfVectorizer,
                       tfidf_weights: np.ndarray) -> List[Tuple[torch.Tensor, int]]:
        """
        Convert entire corpus into a dataset of (embedding, label) pairs.
        """
        dataset = []

        # Process each document with progress tracking
        for idx, (text, label) in tqdm(enumerate(zip(X.values, y.values)),
                                       total=len(X), desc=desc, leave=True):
            # Convert text to embedding vector
            doc_emb = self._get_document_embedding(
                text=text,
                idx=idx,  # Document index for TF-IDF lookup
                model_obj=model_obj,
                embedding_dim=embedding_dim,
                tfidf_vectorizer=tfidf_vectorizer,
                tfidf_weights=tfidf_weights
            )

            # Append (embedding, label) pair to dataset
            # Convert label to int for PyTorch cross-entropy loss
            dataset.append((doc_emb, int(label)))

        return dataset

    def save_model(self):

        save_path = self.config.get('output_dir')
        self.experiment_name = self.config.get('experiment_name')
        """Save all model components."""
        import os
        import json
        import joblib
        import torch

        os.makedirs(save_path, exist_ok=True)

        # 1. Save embedding model (from prepare_features)
        embedding_model = self.results.get('embedding_model')  # You'll need to store this

        if self.embedding_model == 'doc2vec':
            filename = f'{self.experiment_name}_{self.embedding_model}_epoch{self.doc2vec_epochs}_min{self.doc2vec_mincount}.bin'
            self.vectorizer.save(os.path.join(save_path, filename))
        else:  # Word2Vec
            filename = f'{self.experiment_name}_{self.embedding_model}_{self.embedding_name}_{self.aggregation}.pkl'
            joblib.dump(self.vectorizer, os.path.join(save_path, filename))

        # 2. Save PyTorch neural network
        torch.save(self.model.state_dict(), os.path.join(save_path, 'nn_weights.pth'))

        # 3. Save configuration
        config_to_save = {
            'embedding_model': self.embedding_model,
            'embedding_name': self.embedding_name,
            'aggregation': self.aggregation,
            'architecture': self.config.get('architecture', 'simple'),
            'embedding_dim': self.model.embedding_dim,
            'hidden_dim': self.model.hidden_dim,
            'dropout_rate': self.model.dropout,
            'num_classes': self.model.num_classes
        }

        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_to_save, f, indent=2)

        # 4. Save TF-IDF vectorizer if used
        if self.aggregation == 'tfidf_weighted':
            tfidf_vec = self.results.get('tfidf_vectorizer')
            if tfidf_vec:
                joblib.dump(tfidf_vec, os.path.join(save_path, 'tfidf_vectorizer.pkl'))

        print(f"✓ Model saved to {save_path}")

class PyTorchModelWrapper:
    """
    Wrapper to make PyTorch models compatible with sklearn API.

    This wrapper provides predict() and predict_proba() methods that
    the base CEFRClassifier.evaluate_model() expects, while keeping
    the underlying PyTorch model unchanged.
    """

    def __init__(self, model, device):
        """
        Initialize wrapper.
        """
        self.model = model
        self.device = device

    def predict(self, dataloader):
        """
        Make predictions on data.
        """

        self.model.eval()  # Set to evaluation mode
        predictions = []

        with torch.no_grad():  # Disable gradient computation
            for embeddings, _ in dataloader:
                embeddings = embeddings.to(self.device)
                outputs = self.model(embeddings)  # Forward pass
                _, predicted = torch.max(outputs.data, 1)  # Get class with highest score
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, dataloader):
        """
        Predict class probabilities.
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for embeddings, _ in dataloader:
                embeddings = embeddings.to(self.device)
                outputs = self.model(embeddings)  # Forward pass (logits)
                probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)
