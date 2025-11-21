"""
CEFR Level Classifier using Pre-trained Word2Vec Embeddings
Corpus-based evaluation: Train on EFCamDAT, test on other corpora for generalization
Adapted from sentiment analysis approach for ordinal CEFR classification (A1-C2)

Key approach:
1. Load pre-trained Word2Vec (Google News or GloVe)
2. Convert documents to embeddings by averaging word vectors
3. Train simple feedforward classifier on averaged embeddings
4. Evaluate with comprehensive metrics on in-domain and out-of-domain data
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import gensim.downloader as api
import warnings

# Import common utilities
from src.utils.data_utils import load_dataset
from src.utils.evaluation_utils import (
    plot_confusion_matrix, analyze_errors
)

warnings.filterwarnings('ignore')


def log(message):
    """Print a formatted log message."""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class CEFRClassifier(nn.Module):
    """
    Feedforward neural network for CEFR level classification.
    5-class ordinal classification (A1, A2, B1, B2, C1/C2).

    Architecture:
    - Input: Document embedding (averaged word vectors)
    - FC layer: embedding_dim → hidden_dim
    - ReLU activation
    - Dropout (0.3)
    - FC layer: hidden_dim → 5 classes
    """
    def __init__(self, embedding_dim, hidden_dim=128, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================================
# EMBEDDING UTILITIES
# ============================================================================

def load_pretrained_embeddings(embedding_name='glove-wiki-gigaword-300'):
    """
    Load pre-trained embeddings using gensim downloader.

    Parameters:
    -----------
    embedding_name : str
        Pre-trained embedding model name
        Options:
        - 'word2vec-google-news-300'
        - 'glove-wiki-gigaword-300'
        - 'fasttext-wiki-news-subwords-300'

    Returns:
    --------
    KeyedVectors : Loaded pre-trained embeddings
    """
    log("LOADING PRE-TRAINED EMBEDDINGS")
    print(f"Model: {embedding_name}")
    print("Downloading/loading... (first run will download and cache)")

    word2vec_model = api.load(embedding_name)

    print(f"✓ Embeddings loaded successfully")
    print(f"  Vocabulary: {len(word2vec_model):,} words")
    print(f"  Dimensions: {word2vec_model.vector_size}d")

    return word2vec_model


def get_document_embedding(text, word_to_embedding, embedding_dim=300):
    """
    Compute document embedding by averaging word embeddings.

    Parameters:
    -----------
    text : str
        Input text document
    word_to_embedding : KeyedVectors or dict
        Pre-trained embeddings
    embedding_dim : int
        Embedding dimensionality

    Returns:
    --------
    torch.Tensor : Document embedding vector
    """
    embeddings_list = []

    # Get embedding dimension
    if hasattr(word_to_embedding, 'vector_size'):
        embedding_dim = word_to_embedding.vector_size

    # Process each word
    for word in str(text).lower().split():
        if word in word_to_embedding:
            word_emb = word_to_embedding[word]
            # Convert to tensor if needed (for gensim KeyedVectors)
            if not torch.is_tensor(word_emb):
                word_emb = torch.tensor(word_emb, dtype=torch.float)
        else:
            # OOV word - use zero vector
            word_emb = torch.zeros(embedding_dim)

        embeddings_list.append(word_emb)

    # Average all word embeddings
    if len(embeddings_list) == 0:
        return torch.zeros(embedding_dim)

    document_embedding = torch.mean(torch.stack(embeddings_list), dim=0)
    return document_embedding


def create_dataloader_cefr(data, labels, word_to_embedding, batch_size=64, desc="Processing"):
    """
    Create PyTorch DataLoader from raw CEFR texts.

    Parameters:
    -----------
    data : pd.Series or list
        Texts
    labels : pd.Series or list
        CEFR labels
    word_to_embedding : KeyedVectors or dict
        Pre-trained embeddings
    batch_size : int
        Batch size
    desc : str
        Description for progress bar

    Returns:
    --------
    DataLoader : PyTorch DataLoader
    """
    # Label mapping: CEFR → numeric (5 classes with merged C1/C2)
    label_map = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1/C2': 4}

    # Create dataset
    dataset = []

    for i, text in enumerate(tqdm(data, desc=desc)):
        doc_emb = get_document_embedding(text, word_to_embedding)
        label_str = labels.iloc[i] if hasattr(labels, 'iloc') else labels[i]
        label_numeric = label_map.get(label_str, 4)  # Default to C1/C2 if unknown
        dataset.append((doc_emb, label_numeric))

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_cefr_model(model, train_loader, optimizer, loss_function, device):
    """
    Train CEFR classifier for one epoch.

    Parameters:
    -----------
    model : nn.Module
        CEFR classifier
    train_loader : DataLoader
        Training data
    optimizer : torch.optim.Optimizer
        Optimizer
    loss_function : nn.Module
        Loss function
    device : torch.device
        Device

    Returns:
    --------
    float : Average training loss
    """
    model.train()
    total_loss = 0

    for X, y in tqdm(train_loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        log_probs = model(X)
        loss = loss_function(log_probs, y)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)


def evaluate_cefr_model(model, val_loader, device):
    """
    Evaluate CEFR classifier.

    Parameters:
    -----------
    model : nn.Module
        CEFR classifier
    val_loader : DataLoader
        Validation/test data
    device : torch.device
        Device

    Returns:
    --------
    tuple : (accuracy, predictions, true_labels, probabilities)
    """
    model.eval()
    correct, total = 0, 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)

            out = model(X)
            probs = F.softmax(out, dim=1)
            preds = probs.argmax(dim=1)

            correct += (preds == y).sum().item()
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            total += y.size(0)

    acc = correct / total
    return acc, all_predictions, all_labels, all_probabilities


# ============================================================================
# SKLEARN-COMPATIBLE WRAPPER
# ============================================================================

class CEFRModelWrapper:
    """Wrapper to make PyTorch model compatible with sklearn evaluation utilities."""

    def __init__(self, model, device, label_map_inv):
        self.model = model
        self.device = device
        self.label_map_inv = label_map_inv
        self._predictions_cache = {}
        self._proba_cache = {}

    def predict(self, dataloader):
        """Predict CEFR labels."""
        cache_key = id(dataloader)

        if cache_key not in self._predictions_cache:
            _, pred_numeric, _, _ = evaluate_cefr_model(self.model, dataloader, self.device)
            self._predictions_cache[cache_key] = pred_numeric

        predictions = self._predictions_cache[cache_key]

        # Convert numeric to CEFR labels
        return np.array([self.label_map_inv[p] for p in predictions])

    def predict_proba(self, dataloader):
        """Predict class probabilities."""
        cache_key = id(dataloader)

        if cache_key not in self._proba_cache:
            _, _, _, probs = evaluate_cefr_model(self.model, dataloader, self.device)
            self._proba_cache[cache_key] = probs

        return np.array(self._proba_cache[cache_key])


# ============================================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================================

def run_experiment(train_path, test_path, output_dir='../results',
                   embedding_name='glove-wiki-gigaword-300',
                   batch_size=64, hidden_dim=128, epochs=5,
                   learning_rate=0.001, val_size=0.2, random_state=42):
    """
    Run complete CEFR classification experiment with corpus-based evaluation.

    Parameters:
    -----------
    train_path : str
        Path to training dataset CSV (EFCamDAT samples)
    test_path : str
        Path to test dataset CSV (other corpora)
    output_dir : str
        Results directory
    embedding_name : str
        Pre-trained embedding name
    batch_size : int
        Batch size
    hidden_dim : int
        Hidden layer size
    epochs : int
        Training epochs
    learning_rate : float
        Learning rate
    val_size : float
        Validation set proportion (from training data)
    random_state : int
        Random seed

    Returns:
    --------
    dict : Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ===================================================================
    # STEP 1: Load training data (EFCamDAT)
    # ===================================================================
    log("LOADING TRAINING DATA (EFCamDAT)")
    df_train_full = load_dataset(train_path)
    print(f"Total training samples: {len(df_train_full):,}")
    print(f"Level distribution:")
    for level, count in df_train_full['level'].value_counts().sort_index().items():
        pct = (count / len(df_train_full)) * 100
        print(f"  {level}: {count:,} ({pct:.1f}%)")

    # ===================================================================
    # STEP 2: Split training data into train/val
    # ===================================================================
    log("SPLITTING TRAINING DATA")
    print(f"Validation size: {val_size * 100}%")

    X_full = df_train_full['answer']
    y_full = df_train_full['level']

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_full
    )

    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")

    # ===================================================================
    # STEP 3: Load test data (other corpora)
    # ===================================================================
    log("LOADING TEST DATA (Other Corpora)")
    df_test = load_dataset(test_path)

    X_test = df_test['answer']
    y_test = df_test['level']

    print(f"Test set: {len(X_test):,} samples")
    print(f"Test corpora: {list(df_test['source_file'].unique())}")

    # ===================================================================
    # STEP 4: Load pre-trained Word2Vec
    # ===================================================================
    word2vec_model = load_pretrained_embeddings(embedding_name)

    # ===================================================================
    # STEP 5: Create dataloaders
    # ===================================================================
    log("CREATING DATALOADERS")
    print(f"Batch size: {batch_size}")

    train_dataloader = create_dataloader_cefr(
        X_train, y_train, word2vec_model, batch_size=batch_size, desc="Train data"
    )

    val_dataloader = create_dataloader_cefr(
        X_val, y_val, word2vec_model, batch_size=batch_size, desc="Val data"
    )

    test_dataloader = create_dataloader_cefr(
        X_test, y_test, word2vec_model, batch_size=batch_size, desc="Test data"
    )

    print(f"✓ Dataloaders created")
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Val batches: {len(val_dataloader)}")
    print(f"  Test batches: {len(test_dataloader)}")

    # ===================================================================
    # STEP 6: Initialize model
    # ===================================================================
    log("INITIALIZING MODEL")

    EMB_DIM = word2vec_model.vector_size
    NUM_CLASSES = 5  # A1, A2, B1, B2, C1/C2

    print(f"Embedding dimension: {EMB_DIM}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of classes: {NUM_CLASSES}")

    model = CEFRClassifier(
        embedding_dim=EMB_DIM,
        hidden_dim=hidden_dim,
        num_classes=NUM_CLASSES
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ===================================================================
    # STEP 7: Train model
    # ===================================================================
    log("TRAINING MODEL")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")

    best_val_acc = 0
    for epoch in range(epochs):
        train_loss = train_cefr_model(model, train_dataloader, optimizer, loss_function, device)
        val_acc, _, _, _ = evaluate_cefr_model(model, val_dataloader, device)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"✓ Training complete")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # ===================================================================
    # STEP 8: Evaluate on validation set (in-domain)
    # ===================================================================
    log("EVALUATION ON VALIDATION SET (In-Domain EFCamDAT)")

    # Get predictions
    val_acc, val_pred_numeric, val_true_numeric, val_probs = evaluate_cefr_model(
        model, val_dataloader, device
    )

    # Convert numeric back to labels
    label_map_inv = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}
    val_pred_labels = np.array([label_map_inv[p] for p in val_pred_numeric])

    # Evaluate with common utilities (create wrapper)
    model_wrapper = CEFRModelWrapper(model, device, label_map_inv)

    # Use wrapper for evaluation (pass dataloaders as dummy X)
    from src.utils.evaluation_utils import compute_all_metrics, print_metrics

    val_metrics = compute_all_metrics(y_val, val_pred_labels, np.array(val_probs))
    print(f"\n✓ Validation Accuracy: {val_acc:.4f}")
    print_metrics(val_metrics, train_acc=None)

    # ===================================================================
    # STEP 9: Evaluate on test set (out-of-domain)
    # ===================================================================
    log("EVALUATION ON TEST SET (Out-of-Domain Other Corpora)")

    # Get predictions
    test_acc, test_pred_numeric, test_true_numeric, test_probs = evaluate_cefr_model(
        model, test_dataloader, device
    )

    # Convert numeric back to labels
    test_pred_labels = np.array([label_map_inv[p] for p in test_pred_numeric])

    # Evaluate
    test_metrics = compute_all_metrics(y_test, test_pred_labels, np.array(test_probs))
    print(f"\n✓ Test Accuracy: {test_acc:.4f}")
    print_metrics(test_metrics, train_acc=None)

    # Calculate generalization gap
    generalization_gap = val_acc - test_acc
    print(f"\n✓ Generalization Gap: {generalization_gap:.4f} ({generalization_gap*100:.2f}%)")

    # ===================================================================
    # STEP 10: Generate visualizations
    # ===================================================================
    log("GENERATING VISUALIZATIONS")

    # Validation confusion matrix
    val_cm_path = os.path.join(output_dir, 'confusion_matrix_validation.png')
    plot_confusion_matrix(y_val, val_pred_labels, output_path=val_cm_path,
                          title=f'Confusion Matrix - Validation ({embedding_name})')

    # Test confusion matrix
    test_cm_path = os.path.join(output_dir, 'confusion_matrix_test.png')
    plot_confusion_matrix(y_test, test_pred_labels, output_path=test_cm_path,
                          title=f'Confusion Matrix - Test ({embedding_name})')

    # ===================================================================
    # STEP 11: Error analysis
    # ===================================================================
    log("ERROR ANALYSIS")

    # Validation errors
    val_error_path = os.path.join(output_dir, 'misclassified_validation.csv')
    df_val_for_analysis = pd.DataFrame({
        'answer': X_val.values,
        'level': y_val.values
    })
    val_error_df = analyze_errors(X_val, y_val, val_pred_labels,
                                   df_val_for_analysis, output_path=val_error_path)

    # Test errors
    test_error_path = os.path.join(output_dir, 'misclassified_test.csv')
    df_test_for_analysis = df_test[['answer', 'level', 'source_file']].copy()
    test_error_df = analyze_errors(X_test, y_test, test_pred_labels,
                                    df_test_for_analysis, output_path=test_error_path)

    # ===================================================================
    # STEP 12: Per-corpus analysis
    # ===================================================================
    log("PER-CORPUS ANALYSIS (Test Set)")

    corpus_results = {}
    for corpus in df_test['source_file'].unique():
        corpus_mask = df_test['source_file'] == corpus
        corpus_y_true = y_test[corpus_mask].values
        corpus_y_pred = test_pred_labels[corpus_mask]

        corpus_acc = (corpus_y_true == corpus_y_pred).mean()
        corpus_results[corpus] = {
            'accuracy': corpus_acc,
            'samples': len(corpus_y_true)
        }

        print(f"\n{corpus}:")
        print(f"  Samples: {len(corpus_y_true):,}")
        print(f"  Accuracy: {corpus_acc:.4f}")

    # ===================================================================
    # STEP 13: Save comprehensive results
    # ===================================================================
    log("SAVING RESULTS")

    summary_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT: WORD2VEC CEFR - CORPUS-BASED EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASETS:\n")
        f.write(f"  Training: {train_path}\n")
        f.write(f"  Test: {test_path}\n\n")

        f.write("DATA SPLITS:\n")
        f.write(f"  Training: {len(X_train):,} samples (EFCamDAT)\n")
        f.write(f"  Validation: {len(X_val):,} samples (EFCamDAT held-out)\n")
        f.write(f"  Test: {len(X_test):,} samples (Other corpora)\n\n")

        f.write("PRE-TRAINED EMBEDDINGS:\n")
        f.write(f"  Model: {embedding_name}\n")
        f.write(f"  Vocabulary: {len(word2vec_model):,} words\n")
        f.write(f"  Dimensions: {EMB_DIM}d\n\n")

        f.write("MODEL PARAMETERS:\n")
        f.write(f"  Hidden dimension: {hidden_dim}\n")
        f.write(f"  Dropout: 0.3\n")
        f.write(f"  Epochs: {epochs}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Optimizer: Adam\n")
        f.write(f"  Loss: CrossEntropyLoss\n\n")

        f.write("=" * 70 + "\n")
        f.write("VALIDATION RESULTS (In-Domain EFCamDAT)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Accuracy: {val_acc:.4f}\n")
        f.write(f"Adjacent Accuracy (±1): {val_metrics['adjacent_accuracy']:.4f}\n")
        f.write(f"MAE: {val_metrics['mae']:.4f}\n")
        f.write(f"QWK: {val_metrics['qwk']:.4f}\n")
        f.write(f"OCA: {val_metrics['oca']:.4f}\n")
        f.write(f"EMD: {val_metrics['emd']:.4f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("TEST RESULTS (Out-of-Domain Other Corpora)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Adjacent Accuracy (±1): {test_metrics['adjacent_accuracy']:.4f}\n")
        f.write(f"MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"QWK: {test_metrics['qwk']:.4f}\n")
        f.write(f"OCA: {test_metrics['oca']:.4f}\n")
        f.write(f"EMD: {test_metrics['emd']:.4f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("GENERALIZATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Generalization Gap: {generalization_gap:.4f}\n\n")

        f.write("Per-Corpus Results:\n")
        for corpus, results in sorted(corpus_results.items()):
            f.write(f"  {corpus}: {results['accuracy']:.4f} ({results['samples']:,} samples)\n")

    print(f"\n✓ Summary saved to: {summary_path}")

    # ===================================================================
    # STEP 14: Return results
    # ===================================================================
    results = {
        'model': model,
        'word2vec_model': word2vec_model,
        'validation': {
            'y_true': y_val,
            'y_pred': val_pred_labels,
            'accuracy': val_acc,
            'metrics': val_metrics
        },
        'test': {
            'y_true': y_test,
            'y_pred': test_pred_labels,
            'accuracy': test_acc,
            'metrics': test_metrics,
            'corpus_results': corpus_results
        },
        'generalization_gap': generalization_gap,
        'parameters': {
            'embedding_name': embedding_name,
            'embedding_dim': EMB_DIM,
            'hidden_dim': hidden_dim,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }

    log("EXPERIMENT COMPLETE")
    print(f"\nKey Results:")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Generalization Gap: {generalization_gap:.4f}")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # ========== CONFIGURATION ==========
    samples = "150k"
    # Data paths
    TRAIN_PATH = f'../dataset/splits/train_{samples}.csv'
    TEST_PATH = '../../dataset/splits/test_other_corpora.csv'

    # Choose embedding:
    # EMBEDDING = 'glove-wiki-gigaword-300'  # Recommended for CEFR
    EMBEDDING = 'word2vec-google-news-300'  # Alternative
    # EMBEDDING = 'fasttext-wiki-news-subwords-300'  # Good for OOV

    # Model parameters
    BATCH_SIZE = 64
    HIDDEN_DIM = 128
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Training parameters
    VAL_SIZE = 0.2  # 20% of training data for validation
    RANDOM_STATE = 42

    # Output directory
    OUTPUT_DIR = f'../results/Word2Vec_corpus_{samples}_{EMBEDDING.replace("-", "_")}_EPOCH_{EPOCHS}_HIDDENDIM_{HIDDEN_DIM}'

    # ========== RUN EXPERIMENT ==========

    print("=" * 80)
    print("WORD2VEC - CORPUS-BASED GENERALIZATION EXPERIMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Training data: {TRAIN_PATH}")
    print(f"  Test data: {TEST_PATH}")
    print(f"  Embedding: {EMBEDDING}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 80)

    # Run experiment
    results = run_experiment(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        embedding_name=EMBEDDING,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED SUCCESSFULLY")
    print("=" * 80)