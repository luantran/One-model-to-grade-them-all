"""
Word2Vec CEFR Classifier - Standardized Version
================================================
Corpus-based evaluation: Train on EFCamDAT, test on other corpora for generalization

Standardized function names:
- load_data()
- prepare_features()
- train_model()
- evaluate_model()
- run_experiment()
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
from src.utils.data_utils import load_dataset as load_csv
from src.utils.evaluation_utils import (
    compute_all_metrics, print_metrics, plot_confusion_matrix, analyze_errors
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
    """Feedforward neural network for CEFR classification (5 classes)."""

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
# STANDARDIZED FUNCTIONS
# ============================================================================

def load_data(train_path, test_path, val_size=0.2, random_state=42):
    """
    Load and split data into train/val/test sets.
    Uses pre-computed numeric labels from label_numeric column.

    Parameters:
    -----------
    train_path : str
        Path to training CSV (EFCamDAT)
    test_path : str
        Path to test CSV (other corpora)
    val_size : float
        Validation set proportion
    random_state : int
        Random seed

    Returns:
    --------
    dict : Data splits
    """
    log("LOADING DATA")

    # Load datasets
    df_train_full = load_csv(train_path)
    df_test = load_csv(test_path)

    print(f"Training data: {len(df_train_full):,} samples")
    print(f"Test data: {len(df_test):,} samples")

    # Extract text and numeric labels
    X_full = df_train_full['answer']
    y_full_numeric = df_train_full['label_numeric']
    y_full_labels = df_train_full['level']

    X_test = df_test['answer']
    y_test_numeric = df_test['label_numeric']
    y_test_labels = df_test['level']

    # Split train into train/val
    X_train, X_val, y_train, y_val, y_train_labels, y_val_labels = train_test_split(
        X_full, y_full_numeric, y_full_labels,
        test_size=val_size,
        random_state=random_state,
        stratify=y_full_numeric
    )

    print(f"\nSplit sizes:")
    print(f"  Training: {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test: {len(X_test):,}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test_numeric,
        'y_train_labels': y_train_labels,
        'y_val_labels': y_val_labels,
        'y_test_labels': y_test_labels,
        'df_test': df_test
    }


def prepare_features(X_train, X_val, X_test, y_train, y_val, y_test,
                     embedding_name='glove-wiki-gigaword-300', batch_size=64):
    """
    Convert text to Word2Vec features (averaged embeddings) and create dataloaders.

    Parameters:
    -----------
    X_train, X_val, X_test : Text data
    y_train, y_val, y_test : Numeric labels
    embedding_name : str
        Pre-trained embedding model name
    batch_size : int
        Batch size for dataloaders

    Returns:
    --------
    dict : Dataloaders, word2vec model, and embedding_dim
    """
    log("PREPARING FEATURES (Word2Vec Embeddings)")
    print(f"Embedding: {embedding_name}")
    print(f"Batch size: {batch_size}")
    print(f"Loading pre-trained embeddings...")

    # Load Word2Vec model
    word2vec_model = api.load(embedding_name)
    embedding_dim = word2vec_model.vector_size

    print(f"✓ Embeddings loaded")
    print(f"  Vocabulary: {len(word2vec_model):,} words")
    print(f"  Dimensions: {embedding_dim}d")

    # Helper function to get document embedding
    def get_document_embedding(text):
        embeddings_list = []
        for word in str(text).lower().split():
            if word in word2vec_model:
                word_emb = torch.tensor(word2vec_model[word], dtype=torch.float)
            else:
                word_emb = torch.zeros(embedding_dim)
            embeddings_list.append(word_emb)

        if len(embeddings_list) == 0:
            return torch.zeros(embedding_dim)
        return torch.mean(torch.stack(embeddings_list), dim=0)

    # Create datasets with embeddings
    def create_dataset(X, y, desc):
        dataset = []
        for text, label in tqdm(zip(X.values, y.values), total=len(X), desc=desc):
            doc_emb = get_document_embedding(text)
            dataset.append((doc_emb, int(label)))
        return dataset

    print("\nCreating embedded datasets...")
    train_dataset = create_dataset(X_train, y_train, "Train embeddings")
    val_dataset = create_dataset(X_val, y_val, "Val embeddings")
    test_dataset = create_dataset(X_test, y_test, "Test embeddings")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n✓ Dataloaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'word2vec_model': word2vec_model,
        'embedding_dim': embedding_dim
    }


def train_model(train_loader, embedding_dim, hidden_dim=128, epochs=10,
                learning_rate=0.001, device=None):
    """
    Train Word2Vec-based CEFR classifier.

    Parameters:
    -----------
    train_loader : DataLoader
        Training data loader
    embedding_dim : int
        Embedding dimensionality
    hidden_dim : int
        Hidden layer size
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    device : torch.device
        Device for training

    Returns:
    --------
    model : Trained model
    """
    log("TRAINING MODEL")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Architecture: Word2Vec + Feedforward NN")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output classes: 5")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")

    # Initialize model
    model = CEFRClassifier(
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
    best_val_loss = float('inf')

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
    return model


def evaluate_model(model, data_loader, y_numeric, y_labels, dataset_name="Dataset", device=None):
    """
    Evaluate model and return predictions with metrics.

    Parameters:
    -----------
    model : Trained model
    data_loader : DataLoader
        Data to evaluate
    y_numeric : True labels (numeric 0-4)
    y_labels : True labels (CEFR strings)
    dataset_name : str
        Name for logging
    device : torch.device
        Device

    Returns:
    --------
    dict : Evaluation results
    """
    log(f"EVALUATING MODEL ON {dataset_name.upper()}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for X, y in tqdm(data_loader, desc="Evaluating", leave=False):
            X = X.to(device)
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    y_pred_numeric = np.array(all_predictions)
    y_pred_proba = np.array(all_probabilities)
    y_true_numeric = y_numeric.values

    # Convert numeric predictions to CEFR labels
    label_map_inv = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}
    y_pred_labels = np.array([label_map_inv[p] for p in y_pred_numeric])

    # Calculate accuracy
    accuracy = (y_pred_numeric == y_true_numeric).mean()

    # Compute comprehensive metrics
    metrics = compute_all_metrics(y_labels, y_pred_labels, y_pred_proba)

    print(f"\n✓ {dataset_name} Accuracy: {accuracy:.4f}")
    print_metrics(metrics, train_acc=None)

    return {
        'y_true_numeric': y_true_numeric,
        'y_pred_numeric': y_pred_numeric,
        'y_true_labels': y_labels,
        'y_pred_labels': y_pred_labels,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'metrics': metrics
    }


def perform_error_analysis(X_data, y_true_labels, y_pred_labels, df_data, output_dir, dataset_name="validation"):
    """
    Perform error analysis and save misclassified samples.

    Parameters:
    -----------
    X_data : Text data
    y_true_labels : True CEFR labels
    y_pred_labels : Predicted CEFR labels
    df_data : DataFrame with full data
    output_dir : str
        Output directory
    dataset_name : str
        Name for output file

    Returns:
    --------
    str : Path to saved CSV
    """
    log(f"ERROR ANALYSIS - {dataset_name.upper()}")

    error_path = os.path.join(output_dir, f'misclassified_{dataset_name}.csv')
    analyze_errors(X_data, y_true_labels, y_pred_labels, df_data, output_path=error_path)

    print(f"✓ Error analysis saved to {error_path}")
    return error_path


def perform_per_corpus_analysis(df_test, y_test_labels, y_pred_labels, output_dir=None):
    """
    Analyze performance per corpus in test set.

    Parameters:
    -----------
    df_test : DataFrame
        Test dataframe with 'source_file' column
    y_test_labels : True CEFR labels
    y_pred_labels : Predicted CEFR labels
    output_dir : str, optional
        Output directory to save results

    Returns:
    --------
    dict : Per-corpus results
    """
    log("PER-CORPUS ANALYSIS (Test Set)")

    corpus_results = {}
    for corpus in df_test['source_file'].unique():
        corpus_mask = df_test['source_file'] == corpus
        corpus_y_true = y_test_labels[corpus_mask].values
        corpus_y_pred = y_pred_labels[corpus_mask]

        corpus_acc = (corpus_y_true == corpus_y_pred).mean()
        corpus_results[corpus] = {
            'accuracy': corpus_acc,
            'samples': len(corpus_y_true)
        }

        print(f"\n{corpus}:")
        print(f"  Samples: {len(corpus_y_true):,}")
        print(f"  Accuracy: {corpus_acc:.4f}")

    # Optionally save to CSV
    if output_dir:
        corpus_df = pd.DataFrame([
            {'corpus': corpus, 'accuracy': results['accuracy'], 'samples': results['samples']}
            for corpus, results in corpus_results.items()
        ])
        corpus_path = os.path.join(output_dir, 'per_corpus_results.csv')
        corpus_df.to_csv(corpus_path, index=False)
        print(f"\n✓ Per-corpus results saved to {corpus_path}")

    return corpus_results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(train_path, test_path, output_dir='../results',
                   embedding_name='glove-wiki-gigaword-300',
                   batch_size=64, hidden_dim=128, epochs=10,
                   learning_rate=0.001, val_size=0.2, random_state=42):
    """
    Run complete Word2Vec CEFR classification experiment.

    Parameters:
    -----------
    train_path : str
        Path to training CSV
    test_path : str
        Path to test CSV
    output_dir : str
        Output directory
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
        Validation proportion
    random_state : int
        Random seed

    Returns:
    --------
    dict : Complete results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Step 1: Load data
    data = load_data(train_path, test_path, val_size, random_state)

    # Step 2: Prepare features (embeddings + dataloaders)
    features = prepare_features(
        data['X_train'], data['X_val'], data['X_test'],
        data['y_train'], data['y_val'], data['y_test'],
        embedding_name=embedding_name,
        batch_size=batch_size
    )

    # Step 3: Train model
    model = train_model(
        features['train_loader'],
        embedding_dim=features['embedding_dim'],
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device
    )

    # Step 4: Evaluate on validation set (in-domain)
    val_results = evaluate_model(
        model,
        features['val_loader'],
        data['y_val'],
        data['y_val_labels'],
        dataset_name="Validation (In-Domain)",
        device=device
    )

    # Step 5: Evaluate on test set (out-of-domain)
    test_results = evaluate_model(
        model,
        features['test_loader'],
        data['y_test'],
        data['y_test_labels'],
        dataset_name="Test (Out-of-Domain)",
        device=device
    )

    # Step 6: Calculate generalization gap
    generalization_gap = val_results['accuracy'] - test_results['accuracy']
    print(f"\n✓ Generalization Gap: {generalization_gap:.4f} ({generalization_gap * 100:.2f}%)")

    # Step 7: Visualizations
    log("GENERATING VISUALIZATIONS")

    val_cm_path = os.path.join(output_dir, 'confusion_matrix_validation.png')
    plot_confusion_matrix(
        data['y_val_labels'],
        val_results['y_pred_labels'],
        output_path=val_cm_path,
        title=f'Confusion Matrix - Validation (Word2Vec {embedding_name})'
    )

    test_cm_path = os.path.join(output_dir, 'confusion_matrix_test.png')
    plot_confusion_matrix(
        data['y_test_labels'],
        test_results['y_pred_labels'],
        output_path=test_cm_path,
        title=f'Confusion Matrix - Test (Word2Vec {embedding_name})'
    )

    # Step 8: Error analysis
    log("ERROR ANALYSIS")

    val_error_path = os.path.join(output_dir, 'misclassified_validation.csv')
    df_val_analysis = pd.DataFrame({
        'answer': data['X_val'].values,
        'level': data['y_val_labels'].values
    })
    analyze_errors(
        data['X_val'], data['y_val_labels'], val_results['y_pred_labels'],
        df_val_analysis, output_path=val_error_path
    )

    test_error_path = os.path.join(output_dir, 'misclassified_test.csv')
    analyze_errors(
        data['X_test'], data['y_test_labels'], test_results['y_pred_labels'],
        data['df_test'], output_path=test_error_path
    )

    # Step 9: Per-corpus analysis
    log("PER-CORPUS ANALYSIS (Test Set)")

    corpus_results = {}
    for corpus in data['df_test']['source_file'].unique():
        corpus_mask = data['df_test']['source_file'] == corpus
        corpus_y_true = data['y_test_labels'][corpus_mask].values
        corpus_y_pred = test_results['y_pred_labels'][corpus_mask]

        corpus_acc = (corpus_y_true == corpus_y_pred).mean()
        corpus_results[corpus] = {
            'accuracy': corpus_acc,
            'samples': len(corpus_y_true)
        }

        print(f"\n{corpus}:")
        print(f"  Samples: {len(corpus_y_true):,}")
        print(f"  Accuracy: {corpus_acc:.4f}")

    # Step 10: Save summary
    log("SAVING RESULTS")

    summary_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT: WORD2VEC CEFR - CORPUS-BASED EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Embedding: {embedding_name}\n")
        f.write(f"  Embedding dim: {features['embedding_dim']}\n")
        f.write(f"  Hidden dim: {hidden_dim}\n")
        f.write(f"  Epochs: {epochs}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Learning rate: {learning_rate}\n\n")

        f.write("=" * 70 + "\n")
        f.write("VALIDATION RESULTS (In-Domain)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy: {val_results['accuracy']:.4f}\n")
        f.write(f"Adjacent Accuracy: {val_results['metrics']['adjacent_accuracy']:.4f}\n")
        f.write(f"MAE: {val_results['metrics']['mae']:.4f}\n")
        f.write(f"QWK: {val_results['metrics']['qwk']:.4f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("TEST RESULTS (Out-of-Domain)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"Adjacent Accuracy: {test_results['metrics']['adjacent_accuracy']:.4f}\n")
        f.write(f"MAE: {test_results['metrics']['mae']:.4f}\n")
        f.write(f"QWK: {test_results['metrics']['qwk']:.4f}\n\n")

        f.write("=" * 70 + "\n")
        f.write("GENERALIZATION ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generalization Gap: {generalization_gap:.4f}\n\n")

        f.write("Per-Corpus Results:\n")
        for corpus, results in sorted(corpus_results.items()):
            f.write(f"  {corpus}: {results['accuracy']:.4f} ({results['samples']:,} samples)\n")

    print(f"✓ Summary saved to {summary_path}")

    # Return complete results
    return {
        'model': model,
        'word2vec_model': features['word2vec_model'],
        'validation': val_results,
        'test': test_results,
        'corpus_results': corpus_results,
        'generalization_gap': generalization_gap,
        'config': {
            'embedding_name': embedding_name,
            'embedding_dim': features['embedding_dim'],
            'hidden_dim': hidden_dim,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    samples = "100k"
    TRAIN_PATH = f'../dataset/splits/train_{samples}.csv'
    TEST_PATH = '../../dataset/splits/test_other_corpora.csv'

    EMBEDDING_NAME = 'glove-wiki-gigaword-300'  # or 'word2vec-google-news-300'
    BATCH_SIZE = 64
    HIDDEN_DIM = 128
    EPOCHS = 10
    LEARNING_RATE = 0.001

    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    OUTPUT_DIR = f'../results/Word2Vec_corpus_{samples}_{EMBEDDING_NAME.replace("-", "_")}_EPOCH_{EPOCHS}_HD_{HIDDEN_DIM}'

    print("=" * 80)
    print("WORD2VEC - CORPUS-BASED GENERALIZATION EXPERIMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Embedding: {EMBEDDING_NAME}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("=" * 80)

    results = run_experiment(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        embedding_name=EMBEDDING_NAME,
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