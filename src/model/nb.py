"""
Naive Bayes experiment using corpus-based train/test split.
Trains on EFCamDAT, tests on held-out EFCamDAT + other corpora for generalization.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import warnings

# Import common utilities
from src.utils.data_utils import load_dataset
from src.utils.evaluation_utils import (
    evaluate_model, plot_confusion_matrix, analyze_errors,
    compute_all_metrics, print_metrics
)

warnings.filterwarnings('ignore')



def log(message):
    """Print a formatted log message."""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)


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
    dict : Data splits with keys:
        - X_train, X_val, X_test (text)
        - y_train, y_val, y_test (numeric labels 0-4)
        - y_train_labels, y_val_labels, y_test_labels (CEFR labels)
        - df_test (full test dataframe for per-corpus analysis)
    """
    log("LOADING DATA")

    # Load datasets
    df_train_full = load_dataset(train_path)
    df_test = load_dataset(test_path)

    print(f"Training data: {len(df_train_full):,} samples")
    print(f"Test data: {len(df_test):,} samples")

    # Extract text and numeric labels
    X_full = df_train_full['answer']
    y_full_numeric = df_train_full['label_numeric']  # Use pre-computed numeric labels
    y_full_labels = df_train_full['level']  # Keep CEFR labels for display

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

    # Show label distribution
    print(f"\nLabel distribution (training):")
    label_map_inv = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}
    for numeric in sorted(y_train.unique()):
        count = (y_train == numeric).sum()
        cefr = label_map_inv[numeric]
        print(f"  {numeric} ({cefr}): {count:,}")

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


def prepare_features(X_train, X_val, X_test, method='tfidf', max_features=5000, ngram_range=(1, 2)):
    """
    Convert text to numerical features using bag-of-words.

    Parameters:
    -----------
    X_train, X_val, X_test : Text data
    method : str
        'count' or 'tfidf'
    max_features : int
        Maximum number of features
    ngram_range : tuple
        N-gram range (e.g., (1,2) for unigrams + bigrams)

    Returns:
    --------
    dict : Feature matrices and vectorizer
        - X_train_features, X_val_features, X_test_features
        - vectorizer
    """
    log("PREPARING FEATURES")
    print(f"Method: {method}")
    print(f"Max features: {max_features}")
    print(f"N-gram range: {ngram_range}")

    # Create vectorizer
    if method == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
    else:
        raise ValueError("method must be 'count' or 'tfidf'")

    # Fit on training data and transform all sets
    X_train_features = vectorizer.fit_transform(X_train)
    X_val_features = vectorizer.transform(X_val)
    X_test_features = vectorizer.transform(X_test)

    print(f"\nFeature shapes:")
    print(f"  Training: {X_train_features.shape}")
    print(f"  Validation: {X_val_features.shape}")
    print(f"  Test: {X_test_features.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

    return {
        'X_train_features': X_train_features,
        'X_val_features': X_val_features,
        'X_test_features': X_test_features,
        'vectorizer': vectorizer
    }


def train_model(X_train, y_train, alpha=1.0):
    """
    Train Multinomial Naive Bayes classifier.

    Parameters:
    -----------
    X_train : Training features
    y_train : Training labels (numeric 0-4)
    alpha : float
        Smoothing parameter

    Returns:
    --------
    model : Trained classifier
    """
    log("TRAINING MODEL")
    print(f"Algorithm: Multinomial Naive Bayes")
    print(f"Smoothing (alpha): {alpha}")

    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    print(f"✓ Model trained")
    print(f"  Classes: {model.classes_}")

    return model


def evaluate_model(model, X, y_numeric, y_labels, dataset_name="Dataset"):
    """
    Evaluate model and return predictions with metrics.

    Parameters:
    -----------
    model : Trained model
    X : Features
    y_numeric : True labels (numeric 0-4)
    y_labels : True labels (CEFR strings)
    dataset_name : str
        Name for logging

    Returns:
    --------
    dict : Evaluation results
        - y_true_numeric, y_pred_numeric
        - y_true_labels, y_pred_labels
        - accuracy, metrics
    """
    log(f"EVALUATING MODEL ON {dataset_name.upper()}")

    # Predict
    y_pred_numeric = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Convert numeric predictions back to CEFR labels
    label_map_inv = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1/C2'}
    y_pred_labels = np.array([label_map_inv[p] for p in y_pred_numeric])

    # Calculate accuracy
    accuracy = (y_pred_numeric == y_numeric).mean()

    # Compute comprehensive metrics
    metrics = compute_all_metrics(y_labels, y_pred_labels, y_pred_proba)

    print(f"\n✓ {dataset_name} Accuracy: {accuracy:.4f}")
    print_metrics(metrics, train_acc=None)

    return {
        'y_true_numeric': y_numeric,
        'y_pred_numeric': y_pred_numeric,
        'y_true_labels': y_labels,
        'y_pred_labels': y_pred_labels,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'metrics': metrics
    }


def vectorize_text(X_train, X_val, X_test, method='count', max_features=5000, ngram_range=(1, 1)):
    """
    Convert text to numerical features using bag of words.

    Parameters:
    -----------
    X_train : Training text data
    X_val : Validation text data
    X_test : Test text data (other corpora)
    method : str
        'count' for CountVectorizer or 'tfidf' for TfidfVectorizer
    max_features : int
        Maximum number of features to extract
    ngram_range : tuple
        Range of n-grams (e.g., (1,1) for unigrams, (1,2) for uni+bigrams)

    Returns:
    --------
    X_train_vec, X_val_vec, X_test_vec, vectorizer
    """
    log("VECTORIZING TEXT")
    print(f"Method: {method}")
    print(f"Max features: {max_features}")
    print(f"N-gram range: {ngram_range}")

    if method == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
    else:
        raise ValueError("method must be 'count' or 'tfidf'")

    # Fit on training data only
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    print(f"\nTraining features shape: {X_train_vec.shape}")
    print(f"Validation features shape: {X_val_vec.shape}")
    print(f"Test features shape: {X_test_vec.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return X_train_vec, X_val_vec, X_test_vec, vectorizer


def train_naive_bayes(X_train, y_train, alpha=1.0):
    """
    Train Multinomial Naive Bayes classifier.

    Parameters:
    -----------
    X_train : Training features
    y_train : Training labels
    alpha : float
        Smoothing parameter

    Returns:
    --------
    model : Trained classifier
    """
    log("TRAINING MULTINOMIAL NAIVE BAYES")
    print(f"Alpha (smoothing): {alpha}")
    print(f"Training shape: {X_train.shape}")

    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    print("✓ Training complete")

    return model


def run_experiment(train_path, test_path, output_dir='../results',
                   vectorizer_method='count', max_features=5000,
                   ngram_range=(1, 1), val_size=0.2, alpha=1.0,
                   random_state=42):
    """
    Run complete Naive Bayes experiment pipeline with corpus-based evaluation.

    Parameters:
    -----------
    train_path : str
        Path to training dataset CSV (EFCamDAT samples)
    test_path : str
        Path to test dataset CSV (other corpora)
    output_dir : str
        Directory to save results
    vectorizer_method : str
        'count' or 'tfidf'
    max_features : int
        Maximum number of features
    ngram_range : tuple
        N-gram range
    val_size : float
        Validation set proportion (from training data)
    alpha : float
        Naive Bayes smoothing parameter
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict : Results dictionary containing model, predictions, and metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # ===================================================================
    # STEP 1: Load training data (EFCamDAT)
    # ===================================================================
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
    print(f"Training distribution:")
    for level, count in pd.Series(y_train).value_counts().sort_index().items():
        pct = (count / len(y_train)) * 100
        print(f"  {level}: {count:,} ({pct:.1f}%)")

    print(f"\nValidation set: {len(X_val):,} samples")
    print(f"Validation distribution:")
    for level, count in pd.Series(y_val).value_counts().sort_index().items():
        pct = (count / len(y_val)) * 100
        print(f"  {level}: {count:,} ({pct:.1f}%)")

    # ===================================================================
    # STEP 3: Load test data (other corpora)
    # ===================================================================
    log("LOADING TEST DATA (Other Corpora)")
    df_test = load_dataset(test_path)

    X_test = df_test['answer']
    y_test = df_test['level']

    print(f"Test set: {len(X_test):,} samples")
    print(f"Test corpora: {list(df_test['source_file'].unique())}")
    print(f"\nTest distribution:")
    for level, count in y_test.value_counts().sort_index().items():
        pct = (count / len(y_test)) * 100
        print(f"  {level}: {count:,} ({pct:.1f}%)")

    # ===================================================================
    # STEP 4: Feature extraction
    # ===================================================================
    X_train_vec, X_val_vec, X_test_vec, vectorizer = vectorize_text(
        X_train, X_val, X_test,
        method=vectorizer_method,
        max_features=max_features,
        ngram_range=ngram_range
    )

    # ===================================================================
    # STEP 5: Train model
    # ===================================================================
    model = train_naive_bayes(X_train_vec, y_train, alpha=alpha)

    # ===================================================================
    # STEP 6: Evaluate on validation set (in-domain)
    # ===================================================================
    y_val_pred, train_acc_val, val_acc, val_metrics, val_report = evaluate_model(
        model, X_train_vec, X_val_vec, y_train, y_val,
        has_predict_proba=True
    )

    print(f"\n✓ Validation Accuracy: {val_acc:.4f}")
    print(f"✓ Adjacent Accuracy (±1): {val_metrics['adjacent_accuracy']:.4f}")
    print(f"✓ MAE: {val_metrics['mae']:.4f} levels")
    print(f"✓ Quadratic Weighted Kappa: {val_metrics['qwk']:.4f}")

    # ===================================================================
    # STEP 7: Evaluate on test set (out-of-domain)
    # ===================================================================
    y_test_pred, _, test_acc, test_metrics, test_report = evaluate_model(
        model, X_train_vec, X_test_vec, y_train, y_test,
        has_predict_proba=True
    )

    print(f"\n✓ Test Accuracy: {test_acc:.4f}")
    print(f"✓ Adjacent Accuracy (±1): {test_metrics['adjacent_accuracy']:.4f}")
    print(f"✓ MAE: {test_metrics['mae']:.4f} levels")
    print(f"✓ Quadratic Weighted Kappa: {test_metrics['qwk']:.4f}")

    # Calculate generalization gap
    generalization_gap = val_acc - test_acc
    print(f"\n✓ Generalization Gap: {generalization_gap:.4f} ({generalization_gap*100:.2f}%)")

    # ===================================================================
    # STEP 8: Generate visualizations
    # ===================================================================
    # Validation confusion matrix
    val_cm_path = os.path.join(output_dir, 'confusion_matrix_validation.png')
    plot_confusion_matrix(y_val, y_val_pred, output_path=val_cm_path,
                          title='Confusion Matrix - Validation (EFCamDAT)')

    # Test confusion matrix
    test_cm_path = os.path.join(output_dir, 'confusion_matrix_test.png')
    plot_confusion_matrix(y_test, y_test_pred, output_path=test_cm_path,
                          title='Confusion Matrix - Test (Other Corpora)')

    # ===================================================================
    # STEP 9: Error analysis
    # ===================================================================

    # Validation errors
    val_error_path = os.path.join(output_dir, 'misclassified_validation.csv')
    df_val_for_analysis = pd.DataFrame({
        'answer': X_val.values,
        'level': y_val.values
    })
    val_error_df = analyze_errors(X_val, y_val, y_val_pred,
                                   df_val_for_analysis, output_path=val_error_path)

    # Test errors
    test_error_path = os.path.join(output_dir, 'misclassified_test.csv')
    df_test_for_analysis = df_test[['answer', 'level', 'source_file']].copy()
    test_error_df = analyze_errors(X_test, y_test, y_test_pred,
                                    df_test_for_analysis, output_path=test_error_path)

    # ===================================================================
    # STEP 10: Per-corpus analysis (for test set)
    # ===================================================================
    corpus_results = {}
    for corpus in df_test['source_file'].unique():
        corpus_mask = df_test['source_file'] == corpus
        corpus_y_true = y_test[corpus_mask]
        corpus_y_pred = y_test_pred[corpus_mask]

        corpus_acc = (corpus_y_true == corpus_y_pred).mean()
        corpus_results[corpus] = {
            'accuracy': corpus_acc,
            'samples': len(corpus_y_true)
        }

        print(f"\n{corpus}:")
        print(f"  Samples: {len(corpus_y_true):,}")
        print(f"  Accuracy: {corpus_acc:.4f}")

    # Save per-corpus results
    corpus_results_path = os.path.join(output_dir, 'per_corpus_results.txt')
    with open(corpus_results_path, 'w') as f:
        f.write("PER-CORPUS RESULTS (Test Set)\n")
        f.write("=" * 70 + "\n\n")
        for corpus, results in sorted(corpus_results.items()):
            f.write(f"{corpus}:\n")
            f.write(f"  Samples: {results['samples']:,}\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f}\n\n")

    # ===================================================================
    # STEP 11: Save comprehensive results
    # ===================================================================
    log("SAVING RESULTS")

    summary_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT: MULTINOMIAL NAIVE BAYES - CORPUS-BASED EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASETS:\n")
        f.write(f"  Training: {train_path}\n")
        f.write(f"  Test: {test_path}\n\n")

        f.write("DATA SPLITS:\n")
        f.write(f"  Training: {len(X_train):,} samples (EFCamDAT)\n")
        f.write(f"  Validation: {len(X_val):,} samples (EFCamDAT held-out)\n")
        f.write(f"  Test: {len(X_test):,} samples (Other corpora)\n\n")

        f.write("PARAMETERS:\n")
        f.write(f"  Vectorizer: {vectorizer_method}\n")
        f.write(f"  Max features: {max_features}\n")
        f.write(f"  N-gram range: {ngram_range}\n")
        f.write(f"  Alpha: {alpha}\n")
        f.write(f"  Random state: {random_state}\n\n")

        f.write("=" * 70 + "\n")
        f.write("VALIDATION RESULTS (In-Domain EFCamDAT)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Accuracy: {val_acc:.4f}\n")
        f.write(f"Adjacent Accuracy (±1): {val_metrics['adjacent_accuracy']:.4f}\n")
        f.write(f"MAE: {val_metrics['mae']:.4f} levels\n")
        f.write(f"MSE: {val_metrics['mse']:.4f}\n")
        f.write(f"RMSE: {val_metrics['rmse']:.4f} levels\n")
        f.write(f"Quadratic Weighted Kappa: {val_metrics['qwk']:.4f}\n")
        f.write(f"Ordinal Classification Accuracy: {val_metrics['oca']:.4f}\n")
        f.write(f"Earth Mover's Distance: {val_metrics['emd']:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(val_report)
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("TEST RESULTS (Out-of-Domain Other Corpora)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Adjacent Accuracy (±1): {test_metrics['adjacent_accuracy']:.4f}\n")
        f.write(f"MAE: {test_metrics['mae']:.4f} levels\n")
        f.write(f"MSE: {test_metrics['mse']:.4f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.4f} levels\n")
        f.write(f"Quadratic Weighted Kappa: {test_metrics['qwk']:.4f}\n")
        f.write(f"Ordinal Classification Accuracy: {test_metrics['oca']:.4f}\n")
        f.write(f"Earth Mover's Distance: {test_metrics['emd']:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(test_report)
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("GENERALIZATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Validation Accuracy (EFCamDAT): {val_acc:.4f}\n")
        f.write(f"Test Accuracy (Other Corpora): {test_acc:.4f}\n")
        f.write(f"Generalization Gap: {generalization_gap:.4f} ({generalization_gap*100:.2f}%)\n\n")

        f.write("Per-Corpus Results:\n")
        for corpus, results in sorted(corpus_results.items()):
            f.write(f"  {corpus}: {results['accuracy']:.4f} ({results['samples']:,} samples)\n")

    print(f"\n✓ Summary saved to: {summary_path}")
    print(f"✓ Validation confusion matrix: {val_cm_path}")
    print(f"✓ Test confusion matrix: {test_cm_path}")
    print(f"✓ Validation errors: {val_error_path}")
    print(f"✓ Test errors: {test_error_path}")
    print(f"✓ Per-corpus results: {corpus_results_path}")

    # ===================================================================
    # STEP 12: Return results
    # ===================================================================
    results = {
        'model': model,
        'vectorizer': vectorizer,
        'validation': {
            'y_true': y_val,
            'y_pred': y_val_pred,
            'accuracy': val_acc,
            'metrics': val_metrics,
            'report': val_report,
            'error_df': val_error_df
        },
        'test': {
            'y_true': y_test,
            'y_pred': y_test_pred,
            'accuracy': test_acc,
            'metrics': test_metrics,
            'report': test_report,
            'error_df': test_error_df,
            'corpus_results': corpus_results
        },
        'generalization_gap': generalization_gap,
        'parameters': {
            'vectorizer_method': vectorizer_method,
            'max_features': max_features,
            'ngram_range': ngram_range,
            'val_size': val_size,
            'alpha': alpha,
            'random_state': random_state
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

    # Data paths
    samples = "150k"
    TRAIN_PATH = f'../dataset/splits/train_{samples}.csv'
    TEST_PATH = '../../dataset/splits/test_other_corpora.csv'

    # Model parameters
    VECTORIZER_METHOD = 'tfidf'  # 'count' or 'tfidf'
    MAX_FEATURES = 15000
    NGRAM_RANGE = (1, 3)  # (1,1) for unigrams, (1,2) for uni+bigrams, (1,3) for uni+bi+trigrams
    ALPHA = 1.0  # Smoothing parameter

    # Training parameters
    VAL_SIZE = 0.2  # 20% of training data for validation
    RANDOM_STATE = 42

    # Output directory
    OUTPUT_DIR = f'../results/NB_corpus_{samples}_vec_{VECTORIZER_METHOD}_max_{MAX_FEATURES}_ngram_{NGRAM_RANGE}'

    # ========== RUN EXPERIMENT ==========

    print("=" * 80)
    print("NAIVE BAYES - CORPUS-BASED GENERALIZATION EXPERIMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Training data: {TRAIN_PATH}")
    print(f"  Test data: {TEST_PATH}")
    print(f"  Vectorizer: {VECTORIZER_METHOD.upper()}")
    print(f"  Max features: {MAX_FEATURES:,}")
    print(f"  N-gram range: {NGRAM_RANGE}")
    print(f"  Alpha: {ALPHA}")
    print(f"  Validation size: {VAL_SIZE * 100}%")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 80)

    # Run experiment
    results = run_experiment(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        vectorizer_method=VECTORIZER_METHOD,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        val_size=VAL_SIZE,
        alpha=ALPHA,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED SUCCESSFULLY")
    print("=" * 80)