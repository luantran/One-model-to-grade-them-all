"""
Naive Bayes CEFR Classifier - Standardized Version
===================================================
Corpus-based evaluation: Train on EFCamDAT, test on other corpora for generalization

Standardized function names:
- load_data()
- prepare_features()
- train_model()
- evaluate_model()
- run_experiment()
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
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
    dict : Data splits with keys:
        - X_train, X_val, X_test (text)
        - y_train, y_val, y_test (numeric labels 0-4)
        - y_train_labels, y_val_labels, y_test_labels (CEFR labels)
        - df_test (full test dataframe for per-corpus analysis)
    """

    # Load datasets
    df_train_full = load_csv(train_path)
    df_test = load_csv(test_path)

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


def perform_error_analysis(X_data, y_true_labels, y_pred_labels, df_data, output_dir, dataset_name="validation"):
    """
    Perform error analysis and save misclassified samples.

    Parameters:
    -----------
    X_data : Text data
    y_true_labels : True CEFR labels
    y_pred_labels : Predicted CEFR labels
    df_data : DataFrame with full data (for validation, reconstruct; for test, use df_test)
    output_dir : str
        Output directory
    dataset_name : str
        Name for output file (validation/test)

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
    dict : Per-corpus results {corpus: {'accuracy': float, 'samples': int}}
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
                   method='tfidf', max_features=5000, ngram_range=(1, 2),
                   alpha=1.0, val_size=0.2, random_state=42):
    """
    Run complete Naive Bayes CEFR classification experiment.

    Parameters:
    -----------
    train_path : str
        Path to training CSV
    test_path : str
        Path to test CSV
    output_dir : str
        Output directory
    method : str
        'count' or 'tfidf'
    max_features : int
        Maximum vocabulary size
    ngram_range : tuple
        N-gram range
    alpha : float
        Smoothing parameter
    val_size : float
        Validation set proportion
    random_state : int
        Random seed

    Returns:
    --------
    dict : Complete results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    data = load_data(train_path, test_path, val_size, random_state)

    # Step 2: Prepare features
    features = prepare_features(
        data['X_train'], data['X_val'], data['X_test'],
        method=method, max_features=max_features, ngram_range=ngram_range
    )

    # Step 3: Train model
    model = train_model(features['X_train_features'], data['y_train'], alpha=alpha)

    # Step 4: Evaluate on validation set (in-domain)
    val_results = evaluate_model(
        model,
        features['X_val_features'],
        data['y_val'],
        data['y_val_labels'],
        dataset_name="Validation (In-Domain)"
    )

    # Step 5: Evaluate on test set (out-of-domain)
    test_results = evaluate_model(
        model,
        features['X_test_features'],
        data['y_test'],
        data['y_test_labels'],
        dataset_name="Test (Out-of-Domain)"
    )

    # Step 6: Calculate generalization gap
    generalization_gap = val_results['accuracy'] - test_results['accuracy']
    print(f"\n✓ Generalization Gap: {generalization_gap:.4f} ({generalization_gap * 100:.2f}%)")

    # Step 7: Visualizations
    log("GENERATING VISUALIZATIONS")

    # Confusion matrices
    val_cm_path = os.path.join(output_dir, 'confusion_matrix_validation.png')
    plot_confusion_matrix(
        data['y_val_labels'],
        val_results['y_pred_labels'],
        output_path=val_cm_path,
        title=f'Confusion Matrix - Validation (Naive Bayes {method})'
    )

    test_cm_path = os.path.join(output_dir, 'confusion_matrix_test.png')
    plot_confusion_matrix(
        data['y_test_labels'],
        test_results['y_pred_labels'],
        output_path=test_cm_path,
        title=f'Confusion Matrix - Test (Naive Bayes {method})'
    )

    # Step 8: Error analysis
    df_val_analysis = pd.DataFrame({
        'answer': data['X_val'].values,
        'level': data['y_val_labels'].values
    })
    perform_error_analysis(
        data['X_val'], data['y_val_labels'], val_results['y_pred_labels'],
        df_val_analysis, output_dir, dataset_name="validation"
    )
    perform_error_analysis(
        data['X_test'], data['y_test_labels'], test_results['y_pred_labels'],
        data['df_test'], output_dir, dataset_name="test"
    )

    # Step 9: Per-corpus analysis
    corpus_results = perform_per_corpus_analysis(
        data['df_test'], data['y_test_labels'], test_results['y_pred_labels'],
        output_dir=output_dir
    )

    # Step 10: Save summary
    log("SAVING RESULTS")

    summary_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT: NAIVE BAYES CEFR - CORPUS-BASED EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Method: {method}\n")
        f.write(f"  Max features: {max_features}\n")
        f.write(f"  N-gram range: {ngram_range}\n")
        f.write(f"  Alpha (smoothing): {alpha}\n\n")

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
        'vectorizer': features['vectorizer'],
        'validation': val_results,
        'test': test_results,
        'corpus_results': corpus_results,
        'generalization_gap': generalization_gap,
        'config': {
            'method': method,
            'max_features': max_features,
            'ngram_range': ngram_range,
            'alpha': alpha
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

    METHOD = 'tfidf'  # 'count' or 'tfidf'
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)  # unigrams + bigrams
    ALPHA = 1.0

    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    OUTPUT_DIR = f'../results/NaiveBayes_corpus_{samples}_{METHOD}_maxfeat_{MAX_FEATURES}_ngram_{NGRAM_RANGE[0]}_{NGRAM_RANGE[1]}'

    print("=" * 80)
    print("NAIVE BAYES - CORPUS-BASED GENERALIZATION EXPERIMENT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Method: {METHOD}")
    print(f"  Max features: {MAX_FEATURES}")
    print(f"  N-grams: {NGRAM_RANGE}")
    print(f"  Alpha: {ALPHA}")
    print("=" * 80)

    results = run_experiment(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        method=METHOD,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        alpha=ALPHA,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT FINISHED SUCCESSFULLY")
    print("=" * 80)