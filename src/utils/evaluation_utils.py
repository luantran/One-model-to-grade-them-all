"""
Evaluation utilities for ordinal classification tasks.
Common metrics and evaluation functions that can be used across different models
(Naive Bayes, Word2Vec, BERT, etc.)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error,
    cohen_kappa_score, classification_report, confusion_matrix
)
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns


def encode_labels_to_numeric(labels):
    """
    Convert CEFR labels to numeric ordinal scale.
    A1=1, A2=2, B1=3, B2=4, C1/C2=5 (or C1=5, C2=6 if not merged)

    Parameters:
    -----------
    labels : array-like
        CEFR labels (e.g., ['A1', 'B2', 'C1/C2'])

    Returns:
    --------
    np.array : Numeric representation of labels
    """
    label_map = {
        'A1': 1,
        'A2': 2,
        'B1': 3,
        'B2': 4,
        'C1': 5,
        'C2': 6,
        'C1/C2': 5  # Treat merged C1/C2 as level 5
    }

    # Convert to list if needed
    if isinstance(labels, pd.Series):
        labels = labels.values

    numeric_labels = []
    for label in labels:
        if label in label_map:
            numeric_labels.append(label_map[label])
        else:
            # Handle unexpected labels
            print(f"Warning: Unexpected label '{label}', treating as missing")
            numeric_labels.append(np.nan)

    return np.array(numeric_labels)


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate Quadratic Weighted Kappa.

    Parameters:
    -----------
    y_true : array-like
        True labels (numeric)
    y_pred : array-like
        Predicted labels (numeric)

    Returns:
    --------
    float : QWK score
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def off_by_one_accuracy(y_true, y_pred, tolerance=1):
    """
    Calculate percentage of predictions within 'tolerance' levels.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    tolerance : int
        Maximum acceptable distance (default=1 for adjacent accuracy)

    Returns:
    --------
    float : Proportion of predictions within tolerance
    """
    y_true_numeric = encode_labels_to_numeric(y_true)
    y_pred_numeric = encode_labels_to_numeric(y_pred)

    # Remove any NaN values
    valid_mask = ~(np.isnan(y_true_numeric) | np.isnan(y_pred_numeric))
    y_true_numeric = y_true_numeric[valid_mask]
    y_pred_numeric = y_pred_numeric[valid_mask]

    within_tolerance = np.abs(y_true_numeric - y_pred_numeric) <= tolerance
    return within_tolerance.mean()


def ordinal_classification_accuracy(y_true, y_pred, weights=None):
    """
    Weighted accuracy giving partial credit for near-misses.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    weights : list, optional
        Credit given for each distance level
        Default: [1.0, 0.5, 0.25, 0.0, 0.0, 0.0]
        - Exact: 1.0, 1 level off: 0.5, 2 levels off: 0.25, 3+ levels: 0.0

    Returns:
    --------
    float : Weighted accuracy score
    """
    if weights is None:
        weights = [1.0, 0.5, 0.25, 0.0, 0.0, 0.0]

    y_true_numeric = encode_labels_to_numeric(y_true)
    y_pred_numeric = encode_labels_to_numeric(y_pred)

    # Remove any NaN values
    valid_mask = ~(np.isnan(y_true_numeric) | np.isnan(y_pred_numeric))
    y_true_numeric = y_true_numeric[valid_mask]
    y_pred_numeric = y_pred_numeric[valid_mask]

    distances = np.abs(y_true_numeric - y_pred_numeric)
    scores = [weights[int(min(d, len(weights) - 1))] for d in distances]

    return np.mean(scores)


def ordinal_emd(y_true, y_pred):
    """
    Earth Mover's Distance (Wasserstein Distance) for ordinal classification.
    Measures the "cost" of transforming predicted distribution to true distribution.
    Lower is better (0 = perfect).

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels

    Returns:
    --------
    float : EMD score
    """
    y_true_numeric = encode_labels_to_numeric(y_true)
    y_pred_numeric = encode_labels_to_numeric(y_pred)

    # Remove any NaN values
    valid_mask = ~(np.isnan(y_true_numeric) | np.isnan(y_pred_numeric))
    y_true_numeric = y_true_numeric[valid_mask]
    y_pred_numeric = y_pred_numeric[valid_mask]

    return wasserstein_distance(y_true_numeric, y_pred_numeric)


def cumulative_link_loss(y_true, y_pred_proba):
    """
    Cumulative link model loss for ordinal classification.
    Evaluates how well the model respects ordinal structure in probabilities.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class

    Returns:
    --------
    float : Cumulative binary cross-entropy loss (lower is better)
    """
    y_true_numeric = encode_labels_to_numeric(y_true)

    # Remove any NaN values
    valid_mask = ~np.isnan(y_true_numeric)
    y_true_numeric = y_true_numeric[valid_mask]
    y_pred_proba = y_pred_proba[valid_mask]

    n_classes = y_pred_proba.shape[1]

    # Create cumulative probability matrix for true labels
    cumulative_true = np.zeros((len(y_true_numeric), n_classes))
    for i, true_val in enumerate(y_true_numeric):
        cumulative_true[i, :int(true_val)] = 1

    # Get cumulative predicted probabilities
    cumulative_pred = np.cumsum(y_pred_proba, axis=1)
    cumulative_pred = np.clip(cumulative_pred, 1e-10, 1 - 1e-10)  # Avoid log(0)

    # Binary cross-entropy on cumulative probabilities
    loss = -np.mean(
        cumulative_true * np.log(cumulative_pred) +
        (1 - cumulative_true) * np.log(1 - cumulative_pred)
    )

    return loss


def get_metric_interpretation(metric_name, value):
    """
    Return interpretation text for a given metric value.

    Parameters:
    -----------
    metric_name : str
        Name of the metric
    value : float
        Metric value

    Returns:
    --------
    str : Interpretation text with performance level
    """
    interpretations = {
        'test_accuracy': [
            (0.80, "++ Excellent"),
            (0.70, "+  Good"),
            (0.60, "o  Fair"),
            (0.00, "-  Poor")
        ],
        'adjacent_accuracy': [
            (0.95, "++ Excellent"),
            (0.90, "+  Good"),
            (0.85, "o  Fair"),
            (0.00, "-  Poor")
        ],
        'mae': [
            (0.50, "++ Excellent (≤0.5)"),
            (0.75, "+  Good (≤0.75)"),
            (1.00, "o  Fair (≤1.0)"),
            (float('inf'), "-  Poor (>1.0)")
        ],
        'rmse': [
            (0.70, "++ Excellent (≤0.7)"),
            (1.00, "+  Good (≤1.0)"),
            (1.30, "o  Fair (≤1.3)"),
            (float('inf'), "-  Poor (>1.3)")
        ],
        'qwk': [
            (0.80, "++ Excellent"),
            (0.70, "+  Good"),
            (0.60, "o  Fair"),
            (0.00, "-  Poor")
        ],
        'oca': [
            (0.75, "++ Excellent"),
            (0.65, "+  Good"),
            (0.55, "o  Fair"),
            (0.00, "-  Poor")
        ],
        'emd': [
            (0.50, "++ Excellent (≤0.5)"),
            (0.80, "+  Good (≤0.8)"),
            (1.20, "o  Fair (≤1.2)"),
            (float('inf'), "-  Poor (>1.2)")
        ],
        'cumulative_loss': [
            (0.30, "++ Excellent (≤0.3)"),
            (0.50, "+  Good (≤0.5)"),
            (0.70, "o  Fair (≤0.7)"),
            (float('inf'), "-  Poor (>0.7)")
        ]
    }

    if metric_name not in interpretations:
        return ""

    thresholds = interpretations[metric_name]

    # For metrics where lower is better (MAE, RMSE, EMD, cumulative_loss)
    if metric_name in ['mae', 'rmse', 'emd', 'cumulative_loss']:
        for threshold, interpretation in thresholds:
            if value <= threshold:
                return f"[{interpretation}]"
    # For metrics where higher is better
    else:
        for threshold, interpretation in thresholds:
            if value >= threshold:
                return f"[{interpretation}]"

    return ""


def compute_all_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute all ordinal classification metrics at once.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities (for cumulative link loss)

    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    # Convert to numeric
    y_true_numeric = encode_labels_to_numeric(y_true)
    y_pred_numeric = encode_labels_to_numeric(y_pred)

    # Remove any NaN values
    valid_mask = ~(np.isnan(y_true_numeric) | np.isnan(y_pred_numeric))
    y_true_numeric_clean = y_true_numeric[valid_mask]
    y_pred_numeric_clean = y_pred_numeric[valid_mask]

    # For label-based comparisons, use the original labels
    y_true_labels = y_true if isinstance(y_true, np.ndarray) else y_true.values
    y_pred_labels = y_pred if isinstance(y_pred, np.ndarray) else y_pred.values

    # Filter labels to match valid mask
    y_true_labels_clean = y_true_labels[valid_mask]
    y_pred_labels_clean = y_pred_labels[valid_mask]

    # Standard metrics (use labels for exact comparison)
    exact_acc = accuracy_score(y_true_labels_clean, y_pred_labels_clean)

    # Ordinal metrics (use numeric values)
    adjacent_acc = off_by_one_accuracy(y_true, y_pred, tolerance=1)
    mae = mean_absolute_error(y_true_numeric_clean, y_pred_numeric_clean)
    mse = mean_squared_error(y_true_numeric_clean, y_pred_numeric_clean)
    rmse = np.sqrt(mse)
    qwk = quadratic_weighted_kappa(y_true_numeric_clean, y_pred_numeric_clean)
    oca = ordinal_classification_accuracy(y_true, y_pred)
    emd = ordinal_emd(y_true, y_pred)

    metrics = {
        'test_accuracy': exact_acc,
        'adjacent_accuracy': adjacent_acc,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'qwk': qwk,
        'oca': oca,
        'emd': emd,
        'cumulative_loss': None
    }

    # Cumulative link loss (only if probabilities provided)
    if y_pred_proba is not None:
        try:
            # Filter probabilities to match valid mask
            y_pred_proba_clean = y_pred_proba[valid_mask]
            cumulative_loss = cumulative_link_loss(y_true, y_pred_proba_clean)
            metrics['cumulative_loss'] = cumulative_loss
        except Exception as e:
            print(f"Warning: Could not compute cumulative link loss: {str(e)}")

    return metrics


def print_metrics(metrics, train_acc=None):
    """
    Pretty print all metrics with interpretations.

    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from compute_all_metrics
    train_acc : float, optional
        Training accuracy (for computing overfitting gap)
    """

    # Standard classification metrics
    print("\n" + "-" * 70)
    print("STANDARD CLASSIFICATION METRICS")
    print("-" * 70)

    if train_acc is not None:
        print(f"Training Accuracy: {train_acc:.4f}")

    exact_acc = metrics['test_accuracy']
    acc_interp = get_metric_interpretation('test_accuracy', exact_acc)
    print(f"Exact Match Accuracy: {exact_acc:.4f} {acc_interp}")

    # Ordinal classification metrics
    print("\n" + "-" * 70)
    print("ORDINAL CLASSIFICATION METRICS")
    print("-" * 70)
    print()
    adjacent_acc = metrics['adjacent_accuracy']
    adj_interp = get_metric_interpretation('adjacent_accuracy', adjacent_acc)
    print(f"Adjacent Accuracy (±1 level): {adjacent_acc:.4f} {adj_interp}")
    print(f"  → % of predictions within 1 CEFR level")
    print()

    mae = metrics['mae']
    mae_interp = get_metric_interpretation('mae', mae)
    print(f"Mean Absolute Error (MAE): {mae:.4f} levels {mae_interp}")
    print(f"  → Average distance from true level")
    print()

    print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"  → Squared error (penalizes large mistakes)")
    print()

    rmse = metrics['rmse']
    rmse_interp = get_metric_interpretation('rmse', rmse)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} levels {rmse_interp}")
    print(f"  → Like MAE but penalizes larger errors more")
    print()

    qwk = metrics['qwk']
    qwk_interp = get_metric_interpretation('qwk', qwk)
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f} {qwk_interp}")
    print(f"  → Agreement metric with quadratic penalty")
    print()

    oca = metrics['oca']
    oca_interp = get_metric_interpretation('oca', oca)
    print(f"Ordinal Classification Accuracy (OCA): {oca:.4f} {oca_interp}")
    print(f"  → Weighted accuracy (exact=1.0, ±1=0.5, ±2=0.25)")
    print()

    emd = metrics['emd']
    emd_interp = get_metric_interpretation('emd', emd)
    print(f"Earth Mover's Distance (EMD): {emd:.4f} {emd_interp}")
    print(f"  → Cost to transform predicted to true distribution")
    print()

    if metrics['cumulative_loss'] is not None:
        cumulative_loss = metrics['cumulative_loss']
        cum_interp = get_metric_interpretation('cumulative_loss', cumulative_loss)
        print(f"Cumulative Link Loss: {cumulative_loss:.4f} {cum_interp}")
        print(f"  → Ordinal probability loss (lower is better)")
        print()
    else:
        print(f"Cumulative Link Loss: N/A (no probability predictions)")
        print()

def evaluate_model(model, X_train, X_test, y_train, y_test, has_predict_proba=True):
    """
    Comprehensive model evaluation with ordinal classification metrics.
    Works with any model that has predict() method.

    Parameters:
    -----------
    model : object
        Trained model with predict() method
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    has_predict_proba : bool
        Whether model supports predict_proba() for cumulative loss

    Returns:
    --------
    tuple : (test_pred, train_acc, test_acc, metrics, class_report)
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {train_acc:.4f}")

    # Test predictions
    test_pred = model.predict(X_test)

    # Get probabilities if available
    y_pred_proba = None
    if has_predict_proba and hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            pass

    # Compute all metrics
    metrics = compute_all_metrics(y_test, test_pred, y_pred_proba)
    metrics['train_accuracy'] = train_acc

    # Print metrics
    print_metrics(metrics, train_acc=train_acc)

    # Classification report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT (Test Set)")
    print("-" * 70)
    class_report = classification_report(y_test, test_pred, digits=4)
    print(class_report)

    return test_pred, train_acc, metrics['test_accuracy'], metrics, class_report


def plot_confusion_matrix(y_test, y_pred, output_path='confusion_matrix.png', title='Confusion Matrix'):
    """
    Plot and save confusion matrix.

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    output_path : str
        Path to save figure
    title : str
        Plot title
    """
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    # Get unique labels in sorted order
    labels = sorted(pd.Series(y_test).unique())

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Print header
    print(f"{'':8}", end="")
    for label in labels:
        print(f"{label:>8}", end="")
    print()
    print("-" * (8 + 8 * len(labels)))

    # Print matrix rows
    for i, label in enumerate(labels):
        print(f"{label:8}", end="")
        for j in range(len(labels)):
            print(f"{cm[i, j]:8}", end="")
        print()

    print()

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    plt.close()


def analyze_errors(X_test, y_test, y_pred, df_test, output_path=None):
    """
    Analyze misclassified examples.

    Parameters:
    -----------
    X_test : array-like
        Test features (for reference)
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    df_test : pd.DataFrame
        Test set dataframe with original data
    output_path : str, optional
        Path to save error analysis CSV

    Returns:
    --------
    pd.DataFrame : DataFrame with misclassified examples
    """
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    # Convert to numpy arrays if needed
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Find misclassified examples
    misclassified = y_test != y_pred

    # Reset index of df_test to ensure alignment
    df_test_reset = df_test.reset_index(drop=True)

    error_df = df_test_reset[misclassified].copy()
    error_df['predicted'] = y_pred[misclassified]
    error_df['true_label'] = y_test[misclassified]

    print(f"Total misclassified: {misclassified.sum()} / {len(y_test)} "
          f"({misclassified.sum() / len(y_test) * 100:.2f}%)")

    # Show some examples
    print("\nSample misclassifications:")
    for i, row in error_df.head(5).iterrows():
        print(f"\n{'-' * 60}")
        print(f"True: {row['true_label']} | Predicted: {row['predicted']}")
        if 'native_language' in row:
            print(f"Native Language: {row['native_language']}")
        if 'answer' in row:
            print(f"Answer (first 200 chars): {row['answer'][:200]}...")

    # Save if path provided
    if output_path:
        error_df.to_csv(output_path, index=False)
        print(f"\n✓ Misclassified examples saved to: {output_path}")

    return error_df


def print_results_summary(metrics, train_acc, dataset_info, model_name, parameters, output_files):
    """
    Print a comprehensive results summary.

    Parameters:
    -----------
    metrics : dict
        Evaluation metrics
    train_acc : float
        Training accuracy
    dataset_info : dict
        Dictionary with 'total', 'train', 'test' sample counts
    model_name : str
        Name of the model
    parameters : dict
        Model/experiment parameters
    output_files : dict
        Paths to output files
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)

    print("\nDATASET INFORMATION")
    print("-" * 70)
    print(f"  {'Total samples:':<35} {dataset_info['total']:>10,}")
    print(f"  {'Training samples:':<35} {dataset_info['train']:>10,}")
    print(f"  {'Test samples:':<35} {dataset_info['test']:>10,}")

    print("\nMODEL CONFIGURATION")
    print("-" * 70)
    print(f"  {'Algorithm:':<35} {model_name}")
    for param_name, param_value in parameters.items():
        print(f"  {param_name + ':':<35} {str(param_value):>10}")

    print("\nSTANDARD METRICS")
    print("-" * 70)
    test_acc = metrics['test_accuracy']
    acc_interp = get_metric_interpretation('test_accuracy', test_acc)
    print(f"  {'Training Accuracy:':<35} {train_acc:>10.2%}")
    print(f"  {'Test Accuracy:':<35} {test_acc:>10.2%} {acc_interp}")
    print(f"  {'Overfitting Gap:':<35} {(train_acc - test_acc):>10.2%}")

    print("\nORDINAL CLASSIFICATION METRICS")
    print("-" * 70)

    metric_names = [
        ('adjacent_accuracy', 'Adjacent Accuracy (±1):'),
        ('mae', 'Mean Absolute Error:'),
        ('rmse', 'Root Mean Squared Error:'),
        ('qwk', 'Quadratic Weighted Kappa:'),
        ('oca', 'Ordinal Classification Acc:'),
        ('emd', 'Earth Movers Distance:')
    ]

    for key, label in metric_names:
        value = metrics[key]
        interp = get_metric_interpretation(key, value)
        if key == 'adjacent_accuracy':
            print(f"  {label:<35} {value:>10.2%} {interp}")
        else:
            print(f"  {label:<35} {value:>10.4f} {interp}")

    if metrics['cumulative_loss'] is not None:
        cum_interp = get_metric_interpretation('cumulative_loss', metrics['cumulative_loss'])
        print(f"  {'Cumulative Link Loss:':<35} {metrics['cumulative_loss']:>10.4f} {cum_interp}")

    print("\nOUTPUT FILES")
    print("-" * 70)
    for file_type, file_path in output_files.items():
        print(f"  • {file_type}: {file_path}")

    print("\n" + "=" * 70)
    print(" " * 24 + "EXPERIMENT COMPLETE")
    print("=" * 70 + "\n")