"""
Evaluation utilities for ordinal classification tasks.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
)


def encode_labels_to_numeric(labels):
    """
    Convert CEFR labels to numeric ordinal scale.
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
            numeric_labels.append(np.nan)

    return np.array(numeric_labels)

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate Quadratic Weighted Kappa.
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def off_by_one_accuracy(y_true, y_pred, tolerance=1):
    """
    Calculate percentage of predictions within 'tolerance' levels.
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

def compute_classification_metrics(y_true, y_pred):
    """
    Compute per-class and weighted classification metrics.
    """
    # Define CEFR labels in order
    labels = ['A1', 'A2', 'B1', 'B2', 'C1/C2']

    # Get metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Create per-class metrics
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    # Weighted average
    weighted_avg = {
        'precision': float(np.average(precision, weights=support)),
        'recall': float(np.average(recall, weights=support)),
        'f1': float(np.average(f1, weights=support)),
        'support': int(np.sum(support))
    }

    # Macro average
    macro_avg = {
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1': float(np.mean(f1)),
        'support': int(np.sum(support))
    }

    return {
        'per_class': per_class,
        'weighted_avg': weighted_avg,
        'macro_avg': macro_avg
    }

def compute_all_metrics(y_true, y_pred):
    """
    Compute all ordinal classification metrics at once.
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
    qwk = quadratic_weighted_kappa(y_true_numeric_clean, y_pred_numeric_clean)
    oca = ordinal_classification_accuracy(y_true, y_pred)

    # Classification report metrics
    class_metrics = compute_classification_metrics(y_true_labels_clean, y_pred_labels_clean)

    metrics = {
        'test_accuracy': exact_acc,
        'adjacent_accuracy': adjacent_acc,
        'qwk': qwk,
        'oca': oca,
        'classification_metrics': class_metrics
    }
    return metrics

def print_metrics(metrics, train_acc=None):
    """
    Pretty print all metrics with interpretations.
    """
    if train_acc is not None:
        print(f"Training Accuracy: {train_acc:.4f}")

    exact_acc = metrics['test_accuracy']
    print(f"Exact Match Accuracy: {exact_acc:.4f}")

    class_metrics = metrics['classification_metrics']
    per_class = class_metrics['per_class']

    # Print per-class metrics
    print(f"{'Class':<12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 70)

    for label in ['A1', 'A2', 'B1', 'B2', 'C1/C2']:
        if label in per_class:
            metrics_data = per_class[label]
            print(f"{label:<12} {metrics_data['precision']:>12.4f} {metrics_data['recall']:>12.4f} "
                  f"{metrics_data['f1']:>12.4f} {metrics_data['support']:>12d}")

    print("-" * 70)

    # Print weighted and macro averages
    weighted = class_metrics['weighted_avg']
    macro = class_metrics['macro_avg']

    print(f"{'Weighted Avg':<12} {weighted['precision']:>12.4f} {weighted['recall']:>12.4f} "
          f"{weighted['f1']:>12.4f} {weighted['support']:>12d}")
    print(f"{'Macro Avg':<12} {macro['precision']:>12.4f} {macro['recall']:>12.4f} "
          f"{macro['f1']:>12.4f} {macro['support']:>12d}")

    # Ordinal classification metrics
    print("\n" + "-" * 70)
    print("ORDINAL CLASSIFICATION METRICS")
    print("-" * 70)
    print()

    qwk = metrics['qwk']
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print(f"  → Agreement metric with quadratic penalty")
    print()

    adjacent_acc = metrics['adjacent_accuracy']
    print(f"Adjacent Accuracy (±1 level): {adjacent_acc:.4f}")
    print(f"  → % of predictions within 1 CEFR level")
    print()

    oca = metrics['oca']
    print(f"Ordinal Classification Accuracy (OCA): {oca:.4f}")
    print(f"  → Weighted accuracy (exact=1.0, ±1=0.5, ±2=0.25)")

def plot_confusion_matrix(y_out_test, y_pred, output_path='confusion_matrix.png', title='Confusion Matrix'):
    """
    Plot and save confusion matrix.
    """
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    # Get unique labels in sorted order
    labels = sorted(pd.Series(y_out_test).unique())

    # Create confusion matrix
    cm = confusion_matrix(y_out_test, y_pred, labels=labels)

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

def perform_per_corpus_analysis(df_test, y_out_test_labels, y_pred_labels, output_dir=None):
    """
    Analyze performance per corpus in test set.
    """
    corpus_results = {}
    for corpus in df_test['source_file'].unique():
        corpus_mask = df_test['source_file'] == corpus
        corpus_y_true = y_out_test_labels[corpus_mask].values
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
