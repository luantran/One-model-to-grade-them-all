import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

"""
Utility module for saving comprehensive experiment results including classification reports, experiment summaries, and JSON outputs

Functions were reformatted for cleanliness using IDE plugin 
"""

def log_step(message):
    """Print a formatted log message."""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)

def save_classification_report_structured(y_true, y_pred, output_dir, dataset_name="test"):
    """
    Generate classification report and save as both CSV and JSON for easy access.
    """
    from sklearn.metrics import precision_recall_fscore_support
    import json

    # Get detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=['A1', 'A2', 'B1', 'B2', 'C1/C2'],
        zero_division=0
    )

    # Create structured report dictionary
    report_dict = {
        'classes': ['A1', 'A2', 'B1', 'B2', 'C1/C2']
    }

    # Add per-class metrics
    for i, label in enumerate(report_dict['classes']):
        report_dict[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1-score': float(f1[i]),
            'support': int(support[i])
        }

    # Calculate and add weighted/macro averages
    report_dict['weighted_avg'] = {
        'precision': float(np.average(precision, weights=support)),
        'recall': float(np.average(recall, weights=support)),
        'f1-score': float(np.average(f1, weights=support)),
        'support': int(np.sum(support))
    }

    report_dict['macro_avg'] = {
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1-score': float(np.mean(f1)),
        'support': int(np.sum(support))
    }

    # Save as CSV
    csv_path = os.path.join(output_dir, f'classification_report_{dataset_name}.csv')
    rows = []
    for label in report_dict['classes'] + ['weighted_avg', 'macro_avg']:
        if label in report_dict and isinstance(report_dict[label], dict):
            row = {'class': label}
            row.update(report_dict[label])
            rows.append(row)

    df_report = pd.DataFrame(rows)
    df_report.to_csv(csv_path, index=False)
    print(f"✓ Classification report CSV saved to {csv_path}")

    # Save as JSON
    json_path = os.path.join(output_dir, f'classification_report_{dataset_name}.json')
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    print(f"✓ Classification report JSON saved to {json_path}")

    return report_dict, csv_path, json_path

def save_experiment_summary(in_test_results, test_results, corpus_results, generalization_gap,
                           config, model_name, output_dir):
    """
    Generate and save comprehensive experiment summary to text file.
    """
    summary_path = os.path.join(output_dir, 'experiment_summary.txt')

    with open(summary_path, 'w') as f:
        # Header
        f.write(f"EXPERIMENT: {model_name.upper()} CEFR - CORPUS-BASED EVALUATION\n")
        f.write("=" * 80 + "\n\n")

        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        # In-Domain test Results
        f.write("=" * 80 + "\n")
        f.write("IN-DOMAIN TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("Standard Metrics:\n")
        f.write(f"  Exact Match Accuracy: {in_test_results['accuracy']:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write("-" * 80 + "\n")
        val_class_metrics = in_test_results['metrics']['classification_metrics']
        f.write(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}\n")
        f.write("-" * 80 + "\n")

        f.write("Ordinal Classification Metrics:\n")
        f.write(f"  Adjacent Accuracy (±1 level): {in_test_results['metrics']['adjacent_accuracy']:.4f}\n")
        f.write(f"  Quadratic Weighted Kappa (QWK): {in_test_results['metrics']['qwk']:.4f}\n")
        f.write(f"  Ordinal Classification Accuracy (OCA): {in_test_results['metrics']['oca']:.4f}\n\n")

        for label in ['A1', 'A2', 'B1', 'B2', 'C1/C2']:
            if label in val_class_metrics['per_class']:
                m = val_class_metrics['per_class'][label]
                f.write(f"{label:<15} {m['precision']:>12.4f} {m['recall']:>12.4f} "
                       f"{m['f1']:>12.4f} {m['support']:>12d}\n")

        f.write("-" * 80 + "\n")
        weighted = val_class_metrics['weighted_avg']
        f.write(f"{'Weighted Avg':<15} {weighted['precision']:>12.4f} {weighted['recall']:>12.4f} "
               f"{weighted['f1']:>12.4f} {weighted['support']:>12d}\n")
        macro = val_class_metrics['macro_avg']
        f.write(f"{'Macro Avg':<15} {macro['precision']:>12.4f} {macro['recall']:>12.4f} "
               f"{macro['f1']:>12.4f} {macro['support']:>12d}\n\n")

        # Test Results
        f.write("=" * 80 + "\n")
        f.write("TEST RESULTS (Out-of-Domain)\n")
        f.write("=" * 80 + "\n\n")

        f.write("Standard Metrics:\n")
        f.write(f"  Exact Match Accuracy: {test_results['accuracy']:.4f}\n\n")

        f.write("Ordinal Classification Metrics:\n")
        f.write(f"  Adjacent Accuracy (±1 level): {test_results['metrics']['adjacent_accuracy']:.4f}\n")
        f.write(f"  Quadratic Weighted Kappa (QWK): {test_results['metrics']['qwk']:.4f}\n")
        f.write(f"  Ordinal Classification Accuracy (OCA): {test_results['metrics']['oca']:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write("-" * 80 + "\n")
        test_class_metrics = test_results['metrics']['classification_metrics']
        f.write(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}\n")
        f.write("-" * 80 + "\n")

        for label in ['A1', 'A2', 'B1', 'B2', 'C1/C2']:
            if label in test_class_metrics['per_class']:
                m = test_class_metrics['per_class'][label]
                f.write(f"{label:<15} {m['precision']:>12.4f} {m['recall']:>12.4f} "
                       f"{m['f1']:>12.4f} {m['support']:>12d}\n")

        f.write("-" * 80 + "\n")
        weighted = test_class_metrics['weighted_avg']
        f.write(f"{'Weighted Avg':<15} {weighted['precision']:>12.4f} {weighted['recall']:>12.4f} "
               f"{weighted['f1']:>12.4f} {weighted['support']:>12d}\n")
        macro = test_class_metrics['macro_avg']
        f.write(f"{'Macro Avg':<15} {macro['precision']:>12.4f} {macro['recall']:>12.4f} "
               f"{macro['f1']:>12.4f} {macro['support']:>12d}\n\n")

        # Generalization Analysis
        f.write("=" * 80 + "\n")
        f.write("GENERALIZATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generalization Gap (In-Domain Test Accuracy - Out-Domain Test Accuracy): {generalization_gap:.4f} "
               f"({generalization_gap * 100:.2f}%)\n\n")

        if corpus_results:
            f.write("Per-Corpus Performance:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Corpus':<40} {'Accuracy':>15} {'Samples':>15}\n")
            f.write("-" * 80 + "\n")
            for corpus in sorted(corpus_results.keys()):
                results = corpus_results[corpus]
                f.write(f"{corpus:<40} {results['accuracy']:>15.4f} {results['samples']:>15,}\n")
            f.write("\n")

    print(f"✓ Experiment summary saved to {summary_path}")
    return summary_path


def save_results_json(in_test_results, test_results, corpus_results, generalization_gap,
                      config, output_dir):
    """
    Save all metrics and results to a JSON file for comparison analysis.
    """
    # Create comprehensive results dictionary
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'configuration': config,
        'in_domain_test_results': {
            'accuracy': float(in_test_results['accuracy']),
            'adjacent_accuracy': float(in_test_results['metrics']['adjacent_accuracy']),
            'qwk': float(in_test_results['metrics']['qwk']),
            'oca': float(in_test_results['metrics']['oca']),
            'classification_metrics': {
                'per_class': {
                    k: {
                        'precision': float(v['precision']),
                        'recall': float(v['recall']),
                        'f1': float(v['f1']),
                        'support': int(v['support'])
                    } for k, v in in_test_results['metrics']['classification_metrics']['per_class'].items()
                },
                'weighted_avg': {
                    'precision': float(in_test_results['metrics']['classification_metrics']['weighted_avg']['precision']),
                    'recall': float(in_test_results['metrics']['classification_metrics']['weighted_avg']['recall']),
                    'f1': float(in_test_results['metrics']['classification_metrics']['weighted_avg']['f1']),
                    'support': int(in_test_results['metrics']['classification_metrics']['weighted_avg']['support'])
                },
                'macro_avg': {
                    'precision': float(in_test_results['metrics']['classification_metrics']['macro_avg']['precision']),
                    'recall': float(in_test_results['metrics']['classification_metrics']['macro_avg']['recall']),
                    'f1': float(in_test_results['metrics']['classification_metrics']['macro_avg']['f1']),
                    'support': int(in_test_results['metrics']['classification_metrics']['macro_avg']['support'])
                }
            }
        },
        'test_results': {
            'accuracy': float(test_results['accuracy']),
            'adjacent_accuracy': float(test_results['metrics']['adjacent_accuracy']),
            'qwk': float(test_results['metrics']['qwk']),
            'oca': float(test_results['metrics']['oca']),
            'classification_metrics': {
                'per_class': {
                    k: {
                        'precision': float(v['precision']),
                        'recall': float(v['recall']),
                        'f1': float(v['f1']),
                        'support': int(v['support'])
                    } for k, v in test_results['metrics']['classification_metrics']['per_class'].items()
                },
                'weighted_avg': {
                    'precision': float(test_results['metrics']['classification_metrics']['weighted_avg']['precision']),
                    'recall': float(test_results['metrics']['classification_metrics']['weighted_avg']['recall']),
                    'f1': float(test_results['metrics']['classification_metrics']['weighted_avg']['f1']),
                    'support': int(test_results['metrics']['classification_metrics']['weighted_avg']['support'])
                },
                'macro_avg': {
                    'precision': float(test_results['metrics']['classification_metrics']['macro_avg']['precision']),
                    'recall': float(test_results['metrics']['classification_metrics']['macro_avg']['recall']),
                    'f1': float(test_results['metrics']['classification_metrics']['macro_avg']['f1']),
                    'support': int(test_results['metrics']['classification_metrics']['macro_avg']['support'])
                }
            }
        },
        'generalization': {
            'gap': float(generalization_gap),
            'gap_percentage': float(generalization_gap * 100)
        },
        'per_corpus_results': {
            corpus: {
                'accuracy': float(results['accuracy']),
                'samples': int(results['samples'])
            }
            for corpus, results in corpus_results.items()
        }
    }

    # Save to JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"✓ Results saved to {json_path}")
    return json_path
