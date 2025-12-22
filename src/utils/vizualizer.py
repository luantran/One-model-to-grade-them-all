"""
Compare results from multiple experiments and generate comparison visualizations.

This script reads multiple results.json files and creates:
1. Bar chart comparing test metrics (accuracy, adjacent_accuracy, qwk, f1)
2. Bar chart comparing in-domain test metrics (same metrics)
3. Bar chart comparing F1 scores per class for test results + table
4. Bar chart comparing F1 scores per class for in-domain results + table

Many of the chart generation functions were written with the help of Generative AI (Claude)
"""

import json
import pandas as pd
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def plot_data_distribution(splits: Dict[str, Any], output_dir: str, title_prefix: str = ""):
    """
    Generate separate bar charts for CEFR label distributions in train/val/test splits.
    """
    os.makedirs(output_dir, exist_ok=True)

    levels = ['A1', 'A2', 'B1', 'B2', 'C1/C2']
    # Extended color palette for up to 16 experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#393b79']

    split_keys = ['y_train', 'y_in_test', 'y_out_test']
    split_names = ['Training', 'In-Domain Test', 'Out-of-Domain Test']

    for key, name in zip(split_keys, split_names):
        y = splits[key]
        counts = [(y == i).sum() for i in range(5)]
        percentages = [c / sum(counts) * 100 for c in counts]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(levels, counts, color=colors)

        # Compute top margin dynamically
        top_margin = max(counts) * 0.12  # 12% extra space
        ax.set_ylim(0, max(counts) + top_margin)

        # Annotate counts and percentages
        for i, (count, perc) in enumerate(zip(counts, percentages)):
            ax.text(i, count + top_margin*0.05, f"{count:,}\n({perc:.1f}%)",
                    ha='center', va='bottom', fontsize=21)

        ax.set_ylabel("Number of Samples", fontsize=17)
        ax.set_xlabel("CEFR Level", fontsize=17)
        ax.set_title(f"{title_prefix} CEFR Distribution - {name} Set", fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        plt.tight_layout()

        path = os.path.join(output_dir, f"cefr_distribution_{name.lower()}.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved {name} chart to: {path}")

def load_results(json_paths):
    """Load results from JSON files."""
    results = []
    for path in json_paths:
        with open(path, 'r') as f:
            results.append(json.load(f))
    return results


def extract_metrics(results, result_type='test_results'):
    """Extract main metrics from results."""
    metrics = {
        'accuracy': [],
        'adjacent_accuracy': [],
        'qwk': [],
        'f1': []
    }

    for result in results:
        data = result[result_type]
        metrics['accuracy'].append(data['accuracy'])
        metrics['adjacent_accuracy'].append(data['adjacent_accuracy'])
        metrics['qwk'].append(data['qwk'])
        # F1 is the weighted average f1
        metrics['f1'].append(data['classification_metrics']['weighted_avg']['f1'])

    return metrics


def extract_per_class_f1(results, result_type='test_results'):
    """Extract F1 scores per class."""
    classes = ['A1', 'A2', 'B1', 'B2', 'C1/C2']
    per_class_f1 = {cls: [] for cls in classes}

    for result in results:
        data = result[result_type]['classification_metrics']['per_class']
        for cls in classes:
            per_class_f1[cls].append(data[cls]['f1'])

    return per_class_f1


def create_metrics_comparison_chart(metrics, experiment_names, title, output_path):
    """Create grouped bar chart for metrics comparison."""
    metric_names = ['Accuracy', 'Adjacent\nAccuracy', 'QWK', 'F1']
    metric_keys = ['accuracy', 'adjacent_accuracy', 'qwk', 'f1']

    x = np.arange(len(metric_names))
    n_experiments = len(experiment_names)

    # Dynamic width calculation: total width 0.8, divided by number of experiments
    width = 0.8 / n_experiments

    # Adjust figure size based on number of experiments
    fig_width = max(14, 10 + n_experiments * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # Extended color palette for up to 16 experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#393b79']  # Added one more color

    # Create bars for each experiment
    for i, exp_name in enumerate(experiment_names):
        values = [metrics[key][i] for key in metric_keys]
        # Center the bars: offset from -(n-1)/2 to +(n-1)/2
        offset = width * (i - (n_experiments - 1) / 2)
        bars = ax.bar(x + offset, values, width, label=exp_name,
                     alpha=0.8, color=colors[i % len(colors)])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=18)

    ax.set_xlabel('Metrics', fontsize=18, fontweight='bold')
    ax.set_ylabel('Score', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    # Adjust legend columns based on number of experiments
    legend_cols = 1 if n_experiments <= 3 else 2 if n_experiments <= 8 else 3
    ax.legend(loc='lower right', fontsize=14, ncol=legend_cols)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_metrics_table(metrics, experiment_names, title, output_path):
    """Create table with main metrics comparison."""
    metric_names = ['Accuracy', 'Adjacent Accuracy', 'QWK', 'F1 (Weighted)']
    metric_keys = ['accuracy', 'adjacent_accuracy', 'qwk', 'f1']

    # Create DataFrame
    data = {}
    for metric_name, metric_key in zip(metric_names, metric_keys):
        data[metric_name] = metrics[metric_key]

    df = pd.DataFrame(data, index=experiment_names)
    df = df.round(3)

    # Dynamic figure height based on number of experiments
    fig_height = 3 + len(experiment_names) * 0.6
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style row labels
    for i in range(len(df.index)):
        table[(i + 1, -1)].set_facecolor('#E8F5E9')
        table[(i + 1, -1)].set_text_props(weight='bold')

    # Highlight best values in each column
    for col_idx, col in enumerate(df.columns):
        best_idx = df[col].idxmax()
        row_idx = list(df.index).index(best_idx) + 1
        table[(row_idx, col_idx)].set_facecolor('#FFEB3B')

    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Also save as CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")


def create_per_class_f1_chart(per_class_f1, experiment_names, title, output_path):
    """Create grouped bar chart for per-class F1 scores."""
    classes = ['A1', 'A2', 'B1', 'B2', 'C1/C2']

    x = np.arange(len(classes))
    n_experiments = len(experiment_names)

    # Dynamic width calculation
    width = 0.8 / n_experiments

    # Adjust figure size based on number of experiments
    fig_width = max(14, 10 + n_experiments * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # Extended color palette for up to 16 experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#393b79']  # Added one more color

    # Create bars for each experiment
    for i, exp_name in enumerate(experiment_names):
        values = [per_class_f1[cls][i] for cls in classes]
        # Center the bars
        offset = width * (i - (n_experiments - 1) / 2)
        bars = ax.bar(x + offset, values, width, label=exp_name,
                     alpha=0.8, color=colors[i % len(colors)])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=18, rotation=0)

    ax.set_xlabel('CEFR Level', fontsize=18, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    # Adjust legend columns based on number of experiments
    legend_cols = 1 if n_experiments <= 3 else 2 if n_experiments <= 8 else 3
    ax.legend(loc='lower right', fontsize=14, ncol=legend_cols)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_per_class_f1_table(per_class_f1, experiment_names, title, output_path):
    """Create table with per-class F1 scores."""
    classes = ['A1', 'A2', 'B1', 'B2', 'C1/C2']

    # Create DataFrame
    data = {}
    for cls in classes:
        data[cls] = [per_class_f1[cls][i] for i in range(len(experiment_names))]

    df = pd.DataFrame(data, index=experiment_names)

    # Add macro average column
    df['Macro Avg'] = df.mean(axis=1)

    # Format to 3 decimal places
    df = df.round(3)

    # Dynamic figure height based on number of experiments
    fig_height = 3 + len(experiment_names) * 0.6
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style row labels
    for i in range(len(df.index)):
        table[(i + 1, -1)].set_facecolor('#E8F5E9')
        table[(i + 1, -1)].set_text_props(weight='bold')

    # Highlight best values in each column
    for col_idx, col in enumerate(df.columns):
        best_idx = df[col].idxmax()
        row_idx = list(df.index).index(best_idx) + 1
        table[(row_idx, col_idx)].set_facecolor('#FFEB3B')

    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Also save as CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")


def extract_per_corpus_metrics(results):
    """Extract per-corpus accuracy from results."""
    corpora = ['asag', 'icnale_we_learners', 'icnale_we_uae_learners',
               'icnale_wep_learners', 'write_improve_first_and_final']

    corpus_names = {
        'asag': 'ASAG',
        'icnale_we_learners': 'ICNALE WE',
        'icnale_we_uae_learners': 'ICNALE WE UAE',
        'icnale_wep_learners': 'ICNALE WEP',
        'write_improve_first_and_final': 'Write & Improve'
    }

    per_corpus_accuracy = {corpus_names[corpus]: [] for corpus in corpora}

    for result in results:
        corpus_results = result['per_corpus_results']
        for corpus in corpora:
            per_corpus_accuracy[corpus_names[corpus]].append(
                corpus_results[corpus]['accuracy']
            )

    return per_corpus_accuracy


def create_per_corpus_chart(per_corpus_accuracy, experiment_names, title, output_path):
    """Create grouped bar chart for per-corpus accuracy."""
    corpora = list(per_corpus_accuracy.keys())

    x = np.arange(len(corpora))
    n_experiments = len(experiment_names)

    # Dynamic width calculation
    width = 0.8 / n_experiments

    # Adjust figure size based on number of experiments
    fig_width = max(14, 10 + n_experiments * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # Extended color palette for up to 16 experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#393b79']  # Added one more color

    # Create bars for each experiment
    for i, exp_name in enumerate(experiment_names):
        values = [per_corpus_accuracy[corpus][i] for corpus in corpora]
        # Center the bars
        offset = width * (i - (n_experiments - 1) / 2)
        bars = ax.bar(x + offset, values, width, label=exp_name,
                      alpha=0.8, color=colors[i % len(colors)])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=16, rotation=0)

    ax.set_xlabel('Corpus', fontsize=18, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(corpora, rotation=15, ha='right')
    # Adjust legend columns based on number of experiments
    legend_cols = 1 if n_experiments <= 3 else 2 if n_experiments <= 8 else 3
    ax.legend(loc='upper right', fontsize=14, ncol=legend_cols)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_per_corpus_table(per_corpus_accuracy, experiment_names, title, output_path):
    """Create table with per-corpus accuracy."""
    corpora = list(per_corpus_accuracy.keys())

    # Create DataFrame
    data = {}
    for corpus in corpora:
        data[corpus] = [per_corpus_accuracy[corpus][i] for i in range(len(experiment_names))]

    df = pd.DataFrame(data, index=experiment_names)

    # Add average column
    df['Average'] = df.mean(axis=1)

    # Format to 3 decimal places
    df = df.round(3)

    # Dynamic figure height based on number of experiments
    fig_height = 3 + len(experiment_names) * 0.6
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style row labels
    for i in range(len(df.index)):
        table[(i + 1, -1)].set_facecolor('#E8F5E9')
        table[(i + 1, -1)].set_text_props(weight='bold')

    # Highlight best values in each column
    for col_idx, col in enumerate(df.columns):
        best_idx = df[col].idxmax()
        row_idx = list(df.index).index(best_idx) + 1
        table[(row_idx, col_idx)].set_facecolor('#FFEB3B')

    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Also save as CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")


def compare(json_paths, output_dir='comparison_results'):
    """Main function to generate all comparisons."""
    # Validate number of experiments
    if len(json_paths) < 2:
        print("Error: Need at least 2 experiments to compare")
        sys.exit(1)
    if len(json_paths) > 16:  # Changed from 15 to 16
        print("Error: Maximum 16 experiments supported")
        sys.exit(1)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from {len(json_paths)} files...")
    results = load_results(json_paths)

    # Extract experiment names
    experiment_names = [r['configuration']['experiment_name'] for r in results]
    print(f"Experiments: {experiment_names}")

    # 1. Test metrics comparison
    print("\nGenerating test metrics comparison...")
    test_metrics = extract_metrics(results, 'test_results')
    create_metrics_comparison_chart(
        test_metrics,
        experiment_names,
        'Test Results: Metrics Comparison Across Experiments',
        f'{output_dir}/test_metrics_comparison.png'
    )
    # 1a. Test metrics table
    print("Generating test metrics table...")
    create_metrics_table(
        test_metrics,
        experiment_names,
        'Test Results: Main Metrics Comparison',
        f'{output_dir}/test_metrics_table.png'
    )

    # 2. In-domain test metrics comparison
    print("Generating in-domain test metrics comparison...")
    val_metrics = extract_metrics(results, 'in_domain_test_results')
    create_metrics_comparison_chart(
        val_metrics,
        experiment_names,
        'In-Domain Test Results: Metrics Comparison Across Experiments',
        f'{output_dir}/in_domain_test_metrics_comparison.png'
    )
    # 2a. In-Domain metrics table
    print("Generating in-domain test metrics table...")
    create_metrics_table(
        val_metrics,
        experiment_names,
        'In-Domain Test Results: Main Metrics Comparison',
        f'{output_dir}/in_domain_test_metrics_table.png'
    )

    # 3. Test per-class F1 comparison
    print("Generating test per-class F1 comparison...")
    test_per_class_f1 = extract_per_class_f1(results, 'test_results')
    create_per_class_f1_chart(
        test_per_class_f1,
        experiment_names,
        'Test Results: F1 Score per CEFR Level',
        f'{output_dir}/test_f1_per_class.png'
    )
    create_per_class_f1_table(
        test_per_class_f1,
        experiment_names,
        'Test Results: F1 Score per CEFR Level (Table)',
        f'{output_dir}/test_f1_per_class_table.png'
    )

    # 4. In-Domain per-class F1 comparison
    print("Generating in-domain per-class F1 comparison...")
    val_per_class_f1 = extract_per_class_f1(results, 'in_domain_test_results')
    # create_per_class_f1_chart(
    #     val_per_class_f1,
    #     experiment_names,
    #     'In-Domain Results: F1 Score per CEFR Level',
    #     f'{output_dir}/in_domain_test_f1_per_class.png'
    # )
    create_per_class_f1_table(
        val_per_class_f1,
        experiment_names,
        'In-Domain Results: F1 Score per CEFR Level (Table)',
        f'{output_dir}/in_domain_test_f1_per_class_table.png'
    )

    # 5. Per-corpus accuracy comparison
    print("Generating per-corpus accuracy comparison...")
    per_corpus_accuracy = extract_per_corpus_metrics(results)
    create_per_corpus_chart(
        per_corpus_accuracy,
        experiment_names,
        'Per-Corpus Accuracy Comparison',
        f'{output_dir}/per_corpus_accuracy.png'
    )
    create_per_corpus_table(
        per_corpus_accuracy,
        experiment_names,
        'Per-Corpus Accuracy (Table)',
        f'{output_dir}/per_corpus_accuracy_table.png'
    )

    print(f"\nAll comparisons generated successfully in '{output_dir}/'")


if __name__ == '__main__':
    json_paths = [
        'results/all/Experiment0_NaiveBayes_baseline/results.json',
        'results/all/Experiment0_Word2Vec_baseline/results.json',
        'results/all/Experiment0_RoBERTa_baseline/results.json',
    ]
    output_dir = 'results/comparison_results/all/baseline_comparison'

    # json_paths = [
    #     'results/all/Experiment0_NaiveBayes_baseline/results.json',
    #     'results/all/Experiment1_NaiveBayes_no_stopwords/results.json',
    #     'results/all/Experiment2_NaiveBayes_unigrams-only/results.json',
    #     'results/all/Experiment3_NaiveBayes_bigrams-only/results.json',
    #     'results/all/Experiment4_NaiveBayes_count/results.json',
    #     'results/all/Experiment5_NaiveBayes_large_vocab/results.json'
    # ]
    # output_dir = 'results/comparison_results/all/naive_bayes_comparison/'

    # json_paths = [
    #     'results/all/Experiment0_Word2Vec_baseline/results.json',
    #     'results/all/Experiment1_Word2Vec_google_news_w2v/results.json',
    #     'results/all/Experiment2_Doc2Vec/results.json',
    #     'results/all/Experiment3_Word2Vec_deeper_network/results.json',
    #     'results/all/Experiment4_Word2Vec_more_epochs/results.json',
    # ]
    # output_dir = 'results/comparison_results/all/word2vec_comparison/'

    # json_paths = [
    #     'results/all/Experiment0_RoBERTa_baseline/results.json',
    #     'results/all/Experiment1_DistilRoBERTa/results.json',
    #     'results/all/Experiment2_RoBERTa_short_sequences/results.json',
    #     'results/all/Experiment3_RoBERTa_frozen_encoder/results.json',
    #     'results/all/Experiment4_RoBERTa_careful_tuning/results.json',
    # ]

    output_dir = 'results/comparison_results/all/per_corpus_comparison_results/'
    compare(json_paths, output_dir)