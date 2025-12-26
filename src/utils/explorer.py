import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

"""
Explorer that loads csv files and displays statistics and generates chart for them

Many of the chart generation functions were written with the help of Generative AI (Claude)
"""

def explore_data(csv_file, label_column='level', text_column='answer', source_column='source_file',
                 filter_sources=None, plot=True, output_dir='.', dataset_name=''):
    """
    Load and explore a dataset from CSV with optional visualization.
    """
    print("\n" + "=" * 80)
    print(f"EXPLORING DATASET: {csv_file}")
    print("=" * 80)

    # Load dataset
    try:
        df = pd.read_csv(csv_file)
        print(f"\n✓ Successfully loaded: {csv_file}")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"\n✗ ERROR: File not found: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR loading file: {e}")
        sys.exit(1)

    # Filter by source if specified
    if filter_sources is not None and source_column in df.columns:
        original_size = len(df)
        df = df[df[source_column].isin(filter_sources)].copy()
        print(f"\n✓ Filtered to sources: {filter_sources}")
        print(f"  Original size: {original_size:,}")
        print(f"  Filtered size: {len(df):,} ({len(df) / original_size * 100:.1f}%)")

    # Display statistics
    print(f"\nTotal samples: {len(df):,}")

    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    # Check if label column exists
    if label_column in df.columns:
        print(f"\nLevel distribution:")
        level_counts = df[label_column].value_counts().sort_index()
        for level, count in level_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {level}: {count:,} ({pct:.1f}%)")
    else:
        print(f"\nWARNING: Label column '{label_column}' not found in dataset")

    # Check if source column exists
    if source_column in df.columns:
        print(f"\nSource file distribution:")
        source_counts = df[source_column].value_counts().sort_index()
        for source, count in source_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {source}: {count:,} ({pct:.1f}%)")
    else:
        print(f"\nNote: Source column '{source_column}' not found in dataset")


    # Generate plots if requested
    if plot and 'label_numeric' in df.columns:
        print(f"\n" + "-" * 80)
        print("GENERATING DISTRIBUTION PLOT")
        print("-" * 80)

        plot_class_distribution(df, output_dir, dataset_name)

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)

    return df


def plot_class_distribution(df, output_dir='.', dataset_name=''):
    """Generate bar chart for CEFR label distribution."""
    os.makedirs(output_dir, exist_ok=True)

    levels = ['A1', 'A2', 'B1', 'B2', 'C1/C2']
    colors = ['#4daf4a', '#377eb8', '#ff7f00', '#984ea3', '#e41a1c']

    y = df['label_numeric']
    counts = [(y == i).sum() for i in range(5)]
    percentages = [c / sum(counts) * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(levels, counts, color=colors)

    # Compute top margin dynamically
    top_margin = max(counts) * 0.12
    ax.set_ylim(0, max(counts) + top_margin)

    # Annotate counts and percentages (font size increased by 50%: 9 -> 13.5)
    for i, (count, perc) in enumerate(zip(counts, percentages)):
        ax.text(i, count + top_margin * 0.05, f"{count:,}\n({perc:.1f}%)",
                ha='center', va='bottom', fontsize=13.5)

    # Increase axis label font sizes by 50%
    ax.set_ylabel("Number of Samples", fontsize=18)
    ax.set_xlabel("CEFR Level", fontsize=18)

    # Increase tick label font sizes by 50%
    ax.tick_params(axis='both', which='major', labelsize=15)

    total_samples = len(df)
    # Increase title font size by 50%
    ax.set_title(f"{dataset_name}\n(Total: {total_samples:,} samples)", fontsize=18)
    plt.tight_layout()

    path = os.path.join(output_dir, f"cefr_distribution_{dataset_name}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {path}")
    plt.show()

if __name__ == "__main__":
    # ========== RUN ==========

    # df = explore_data(
    #     csv_file='dataset/splits/test_other_corpora.csv',
    #     label_column='level',
    #     text_column='answer',
    #     source_column='source_file',
    #     filter_sources=['asag', 'icnale_we_learners', 'icnale_we_uae_learners', 'icnale_wep_learners', 'write_improve_first_and_final'],
    #     output_dir='dataset/distribution',
    #     dataset_name='Aggregate Corpora - Out-Of-Domain Test Dataset'
    # )

    df = explore_data(
        csv_file='dataset/splits/train_100k.csv',
        label_column='level',
        text_column='answer',
        source_column='source_file',
        filter_sources=['efcamdat_main', 'efcamdat_alternate'],
        output_dir='dataset/distribution/',
        dataset_name='EFCAmDAT SubCorpus - Experiment Dataset'
    )

    print(f"\n✓ Dataset successfully loaded and explored!")
    print(f"  Shape: {df.shape}")

