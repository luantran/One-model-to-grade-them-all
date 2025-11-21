"""
Data utilities for text classification experiments.
Common functions for loading, cleaning, exploring, and splitting datasets.
Creates:
1. One stratified training file from EFCamDAT (for scikit-learn to split later)
2. One test file with all other corpora (for final generalization testing)
"""
import sys
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def log(message):
    """Print a formatted log message."""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)


def load_dataset(file_path, text_column='answer', label_column='level'):
    """
    Load a dataset from CSV.

    Parameters:
    -----------
    file_path : str
        Path to CSV file
    text_column : str
        Name of the text/answer column
    label_column : str
        Name of the label column

    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    log('Loading Dataset...')
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Verify required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataset")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")

    return df


def regenerate_id_column(df, id_col='id', start_id=0):
    """
    Regenerate IDs starting from a specified number.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to regenerate IDs for
    id_col : str
        Name of the ID column
    start_id : int
        Starting ID number (default: 0)

    Returns:
    --------
    pd.DataFrame : DataFrame with regenerated IDs
    """
    df_with_new_ids = df.copy()

    # Generate new sequential IDs
    df_with_new_ids[id_col] = range(start_id, start_id + len(df_with_new_ids))

    # Move id column to the front
    cols = list(df_with_new_ids.columns)
    if id_col in cols:
        cols.remove(id_col)
    new_cols = [id_col] + cols

    return df_with_new_ids[new_cols]


def split_by_corpus(df, train_corpus_pattern='efcamdat', source_col='source_file',
                   level_col='level', verbose=True):
    """
    Split dataset by corpus: training corpus vs. all other corpora for testing.

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset with source_file column
    train_corpus_pattern : str
        Pattern to identify training corpus (e.g., 'efcamdat')
    source_col : str
        Column name containing source file information
    level_col : str
        Column name containing level information
    verbose : bool
        Whether to print detailed information

    Returns:
    --------
    dict : Dictionary with train and test dataframes
        {
            'train': df_train (full training corpus),
            'test': df_test (all non-training corpora)
        }
    """
    if verbose:
        log("CORPUS-BASED SPLIT")
        print(f"Training corpus: '{train_corpus_pattern}'")
        print(f"Test corpora: All others")

    # Split into train and test by corpus
    df_train = df[df[source_col].str.contains(train_corpus_pattern, case=False)].copy()
    df_test = df[~df[source_col].str.contains(train_corpus_pattern, case=False)].copy()

    if verbose:
        print(f"\nTraining corpus ({train_corpus_pattern}):")
        print(f"  Total samples: {len(df_train):,}")

        train_sources = df_train[source_col].value_counts().sort_index()
        for source, count in train_sources.items():
            print(f"    {source}: {count:,}")

        print(f"\n  Level distribution:")
        train_levels = df_train[level_col].value_counts().sort_index()
        for level, count in train_levels.items():
            pct = (count / len(df_train)) * 100
            print(f"    {level}: {count:,} ({pct:.1f}%)")

        print(f"\nTest corpora (all others):")
        print(f"  Total samples: {len(df_test):,}")

        test_sources = df_test[source_col].value_counts().sort_index()
        for source, count in test_sources.items():
            print(f"    {source}: {count:,}")

        print(f"\n  Level distribution:")
        test_levels = df_test[level_col].value_counts().sort_index()
        for level, count in test_levels.items():
            pct = (count / len(df_test)) * 100
            print(f"    {level}: {count:,} ({pct:.1f}%)")

    return {
        'train': df_train,
        'test': df_test
    }


def stratified_sample(df, total_samples, level_col='level', source_col='source_file',
                     topic_col='prompt', random_state=42, verbose=True):
    """
    Sample data with hierarchical stratification: Level → Source → Topic

    Priority order:
    1. Equal distribution across levels
    2. Within each level, equal distribution across source files
    3. Within each level-source combination, equal distribution across topics

    Parameters:
    -----------
    df : pd.DataFrame
        The full dataset
    total_samples : int
        Total number of samples to extract
    level_col : str
        Column name for proficiency level
    source_col : str
        Column name for source file (provenance)
    topic_col : str
        Column name for topic/prompt
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print detailed sampling information

    Returns:
    --------
    pd.DataFrame : Stratified sample
    """
    if verbose:
        log(f"HIERARCHICAL STRATIFIED SAMPLING: {total_samples:,} samples")
        print(f"Priority: {level_col} → {source_col} → {topic_col}")

    # Get unique values at each level
    levels = sorted(df[level_col].unique())

    # Calculate samples per level (priority 1)
    samples_per_level = total_samples // len(levels)
    remainder_level = total_samples % len(levels)

    if verbose:
        print(f"\nLevel 1 - Distribution by {level_col}:")
        print(f"  Unique levels: {len(levels)} {levels}")
        print(f"  Samples per level: {samples_per_level}")
        print(f"  Remainder: {remainder_level}")

    all_sampled = []

    # For each level
    for level_idx, level in enumerate(levels):
        level_df = df[df[level_col] == level]

        # Allocate samples for this level (distribute remainder to first levels)
        level_target = samples_per_level + (1 if level_idx < remainder_level else 0)

        if verbose:
            print(f"\n  {level}: {len(level_df):,} available → targeting {level_target} samples")

        # Get unique sources within this level
        sources = sorted(level_df[source_col].unique())

        # Calculate samples per source (priority 2)
        samples_per_source = level_target // len(sources)
        remainder_source = level_target % len(sources)

        if verbose:
            print(f"    Level 2 - {len(sources)} source(s): {sources}")
            print(f"    Samples per source: {samples_per_source}")

        # For each source within this level
        for source_idx, source in enumerate(sources):
            level_source_df = level_df[level_df[source_col] == source]

            # Allocate samples for this level-source combination
            source_target = samples_per_source + (1 if source_idx < remainder_source else 0)

            if verbose:
                print(f"      {source}: {len(level_source_df):,} available → targeting {source_target}")

            # Get unique topics within this level-source combination
            topics = sorted(level_source_df[topic_col].unique())

            # Calculate samples per topic (priority 3)
            samples_per_topic = source_target // len(topics)
            remainder_topic = source_target % len(topics)

            if verbose and len(topics) <= 5:
                print(f"        Level 3 - {len(topics)} topic(s), {samples_per_topic} per topic")
            elif verbose:
                print(f"        Level 3 - {len(topics)} topics, {samples_per_topic} per topic")

            # For each topic within this level-source combination
            for topic_idx, topic in enumerate(topics):
                combo_df = level_source_df[level_source_df[topic_col] == topic]

                # Allocate samples for this level-source-topic combination
                topic_target = samples_per_topic + (1 if topic_idx < remainder_topic else 0)

                available = len(combo_df)

                # Sample (take all if insufficient, otherwise sample randomly)
                if available <= topic_target:
                    sample = combo_df
                else:
                    sample = combo_df.sample(n=topic_target, random_state=random_state)

                all_sampled.append(sample)

    # Combine all samples
    stratified_df = pd.concat(all_sampled, ignore_index=True)

    # Shuffle the final dataset
    stratified_df = stratified_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if verbose:
        print(f"\n" + "-" * 80)
        print(f"SAMPLING SUMMARY")
        print("-" * 80)
        print(f"Requested: {total_samples:,} samples")
        print(f"Actually sampled: {len(stratified_df):,} samples")
        print(f"Difference: {total_samples - len(stratified_df):,} samples")

        # Show distribution by level
        print(f"\nFinal distribution by {level_col}:")
        for level in levels:
            count = len(stratified_df[stratified_df[level_col] == level])
            print(f"  {level}: {count:,} samples")

        # Show distribution by source
        print(f"\nFinal distribution by {source_col}:")
        source_counts = stratified_df[source_col].value_counts().sort_index()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} samples")

    return stratified_df


def clean_data(df, text_column='answer', label_column='level', remove_empty=True):
    """
    Clean the dataset by removing missing values.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to clean
    text_column : str
        Name of the text column
    label_column : str
        Name of the label column
    remove_empty : bool
        Whether to remove empty text entries

    Returns:
    --------
    pd.DataFrame : Cleaned dataset
    """
    log("CLEANING DATA")

    initial_len = len(df)

    # Remove rows with missing text or labels
    df = df.dropna(subset=[text_column, label_column])

    # Remove empty text
    if remove_empty:
        df = df[df[text_column].str.strip() != '']

    final_len = len(df)
    removed = initial_len - final_len

    print(f"Removed {removed} rows with missing/empty data")
    print(f"Final dataset size: {final_len:,}")

    return df


def explore_data(df, label_column='level', text_column='answer', source_column='source_file'):
    """
    Explore the dataset and show statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to explore
    label_column : str
        Name of the label column
    text_column : str
        Name of the text column
    source_column : str
        Name of the source/provenance column

    Returns:
    --------
    pd.DataFrame : Same dataframe with added statistics
    """
    print(f"\nTotal samples: {len(df):,}")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.any():
        print(missing[missing > 0])
    else:
        print("  No missing values")

    print(f"\nLevel distribution:")
    level_counts = df[label_column].value_counts().sort_index()
    for level, count in level_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {level}: {count:,} ({pct:.1f}%)")

    if source_column in df.columns:
        print(f"\nSource file distribution:")
        source_counts = df[source_column].value_counts().sort_index()
        for source, count in source_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {source}: {count:,} ({pct:.1f}%)")


    if 'prompt' in df.columns:
        unique_prompts = df['prompt'].nunique()
        print(f"\nUnique prompts/topics: {unique_prompts}")

    print(f"\nText length statistics:")
    df['text_length'] = df[text_column].str.len()
    stats = df['text_length'].describe()
    print(f"  Mean: {stats['mean']:.1f} characters")
    print(f"  Median: {stats['50%']:.1f} characters")
    print(f"  Min: {stats['min']:.0f} characters")
    print(f"  Max: {stats['max']:.0f} characters")

    return df


def create_train_test_files(
    input_file,
    output_dir='../dataset/splits/',
    train_samples=100000,
    train_corpus='efcamdat',
    regenerate_ids=True,
    add_numeric_labels=True,
    random_state=42,
    verbose=True
):
    """
    Create training and test files for model training.

    Creates two files:
    1. train_{n}k.csv: Stratified samples from training corpus (for scikit-learn)
    2. test_other_corpora.csv: All samples from other corpora (for generalization)

    Parameters:
    -----------
    input_file : str
        Path to input CSV with all corpora
    output_dir : str
        Directory to save output files
    train_samples : int
        Number of samples for training (from training corpus)
    train_corpus : str
        Pattern to identify training corpus (default: 'efcamdat')
    regenerate_ids : bool
        Whether to regenerate IDs starting from 0 (default: True)
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print detailed information

    Returns:
    --------
    tuple : (train_df, test_df)
    """
    log("CREATING TRAINING AND TEST FILES")
    print(f"Input: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Training samples: {train_samples:,} (from {train_corpus})")
    if regenerate_ids:
        print(f"Regenerating IDs: Starting from 0")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load dataset
    df = load_dataset(input_file)

    # Step 2: Split by corpus
    corpus_split = split_by_corpus(
        df,
        train_corpus_pattern=train_corpus,
        source_col='source_file',
        level_col='level',
        verbose=verbose
    )

    df_train_full = corpus_split['train']
    df_test_all = corpus_split['test']

    # Step 3: Stratified sample from training corpus
    df_train = stratified_sample(
        df_train_full,
        total_samples=train_samples,
        level_col='level',
        source_col='source_file',
        topic_col='prompt',
        random_state=random_state,
        verbose=verbose
    )

    # Step 4: Clean both datasets
    log("CLEANING DATASETS")

    print("\nCleaning training data...")
    df_train_clean = clean_data(df_train, text_column='answer', label_column='level')

    print("\nCleaning test data...")
    df_test_clean = clean_data(df_test_all, text_column='answer', label_column='level')

    # Step 5: Add numeric label mapping
    if add_numeric_labels:
        log("ADDING NUMERIC LABEL MAPPING")

        # Label mapping: CEFR → numeric (5 classes with merged C1/C2)
        label_map = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1/C2': 4}

        print("\nLabel mapping:")
        for cefr, numeric in label_map.items():
            print(f"  {cefr} → {numeric}")

        # Add numeric label column
        df_train_clean['label_numeric'] = df_train_clean['level'].map(label_map)
        df_test_clean['label_numeric'] = df_test_clean['level'].map(label_map)

        # Verify mapping
        unmapped_train = df_train_clean['label_numeric'].isna().sum()
        unmapped_test = df_test_clean['label_numeric'].isna().sum()

        if unmapped_train > 0 or unmapped_test > 0:
            print(f"\nWARNING: Found unmapped labels!")
            print(f"  Training: {unmapped_train} unmapped")
            print(f"  Test: {unmapped_test} unmapped")

            # Show which labels couldn't be mapped
            if unmapped_train > 0:
                unmapped_labels = df_train_clean[df_train_clean['label_numeric'].isna()]['level'].unique()
                print(f"  Unmapped training labels: {list(unmapped_labels)}")
            if unmapped_test > 0:
                unmapped_labels = df_test_clean[df_test_clean['label_numeric'].isna()]['level'].unique()
                print(f"  Unmapped test labels: {list(unmapped_labels)}")
        else:
            print(f"\n✓ All labels mapped successfully")
            print(f"  Training: {len(df_train_clean):,} samples")
            print(f"  Test: {len(df_test_clean):,} samples")

    # Step 5: Regenerate IDs if requested
    if regenerate_ids:
        log("REGENERATING IDs")

        print("\nRegenerating training IDs...")
        df_train_clean = regenerate_id_column(df_train_clean, id_col='id', start_id=0)
        print(f"Training IDs: 0 to {len(df_train_clean) - 1}")

        print("\nRegenerating test IDs...")
        df_test_clean = regenerate_id_column(df_test_clean, id_col='id', start_id=0)
        print(f"Test IDs: 0 to {len(df_test_clean) - 1}")

    # Step 6: Save files
    log("SAVING FILES")

    # Determine filename based on sample size
    train_size_label = f"{train_samples // 1000}k" if train_samples >= 1000 else str(train_samples)
    train_file = os.path.join(output_dir, f'train_{train_size_label}.csv')
    test_file = os.path.join(output_dir, 'test_other_corpora.csv')

    df_train_clean.to_csv(train_file, index=False, encoding='utf-8')
    print(f"\nTraining file: {train_file}")
    print(f"  Samples: {len(df_train_clean):,}")
    print(f"  Levels: {dict(df_train_clean['level'].value_counts().sort_index())}")
    if regenerate_ids:
        print(f"  ID range: 0 to {len(df_train_clean) - 1}")

    df_test_clean.to_csv(test_file, index=False, encoding='utf-8')
    print(f"\nTest file: {test_file}")
    print(f"  Samples: {len(df_test_clean):,}")
    print(f"  Levels: {dict(df_test_clean['level'].value_counts().sort_index())}")
    print(f"  Corpora: {list(df_test_clean['source_file'].unique())}")
    if regenerate_ids:
        print(f"  ID range: 0 to {len(df_test_clean) - 1}")

    # Step 7: Explore both datasets
    if verbose:
        log("TRAINING DATA EXPLORATION")
        explore_data(df_train_clean, label_column='level',
                    text_column='answer', source_column='source_file')

        log("TEST DATA EXPLORATION")
        explore_data(df_test_clean, label_column='level',
                    text_column='answer', source_column='source_file')

    return df_train_clean, df_test_clean


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to create training and test datasets.
    """

    # ========== CONFIGURATION ==========

    INPUT_FILE = '../../dataset/merged/dataset_merged.csv'
    OUTPUT_DIR = '../../dataset/splits/'

    # Training parameters
    TRAIN_SAMPLES = 100000  # Change this to 50000, 150000, 200000, etc.
    TRAIN_CORPUS = 'efcamdat'  # Train only on EFCamDAT

    # ID regeneration
    REGENERATE_IDS = True  # Set to False to keep original IDs

    # Random seed
    RANDOM_STATE = 42

    # Verbose output
    VERBOSE = True

    # ========== RUN ==========

    try:
        train_df, test_df = create_train_test_files(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            train_samples=TRAIN_SAMPLES,
            train_corpus=TRAIN_CORPUS,
            regenerate_ids=REGENERATE_IDS,
            random_state=RANDOM_STATE,
            verbose=VERBOSE
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()