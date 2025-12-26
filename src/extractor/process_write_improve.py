import os

import pandas as pd

"""
Parser for Write & Improve corpus that filters versions, maps prompts, combines human/automated CEFR annotations,
 standardizes levels, and exports.
 """

def load_data(corpus_path, prompts_path):
    """
    Load the corpus and prompts data.
    """
    print("Loading data...")
    df = pd.read_csv(corpus_path, index_col=0)
    prompts = pd.read_csv(prompts_path, sep='\t')
    print(f"  Corpus loaded: {df.shape}")
    print(f"  Prompts loaded: {prompts.shape}")
    return df, prompts


def filter_versions(df, version_type='first_and_final'):
    """
    Filter dataframe by version type.
    """
    print(f"\nFiltering versions: {version_type}")

    if version_type == 'first_and_final':
        filtered = df[
            (df['is_first_version'] == True) |
            (df['is_final_version'] == True)
            ]
    elif version_type == 'first':
        filtered = df[df['is_first_version'] == True]
    elif version_type == 'final':
        filtered = df[df['is_final_version'] == True]
    elif version_type == 'all':
        filtered = df.copy()
    else:
        raise ValueError(
            f"Invalid version_type: {version_type}. Choose from: 'first_and_final', 'first', 'final', 'all'")

    print(f"  Rows after filtering: {len(filtered)}")
    return filtered


def select_columns(df, columns=None):
    """
    Select specific columns from the dataframe.
    """
    if columns is None:
        columns = ['language', 'public_prompt_id', 'text', 'automarker_cefr_level', 'humannotator_cefr_level']

    print(f"\nSelecting columns: {columns}")
    return df[columns].copy()


def map_prompts(df, prompts_df, prompt_id_col='public_prompt_id'):
    """
    Map prompt IDs to actual prompt text.
    """
    print("\nMapping prompts...")
    lookup = prompts_df.set_index("public_prompt_id")["prompt"]
    df[prompt_id_col] = df[prompt_id_col].map(lookup)

    # Count how many prompts were successfully mapped
    mapped_count = df[prompt_id_col].notna().sum()
    print(f"  Successfully mapped {mapped_count}/{len(df)} prompts")

    return df


def combine_levels(df, human_col='humannotator_cefr_level', auto_col='automarker_cefr_level'):
    """
    Combine human and automated CEFR levels, prioritizing human annotations.
    """
    print("\nCombining CEFR levels...")
    df["raw_level"] = df[human_col].fillna(df[auto_col])

    human_count = df[human_col].notna().sum()
    auto_count = df[auto_col].notna().sum() - human_count
    total_count = df["raw_level"].notna().sum()

    print(f"  Human annotations: {human_count}")
    print(f"  Auto annotations used: {auto_count}")
    print(f"  Total with levels: {total_count}")

    # Drop the original columns
    df = df.drop(columns=[human_col, auto_col])

    return df


def standardize_cefr_level(level):
    """
    Standardize CEFR levels by removing plus modifiers.
    """
    if pd.isna(level):
        return level

    # Remove the '+' suffix
    return str(level).replace('+', '')


def standardize_levels(df, raw_level_col='raw_level'):
    """
    Create a standardized level column from raw levels.
    """
    print("\nStandardizing CEFR levels...")

    # Create standardized level column
    df['level'] = df[raw_level_col].apply(standardize_cefr_level)

    # Count changes
    changes = (df[raw_level_col] != df['level']).sum()
    print(f"  Standardized {changes} levels (removed '+' modifiers)")

    # Show before/after distribution
    print(f"\n  Raw level distribution:")
    raw_counts = df[raw_level_col].value_counts().sort_index()
    for level, count in raw_counts.items():
        print(f"    {level}: {count}")

    print(f"\n  Standardized level distribution:")
    std_counts = df['level'].value_counts().sort_index()
    for level, count in std_counts.items():
        print(f"    {level}: {count}")

    return df


def rename_columns(df, rename_dict=None):
    """
    Rename columns in the dataframe.
    """
    if rename_dict is None:
        rename_dict = {
            'public_prompt_id': 'prompt',
            'language': 'native_language',
            'text': 'answer'
        }

    print(f"\nRenaming columns: {rename_dict}")
    return df.rename(columns=rename_dict)


def add_id_column(df, id_col_name='id', position=0):
    """
    Add a sequential ID column to the dataframe.
    """
    print(f"\nAdding ID column '{id_col_name}' at position {position}")
    df.insert(position, id_col_name, range(len(df)))
    return df


def reorder_columns(df, column_order=None):
    """
    Reorder columns in the dataframe.
    """
    if column_order is None:
        column_order = ['id', 'native_language', 'prompt', 'answer', 'level', 'raw_level']

    # Only reorder columns that exist
    available_columns = [col for col in column_order if col in df.columns]

    print(f"\nReordering columns: {available_columns}")
    return df[available_columns]


def save_dataframe(df, output_path, index=False):
    """
    Save dataframe to CSV.
    """
    print(f"\nSaving to: {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=index)
    print(f"  ✓ Saved {len(df)} rows")


def process_write_improve(corpus_path, prompts_path, output_path, version_type='first_and_final'):
    """
    Complete pipeline to process Write & Improve corpus.
    """
    print("=" * 60)
    print("WRITE & IMPROVE CORPUS PROCESSOR")
    print("=" * 60)

    # Step 1: Load data
    df, prompts = load_data(corpus_path, prompts_path)

    # Step 2: Filter versions
    df = filter_versions(df, version_type=version_type)

    # Step 3: Select columns
    df = select_columns(df)

    # Step 4: Map prompts
    df = map_prompts(df, prompts)

    # Step 5: Rename columns
    df = rename_columns(df)

    # Step 6: Combine levels (creates raw_level)
    df = combine_levels(df)

    # Step 7: Standardize levels (creates level from raw_level)
    df = standardize_levels(df)

    # remove pre-A1
    df = df[df.level != 'pre-A1']

    # Step 8: Add ID column
    df = add_id_column(df)

    # Step 9: Reorder columns
    df = reorder_columns(df)

    # Step 10: Save
    save_dataframe(df, output_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nStandardized level distribution:")
    print(df['level'].value_counts().sort_index())
    print(f"\nRaw level distribution:")
    print(df['raw_level'].value_counts().sort_index())
    print(f"\nNative language distribution:")
    print(df['native_language'].value_counts())
    print("=" * 60)

    return df


# Main execution
if __name__ == "__main__":
    # Choose version type: 'first_and_final', 'first', 'final', or 'all'
    VERSION_TYPE = 'first_and_final'

    # Configuration
    CORPUS_PATH = '../../assets/write-improve/whole-corpus/en-writeandimprove2024-corpus.csv'
    PROMPTS_PATH = '../../assets/write-improve/whole-corpus/en-writeandimprove2024-prompts-info.tsv'
    OUTPUT_PATH = f'../dataset/write_improve_{VERSION_TYPE}.csv'

    # Process the data
    df = process_write_improve(
        corpus_path=CORPUS_PATH,
        prompts_path=PROMPTS_PATH,
        output_path=OUTPUT_PATH,
        version_type=VERSION_TYPE
    )