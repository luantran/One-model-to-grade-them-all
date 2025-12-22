import pandas as pd
import os
import sys
from pathlib import Path

"""Utility for merging multiple CSV files with provenance tracking, ID regeneration, and level standardization."""

def get_csv_files(input_directory, exclude_patterns=None):
    """
    Get list of CSV files from directory, optionally excluding files matching patterns.
    """
    if not os.path.exists(input_directory):
        print(f"Error: Directory '{input_directory}' does not exist")
        return []

    # Get all CSV files
    all_csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

    # Filter out excluded patterns
    if exclude_patterns:
        csv_files = []
        excluded_files = []

        for file in all_csv_files:
            # Check if file contains any exclude pattern
            should_exclude = any(pattern in file for pattern in exclude_patterns)

            if should_exclude:
                excluded_files.append(file)
            else:
                csv_files.append(file)

        if excluded_files:
            print(f"\nExcluded {len(excluded_files)} files based on patterns {exclude_patterns}:")
            for file in sorted(excluded_files):
                print(f"  - {file}")
    else:
        csv_files = all_csv_files

    return sorted(csv_files)


def read_csv_file(file_path):
    """
    Read a single CSV file with error handling.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"  Error reading file: {e}")
        return None


def add_provenance_column(df, filename, provenance_col='source_file'):
    """
    Add a provenance column to track which file the data came from.
    """
    # Remove .csv extension for cleaner provenance names
    file_label = filename.replace('.csv', '')

    df_with_provenance = df.copy()

    # Add provenance column
    df_with_provenance[provenance_col] = file_label

    # Reorder columns: source_file first, then everything else (except id which will be added later)
    cols = list(df_with_provenance.columns)

    # Remove provenance_col and id (if exists) from their current positions
    cols.remove(provenance_col)
    if 'id' in cols:
        cols.remove('id')

    # Put provenance first, then the rest (id will be regenerated later)
    new_cols = [provenance_col] + cols

    return df_with_provenance[new_cols]


def regenerate_ids(df, id_col='id', start_id=0):
    """
    Regenerate IDs starting from a specified number.
    """
    df_with_new_ids = df.copy()

    # Generate new sequential IDs
    df_with_new_ids[id_col] = range(start_id, start_id + len(df_with_new_ids))

    # Move id column to the front
    cols = list(df_with_new_ids.columns)
    cols.remove(id_col)
    new_cols = [id_col] + cols

    return df_with_new_ids[new_cols]


def merge_c1_c2_levels(df, level_col='level'):
    """
    Merge C1 and C2 levels into a single 'C1/C2' label.
    """
    if level_col not in df.columns:
        return df

    df_merged = df.copy()

    # Count original C1 and C2
    c1_count = len(df_merged[df_merged[level_col] == 'C1'])
    c2_count = len(df_merged[df_merged[level_col] == 'C2'])

    # Replace both C1 and C2 with 'C1/C2'
    df_merged[level_col] = df_merged[level_col].replace(['C1', 'C2'], 'C1/C2')

    if c1_count > 0 or c2_count > 0:
        print(f"  Merged C1 ({c1_count:,}) and C2 ({c2_count:,}) → C1/C2 ({c1_count + c2_count:,})")

    return df_merged


def print_file_info(filename, df, has_provenance=False):
    """
    Print information about a loaded CSV file.
    """
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    if has_provenance:
        print(f"  Column names (with provenance): {list(df.columns)}")
    else:
        print(f"  Column names: {list(df.columns)}")


def print_summary_statistics(combined_df, num_files, provenance_col='source_file', level_col='level', id_col='id'):
    """
    Print summary statistics for the combined dataframe.
    """
    print("\n" + "=" * 80)
    print("COMBINED DATASET STATISTICS")
    print("=" * 80)
    print(f"Files combined: {num_files}")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Total columns: {len(combined_df.columns)}")
    print(f"Column names: {list(combined_df.columns)}")

    # Show ID range
    if id_col in combined_df.columns:
        print(f"\n" + "-" * 80)
        print(f"ID INFORMATION")
        print("-" * 80)
        print(f"  ID range: {combined_df[id_col].min()} to {combined_df[id_col].max()}")
        print(f"  Total IDs: {combined_df[id_col].nunique():,}")

    # Show distribution by source file
    if provenance_col in combined_df.columns:
        print(f"\n" + "-" * 80)
        print(f"DISTRIBUTION BY SOURCE FILE")
        print("-" * 80)
        source_counts = combined_df[provenance_col].value_counts().sort_index()
        for source, count in source_counts.items():
            pct = (count / len(combined_df)) * 100
            print(f"  {source}: {count:,} samples ({pct:.2f}%)")

    # Show distribution by level
    if level_col in combined_df.columns:
        print(f"\n" + "-" * 80)
        print(f"DISTRIBUTION BY LEVEL")
        print("-" * 80)
        level_counts = combined_df[level_col].value_counts().sort_index()
        for level, count in level_counts.items():
            pct = (count / len(combined_df)) * 100
            print(f"  {level}: {count:,} samples ({pct:.2f}%)")

    # Check for missing values
    missing_counts = combined_df.isnull().sum()
    if missing_counts.any():
        print(f"\n" + "-" * 80)
        print(f"MISSING VALUES")
        print("-" * 80)
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count:,} ({count / len(combined_df) * 100:.1f}%)")


def print_sample_data(combined_df, num_rows=3):
    """
    Print sample rows from the combined dataframe.
    """
    print(f"\n" + "-" * 80)
    print(f"SAMPLE DATA")
    print("-" * 80)
    print(f"\nFirst {num_rows} rows:")
    print(combined_df.head(num_rows).to_string())
    print(f"\nLast {num_rows} rows:")
    print(combined_df.tail(num_rows).to_string())


def save_combined_csv(combined_df, output_file):
    """
    Save the combined dataframe to CSV.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated output directory: {output_dir}")

    # Save to CSV
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nCombined file saved to: {output_file}")
    print(f"Total samples saved: {len(combined_df):,}")


def combine_csv_files(input_directory, output_file, exclude_patterns=None,
                      provenance_col='source_file', add_provenance=True,
                      id_col='id', regenerate_id=True, start_id=0,
                      level_col='level', merge_c1_c2=True):
    """
    Combine multiple CSV files from a directory into a single file with provenance tracking.
    """
    print("=" * 80)
    print("CSV FILE COMBINER WITH PROVENANCE TRACKING")
    print("=" * 80)
    print(f"Input directory: {input_directory}")
    print(f"Output file: {output_file}")
    if exclude_patterns:
        print(f"Excluding patterns: {exclude_patterns}")
    if add_provenance:
        print(f"Provenance column: '{provenance_col}'")
    if regenerate_id:
        print(f"Regenerating IDs: Starting from {start_id}")
    if merge_c1_c2:
        print(f"Merging C1 and C2 levels into: 'C1/C2'")
    print("=" * 80)

    # Get CSV files (excluding specified patterns)
    csv_files = get_csv_files(input_directory, exclude_patterns)

    if len(csv_files) == 0:
        print(f"\nNo CSV files found in {input_directory} after applying filters")
        return None

    print(f"\nFound {len(csv_files)} CSV files to combine:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")

    # Read and combine all CSV files
    dataframes = []
    successful_files = []

    print("\n" + "=" * 80)
    print("READING FILES AND PROCESSING")
    print("=" * 80)

    for file in csv_files:
        file_path = os.path.join(input_directory, file)
        print(f"\nReading: {file}")

        df = read_csv_file(file_path)

        if df is not None:
            # Merge C1/C2 if requested
            if merge_c1_c2 and level_col in df.columns:
                df = merge_c1_c2_levels(df, level_col)

            # Add provenance column
            if add_provenance:
                df_with_prov = add_provenance_column(df, file, provenance_col)
                print(f"  Added provenance column: '{provenance_col}' = '{file.replace('.csv', '')}'")
                print_file_info(file, df_with_prov, has_provenance=True)
                dataframes.append(df_with_prov)
            else:
                print_file_info(file, df, has_provenance=False)
                dataframes.append(df)

            successful_files.append(file)

    if not dataframes:
        print("\nNo dataframes to combine!")
        return None

    # Concatenate all dataframes
    print("\n" + "=" * 80)
    print("COMBINING FILES")
    print("=" * 80)

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Regenerate IDs if requested
    if regenerate_id:
        print("\n" + "=" * 80)
        print("REGENERATING IDs")
        print("=" * 80)
        print(f"Generating sequential IDs starting from {start_id}")
        combined_df = regenerate_ids(combined_df, id_col, start_id)
        print(f"New ID range: {start_id} to {start_id + len(combined_df) - 1}")
        print(f"Column order: ['{id_col}', '{provenance_col}', ...]")

    # Print statistics
    print_summary_statistics(combined_df, len(successful_files), provenance_col, level_col, id_col)

    # Print sample data
    print_sample_data(combined_df, num_rows=3)

    # Save the combined dataframe
    print("\n" + "=" * 80)
    print("SAVING COMBINED FILE")
    print("=" * 80)
    save_combined_csv(combined_df, output_file)

    return combined_df


def main():
    """
    Main function to handle command line arguments and execute combination.
    """
    # Default config
    input_directory = '../dataset/'
    output_file = '../../dataset/merged/dataset_merged.csv'

    # Patterns exclude
    exclude_patterns = ['_all', '_native']

    # Column names
    id_column_name = 'id'
    provenance_column_name = 'source_file'
    level_column_name = 'level'

    # Settings
    add_provenance = True
    regenerate_ids = True
    start_id = 0
    merge_c1_c2 = True

    # Combine files
    combined_df = combine_csv_files(
        input_directory=input_directory,
        output_file=output_file,
        exclude_patterns=exclude_patterns,
        provenance_col=provenance_column_name,
        add_provenance=add_provenance,
        id_col=id_column_name,
        regenerate_id=regenerate_ids,
        start_id=start_id,
        level_col=level_column_name,
        merge_c1_c2=merge_c1_c2
    )

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print(f"\nThe combined dataset includes:")
    print(f"  {len(combined_df):,} total samples")
    print(f"  {combined_df[provenance_column_name].nunique()} source files")
    print(f"  Column order: '{id_column_name}' (first), '{provenance_column_name}' (second)")
    if regenerate_ids:
        print(f"  IDs regenerated: {start_id} to {start_id + len(combined_df) - 1}")
    if merge_c1_c2 and level_column_name in combined_df.columns:
        c1c2_count = len(combined_df[combined_df[level_column_name] == 'C1/C2'])
        if c1c2_count > 0:
            print(f"  C1 and C2 merged into 'C1/C2': {c1c2_count:,} samples")
    print(f"\nYou can now:")
    print(f"  Filter by source: df[df['{provenance_column_name}'] == 'filename']")
    print(f"  Group by source: df.groupby('{provenance_column_name}')")
    print(f"  Filter by level: df[df['{level_column_name}'] == 'C1/C2']")
    print(f"  Filter by id: df[df['{id_column_name}'] == some_number]")
    print("=" * 80)


if __name__ == "__main__":
    main()