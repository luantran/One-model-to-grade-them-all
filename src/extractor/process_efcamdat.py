import pandas as pd

"""EFCAMDAT corpus extractor that loads learner writing datasets (CSV/TSV/Excel),
 standardizes columns, and exports to unified format."""

def load_corpus_file(filepath, file_format='csv'):
    """
    Load corpus file into a pandas DataFrame
    """
    if file_format == 'csv':
        df = pd.read_csv(filepath)
    elif file_format == 'tsv':
        df = pd.read_csv(filepath, sep='\t')
    elif file_format == 'xlsx':
        df = pd.read_excel(filepath)


    print(f"Successfully loaded file: {filepath}")
    print(f"Total samples: {len(df):,}")
    print(f"Columns found: {list(df.columns)}\n")
    return df

def extract_and_prepare_corpus(df, native_lang_col=None, prompt_col='prompt',
                               answer_col='answer', level_col='level',
                               raw_level_col=None, id_prefix='sample'):
    """
    Extract required columns and auto-generate IDs
    """
    print("Extracting and preparing corpus...")

    # Create output dataframe
    output_df = pd.DataFrame()

    # Auto-generate IDs
    output_df['id'] = [f"{id_prefix}_{i+1:06d}" for i in range(len(df))]
    print(f"  ✓ Generated {len(df):,} IDs (format: {id_prefix}_XXXXXX)")

    # Native language
    if native_lang_col and native_lang_col in df.columns:
        output_df['native_language'] = df[native_lang_col]
        print(f"  ✓ Extracted native_language from column '{native_lang_col}'")
    else:
        output_df['native_language'] = 'Unknown'
        print(f"No native_language column specified, using 'Unknown'")

    # Prompt
    output_df['prompt'] = df[prompt_col]
    print(f"  ✓ Extracted prompt from column '{prompt_col}'")

    # Answer
    output_df['answer'] = df[answer_col]
    print(f"  ✓ Extracted answer from column '{answer_col}'")

    # Level
    output_df['level'] = df[level_col]
    print(f"  ✓ Extracted level from column '{level_col}'")

    # Raw level
    if raw_level_col and raw_level_col in df.columns:
        output_df['raw_level'] = df[raw_level_col]
        print(f"  ✓ Extracted raw_level from column '{raw_level_col}'")
    else:
        output_df['raw_level'] = df[level_col]  # Use same as level if not specified
        print(f"No raw_level column specified, copying from level")

    print(f"\nOutput DataFrame created with {len(output_df):,} samples")
    return output_df


def get_statistics(df, level_col='level', prompt_col='prompt'):
    """
    Print basic statistics about the corpus
    """
    print("\n" + "=" * 80)
    print("CORPUS STATISTICS")
    print("=" * 80)

    print(f"\nTotal samples: {len(df):,}")

    # Level distribution
    print(f"\nLevel Distribution:")
    print("-" * 40)
    level_counts = df[level_col].value_counts().sort_index()
    for level, count in level_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {level}: {count:,} ({pct:.2f}%)")

    # Prompt distribution (top 10)
    print(f"\nTop 10 Prompts:")
    print("-" * 40)
    prompt_counts = df[prompt_col].value_counts().head(10)
    for prompt, count in prompt_counts.items():
        pct = (count / len(df)) * 100
        prompt_short = str(prompt)[:50] + "..." if len(str(prompt)) > 50 else str(prompt)
        print(f"  {prompt_short}: {count:,} ({pct:.2f}%)")

    total_prompts = df[prompt_col].nunique()
    if total_prompts > 10:
        print(f"  ... and {total_prompts - 10} more prompts")

    # Native language distribution
    if 'native_language' in df.columns:
        unique_langs = df['native_language'].nunique()
        print(f"\nNative Languages: {unique_langs} unique language(s)")
        if unique_langs <= 20:
            print("-" * 40)
            lang_counts = df['native_language'].value_counts()
            for lang, count in lang_counts.items():
                pct = (count / len(df)) * 100
                print(f"  {lang}: {count:,} ({pct:.2f}%)")

    print("\n" + "=" * 80)


def save_corpus(df, output_filepath, format='csv'):
    """
    Save corpus to file
    """
    if format == 'csv':
        df.to_csv(output_filepath, index=False, encoding='utf-8')
    elif format == 'tsv':
        df.to_csv(output_filepath, index=False, encoding='utf-8', sep='\t')


    print(f"\n✓ Corpus saved to: {output_filepath}")
    print(f"✓ Total samples saved: {len(df):,}")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Format: {format.upper()}")



def process_corpus(input_file, output_file, file_format='csv', output_format='csv',
                  native_lang_col=None, prompt_col='prompt', answer_col='answer',
                  level_col='level', raw_level_col=None, id_prefix='sample',
                  show_statistics=True):
    """
    Complete workflow: load, extract, prepare, and save corpus
    """
    print("\n" + "=" * 80)
    print("CORPUS EXTRACTION WORKFLOW")
    print("=" * 80 + "\n")

    # Step 1: Load corpus
    print("Step 1: Loading corpus...")
    df = load_corpus_file(input_file, file_format)

    # Step 2: Extract and prepare
    print("\nStep 2: Extracting required columns and generating IDs...")
    output_df = extract_and_prepare_corpus(
        df, native_lang_col, prompt_col, answer_col,
        level_col, raw_level_col, id_prefix
    )

    # Step 3: Show statistics
    if show_statistics:
        print("\nStep 3: Analyzing corpus...")
        get_statistics(output_df, 'level', 'prompt')

    # Step 4: Save
    print("\nStep 4: Saving corpus...")
    save_corpus(output_df, output_file, output_format)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80 + "\n")

    return output_df

if __name__ == "__main__":
    # ========== CONFIG ==========

    # Input file settings
    INPUT_FILE = "assets/EFCAMDAT/Final database (alternative prompts).xlsx"
    FILE_FORMAT = "xlsx"  # Options: 'csv', 'tsv', 'xlsx'

    # Output file settings
    OUTPUT_FILE = "dataset/efcamdat_alternate.csv"
    OUTPUT_FORMAT = "csv"  # Options: 'csv', 'tsv'

    NATIVE_LANGUAGE_COLUMN = "l1"
    PROMPT_COLUMN = "topic"
    ANSWER_COLUMN = "text"
    LEVEL_COLUMN = "cefr"
    RAW_LEVEL_COLUMN = None

    ID_PREFIX = ""

    # Display settings
    SHOW_STATISTICS = True

    # ========== RUN EXTRACTION (ALTERNATE) ==========

    corpus = process_corpus(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        file_format=FILE_FORMAT,
        output_format=OUTPUT_FORMAT,
        native_lang_col=NATIVE_LANGUAGE_COLUMN,
        prompt_col=PROMPT_COLUMN,
        answer_col=ANSWER_COLUMN,
        level_col=LEVEL_COLUMN,
        raw_level_col=RAW_LEVEL_COLUMN,
        id_prefix=ID_PREFIX,
        show_statistics=SHOW_STATISTICS
    )

    print(f"✓ Processed corpus available as 'corpus' DataFrame")
    print(f"✓ Contains {len(corpus):,} samples with standardized columns")
    print(f"✓ Output saved to: {OUTPUT_FILE}\n")

    # ========== RUN EXTRACTION (MAIN)==========

    INPUT_FILE = "assets/EFCAMDAT/Final database (main prompts).xlsx"
    OUTPUT_FILE = "dataset/efcamdat_main.csv"

    corpus = process_corpus(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        file_format=FILE_FORMAT,
        output_format=OUTPUT_FORMAT,
        native_lang_col=NATIVE_LANGUAGE_COLUMN,
        prompt_col=PROMPT_COLUMN,
        answer_col=ANSWER_COLUMN,
        level_col=LEVEL_COLUMN,
        raw_level_col=RAW_LEVEL_COLUMN,
        id_prefix=ID_PREFIX,
        show_statistics=SHOW_STATISTICS
    )

    print(f"✓ Processed corpus available as 'corpus' DataFrame")
    print(f"✓ Contains {len(corpus):,} samples with standardized columns")
    print(f"✓ Output saved to: {OUTPUT_FILE}\n")
