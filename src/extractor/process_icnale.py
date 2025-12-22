import csv
import os
from pathlib import Path

"""Parser for ICNALE corpus text files that extracts metadata from filenames 
reads essay content, separates native/learner samples, and exports standardized CSV."""

def standardize_cefr_level(level):
    """
    Standardize CEFR levels by removing sub-level distinctions.
    """

    # Don't standardize XX levels (native speakers)
    if level.startswith('XX'):
        return level

    # Remove the underscore and everything after it for CEFR levels
    base_level = level.split('_')[0]
    return base_level


def is_native_speaker_sample(level):
    """
    Check if a sample is from a native speaker (XX level).
    """
    return level and level.startswith('XX')


def get_full_prompt_text(prompt_code):
    """
    Convert prompt code to full prompt text.
    """
    if 'PTJ' in prompt_code:
        prompt_code = 'PTJ'
    if 'SMK' in prompt_code:
        prompt_code = 'SMK'
    prompt_mapping = {
        'PTJ': 'It is important for college students to have a part-time job.',
        'SMK': 'Smoking should be completely banned at all the restaurants in the country.',
    }

    return prompt_mapping.get(prompt_code, None)


def parse_filename(filename):
    """
    Parse filename to extract metadata.
    """
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]

    # Split by underscore
    parts = name_without_ext.split('_')

    if len(parts) < 6:
        return {
            'corpus': None,
            'native_language': None,
            'prompt': None,
            'level': None,
            'raw_level': None,
            'is_native': False,
            'id': None,
        }

    corpus = parts[0]  # WE
    native_language = parts[1]  # CHN
    prompt_code = parts[2]  # PTJ
    file_id = int(parts[3])
    level_base = parts[4]  # A2 or XX
    level_sub = parts[5]  # 0, 1, 2, or 3

    # Combine level parts to get raw level
    raw_level = f"{level_base}_{level_sub}"

    # Check if native speaker sample
    is_native = is_native_speaker_sample(raw_level)

    # Standardize the level (keeps XX_* unchanged, removes suffix for CEFR)
    standardized_level = standardize_cefr_level(raw_level)

    # Get full prompt text
    full_prompt = get_full_prompt_text(prompt_code)

    return {
        'corpus': corpus,
        'native_language': native_language,
        'prompt': full_prompt,
        'level': standardized_level,
        'raw_level': raw_level,
        'is_native': is_native,
        'id': file_id,
    }


def parse_text_file(file_path):
    """
    Parse a single text file and extract content with metadata.
    """
    try:
        filename = os.path.basename(file_path)
        metadata = parse_filename(filename)

        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        entry = {
            'id': metadata['id'],
            'native_language': metadata['native_language'],
            'prompt': metadata['prompt'],
            'answer': content,
            'level': metadata['level'],
            'raw_level': metadata['raw_level'],
            'is_native': metadata['is_native'],
        }

        return entry

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def parse_text_directory(directory_path):
    """
    Parse all .txt files in a directory and extract their content.
    """
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return []

    # Find all text files
    txt_files = list(directory.rglob('*.txt'))

    if not txt_files:
        print(f"No text files found in {directory_path}")
        return []

    print(f"Found {len(txt_files)} text files in {directory_path}")

    all_results = []
    for txt_file in txt_files:
        result = parse_text_file(txt_file)
        if result:  # Only add successful parses
            all_results.append(result)

    return all_results


def separate_native_and_learner(results):
    """
    Separate results into native speaker samples (XX_*) and learner samples.
    """
    learner_samples = []
    native_samples = []

    for entry in results:
        if entry.get('is_native'):
            native_samples.append(entry)
        else:
            learner_samples.append(entry)

    return learner_samples, native_samples


def save_to_csv(results, output_file='output.csv'):
    """
    Save results to a CSV file.
    """

    fieldnames = ['id', 'native_language', 'prompt', 'answer', 'level', 'raw_level']

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} entries to {output_file}")


def print_sample_entries(results, num_samples=3):
    """
    Print sample entries from results for verification.
    """
    print(f"\n{'=' * 80}")
    print(f"Processed {len(results)} text entries successfully")
    print(f"{'=' * 80}\n")

    for i, data in enumerate(results[:num_samples], 1):
        native_str = " [NATIVE SPEAKER]" if data.get('is_native') else ""
        print(f"\n--- Entry {i}: ID={data['id']}{native_str} ---")
        print(f"Native Language: {data['native_language']}")
        print(f"Prompt: {data['prompt']}")
        print(f"Answer: {data['answer'][:100]}...")
        print(f"Level: {data['level']}")
        print(f"Raw Level: {data.get('raw_level', 'N/A')}")


def print_separation_statistics(learner_samples, native_samples):
    """
    Print statistics about learner vs native speaker samples.
    """
    print(f"\n{'=' * 80}")
    print("Sample Distribution:")
    print(f"{'=' * 80}")
    print(f"  Learner samples (CEFR levels): {len(learner_samples)} entries")
    print(f"  Native speaker samples (XX_*): {len(native_samples)} entries")
    print(f"  Total: {len(learner_samples) + len(native_samples)} entries")


def process_dataset(directory_path, output_csv, dataset_name="", separate_native=False):
    """
    Process a complete dataset: parse directory and save to CSV.
    """
    if dataset_name:
        print(f"\n{'#' * 80}")
        print(f"Processing {dataset_name}")
        print(f"{'#' * 80}")

    results = parse_text_directory(directory_path)

    if not results:
        print(f"No entries found for {dataset_name}")
        return

    print_sample_entries(results)

    if separate_native:
        # Separate native and learner samples
        learner_samples, native_samples = separate_native_and_learner(results)
        print_separation_statistics(learner_samples, native_samples)

        # Save to separate files
        base_path = Path(output_csv)
        base_dir = base_path.parent
        base_name = base_path.stem
        base_ext = base_path.suffix

        # Save learner samples
        learner_output = base_dir / f"{base_name}_learners{base_ext}"
        save_to_csv(learner_samples, learner_output)

        # Save native speaker samples
        if native_samples:
            native_output = base_dir / f"{base_name}_native{base_ext}"
            save_to_csv(native_samples, native_output)
        else:
            print("No native speaker samples found")

    else:
        # Save all to single file
        save_to_csv(results, output_csv)


def main():
    """
    Main function to process all ICNALE datasets.
    """
    datasets = [
        {
            'name': 'ICNALE WE 2.6',
            'path': '../assets/icnale/ICNALE_WE_2.6/WE_1_Classified_Unmerged/',
            'output': '../dataset/icnale_we.csv',
            'separate_native': True  # Set to True to separate native speakers
        },
        {
            'name': 'ICNALE WE UAE 1.0',
            'path': '../assets/icnale/ICNALE_Written_Essays_UAE_1.0/Unmerged',
            'output': '../dataset/icnale_we_uae.csv',
            'separate_native': True
        },
        {
            'name': 'ICNALE WEP 0.5',
            'path': '../assets/icnale/ICNALE_WEP_0.5/WEP_1_Classified_Unmerged',
            'output': '../dataset/icnale_wep.csv',
            'separate_native': True
        }
    ]

    for dataset in datasets:
        process_dataset(
            directory_path=dataset['path'],
            output_csv=dataset['output'],
            dataset_name=dataset['name'],
            separate_native=dataset.get('separate_native', False)
        )


if __name__ == "__main__":
    main()