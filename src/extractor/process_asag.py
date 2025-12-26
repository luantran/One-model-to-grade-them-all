import csv
import os
import xml.etree.ElementTree as ET
from pathlib import Path

"""
XML parser for extracting learner metadata 
from ASAG corpus files and exporting to CSV.
"""

def extract_xml_data(xml_file):
    """
    Extract age, native language, prompt, answer, and majority vote from XML file.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define namespace (TEI uses a default namespace)
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'} if root.tag.startswith('{') else {}

    # Extract filename and convert to int (removing extension)
    filename = os.path.basename(xml_file)
    filename_without_ext = os.path.splitext(filename)[0]
    try:
        filename_int = int(filename_without_ext)
    except ValueError:
        # If filename is not a number, keep it as string
        filename_int = filename_without_ext

    # Extract age
    person = root.find('.//person[@role="participant"]', ns)
    if person is None:
        person = root.find('.//person[@role="participant"]')
    age_str = person.get('age') if person is not None else None
    age = int(age_str) if age_str is not None else None

    # Extract native language
    lang_known = root.find('.//langKnown[@level="native"]', ns)
    if lang_known is None:
        lang_known = root.find('.//langKnown[@level="native"]')
    native_language = lang_known.get('tag') if lang_known is not None else None

    # Extract prompt (question)
    question_div = root.find('.//div[@type="question"]', ns)
    if question_div is None:
        question_div = root.find('.//div[@type="question"]')
    prompt = question_div.find('.//p', ns) if question_div is not None else None
    prompt_text = prompt.text.strip() if prompt is not None and prompt.text else None

    # Extract answer - GET ALL <p> TAGS
    answer_div = root.find('.//div[@type="answer"]', ns)
    if answer_div is None:
        answer_div = root.find('.//div[@type="answer"]')

    answer_text = None
    if answer_div is not None:
        # Find all <p> tags within the answer div
        paragraphs = answer_div.findall('.//p', ns)
        if not paragraphs:
            paragraphs = answer_div.findall('p')

        # Extract text from all paragraphs and join with newlines
        if paragraphs:
            paragraph_texts = []
            for p in paragraphs:
                if p.text:
                    paragraph_texts.append(p.text.strip())
            answer_text = '\n'.join(paragraph_texts) if paragraph_texts else None

    # Extract majority vote
    majority_label = root.find('.//label[@subtype="majority-vote"]', ns)
    if majority_label is None:
        majority_label = root.find('.//label[@subtype="majority-vote"]')
    majority_span = majority_label.find('.//span', ns) if majority_label is not None else None
    if majority_span is None and majority_label is not None:
        majority_span = majority_label.find('span')
    majority_vote = majority_span.text.strip() if majority_span is not None and majority_span.text else None

    return {
        'id': filename_int,
        'native_language': native_language,
        'prompt': prompt_text,
        'answer': answer_text,
        'level': majority_vote
    }

def process_directory(directory_path):
    """
    Process all XML files in a directory.
    """
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return []

    # Find all XML files
    xml_files = list(directory.glob('*.xml'))

    if not xml_files:
        print(f"No XML files found in {directory_path}")
        return []

    print(f"Found {len(xml_files)} XML files")

    results = []
    for xml_file in xml_files:
        print(f"Processing: {xml_file.name}")
        data = extract_xml_data(xml_file)
        if data is not None:
            results.append(data)

    return results


def save_to_csv(results, output_file='output.csv'):
    """
    Save results to a CSV file.
    """

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,
                                fieldnames=['id', 'native_language', 'prompt', 'answer', 'level', 'raw_level'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Replace with your directory path
    directory_path = 'assets/asag/corpus/release-1.0/labelled'

    # Process all XML files
    results = process_directory(directory_path)

    # Print results
    print(f"\n{'=' * 80}")
    print(f"Processed {len(results)} files successfully")
    print(f"{'=' * 80}\n")

    for i, data in enumerate(results, 1):
        print(f"\n--- File {i}: {data['id']} ---")
        print(f"Native Language: {data['native_language']}")
        print(f"Prompt: {data['prompt']}")
        print(f"Answer: {data['answer'][:100]}...")
        print(f"Level: {data['level']}")

    # Optionally save to CSV
    save_to_csv(results, 'dataset/asag.csv')