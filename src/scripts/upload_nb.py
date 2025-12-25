"""
Upload Naive Bayes model to HuggingFace Hub
"""
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import argparse
from pathlib import Path
from datetime import datetime
import logging
import os

from src.scripts.upload import upload_directory_to_hf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_naive_bayes(
        model_path: str,
        vectorizer_path: str,
        repo_id: str,
        private: bool = False,
        token: str = None
):
    """
    Upload Naive Bayes model to HuggingFace Hub

    Args:
        model_path: Path to Naive Bayes model (.joblib file)
        repo_id: HuggingFace repo ID (format: username/repo-name)
        private: Whether to make repo private (default: False)
        token: HuggingFace token (if not provided, uses HF_TOKEN env var)

    Returns:
        URL of the uploaded model
    """
    # Get token from env if not provided
    if token is None:
        token = os.getenv('HF_TOKEN')
        if not token:
            raise ValueError("HF_TOKEN not found. Set it as env var or pass --token")

    api = HfApi(token=token)

    # Step 1: Create repo if it doesn't exist
    logger.info(f"Checking repo '{repo_id}'...")
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        logger.info(f"✓ Repo exists")
    except RepositoryNotFoundError:
        logger.info(f"Creating repo '{repo_id}'...")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token
        )
        logger.info(f"✓ Created {'private' if private else 'public'} repo")

    # Step 2: Upload model
    logger.info("Uploading Naive Bayes model...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.pkl",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    logger.info("Uploading vectorizer...")
    api.upload_file(
        path_or_fileobj=vectorizer_path,
        path_in_repo="vectorizer.pkl",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update vectorizer - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    logger.info("✓ Model uploaded")

    # Step 3: Create README
    logger.info("Creating README...")
    readme = \
f"""---
tags:
- text-classification
- cefr
- naive-bayes
language:
- en
license: mit
---

# CEFR Naive Bayes Classifier

Naive Bayes model for classifying text by CEFR proficiency levels.

## Labels
- **A1**: Beginner
- **A2**: Elementary  
- **B1**: Intermediate
- **B2**: Upper Intermediate
- **C1/C2**: Advanced

## Usage
```python
from huggingface_hub import hf_hub_download
import joblib

# Download model
model_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="model.joblib"
)

# Load model
model = joblib.load(model_path)

# Predict
text = "This is a sample text to classify"
prediction = model.predict([text])[0]
probabilities = model.predict_proba([text])[0]

print(f"Predicted level: {{prediction}}")
print(f"Confidence: {{max(probabilities):.2%}}")
```

## Model Info
- **Type**: Multinomial Naive Bayes
- **Framework**: scikit-learn
- **Task**: Multi-class text classification

Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update README"
    )
    logger.info("✓ README uploaded")

    # Success
    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"✓ SUCCESS!")
    logger.info(f"✓ Model URL: {url}")
    logger.info(f"✓ Visibility: {'Private' if private else 'Public'}")
    logger.info(f"{'=' * 60}\n")

    return url


def main():
    parser = argparse.ArgumentParser(
        description="Upload Naive Bayes model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to Naive Bayes model (.pkl file)"
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        required=True,
        help="Path to vectorizer (.pkl file)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="theluantran/cefr-naive-bayes",
        help="HuggingFace repo ID (default: theluantran/cefr-naive-bayes)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repo private (default: public)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (default: uses HF_TOKEN env var)"
    )

    args = parser.parse_args()

    # Validate model exists
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Validate vectorizer exists
    vectorizer_path = Path(args.vectorizer)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {vectorizer_path}")

    # Upload
    upload_naive_bayes(
        model_path=str(model_path),
        vectorizer_path=str(vectorizer_path),
        repo_id=args.repo_id,
        private=args.private,
        token=args.token
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload directory to HuggingFace Hub"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to directory to upload"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default='theluantran/cefr-naive-bayes',
        help="HuggingFace repo ID (e.g., luantran/my-model)"
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default=".",
        help="Path within repo to place files (default: root)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repo private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (default: uses HF_TOKEN env var)"
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Patterns to ignore (e.g., *.pyc __pycache__)"
    )

    args = parser.parse_args()

    upload_directory_to_hf(
        directory_path=args.dir,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        private=args.private,
        token=args.token,
        ignore_patterns=args.ignore
    )

    # Step 3: Create README
    logger.info("Creating README...")
    readme = \
    f"""---
tags:
- text-classification
- cefr
- naive-bayes
language:
- en
license: mit
---

# CEFR Naive Bayes Classifier

Naive Bayes model for classifying text by CEFR proficiency levels.

## Labels
- **A1**: Beginner
- **A2**: Elementary  
- **B1**: Intermediate
- **B2**: Upper Intermediate
- **C1/C2**: Advanced

## Usage
```python
from huggingface_hub import hf_hub_download
import joblib

# Download model
model_path = hf_hub_download(
    repo_id="{args.repo_id}",
    filename="model.joblib"
)

# Load model
model = joblib.load(model_path)

# Predict
text = "This is a sample text to classify"
prediction = model.predict([text])[0]
probabilities = model.predict_proba([text])[0]

print(f"Predicted level: {{prediction}}")
print(f"Confidence: {{max(probabilities):.2%}}")
```

## Model Info
- **Type**: Multinomial Naive Bayes
- **Framework**: scikit-learn
- **Task**: Multi-class text classification

Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    api = HfApi(token=args.token)

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Update README"
    )
    logger.info("✓ README uploaded")

    # Success
    url = f"https://huggingface.co/{args.repo_id}"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"✓ SUCCESS!")
    logger.info(f"✓ Model URL: {url}")
    logger.info(f"✓ Visibility: {'Private' if args.private else 'Public'}")
    logger.info(f"{'=' * 60}\n")