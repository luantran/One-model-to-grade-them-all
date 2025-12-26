"""
Upload entire directory to HuggingFace Hub
"""
import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_directory_to_hf(
        directory_path: str,
        repo_id: str,
        path_in_repo: str = ".",
        private: bool = False,
        token: str = None,
        ignore_patterns: list = None
):
    """
    Upload entire directory to HuggingFace Hub

    Args:
        directory_path: Local directory path to upload
        repo_id: HuggingFace repo ID (format: username/repo-name)
        path_in_repo: Where to place files in the repo (default: root ".")
        private: Whether to make repo private
        token: HuggingFace token
        ignore_patterns: List of patterns to ignore (e.g., ["*.pyc", "__pycache__"])

    Returns:
        URL of the repo
    """
    # Get token
    if token is None:
        token = os.getenv('HF_TOKEN')
        if not token:
            raise ValueError("HF_TOKEN not found")

    api = HfApi(token=token)

    # <root_dir>/src/scripts/upload_script.py
    ROOT_DIR = Path(__file__).resolve().parents[2]

    abs_dir_path = (
            ROOT_DIR
            / directory_path
    )

    # Validate directory exists
    if not abs_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {abs_dir_path}")

    if not abs_dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {abs_dir_path}")

    # Validate directory exists
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Create repo if doesn't exist
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

    # Upload entire folder
    logger.info(f"Uploading directory '{directory_path}' to '{repo_id}/{path_in_repo}'...")

    api.upload_folder(
        folder_path=str(dir_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload directory: {dir_path.name}",
        ignore_patterns=ignore_patterns or []
    )

    logger.info("✓ Directory uploaded successfully")


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

    # Success
    url = f"https://huggingface.co/{args.repo_id}"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"✓ SUCCESS!")
    logger.info(f"✓ Model URL: {url}")
    logger.info(f"✓ Visibility: {'Private' if args.private else 'Public'}")
    logger.info(f"{'=' * 60}\n")