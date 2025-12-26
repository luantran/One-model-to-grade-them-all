"""
Upload Naive Bayes model to HuggingFace Hub
"""
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from src.scripts.upload import upload_directory_to_hf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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