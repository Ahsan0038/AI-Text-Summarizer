"""
nltk_setup.py – One-time NLTK data downloader
================================================
Run this script once before starting the app locally to ensure the
required NLTK corpora are available.

    python nltk_setup.py
"""

import nltk
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_CORPORA = [
    "stopwords",      # Used by the summarization pipeline
]

def download_nltk_data():
    """Download all required NLTK corpora if not already present."""
    for corpus in REQUIRED_CORPORA:
        try:
            nltk.data.find(f"corpora/{corpus}")
            logger.info("'%s' already downloaded – skipping.", corpus)
        except LookupError:
            logger.info("Downloading '%s' …", corpus)
            nltk.download(corpus, quiet=False)
            logger.info("'%s' downloaded successfully.", corpus)


if __name__ == "__main__":
    download_nltk_data()
    logger.info("NLTK setup complete.")
