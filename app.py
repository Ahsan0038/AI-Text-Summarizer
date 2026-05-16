"""
Text Summarizer - Flask Web Application
========================================
Uses TextRank algorithm (NLTK + NetworkX) to extract key sentences.
Supports both HTML form-based usage and a JSON API endpoint for Postman testing.
Threading is used to run the summarization concurrently without blocking the main thread.
"""

import threading
import logging
import re

import numpy as np
import networkx as nx
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, render_template, jsonify
from typing import List, Optional

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK data – download stopwords once if not already present
# ---------------------------------------------------------------------------
try:
    stopwords.words("english")
except LookupError:
    logger.info("Downloading NLTK stopwords …")
    nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Core NLP helper functions
# ---------------------------------------------------------------------------

def read_article(text: str) -> List[List[str]]:
    """
    Split raw text into a list of tokenised sentences.

    Args:
        text (str): The raw input text.

    Returns:
        List[List[str]]: A list of sentences, each sentence being a list of
                         cleaned word tokens.

    Raises:
        ValueError: If text is not a non-empty string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    # Split on '. ' so we preserve sentence boundaries
    article = text.split(". ")
    sentences: List[List[str]] = []

    for sentence in article:
        # Keep only alphanumeric characters and spaces
        cleaned = re.sub(r"[^A-Za-z0-9]", " ", sentence)
        tokens = cleaned.split()
        if tokens:                      # Skip blank sentences
            sentences.append(tokens)

    return sentences


def sentence_similarity(sent1: List[str], sent2: List[str],
                         stop_words: Optional[List[str]] = None) -> float:
    """
    Compute cosine similarity between two tokenised sentences.

    Args:
        sent1 (List[str]): First sentence tokens.
        sent2 (List[str]): Second sentence tokens.
        stop_words (Optional[List[str]]): Words to ignore when building vectors.

    Returns:
        float: Cosine similarity score in [0, 1].
    """
    if stop_words is None:
        stop_words = []

    # Normalise to lowercase
    s1 = [w.lower() for w in sent1]
    s2 = [w.lower() for w in sent2]

    all_words = list(set(s1 + s2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # Build bag-of-words vectors, skipping stop words
    for w in s1:
        if w not in stop_words:
            vector1[all_words.index(w)] += 1

    for w in s2:
        if w not in stop_words:
            vector2[all_words.index(w)] += 1

    # 1 - cosine_distance  ==  cosine_similarity
    return 1 - nltk.cluster.util.cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences: List[List[str]],
                              stop_words: List[str]) -> np.ndarray:
    """
    Build an N×N similarity matrix for a list of sentences.

    Args:
        sentences (List[List[str]]): Tokenised sentences.
        stop_words (List[str]): Words to ignore during similarity scoring.

    Returns:
        np.ndarray: Square matrix where entry [i][j] is the similarity between
                    sentence i and sentence j.
    """
    n = len(sentences)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:           # Diagonal stays 0 (no self-loops)
                matrix[i][j] = sentence_similarity(
                    sentences[i], sentences[j], stop_words
                )

    return matrix


def generate_summary(text: str, top_n: int = 5) -> str:
    """
    Generate an extractive summary using the TextRank algorithm.

    Steps:
      1. Tokenise text into sentences.
      2. Build a sentence-similarity matrix.
      3. Apply PageRank on the resulting graph.
      4. Return the top-N highest-ranked sentences joined by '. '.

    Args:
        text (str): Raw input text to summarise.
        top_n (int): Number of sentences to include in the summary (default 5).

    Returns:
        str: Summarised text.

    Raises:
        ValueError: If text is invalid or top_n is not a positive integer.
    """
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    stop_words = stopwords.words("english")

    # Step 1 – Parse text into sentence tokens
    sentences = read_article(text)

    if not sentences:
        return ""

    # Step 2 – Build similarity matrix
    sim_matrix = build_similarity_matrix(sentences, stop_words)

    # Step 3 – PageRank on the similarity graph
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)

    # Step 4 – Pick top-N sentences (guard against requesting more than exist)
    top_n = min(top_n, len(sentences))
    ranked = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    summary_sentences = [" ".join(ranked[i][1]) for i in range(top_n)]
    return ". ".join(summary_sentences)

# ---------------------------------------------------------------------------
# Threading wrapper
# ---------------------------------------------------------------------------

def summarize_in_thread(text: str, top_n: int,
                         result_container: dict, error_container: dict) -> None:
    """
    Run generate_summary in a background thread and store the result.

    Args:
        text (str): Input text to summarise.
        top_n (int): Number of sentences for the summary.
        result_container (dict): Mutable dict; key 'summary' receives the output.
        error_container (dict): Mutable dict; key 'error' receives any exception.
    """
    try:
        result_container["summary"] = generate_summary(text, top_n)
    except Exception as exc:        # noqa: BLE001
        error_container["error"] = str(exc)


def run_summary_threaded(text: str, top_n: int) -> str:
    """
    Convenience wrapper: spawn a daemon thread, wait for it, and return result.

    Using threading here demonstrates concurrent execution; for the current
    synchronous Flask setup the thread is joined immediately, but the pattern
    is ready to be extended to async workers or task queues.

    Args:
        text (str): Input text.
        top_n (int): Desired number of summary sentences.

    Returns:
        str: The generated summary text.

    Raises:
        RuntimeError: Propagates any error raised inside the thread.
    """
    result: dict = {}
    error:  dict = {}

    worker = threading.Thread(
        target=summarize_in_thread,
        args=(text, top_n, result, error),
        daemon=True,
        name="SummarizerThread"
    )
    worker.start()
    worker.join()       # Block until summarisation completes

    if "error" in error:
        raise RuntimeError(error["error"])

    return result.get("summary", "")

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False   # Preserve JSON key order


@app.route("/")
def homepage():
    """Serve the main HTML interface."""
    return render_template("index.html", title="AI Text Summarizer")


@app.route("/summarize", methods=["POST"])
def summarize_form():
    """
    Handle HTML form submissions.

    Expects:
        input_text    (str)  – The text to summarise.
        num_sentences (int)  – How many sentences to include.

    Returns:
        Rendered HTML page with original text and summary.
    """
    input_text = request.form.get("input_text", "").strip()
    num_sentences_raw = request.form.get("num_sentences", "3").strip()

    # --- Validate inputs ---
    errors = []
    if not input_text:
        errors.append("Input text cannot be empty.")

    top_n = 3
    if num_sentences_raw:
        try:
            top_n = int(num_sentences_raw)
            if top_n < 1:
                errors.append("Number of sentences must be at least 1.")
        except ValueError:
            errors.append("Number of sentences must be a valid integer.")

    if errors:
        return render_template(
            "index.html",
            title="AI Text Summarizer",
            error="; ".join(errors)
        ), 400

    try:
        # Run summarisation in a separate thread
        summary = run_summary_threaded(input_text, top_n)
    except RuntimeError as exc:
        logger.error("Summarisation failed: %s", exc)
        return render_template(
            "index.html",
            title="AI Text Summarizer",
            error=f"Summarisation error: {exc}"
        ), 500

    logger.info("Summary generated: %d sentences requested, %d produced.",
                top_n, summary.count(".") + 1 if summary else 0)

    return render_template(
        "index.html",
        title="AI Text Summarizer",
        original_text=input_text,
        output_summary=summary,
        num_sentences=top_n
    )


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """
    JSON API endpoint – suitable for Postman or any REST client.

    Request body (JSON):
        {
            "text": "<your text here>",
            "num_sentences": 3        // optional, default 3
        }

    Response (JSON):
        {
            "success": true,
            "summary": "<summary text>",
            "num_sentences": 3,
            "original_length": 120,
            "summary_length": 45
        }

    Error response:
        {
            "success": false,
            "error": "<error message>"
        }
    """
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"success": False,
                        "error": "Request body must be valid JSON."}), 400

    text = data.get("text", "").strip()
    num_sentences = data.get("num_sentences", 3)

    # --- Validate ---
    if not text:
        return jsonify({"success": False,
                        "error": "Field 'text' is required and cannot be empty."}), 400

    if not isinstance(num_sentences, int) or num_sentences < 1:
        return jsonify({"success": False,
                        "error": "'num_sentences' must be a positive integer."}), 400

    try:
        summary = run_summary_threaded(text, num_sentences)
    except RuntimeError as exc:
        logger.error("API summarisation failed: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify({
        "success": True,
        "summary": summary,
        "num_sentences": num_sentences,
        "original_length": len(text.split()),
        "summary_length": len(summary.split())
    })


@app.errorhandler(404)
def not_found(error):
    """Custom 404 handler."""
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Custom 405 handler."""
    return jsonify({"success": False,
                    "error": "HTTP method not allowed on this endpoint."}), 405


@app.errorhandler(500)
def internal_error(error):
    """Custom 500 handler."""
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)