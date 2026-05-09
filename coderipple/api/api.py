"""
api.py

Flask REST API for CodeRipple Semantic Change Impact Analyzer.

Performance optimizations:
  - GraphCodeBERT is loaded ONCE at startup (not per request)
  - git clone uses --depth=10 (only recent history needed)
  - Clone cache persists for the lifetime of the server process
  - Analyzer cache persists dependency graph per repo
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from ..core.analyzer import CommitAnalyzer
from ..modules.semantic_analyzer import GraphCodeBERTEmbedder

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

#  Caches 
_analyzer_cache: dict[str, CommitAnalyzer] = {}
_clone_cache:    dict[str, str]            = {}   # repoFullName → local path


#  Warm up GraphCodeBERT at startup so first request isn't slow 
def _warmup_model() -> None:
    logger.info("Pre-loading GraphCodeBERT model at startup…")
    try:
        embedder = GraphCodeBERTEmbedder.get()
        # Run one dummy embed to force full model load
        embedder.embed("def hello(): pass")
        logger.info("GraphCodeBERT ready (%s)", embedder.mode)
    except Exception as exc:
        logger.warning("Model warmup failed (%s) — will load on first request", exc)


#  Helpers 

def _get_analyzer(repo_path: str) -> CommitAnalyzer:
    if repo_path not in _analyzer_cache:
        _analyzer_cache[repo_path] = CommitAnalyzer(repo_path)
    return _analyzer_cache[repo_path]


def _get_or_clone(repo_full_name: str) -> str:
    """
    Returns a local path to the repo.
    Clones from GitHub if not already cached.
    Uses --depth=10 — fast and enough for recent commits.
    """
    if repo_full_name in _clone_cache:
        cached = _clone_cache[repo_full_name]
        if Path(cached).exists():
            logger.info("Cache hit — reusing clone for %s", repo_full_name)
            return cached
        del _clone_cache[repo_full_name]

    clone_dir = tempfile.mkdtemp(prefix="cr_")
    url = f"https://github.com/{repo_full_name}.git"
    logger.info("Cloning %s (shallow) …", url)

    try:
        subprocess.run(
            ["git", "clone", "--depth=50", url, clone_dir],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise RuntimeError(f"git clone failed for {repo_full_name}: {exc.stderr}") from exc
    except subprocess.TimeoutExpired:
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise RuntimeError(f"git clone timed out for {repo_full_name}")

    _clone_cache[repo_full_name] = clone_dir
    logger.info("Cloned %s → %s", repo_full_name, clone_dir)
    return clone_dir


def _resolve_repo_path(body: dict) -> str:
    repo_full_name  = body.get("repo_full_name",  "").strip()
    repository_path = body.get("repository_path", "").strip()

    if repo_full_name:
        return _get_or_clone(repo_full_name)
    if repository_path:
        if not Path(repository_path).is_dir():
            raise ValueError(f"Path does not exist: {repository_path}")
        return repository_path

    raise ValueError("Provide 'repo_full_name' (e.g. 'owner/repo') or 'repository_path'")


def error(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


#  Routes 

@app.route("/health", methods=["GET"])
def health():
    embedder = GraphCodeBERTEmbedder.get()
    return jsonify({
        "status":       "ok",
        "service":      "CodeRipple Semantic Analyzer",
        "model":        embedder.mode,
        "cached_repos": list(_clone_cache.keys()),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    })


@app.route("/build-graph", methods=["POST"])
def build_graph():
    body = request.get_json(silent=True) or {}
    try:
        repo_path = _resolve_repo_path(body)
    except (ValueError, RuntimeError) as exc:
        return error(str(exc))

    try:
        analyzer = _get_analyzer(repo_path)
        analyzer.build_graph()
        dg = analyzer._dep_graph
        return jsonify({
            "status":   "graph_built",
            "repo":     repo_path,
            "nodes":    dg.node_count(),
            "edges":    dg.edge_count(),
            "built_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.exception("build-graph failed")
        return error(f"Graph build failed: {exc}", 500)


@app.route("/change-impact-commit", methods=["POST"])
def change_impact_commit():
    """
    Full analysis pipeline for a single commit.

    Body:
    {
        "repo_full_name": "owner/repo",   ← auto-cloned from GitHub
        "commit_hash":    "abc123..."
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return error("Request body must be JSON")

    commit_hash = body.get("commit_hash", "").strip()
    if not commit_hash:
        return error("'commit_hash' is required")

    try:
        repo_path = _resolve_repo_path(body)
    except (ValueError, RuntimeError) as exc:
        return error(str(exc))

    try:
        analyzer = _get_analyzer(repo_path)
        result   = analyzer.analyze(commit_hash)
        return jsonify(result)
    except Exception as exc:
        logger.exception("change-impact-commit failed for %s @ %s", repo_path, commit_hash)
        return error(f"Analysis failed: {exc}", 500)


@app.route("/batch-impact", methods=["POST"])
def batch_impact():
    body          = request.get_json(silent=True) or {}
    commit_hashes = body.get("commit_hashes", [])

    if not commit_hashes:
        return error("'commit_hashes' is required")

    try:
        repo_path = _resolve_repo_path(body)
    except (ValueError, RuntimeError) as exc:
        return error(str(exc))

    analyzer = _get_analyzer(repo_path)
    results  = []
    for sha in commit_hashes[:20]:
        try:
            results.append(analyzer.analyze(sha))
        except Exception as exc:
            results.append({"commit": sha, "error": str(exc)})

    return jsonify({"repository": repo_path, "results": results})


#  Entry point 

if __name__ == "__main__":
    port = int(os.environ.get("SEMANTIC_PORT", 5001))
    logger.info("Starting CodeRipple Semantic Analyzer on port %d", port)
    _warmup_model()   # ← load model NOW before any requests come in
    app.run(host="0.0.0.0", port=port, debug=False)