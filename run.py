"""
run.py  —  launch the CodeRipple API from the project root
Usage:
    python run.py
"""
import sys
import os

# Make sure the parent of 'coderipple/' is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coderipple.api.api import app, _warmup_model

if __name__ == "__main__":
    port = int(os.environ.get("SEMANTIC_PORT", 5001))
    print(f"Starting CodeRipple Semantic Analyzer on http://localhost:{port}")
    _warmup_model()   # pre-load GraphCodeBERT before accepting requests
    app.run(host="0.0.0.0", port=port, debug=False)