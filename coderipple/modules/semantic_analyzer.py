"""
semantic_analyzer.py

Stage 2: Semantic Analyzer
Uses GraphCodeBERT to produce code embeddings, then computes cosine similarity
between the old and new versions of each changed function.

Change Classification (Fix 8 — per-language thresholds applied at call time)
  Python/Java/Go/etc.:
    similarity >= 0.92  →  FORMAT_CHANGE
    0.75 <= sim < 0.92  →  REFACTOR
    sim < 0.75          →  LOGIC_CHANGE

  JS/TS bundles (minified code normalised before embedding):
    similarity >= 0.92  →  FORMAT_CHANGE
    0.60 <= sim < 0.92  →  REFACTOR
    sim < 0.60          →  LOGIC_CHANGE
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Pure-Python fallback types
    import math
    class _FakeTensor:
        def __init__(self, data: list[float]):
            self._d = list(data)
        def __getitem__(self, i): return self._d[i]
        def __setitem__(self, i, v): self._d[i] = v
        def __len__(self): return len(self._d)
        @property
        def shape(self): return (len(self._d),)

    class torch:  # noqa: F811
        @staticmethod
        def zeros(n): return _FakeTensor([0.0] * n)
        @staticmethod
        def no_grad():
            import contextlib; return contextlib.nullcontext()
        cuda = type("cuda", (), {"is_available": staticmethod(lambda: False)})()

    class F:  # noqa: F811
        @staticmethod
        def cosine_similarity(a, b, dim=0):
            va = a._d if hasattr(a, "_d") else a
            vb = b._d if hasattr(b, "_d") else b
            dot  = sum(x*y for x,y in zip(va, vb))
            na   = math.sqrt(sum(x*x for x in va)) or 1e-9
            nb   = math.sqrt(sum(x*x for x in vb)) or 1e-9
            return type("T", (), {"item": lambda self: dot/(na*nb)})()
        @staticmethod
        def normalize(v, dim=0):
            data = v._d if hasattr(v, "_d") else v
            norm = math.sqrt(sum(x*x for x in data)) or 1e-9
            return _FakeTensor([x/norm for x in data])

logger = logging.getLogger(__name__)

#  Thresholds (Fix 8 — per-language) 
# Default thresholds calibrated for readable Python / Java / Go source.
_DEFAULT_LOGIC_CHANGE_THRESHOLD = 0.75
_DEFAULT_REFACTOR_THRESHOLD     = 0.92

# JS/TS bundles contain many tiny functions that come out of minified code.
# Even after normalization, GraphCodeBERT produces lower similarity scores
# for these, so we use a less aggressive LOGIC_CHANGE boundary.
_JS_LOGIC_CHANGE_THRESHOLD = 0.60
_JS_REFACTOR_THRESHOLD     = 0.92  # keep FORMAT_CHANGE boundary the same

# Threshold sets keyed by file extension
_THRESHOLDS: dict[str, tuple[float, float]] = {
    # (logic_change_threshold, refactor_threshold)
    ".py":   (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    ".java": (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    ".go":   (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    ".cpp":  (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    ".c":    (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    ".cs":   (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    ".rb":   (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    ".swift":(_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    ".kt":   (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    ".rs":   (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD),
    # JS/TS family — relaxed LOGIC_CHANGE boundary
    ".js":   (_JS_LOGIC_CHANGE_THRESHOLD, _JS_REFACTOR_THRESHOLD),
    ".jsx":  (_JS_LOGIC_CHANGE_THRESHOLD, _JS_REFACTOR_THRESHOLD),
    ".ts":   (_JS_LOGIC_CHANGE_THRESHOLD, _JS_REFACTOR_THRESHOLD),
    ".tsx":  (_JS_LOGIC_CHANGE_THRESHOLD, _JS_REFACTOR_THRESHOLD),
}


def _get_thresholds(file_path: str = "") -> tuple[float, float]:
    """Return (logic_change_threshold, refactor_threshold) for *file_path*."""
    import os
    ext = os.path.splitext(file_path)[1].lower() if file_path else ""
    return _THRESHOLDS.get(ext, (_DEFAULT_LOGIC_CHANGE_THRESHOLD, _DEFAULT_REFACTOR_THRESHOLD))


def _classify(similarity: float, logic_thr: float, refactor_thr: float) -> str:
    """Map a similarity score to a change-type label."""
    if similarity >= refactor_thr:
        return "FORMAT_CHANGE"
    if similarity >= logic_thr:
        return "REFACTOR"
    return "LOGIC_CHANGE"


@dataclass
class SemanticResult:
    semantic_change_score: float   # 0 = identical, 1 = completely different
    similarity: float              # raw cosine similarity
    change_type: str               # FORMAT_CHANGE | REFACTOR | LOGIC_CHANGE
    old_embedding_dim: int
    new_embedding_dim: int
    model_used: str


class GraphCodeBERTEmbedder:
    """
    Lazy-loads microsoft/graphcodebert-base.
    Falls back to a TF-IDF-style token embedding when the model is unavailable
    (e.g., no internet / GPU) so the rest of the pipeline still works.
    """

    MODEL_NAME = "microsoft/graphcodebert-base"
    _instance: Optional["GraphCodeBERTEmbedder"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.tokenizer = None
        self.model     = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded   = False
        self._mode     = "unloaded"

    @classmethod
    def get(cls) -> "GraphCodeBERTEmbedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls()
                    try:
                        inst._try_load()
                    except Exception as exc:
                        logger.warning("Failed to load GraphCodeBERT, falling back to TF-IDF: %s", exc)
                        inst._mode = "tfidf_fallback"
                    cls._instance = inst
        return cls._instance

    def _try_load(self) -> None:
        import os
        if os.environ.get("DISABLE_GRAPHCODEBERT", "false").lower() == "true":
            raise RuntimeError("GraphCodeBERT disabled via environment variable")
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading GraphCodeBERT …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        
        # CPU OPTIMIZATION: Dynamically quantize Linear layers to INT8
        # This makes CPU inference ~2-3x faster and significantly reduces RAM usage.
        import torch
        if self.device == "cpu":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
        self.model.to(self.device)
        self._loaded = True
        self._mode   = "graphcodebert_base"
        logger.info("GraphCodeBERT loaded on %s (quantized=%s)", self.device, str(self.device == "cpu"))

    #  Public API 

    def embed(self, code: str):
        """Return a normalised embedding for *code*."""
        if self._mode == "graphcodebert_base":
            return self._bert_embed(code)
        else:
            return self._tfidf_embed(code)

    def batch_embed(self, codes: list, batch_size: int = 16) -> list:
        """
        Embed a list of code snippets efficiently.
        Uses batched BERT inference when available — much faster than
        calling embed() one-by-one for large commits.
        """
        if not codes:
            return []
        if self._mode == "graphcodebert_base":
            return self._bert_batch_embed(codes, batch_size)
        else:
            return [self._tfidf_embed(c) for c in codes]

    @property
    def mode(self) -> str:
        return self._mode

    #  BERT path 

    def _bert_embed(self, code: str) -> torch.Tensor:
        """CLS-token embedding from GraphCodeBERT, L2-normalised."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_vec = outputs.last_hidden_state[:, 0, :]   # (1, 768)
        return F.normalize(cls_vec.squeeze(0), dim=0)  # (768,)

    def _bert_batch_embed(self, codes: list, batch_size: int = 16) -> list:
        """Batch inference: process multiple snippets in one forward pass."""
        results = []
        for i in range(0, len(codes), batch_size):
            chunk = codes[i : i + batch_size]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # outputs.last_hidden_state: (batch, seq, 768)
            cls_vecs = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
            normed   = F.normalize(cls_vecs, dim=1)        # (batch, 768)
            results.extend([normed[j] for j in range(normed.shape[0])])
        return results

    #  Fallback path 

    @staticmethod
    def _tokenize(code: str) -> list[str]:
        """Split code into sub-word tokens for the fallback."""
        # camelCase / snake_case splitting + lowercase
        code = re.sub(r"[^a-zA-Z0-9_\s]", " ", code)
        tokens = []
        for word in code.split():
            # camelCase → ["camel", "Case"]
            parts = re.sub(r"([A-Z])", r" \1", word).split()
            tokens.extend(p.lower() for p in parts if p)
        return tokens

    def _tfidf_embed(self, code: str, dim: int = 768):
        """
        Deterministic hash-based bag-of-words embedding (no external deps).
        Dimension is fixed to 768 to match BERT shape.
        """
        import math as _math
        tokens = self._tokenize(code)
        if _TORCH_AVAILABLE:
            vec = torch.zeros(dim)
            for tok in tokens:
                vec[hash(tok) % dim] += 1.0
            n = float((vec * vec).sum() ** 0.5)
            return F.normalize(vec, dim=0) if n > 0 else vec
        else:
            # Pure-Python path
            data: list[float] = [0.0] * dim
            for tok in tokens:
                data[hash(tok) % dim] += 1.0
            norm = _math.sqrt(sum(x * x for x in data)) or 1e-9
            return _FakeTensor([x / norm for x in data])


#  Top-level helpers 

def compute_semantic_similarity_bulk(
    pairs: list,  # list of (old_code, new_code) tuples
    return_changed_flags: list = None,  # optional list of bool per pair
    file_paths: list = None,            # optional list of file paths for per-language thresholds
    batch_size: int = 16,
) -> list:
    """
    Fast bulk analysis — embeds ALL old+new codes in one batched pass,
    then computes similarities. Much faster than calling
    compute_semantic_similarity() per function on large commits.

    Returns a list of SemanticResult objects in the same order.
    """
    if not pairs:
        return []

    embedder   = GraphCodeBERTEmbedder.get()
    old_codes  = [p[0] or "" for p in pairs]
    new_codes  = [p[1] or "" for p in pairs]
    all_codes  = old_codes + new_codes  # embed everything at once

    logger.info("Batch-embedding %d code pairs (%d snippets)…", len(pairs), len(all_codes))
    all_embeddings = embedder.batch_embed(all_codes, batch_size=batch_size)

    old_embs = all_embeddings[: len(pairs)]
    new_embs = all_embeddings[len(pairs) :]

    results = []
    for idx, (old_emb, new_emb) in enumerate(zip(old_embs, new_embs)):
        if _TORCH_AVAILABLE:
            similarity = float(
                F.cosine_similarity(old_emb.unsqueeze(0), new_emb.unsqueeze(0)).item()
            )
        else:
            similarity = float(F.cosine_similarity(old_emb, new_emb).item())
        similarity = max(0.0, min(1.0, similarity))
        semantic_change_score = round(1.0 - similarity, 4)

        # Fix 8 — per-language thresholds
        fp = file_paths[idx] if file_paths and idx < len(file_paths) else ""
        logic_thr, refactor_thr = _get_thresholds(fp)
        change_type = _classify(similarity, logic_thr, refactor_thr)

        # Apply return-value upgrade if flagged
        if return_changed_flags and return_changed_flags[idx] and change_type in ("FORMAT_CHANGE", "REFACTOR"):
            change_type = "LOGIC_CHANGE"

        results.append(SemanticResult(
            semantic_change_score = semantic_change_score,
            similarity            = round(similarity, 4),
            change_type           = change_type,
            old_embedding_dim     = old_emb.shape[0],
            new_embedding_dim     = new_emb.shape[0],
            model_used            = embedder.mode,
        ))
    return results


def compute_semantic_similarity(
    old_code: str,
    new_code: str,
    file_path: str = "",
) -> SemanticResult:
    """
    Primary entry-point.  Given the old and new source of a function,
    return a SemanticResult with the change classification.
    Pass file_path to apply per-language thresholds (Fix 8).
    """
    embedder   = GraphCodeBERTEmbedder.get()
    old_emb    = embedder.embed(old_code)
    new_emb    = embedder.embed(new_code)

    if _TORCH_AVAILABLE:
        similarity = float(F.cosine_similarity(
            old_emb.unsqueeze(0), new_emb.unsqueeze(0)
        ).item())
    else:
        similarity = float(F.cosine_similarity(old_emb, new_emb).item())
    similarity = max(0.0, min(1.0, similarity))   # clamp to [0, 1]

    semantic_change_score = round(1.0 - similarity, 4)

    # Fix 8 — per-language thresholds
    logic_thr, refactor_thr = _get_thresholds(file_path)
    change_type = _classify(similarity, logic_thr, refactor_thr)

    return SemanticResult(
        semantic_change_score = semantic_change_score,
        similarity            = round(similarity, 4),
        change_type           = change_type,
        old_embedding_dim     = old_emb.shape[0],
        new_embedding_dim     = new_emb.shape[0],
        model_used            = embedder.mode,
    )
