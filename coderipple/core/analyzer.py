"""
analyzer.py

Core orchestrator — ties Stages 1-5 together into a single
`analyze_commit()` call.

Usage:
    from coderipple.core.analyzer import CommitAnalyzer

    analyzer = CommitAnalyzer("/path/to/repo")
    result   = analyzer.analyze("abc123")
"""

from __future__ import annotations

import logging
import os
import difflib
from datetime import datetime, timezone
from typing import Any

from ..modules.commit_parser    import (
    CommitParser,
    SUPPORTED_EXTENSIONS,
    NON_CODE_EXTENSIONS,
    detect_patch_logic_signals,
)
from ..modules.semantic_analyzer import (
    compute_semantic_similarity,
    compute_semantic_similarity_bulk,
)
from ..modules.dependency_graph  import DependencyGraphBuilder, DependencyGraph
from ..modules.ripple_engine     import RippleEngine, RippleResult

logger = logging.getLogger(__name__)


class CommitAnalyzer:
    """
    Main analysis pipeline.

    The dependency graph is built lazily and cached for re-use across
    multiple commit analyses on the same repository.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path    = repo_path
        self._parser      = CommitParser(repo_path)
        self._dep_graph: DependencyGraph | None = None
        self._ripple: RippleEngine | None = None
        # Cap: skip semantic embedding beyond this many functions to avoid
        # multi-hour runs on giant minified JS bundles.
        self.max_functions = int(os.environ.get("CR_MAX_FUNCTIONS", 500))

    #  Graph lifecycle 

    def build_graph(self) -> None:
        """Pre-build the dependency graph (call once for batch analysis)."""
        builder          = DependencyGraphBuilder(self.repo_path)
        self._dep_graph  = builder.build()
        self._ripple     = RippleEngine(self._dep_graph)

    def _ensure_graph(self) -> None:
        if self._dep_graph is None:
            self.build_graph()

    def _fallback_file_functions(self, file_diffs, repo_commit) -> list:
        """
        When the parser extracts zero functions, treat each changed *code*
        file as one synthetic ChangedFunction.

        Key fixes vs. the original:
        - Only processes files with SUPPORTED_EXTENSIONS (skips .md, .json, etc.)
        - Reads actual old/new blob content from git so BERT gets a real
          code-vs-code comparison instead of '"" vs raw patch text'
        - If blob read fails, auto-classifies without BERT rather than
          producing garbage embeddings
        """
        from ..modules.commit_parser import ChangedFunction
        from pathlib import Path
        results = []
        for fd in file_diffs:
            if fd.status not in ("modified", "added") or not fd.patch:
                continue
            ext = Path(fd.new_path or fd.old_path).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue   # skip docs, configs, assets — never BERT

            # Try to read actual file contents from git for a clean comparison
            old_src = ""
            new_src = ""
            if repo_commit is not None:
                try:
                    parent = repo_commit.parents[0] if repo_commit.parents else None
                    if parent and fd.old_path:
                        blob = parent.tree[fd.old_path]
                        old_src = blob.data_stream.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                try:
                    if fd.new_path:
                        blob = repo_commit.tree[fd.new_path]
                        new_src = blob.data_stream.read().decode("utf-8", errors="replace")
                except Exception:
                    pass

            # If we still have nothing useful to compare, use the patch as a
            # last resort but only when both sides have content. A blank
            # old_source vs. raw patch gives BERT garbage input.
            if not old_src and not new_src:
                new_src = fd.patch  # diff as proxy; will embed both sides as patch vs ""

            added   = sum(1 for l in fd.patch.splitlines() if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in fd.patch.splitlines() if l.startswith("-") and not l.startswith("---"))
            
            # Generate a clean diff for logic signal detection
            clean_diff = "".join(difflib.unified_diff(
                (old_src or "").splitlines(keepends=True),
                (new_src or "").splitlines(keepends=True),
                n=3
            ))
            logic_signal = detect_patch_logic_signals(clean_diff)

            results.append(ChangedFunction(
                file_path             = fd.new_path or fd.old_path,
                function_name         = "__file__",
                old_source            = old_src,
                new_source            = new_src,
                added_lines           = added,
                removed_lines         = removed,
                patch_fragment        = clean_diff or fd.patch,
                return_changed        = False,
                patch_has_logic_signal= logic_signal,
            ))
        return results

    #  Main entry point 

    def analyze(self, commit_hash: str) -> dict[str, Any]:
        """
        Full pipeline: parse → embed → graph → ripple → score → JSON.
        """
        self._ensure_graph()

        #  Stage 1: Parse commit 
        logger.info("Stage 1 — parsing commit %s", commit_hash)
        file_diffs, changed_fns = self._parser.parse_commit(commit_hash)

        #  Guard: doc/config-only commits 
        # If every changed file is a non-code file (.md, .json, .yml, …)
        # there is nothing to embed semantically — return FORMAT_CHANGE
        # immediately without touching BERT at all.
        if not changed_fns:
            from pathlib import Path
            code_diffs = [
                fd for fd in file_diffs
                if Path(fd.new_path or fd.old_path).suffix.lower()
                not in NON_CODE_EXTENSIONS
            ]
            if not code_diffs:
                logger.info(
                    "Commit %s touches only documentation / config files — "
                    "classifying as FORMAT_CHANGE without BERT.",
                    commit_hash,
                )
                return self._trivial_result(commit_hash, file_diffs)

        #  Stage 1b: Fallback to file-level analysis 
        if not changed_fns:
            # Pass the git commit object so the fallback can read actual blobs.
            repo_commit = self._parser._get_repo().commit(commit_hash)
            logger.warning("Function extractor found nothing — falling back to file-level diff analysis")
            changed_fns = self._fallback_file_functions(file_diffs, repo_commit=repo_commit)
            if not changed_fns:
                return self._trivial_result(commit_hash, file_diffs)

        #  Stage 2: Semantic analysis (batched) 
        # Filter to functions that have actual source to compare
        embeddable = [(cf, cf.old_source or "", cf.new_source or "")
                      for cf in changed_fns
                      if cf.old_source or cf.new_source]

        # Safety cap: for huge minified bundles (e.g. lodash, Firebase)
        # limit analysis to the most-changed functions to keep runtime
        # reasonable. Anything beyond cap is still counted in stats.
        if len(embeddable) > self.max_functions:
            logger.warning(
                "Commit has %d embeddable functions — capping to %d "
                "(set CR_MAX_FUNCTIONS env var to override).",
                len(embeddable), self.max_functions
            )
            embeddable = embeddable[: self.max_functions]

        logger.info(
            "Stage 2 — batch semantic analysis (%d / %d functions)",
            len(embeddable), len(changed_fns)
        )

        pairs              = [(old, new) for _, old, new in embeddable]
        return_flags       = [cf.return_changed for cf, _, _ in embeddable]
        file_paths         = [cf.file_path for cf, _, _ in embeddable]
        bulk_results       = compute_semantic_similarity_bulk(
            pairs,
            return_changed_flags=return_flags,
            file_paths=file_paths,
        )

        semantic_results = []
        for (cf, _, _), sr in zip(embeddable, bulk_results):
            # — return-value heuristic upgrade —
            if cf.return_changed and sr.change_type in ("FORMAT_CHANGE", "REFACTOR"):
                sr.change_type = "LOGIC_CHANGE"
                logger.info(
                    "Upgraded %s to LOGIC_CHANGE (return value structure changed)",
                    cf.function_name,
                )

            # — patch-signal heuristic upgrade (catches operator swaps, new —
            #   conditionals, etc. that BERT misses in high-similarity diffs) —
            if cf.patch_has_logic_signal and sr.change_type in ("FORMAT_CHANGE", "REFACTOR"):
                sr.change_type = "LOGIC_CHANGE"
                logger.info(
                    "Upgraded %s to LOGIC_CHANGE (patch logic signal: operator/branch change detected)",
                    cf.function_name,
                )

            semantic_results.append((cf, sr))

        if not semantic_results:
            return self._trivial_result(commit_hash, file_diffs)

        # Pick the worst-case semantic result (highest change score)
        primary_fn, primary_semantic = max(
            semantic_results,
            key=lambda x: x[1].semantic_change_score,
        )

        #  Stage 3+4: Ripple propagation 
        logger.info("Stage 3+4 — ripple propagation")
        changed_node_ids = [
            f"{cf.file_path}::{cf.function_name}"
            for cf in changed_fns
        ]
        ripple: RippleResult = self._ripple.propagate(changed_node_ids)

        #  Stage 5: Stats 
        logger.info("Stage 5 — gathering stats")
        total_lines = sum(
            cf.added_lines + cf.removed_lines for cf in changed_fns
        )

        #  Build output 
        all_fn_details = []
        for cf, sr in semantic_results:
            all_fn_details.append({
                "function":             cf.function_name,
                "file":                 cf.file_path,
                "semantic_change_score": sr.semantic_change_score,
                "similarity":           sr.similarity,
                "change_type":          sr.change_type,
                "added_lines":          cf.added_lines,
                "removed_lines":        cf.removed_lines,
            })

        graph_view = self._ripple.subgraph_for_display(ripple)

        return {
            # Identifiers
            "commit":              commit_hash,
            "repository":          self.repo_path,
            "analyzed_at":         datetime.now(timezone.utc).isoformat(),

            # Primary changed function (worst-case)
            "changed_function":    f"{primary_fn.file_path}::{primary_fn.function_name}",
            "changed_functions":   all_fn_details,

            # Semantic analysis
            "semantic_change_score": primary_semantic.semantic_change_score,
            "similarity":            primary_semantic.similarity,
            "change_type":           primary_semantic.change_type,
            "model_used":            primary_semantic.model_used,

            # Ripple effect
            "direct_impact":    ripple.direct_impact,
            "indirect_impact":  ripple.indirect_impact,
            "impacted_files":   ripple.impacted_files,
            "ripple_depth":     ripple.ripple_depth,
            "ripple_size":      ripple.ripple_size,

            # Risk prediction (disabled per user request)
            "risk_prediction":        None,
            "risk_score":             None,
            "risk_confidence":        None,
            "contributing_factors":   [],
            "feature_breakdown":      {},

            # Stats
            "files_changed":          len(file_diffs),
            "functions_changed":      len(changed_fns),
            "total_lines_changed":    total_lines,

            # Graph snapshot (for visualisation)
            "dependency_graph":       graph_view,
        }

    #  Helpers 

    @staticmethod
    def _trivial_result(commit_hash: str, file_diffs) -> dict[str, Any]:
        return {
            "commit":               commit_hash,
            "changed_function":     None,
            "changed_functions":    [],
            "semantic_change_score": 0.0,
            "similarity":           1.0,
            "change_type":          "FORMAT_CHANGE",
            "model_used":           "graphcodebert_base",
            "direct_impact":        [],
            "indirect_impact":      [],
            "impacted_files":       [],
            "ripple_depth":         0,
            "ripple_size":          0,
            "risk_prediction":      None,
            "risk_score":           None,
            "risk_confidence":      None,
            "contributing_factors": [],
            "feature_breakdown":    {},
            "files_changed":        len(file_diffs),
            "functions_changed":    0,
            "total_lines_changed":  0,
            "dependency_graph":     {"nodes": [], "edges": []},
            "analyzed_at":          datetime.now(timezone.utc).isoformat(),
        }
