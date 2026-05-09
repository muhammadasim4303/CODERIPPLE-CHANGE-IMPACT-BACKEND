"""
risk_predictor.py
─────────────────
Stage 5: Risk Scoring Module

Combines signals from all four upstream stages:
  1. Semantic change score     (GraphCodeBERT cosine distance)
  2. Lines changed             (raw diff size)
  3. Ripple size               (nodes in blast radius)
  4. Ripple depth              (BFS hops from changed function)
  5. Change type               (FORMAT_CHANGE / REFACTOR / LOGIC_CHANGE)
  6. Number of changed functions

Classifies each commit as:  LOW | MEDIUM | HIGH
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .semantic_analyzer import SemanticResult
from .ripple_engine      import RippleResult


@dataclass
class RiskScore:
    score:           float          # 0.0 – 1.0 continuous
    label:           str            # "LOW" | "MEDIUM" | "HIGH"
    confidence:      float
    contributing_factors: list[str]
    feature_breakdown:    dict[str, float]


# ── Weights (tuned empirically) ───────────────────────────────────────────────

W_SEMANTIC     = 0.20   # how semantically different is the code?
W_DIFF_SIZE    = 0.15   # raw lines added + removed
W_RIPPLE_SIZE  = 0.25   # how many nodes are in the blast radius?
W_RIPPLE_DEPTH = 0.15   # how deep does the change propagate?
W_CHANGE_TYPE  = 0.20   # type of change (logic > refactor > format)
W_FN_COUNT     = 0.05   # number of changed functions

# Thresholds
HIGH_THRESHOLD   = 0.60
MEDIUM_THRESHOLD = 0.30


class RiskPredictor:

    @staticmethod
    def predict(
        semantic:        SemanticResult,
        ripple:          RippleResult,
        total_lines_changed: int = 0,
        changed_fn_count:    int = 1,
        return_changed:      bool = False,
    ) -> RiskScore:
        features: dict[str, float] = {}

        # ── 1. Semantic component ─────────────────────────────────────────
        features["semantic"] = semantic.semantic_change_score  # already [0,1]

        # ── 2. Diff size (log-normalised, cap at 500 lines) ───────────────
        features["diff_size"] = min(
            math.log1p(total_lines_changed) / math.log1p(500),
            1.0
        )

        # ── 3. Ripple size (log-normalised, cap at 200 nodes) ─────────────
        features["ripple_size"] = min(
            math.log1p(ripple.ripple_size) / math.log1p(200),
            1.0
        )

        # ── 4. Ripple depth (linear, cap at 6) ────────────────────────────
        features["ripple_depth"] = min(ripple.ripple_depth / 6.0, 1.0)

        # ── 5. Change type ────────────────────────────────────────────────
        features["change_type"] = {
            "LOGIC_CHANGE":   1.0,
            "REFACTOR":       0.5,
            "FORMAT_CHANGE":  0.1,
        }.get(semantic.change_type, 0.5)

        # ── 6. Changed function count (log, cap at 20) ────────────────────
        features["fn_count"] = min(
            math.log1p(changed_fn_count) / math.log1p(20),
            1.0
        )

        # Short function amplifier — small functions have high change ratio
        avg_fn_length = total_lines_changed / max(changed_fn_count, 1)
        if avg_fn_length < 8:
            # Boost semantic score for very short functions
            features["semantic"] = min(features["semantic"] * 2.5, 1.0)

        # ── Weighted sum ──────────────────────────────────────────────────
        score = (
            W_SEMANTIC     * features["semantic"]     +
            W_DIFF_SIZE    * features["diff_size"]    +
            W_RIPPLE_SIZE  * features["ripple_size"]  +
            W_RIPPLE_DEPTH * features["ripple_depth"] +
            W_CHANGE_TYPE  * features["change_type"]  +
            W_FN_COUNT     * features["fn_count"]
        )
        score = round(max(0.0, min(1.0, score)), 4)

        # ── Label + confidence ────────────────────────────────────────────
        if score >= HIGH_THRESHOLD:
            label      = "HIGH"
            confidence = 0.70 + 0.28 * (score - HIGH_THRESHOLD) / (1.0 - HIGH_THRESHOLD)
        elif score >= MEDIUM_THRESHOLD:
            label      = "MEDIUM"
            confidence = 0.60 + 0.15 * (score - MEDIUM_THRESHOLD) / (HIGH_THRESHOLD - MEDIUM_THRESHOLD)
        else:
            label      = "LOW"
            confidence = 0.65 + 0.30 * (1.0 - score / MEDIUM_THRESHOLD)

        confidence = round(min(confidence, 0.97), 4)

        # ── Human-readable factors ────────────────────────────────────────
        factors = _build_factors(semantic, ripple, features, total_lines_changed, changed_fn_count)

        return RiskScore(
            score                = score,
            label                = label,
            confidence           = confidence,
            contributing_factors = factors,
            feature_breakdown    = {k: round(v, 4) for k, v in features.items()},
        )


def _build_factors(
    semantic: SemanticResult,
    ripple:   RippleResult,
    features: dict[str, float],
    lines:    int,
    fn_count: int,
) -> list[str]:
    factors = []

    if semantic.change_type == "LOGIC_CHANGE":
        factors.append(
            f"Semantic logic change detected (similarity={semantic.similarity:.2f})"
        )
    elif semantic.change_type == "REFACTOR":
        factors.append(
            f"Code refactoring detected, logic likely unchanged (similarity={semantic.similarity:.2f})"
        )
    else:
        factors.append(
            f"Formatting / whitespace change only (similarity={semantic.similarity:.2f})"
        )

    if lines > 0:
        factors.append(f"{lines} lines changed across the commit")

    if ripple.ripple_size > 1:
        factors.append(
            f"Ripple effect reaches {ripple.ripple_size} nodes "
            f"(depth {ripple.ripple_depth})"
        )
    else:
        factors.append("No downstream dependents found — isolated change")

    if ripple.direct_impact:
        factors.append(
            f"{len(ripple.direct_impact)} directly impacted function(s): "
            + ", ".join(ripple.direct_impact[:3])
            + ("…" if len(ripple.direct_impact) > 3 else "")
        )

    if ripple.indirect_impact:
        factors.append(
            f"{len(ripple.indirect_impact)} indirectly impacted node(s)"
        )

    if fn_count > 1:
        factors.append(f"{fn_count} functions modified in this commit")

    if ripple.impacted_files:
        factors.append(
            f"Impacted files: " + ", ".join(ripple.impacted_files[:5])
            + ("…" if len(ripple.impacted_files) > 5 else "")
        )

    return factors
