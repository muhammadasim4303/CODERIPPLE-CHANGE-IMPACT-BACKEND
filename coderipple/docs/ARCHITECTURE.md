# CodeRipple — Architecture Documentation
## AI Code Change Impact and Risk Analyzer

---

## 1. System Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CodeRipple — 5-Stage Analysis Pipeline                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌──────────────────────────────────────────────────────────────────────┐  ║
║   │  POST /change-impact-commit                                          │  ║
║   │  { repository_path: "...", commit_hash: "abc123" }                   │  ║
║   └────────────────────────┬─────────────────────────────────────────────┘  ║
║                             │                                                ║
║   ┌─────────────────────────▼──────────────────────────────────────────────┐ ║
║   │  STAGE 1 — Git Commit Parser       (commit_parser.py)                 │ ║
║   │                                                                        │ ║
║   │   GitPython ──► diff extraction ──► function extractor                │ ║
║   │                                     (tree-sitter / regex)             │ ║
║   │                                                                        │ ║
║   │   Output: List[ChangedFunction]                                       │ ║
║   │           { file, function_name, old_source, new_source, patch }      │ ║
║   └─────────────────────────┬──────────────────────────────────────────────┘ ║
║                             │                                                ║
║   ┌─────────────────────────▼──────────────────────────────────────────────┐ ║
║   │  STAGE 2 — Semantic Analyzer       (semantic_analyzer.py)             │ ║
║   │                                                                        │ ║
║   │   old_source ──► GraphCodeBERT ──► embedding_A (768-dim)              │ ║
║   │   new_source ──► GraphCodeBERT ──► embedding_B (768-dim)              │ ║
║   │                                                                        │ ║
║   │   cosine_similarity(A, B) ──► similarity score ──► change_type        │ ║
║   │                                                                        │ ║
║   │   Thresholds:  sim >= 0.92 → FORMAT_CHANGE                            │ ║
║   │                sim >= 0.75 → REFACTOR                                 │ ║
║   │                sim <  0.75 → LOGIC_CHANGE                             │ ║
║   │                                                                        │ ║
║   │   Output: SemanticResult { score, similarity, change_type }           │ ║
║   └─────────────────────────┬──────────────────────────────────────────────┘ ║
║                             │                                                ║
║   ┌─────────────────────────▼──────────────────────────────────────────────┐ ║
║   │  STAGE 3 — Dependency Graph Builder  (dependency_graph.py)            │ ║
║   │                                                                        │ ║
║   │   Repository files ──► AST parser (tree-sitter) ──► NetworkX DiGraph  │ ║
║   │                                                                        │ ║
║   │   Nodes:   functions, classes, files                                  │ ║
║   │   Edges:   calls, imports, inherits, contains                         │ ║
║   │                                                                        │ ║
║   │   Node ID format:  "filepath::ClassName.method_name"                  │ ║
║   │                                                                        │ ║
║   │   Output: DependencyGraph (cached across commits)                     │ ║
║   └─────────────────────────┬──────────────────────────────────────────────┘ ║
║                             │                                                ║
║   ┌─────────────────────────▼──────────────────────────────────────────────┐ ║
║   │  STAGE 4 — Ripple Propagation Engine  (ripple_engine.py)              │ ║
║   │                                                                        │ ║
║   │   changed_nodes ──► BFS on REVERSE graph ──► impact map               │ ║
║   │                                                                        │ ║
║   │   distance == 1 → direct_impact                                       │ ║
║   │   distance >= 2 → indirect_impact                                     │ ║
║   │                                                                        │ ║
║   │   Output: RippleResult                                                │ ║
║   │           { direct, indirect, depth, size, impacted_files }           │ ║
║   └─────────────────────────┬──────────────────────────────────────────────┘ ║
║                             │                                                ║
║   ┌─────────────────────────▼──────────────────────────────────────────────┐ ║
║   │  STAGE 5 — Risk Scoring Module  (risk_predictor.py)                   │ ║
║   │                                                                        │ ║
║   │   Weighted combination:                                                │ ║
║   │   semantic_score  × 0.30                                              │ ║
║   │   diff_size       × 0.15   (log-normalised)                           │ ║
║   │   ripple_size     × 0.25   (log-normalised)                           │ ║
║   │   ripple_depth    × 0.15                                              │ ║
║   │   change_type     × 0.10                                              │ ║
║   │   fn_count        × 0.05                                              │ ║
║   │                                                                        │ ║
║   │   Thresholds: >= 0.60 → HIGH | >= 0.30 → MEDIUM | else → LOW         │ ║
║   │                                                                        │ ║
║   │   Output: RiskScore { label, score, confidence, factors }             │ ║
║   └─────────────────────────┬──────────────────────────────────────────────┘ ║
║                             │                                                ║
║   ┌─────────────────────────▼──────────────────────────────────────────────┐ ║
║   │  JSON Response                                                         │ ║
║   │  { commit, changed_function, semantic_change_score, change_type,      │ ║
║   │    direct_impact, indirect_impact, ripple_depth, ripple_size,         │ ║
║   │    risk_prediction, contributing_factors, dependency_graph }          │ ║
║   └──────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Module Overview

| Module | Stage | Responsibility |
|--------|-------|----------------|
| `commit_parser.py` | 1 | Extract changed functions from git diff |
| `semantic_analyzer.py` | 2 | GraphCodeBERT embeddings + cosine similarity |
| `dependency_graph.py` | 3 | AST-based repo-wide call graph |
| `ripple_engine.py` | 4 | BFS ripple propagation through reverse graph |
| `risk_predictor.py` | 5 | Weighted risk score + label |
| `core/analyzer.py` | — | Orchestrator tying all stages together |
| `api/api.py` | — | Flask REST API |

---

## 3. Semantic Change Detection — Key Design Decision

### Why GraphCodeBERT?

Standard text similarity (Levenshtein, token overlap) fails on cases like:

```python
# Case 1: LOOKS similar, DIFFERENT logic
if x > 5:   →   if x >= 5:   # off-by-one — completely different behaviour

# Case 2: LOOKS different, SAME logic
total = price * quantity
  →
def calculate_total(p, q): return p * q
total = calculate_total(price, quantity)
```

GraphCodeBERT is trained on code using data flow graphs, making its embeddings sensitive to **semantic meaning** rather than surface syntax. Two code snippets with the same logic produce similar embeddings even if restructured; a boundary condition change shifts the embedding.

### Embedding Pipeline

```
source code
    │
    ▼
tokenise (sub-word BPE, max 512 tokens)
    │
    ▼
GraphCodeBERT (12-layer transformer, 768 hidden)
    │
    ▼
CLS token vector  [1 × 768]
    │
    ▼
L2 normalise
    │
    ▼
768-dim semantic embedding
```

### Cosine Similarity

```
similarity = (A · B) / (|A| × |B|)

semantic_change_score = 1 − similarity

FORMAT_CHANGE  : similarity ≥ 0.92  (whitespace, rename)
REFACTOR       : 0.75 ≤ sim < 0.92  (restructured, same logic)
LOGIC_CHANGE   : similarity < 0.75  (semantic drift)
```

**Fallback**: When GraphCodeBERT is not available (no GPU/internet), a hash-based TF-IDF bag-of-words embedding is used. This correctly identifies refactors and format changes but **cannot detect operator-level logic changes** (e.g., `>` vs `>=`) since those require learned semantic representations. The full system requires GraphCodeBERT for production use.

---

## 4. Dependency Graph — Data Model

```
Node types:
  function  —  "src/auth.py::UserService.authenticate"
  class     —  "src/auth.py::UserService"
  file      —  "src/auth.py::__file__"

Edge types:
  calls     —  function A calls function B
  imports   —  file A imports from file B
  inherits  —  class A inherits from class B
  contains  —  file/class contains a function/method
```

### Example Graph

```
FileA.py::__file__
    │ contains
    ▼
FileA.py::funcA  ◄────── FileB.py::funcB ◄────── FileC.py::funcC
                  calls               calls
                         ◄────── FileD.py::funcD
                          calls
```

---

## 5. Ripple Propagation Algorithm

```python
def propagate(changed_nodes, max_depth=10):
    # Build reverse graph: edges point FROM callee TO caller
    rev = dependency_graph.reverse()
    
    # BFS from every changed node
    queue = [(node, distance=0) for node in changed_nodes]
    impact_map = {}
    
    while queue:
        node, depth = dequeue()
        if depth >= max_depth: continue
        
        for caller in rev.successors(node):
            if caller not in impact_map:
                impact_map[caller] = depth + 1
                enqueue(caller, depth + 1)
    
    direct   = {n for n,d in impact_map if d == 1}
    indirect = {n for n,d in impact_map if d >= 2}
    
    return RippleResult(direct, indirect, max(impact_map.values()), len(impact_map))
```

---

## 6. Risk Scoring Formula

```
score = 0.30 × semantic_score
      + 0.15 × log_norm(lines_changed, cap=500)
      + 0.25 × log_norm(ripple_size,   cap=200)
      + 0.15 × min(ripple_depth / 6, 1.0)
      + 0.10 × change_type_weight        # LOGIC=1.0 REFACTOR=0.5 FORMAT=0.1
      + 0.05 × log_norm(fn_count, cap=20)

Labels:
  score >= 0.60  →  HIGH RISK
  score >= 0.30  →  MEDIUM RISK
  score <  0.30  →  LOW RISK
```

---

## 7. API Reference

### `POST /change-impact-commit`
```json
Request:
{
  "repository_path": "/path/to/repo",
  "commit_hash": "abc123..."
}

Response:
{
  "commit": "abc123",
  "changed_function": "src/billing.py::calculate_discount",
  "semantic_change_score": 0.62,
  "change_type": "LOGIC_CHANGE",
  "direct_impact": ["src/checkout.py::process_payment"],
  "indirect_impact": ["src/reports.py::generate_invoice"],
  "ripple_depth": 2,
  "ripple_size": 3,
  "risk_prediction": "HIGH",
  "risk_score": 0.71,
  "risk_confidence": 0.84,
  "contributing_factors": ["..."],
  "dependency_graph": { "nodes": [...], "edges": [...] }
}
```

### `POST /build-graph`
Pre-build the dependency graph for a repository.

### `GET /health`
Liveness check.

---

## 8. Directory Structure

```
coderipple/
├── __init__.py
├── requirements.txt
├── example_run.py
├── api/
│   ├── __init__.py
│   └── api.py                  Flask REST API
├── core/
│   ├── __init__.py
│   └── analyzer.py             Pipeline orchestrator
├── modules/
│   ├── __init__.py
│   ├── commit_parser.py        Stage 1 — Git diff + function extraction
│   ├── semantic_analyzer.py    Stage 2 — GraphCodeBERT embeddings
│   ├── dependency_graph.py     Stage 3 — AST call graph builder
│   ├── ripple_engine.py        Stage 4 — BFS ripple propagation
│   └── risk_predictor.py       Stage 5 — Weighted risk scoring
├── tests/
│   └── __init__.py
└── docs/
    └── ARCHITECTURE.md         (this file)
```
