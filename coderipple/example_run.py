#!/usr/bin/env python3
"""
example_run.py
──────────────
Demonstrates the full CodeRipple semantic analysis pipeline
WITHOUT needing a real git repository.

Run:
    python example_run.py
"""

from __future__ import annotations
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from modules.semantic_analyzer import compute_semantic_similarity
from modules.dependency_graph  import DependencyGraph, NodeInfo, DependencyGraphBuilder
from modules.ripple_engine     import RippleEngine
from modules.risk_predictor    import RiskPredictor, RiskScore

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BLUE   = "\033[34m"


def banner(text: str) -> None:
    w = 70
    print(f"\n{BOLD}{BLUE}{'─' * w}")
    print(f"  {text}")
    print(f"{'─' * w}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 1 — Semantic change detection
# ─────────────────────────────────────────────────────────────────────────────

def demo_semantic() -> None:
    banner("DEMO 1 — Semantic Change Detection (GraphCodeBERT)")

    cases = [
        (
            "Boundary condition change  (x > 5  →  x >= 5)",
            "def is_eligible(x):\n    if x > 5:\n        return True\n    return False",
            "def is_eligible(x):\n    if x >= 5:\n        return True\n    return False",
        ),
        (
            "Refactor — same logic, different structure",
            "total = price * quantity",
            "def calculate_total(p, q):\n    return p * q\ntotal = calculate_total(price, quantity)",
        ),
        (
            "Whitespace / formatting only",
            "def add(a,b):\n  return a+b",
            "def add(a, b):\n    return a + b",
        ),
        (
            "Auth bypass — security-critical change",
            "def authenticate(user, pwd):\n    return check_password(user, pwd)",
            "def authenticate(user, pwd):\n    return True  # TODO: fix later",
        ),
    ]

    for title, old_code, new_code in cases:
        result = compute_semantic_similarity(old_code, new_code)
        colour = RED if result.change_type == "LOGIC_CHANGE" else (
                 YELLOW if result.change_type == "REFACTOR" else GREEN)
        print(f"\n  {BOLD}{title}{RESET}")
        print(f"    similarity           : {result.similarity:.4f}")
        print(f"    semantic_change_score: {result.semantic_change_score:.4f}")
        print(f"    change_type          : {colour}{result.change_type}{RESET}")
        print(f"    model                : {result.model_used}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 2 — Dependency graph + ripple propagation
# ─────────────────────────────────────────────────────────────────────────────

def demo_ripple() -> None:
    banner("DEMO 2 — Dependency Graph & Ripple Propagation")

    # Manually build the example graph from the spec:
    #   FileA.funcA  →  FileB.funcB  →  FileC.funcC
    dg = DependencyGraph()

    nodes = [
        NodeInfo("FileA.py::funcA", "function", "FileA.py", "funcA", 10),
        NodeInfo("FileB.py::funcB", "function", "FileB.py", "funcB", 20),
        NodeInfo("FileC.py::funcC", "function", "FileC.py", "funcC", 30),
        NodeInfo("FileD.py::funcD", "function", "FileD.py", "funcD", 40),
        NodeInfo("FileE.py::funcE", "function", "FileE.py", "funcE", 50),
    ]
    for n in nodes:
        dg.add_node(n)

    # funcB calls funcA, funcC calls funcB, funcD calls funcA, funcE calls funcC
    dg.add_edge("FileB.py::funcB", "FileA.py::funcA", "calls")
    dg.add_edge("FileC.py::funcC", "FileB.py::funcB", "calls")
    dg.add_edge("FileD.py::funcD", "FileA.py::funcA", "calls")
    dg.add_edge("FileE.py::funcE", "FileC.py::funcC", "calls")

    print(f"\n  Graph: {dg.node_count()} nodes, {dg.edge_count()} edges")
    print("\n  Call graph:")
    print("    FileB.funcB ──calls──► FileA.funcA  (CHANGED)")
    print("    FileC.funcC ──calls──► FileB.funcB")
    print("    FileD.funcD ──calls──► FileA.funcA")
    print("    FileE.funcE ──calls──► FileC.funcC")

    engine = RippleEngine(dg)
    ripple = engine.propagate(["FileA.py::funcA"])

    print(f"\n  {BOLD}Change detected in:{RESET} FileA.py::funcA")
    print(f"\n  {RED}Direct impact{RESET}   : {ripple.direct_impact}")
    print(f"  {YELLOW}Indirect impact{RESET} : {ripple.indirect_impact}")
    print(f"  {CYAN}Impacted files{RESET}  : {ripple.impacted_files}")
    print(f"\n  Ripple depth : {ripple.ripple_depth}")
    print(f"  Ripple size  : {ripple.ripple_size} nodes")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 3 — Full risk scoring
# ─────────────────────────────────────────────────────────────────────────────

def demo_risk() -> None:
    banner("DEMO 3 — Risk Prediction (combined signal)")

    scenarios = [
        {
            "name":     "HIGH RISK — auth bypass + wide ripple",
            "old_code": "def authenticate(user, pwd):\n    return check_password(user, pwd)",
            "new_code": "def authenticate(user, pwd):\n    return True",
            "lines":    5,
            "ripple_size": 15,
            "ripple_depth": 4,
            "fn_count": 1,
        },
        {
            "name":     "MEDIUM RISK — refactor with moderate blast radius",
            "old_code": "def fetch_data(url):\n    resp = requests.get(url)\n    return resp.json()",
            "new_code": "async def fetch_data(url):\n    async with aiohttp.ClientSession() as s:\n        async with s.get(url) as r:\n            return await r.json()",
            "lines":    50,
            "ripple_size": 6,
            "ripple_depth": 2,
            "fn_count": 2,
        },
        {
            "name":     "LOW RISK — docs + trivial change",
            "old_code": "def get_name():\n    return self.name",
            "new_code": 'def get_name():\n    """Return the name."""\n    return self.name',
            "lines":    3,
            "ripple_size": 1,
            "ripple_depth": 0,
            "fn_count": 1,
        },
    ]

    from coderipple.modules.ripple_engine import RippleResult

    for s in scenarios:
        semantic = compute_semantic_similarity(s["old_code"], s["new_code"])

        # Fake ripple result matching the scenario
        from dataclasses import replace
        ripple = RippleResult(
            changed_nodes   = ["FileA.py::funcA"],
            direct_impact   = [f"FileB.py::fn{i}" for i in range(min(s["ripple_size"] - 1, 3))],
            indirect_impact = [f"FileC.py::fn{i}" for i in range(max(0, s["ripple_size"] - 4))],
            impacted_files  = [f"File{chr(66+i)}.py" for i in range(min(s["ripple_size"] - 1, 3))],
            ripple_depth    = s["ripple_depth"],
            ripple_size     = s["ripple_size"],
            impact_map      = {},
        )

        risk = RiskPredictor.predict(
            semantic            = semantic,
            ripple              = ripple,
            total_lines_changed = s["lines"],
            changed_fn_count    = s["fn_count"],
        )

        colour = RED if risk.label == "HIGH" else (YELLOW if risk.label == "MEDIUM" else GREEN)
        print(f"\n  {BOLD}{s['name']}{RESET}")
        print(f"    change_type  : {semantic.change_type}")
        print(f"    similarity   : {semantic.similarity:.4f}")
        print(f"    risk_score   : {risk.score:.4f}")
        print(f"    risk_label   : {colour}{risk.label}{RESET}")
        print(f"    confidence   : {risk.confidence:.4f}")
        print(f"    factors      :")
        for f in risk.contributing_factors:
            print(f"      • {f}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO 4 — Full JSON output (spec format)
# ─────────────────────────────────────────────────────────────────────────────

def demo_json_output() -> None:
    banner("DEMO 4 — Full JSON Output (specification format)")

    from coderipple.modules.ripple_engine import RippleResult

    old_code = "def calculate_discount(price, user):\n    if user.tier == 'gold':\n        return price * 0.20\n    return 0"
    new_code = "def calculate_discount(price, user):\n    if user.tier == 'gold' or user.tier == 'silver':\n        return price * 0.20\n    return 0"

    semantic = compute_semantic_similarity(old_code, new_code)
    ripple   = RippleResult(
        changed_nodes   = ["billing.py::calculate_discount"],
        direct_impact   = ["checkout.py::process_payment"],
        indirect_impact = ["reports.py::generate_invoice"],
        impacted_files  = ["checkout.py", "reports.py"],
        ripple_depth    = 2,
        ripple_size     = 3,
        impact_map      = {
            "checkout.py::process_payment": 1,
            "reports.py::generate_invoice": 2,
        },
    )
    risk = RiskPredictor.predict(semantic, ripple, 8, 1)

    output = {
        "commit":               "abc123",
        "changed_function":     "billing.py::calculate_discount",
        "semantic_change_score": semantic.semantic_change_score,
        "change_type":          semantic.change_type,
        "direct_impact":        ripple.direct_impact,
        "indirect_impact":      ripple.indirect_impact,
        "impacted_files":       ripple.impacted_files,
        "ripple_depth":         ripple.ripple_depth,
        "ripple_size":          ripple.ripple_size,
        "risk_prediction":      risk.label,
        "risk_score":           risk.score,
        "contributing_factors": risk.contributing_factors,
    }

    print(json.dumps(output, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{BOLD}{'═' * 70}")
    print("  CodeRipple — Semantic Change Impact Analyzer")
    print("  Example Run")
    print(f"{'═' * 70}{RESET}")

    demo_semantic()
    demo_ripple()
    demo_risk()
    demo_json_output()

    print(f"\n{GREEN}{BOLD}✓  All demos completed.{RESET}\n")
