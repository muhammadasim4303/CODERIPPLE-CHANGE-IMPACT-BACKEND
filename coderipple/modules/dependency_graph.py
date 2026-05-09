"""
dependency_graph.py
───────────────────
Stage 3: Dependency Graph Builder

Parses a repository and constructs a directed dependency graph where:
  Nodes  →  functions, classes, files
  Edges  →  function calls, imports, inheritance

The graph is stored as a NetworkX DiGraph.
Node IDs use the convention:  "file_path::ClassName.method_name"
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx",
                        ".java", ".go", ".cpp", ".c", ".cs",
                        ".rb", ".php", ".swift", ".kt", ".rs"}


@dataclass
class NodeInfo:
    node_id:   str
    kind:      str        # "function" | "class" | "file"
    file_path: str
    name:      str
    lineno:    int = 0


@dataclass
class DependencyGraph:
    graph:    nx.DiGraph = field(default_factory=nx.DiGraph)
    node_map: dict[str, NodeInfo] = field(default_factory=dict)

    def add_node(self, info: NodeInfo) -> None:
        self.node_map[info.node_id] = info
        self.graph.add_node(info.node_id, **{
            "kind":      info.kind,
            "file_path": info.file_path,
            "name":      info.name,
            "lineno":    info.lineno,
        })

    def add_edge(self, src: str, dst: str, edge_type: str = "calls") -> None:
        if src in self.graph and dst in self.graph:
            self.graph.add_edge(src, dst, type=edge_type)

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()


# ── Repository scanner ────────────────────────────────────────────────────────

class DependencyGraphBuilder:
    """
    Walk every Python / JS / TS file in the repo and build a unified
    dependency graph.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path).resolve()

    def build(self) -> DependencyGraph:
        dg = DependencyGraph()
        files = self._collect_files()
        logger.info("Building dependency graph for %d source files …", len(files))

        for fp in files:
            try:
                self._parse_file(fp, dg)
            except Exception as exc:
                logger.debug("Skipping %s: %s", fp, exc)

        logger.info(
            "Graph ready: %d nodes, %d edges",
            dg.node_count(), dg.edge_count()
        )
        return dg

    def _collect_files(self) -> list[Path]:
        files = []
        skip  = {"node_modules", ".git", "__pycache__", ".venv", "venv",
                  "dist", "build", ".next", "vendor"}
        for fp in self.repo_path.rglob("*"):
            if any(p in fp.parts for p in skip):
                continue
            if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(fp)
        return files

    def _rel(self, fp: Path) -> str:
        try:
            return str(fp.relative_to(self.repo_path))
        except ValueError:
            return str(fp)

    # ── File-level dispatch ───────────────────────────────────────────────

    def _parse_file(self, fp: Path, dg: DependencyGraph) -> None:
        rel = self._rel(fp)
        # File node
        file_id = f"{rel}::__file__"
        dg.add_node(NodeInfo(file_id, "file", rel, rel))

        if fp.suffix == ".py":
            self._parse_python(fp, rel, dg)
        elif fp.suffix in {".js", ".ts", ".jsx", ".tsx"}:
            self._parse_js(fp, rel, dg)

    # ── Python parser ─────────────────────────────────────────────────────

    def _parse_python(self, fp: Path, rel: str, dg: DependencyGraph) -> None:
        source = fp.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(fp))
        except SyntaxError:
            return

        visitor = _PythonVisitor(rel, dg)
        visitor.visit(tree)

    # ── JS / TS parser (regex-based) ──────────────────────────────────────

    def _parse_js(self, fp: Path, rel: str, dg: DependencyGraph) -> None:
        source = fp.read_text(encoding="utf-8", errors="replace")
        _parse_js_source(source, rel, dg)


# ── Python AST visitor ────────────────────────────────────────────────────────

class _PythonVisitor(ast.NodeVisitor):

    def __init__(self, file_path: str, dg: DependencyGraph) -> None:
        self.file_path   = file_path
        self.dg          = dg
        self._class_stack: list[str] = []

    def _node_id(self, name: str) -> str:
        if self._class_stack:
            return f"{self.file_path}::{self._class_stack[-1]}.{name}"
        return f"{self.file_path}::{name}"

    # ── Classes ───────────────────────────────────────────────────────────

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_id = f"{self.file_path}::{node.name}"
        self.dg.add_node(NodeInfo(class_id, "class", self.file_path, node.name, node.lineno))

        # Inheritance edges
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_id = f"{self.file_path}::{base.id}"
                self.dg.add_node(NodeInfo(base_id, "class", self.file_path, base.id))
                self.dg.add_edge(class_id, base_id, "inherits")

        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    # ── Functions ─────────────────────────────────────────────────────────

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        fn_id = self._node_id(node.name)
        self.dg.add_node(NodeInfo(fn_id, "function", self.file_path, node.name, node.lineno))

        # File → function containment
        file_id = f"{self.file_path}::__file__"
        self.dg.add_edge(file_id, fn_id, "contains")

        # Class → method containment
        if self._class_stack:
            class_id = f"{self.file_path}::{self._class_stack[-1]}"
            self.dg.add_edge(class_id, fn_id, "contains")

        # Call edges
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee = _resolve_call(child)
                if callee:
                    callee_id = f"{self.file_path}::{callee}"
                    self.dg.add_node(NodeInfo(callee_id, "function", self.file_path, callee))
                    self.dg.add_edge(fn_id, callee_id, "calls")

        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # same treatment

    # ── Imports ───────────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        file_id = f"{self.file_path}::__file__"
        for alias in node.names:
            dep_id = f"{alias.name}::__file__"
            self.dg.add_node(NodeInfo(dep_id, "file", alias.name, alias.name))
            self.dg.add_edge(file_id, dep_id, "imports")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        file_id = f"{self.file_path}::__file__"
        module  = node.module or ""
        dep_id  = f"{module}::__file__"
        self.dg.add_node(NodeInfo(dep_id, "file", module, module))
        self.dg.add_edge(file_id, dep_id, "imports")


def _resolve_call(node: ast.Call) -> Optional[str]:
    """Best-effort extraction of a callee name from an ast.Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


# ── JS/TS regex parser ────────────────────────────────────────────────────────

_JS_FN_RE     = re.compile(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(")
_JS_ARROW_RE  = re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(")
_JS_CLASS_RE  = re.compile(r"class\s+(\w+)(?:\s+extends\s+(\w+))?")
_JS_METHOD_RE = re.compile(r"^\s{2,}(?:async\s+)?(\w+)\s*\(", re.MULTILINE)
_JS_IMPORT_RE = re.compile(r"""(?:import|require)\s*[\({]?['"]([^'"]+)['"]""")
_JS_CALL_RE   = re.compile(r"(\w+)\s*\(")


def _parse_js_source(source: str, file_path: str, dg: DependencyGraph) -> None:
    file_id = f"{file_path}::__file__"

    # Functions
    for m in _JS_FN_RE.finditer(source):
        name  = m.group(1)
        fn_id = f"{file_path}::{name}"
        lineno = source[:m.start()].count("\n") + 1
        dg.add_node(NodeInfo(fn_id, "function", file_path, name, lineno))
        dg.add_edge(file_id, fn_id, "contains")

    for m in _JS_ARROW_RE.finditer(source):
        name  = m.group(1)
        fn_id = f"{file_path}::{name}"
        lineno = source[:m.start()].count("\n") + 1
        dg.add_node(NodeInfo(fn_id, "function", file_path, name, lineno))
        dg.add_edge(file_id, fn_id, "contains")

    # Classes
    for m in _JS_CLASS_RE.finditer(source):
        name     = m.group(1)
        base     = m.group(2)
        class_id = f"{file_path}::{name}"
        lineno   = source[:m.start()].count("\n") + 1
        dg.add_node(NodeInfo(class_id, "class", file_path, name, lineno))
        dg.add_edge(file_id, class_id, "contains")
        if base:
            base_id = f"{file_path}::{base}"
            dg.add_node(NodeInfo(base_id, "class", file_path, base))
            dg.add_edge(class_id, base_id, "inherits")

    # Imports
    for m in _JS_IMPORT_RE.finditer(source):
        dep    = m.group(1)
        dep_id = f"{dep}::__file__"
        dg.add_node(NodeInfo(dep_id, "file", dep, dep))
        dg.add_edge(file_id, dep_id, "imports")
