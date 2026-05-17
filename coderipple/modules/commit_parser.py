"""
commit_parser.py

Stage 1: Git Commit Parser

Extracts the diff from a commit, then locates the old/new source of every
modified function so the semantic analyser has clean code to embed.

Returns a list of ChangedFunction objects — one per function that was
added, removed, or modified in the commit.
"""

from __future__ import annotations

import hashlib
import logging
import re
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# File extensions we care about for semantic analysis
SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx",
                        ".java", ".go", ".cpp", ".c", ".cs",
                        ".rb", ".php", ".swift", ".kt", ".rs"}

# Extensions that are purely documentation / config — never need BERT analysis.
# If a commit ONLY touches these, it is a FORMAT_CHANGE by definition.
NON_CODE_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".rst",   # documentation
    ".json", ".yaml", ".yml", ".toml",    # config files
    ".lock", ".gitignore", ".gitattributes",
    ".env", ".env.example",
    ".editorconfig", ".prettierrc", ".eslintrc",
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico",  # assets
    ".csv", ".xml", ".html", ".htm",
}


@dataclass
class FileDiff:
    old_path:  str
    new_path:  str
    patch:     str          # raw unified diff
    additions: int
    deletions: int
    status:    str          # added | modified | deleted | renamed


@dataclass
class ChangedFunction:
    file_path:      str
    function_name:  str     # "ClassName.method" or "function_name"
    old_source:     str     # code BEFORE the commit (empty if new function)
    new_source:     str     # code AFTER the commit  (empty if deleted)
    added_lines:    int
    removed_lines:  int
    patch_fragment: str     # the diff lines touching this function
    return_changed:       bool = False
    is_minified:          bool = False   # True when source is minified/bundled JS
    patch_has_logic_signal: bool = False # True when patch shows high-confidence logic change


#  Git extraction 

class CommitParser:

    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path).resolve()
        self._repo     = None

    def _get_repo(self):
        if self._repo is None:
            try:
                import git
                self._repo = git.Repo(str(self.repo_path))
            except Exception as exc:
                raise RuntimeError(f"Cannot open git repo at {self.repo_path}: {exc}")
        return self._repo

    #  Public API 

    def parse_commit(self, commit_hash: str) -> tuple[list[FileDiff], list[ChangedFunction]]:
        """
        Returns (file_diffs, changed_functions) for the given commit.
        """
        repo   = self._get_repo()
        try:
            commit = repo.commit(commit_hash)
        except (ValueError, Exception):
            # Commit not found in shallow history — progressively deepen the clone
            import subprocess
            logger.info("Commit %s not in shallow history — deepening clone…", commit_hash)
            fetched = False
            for depth in (100, 500, 2147483647):  # 2147483647 = full unshallow
                depth_arg = f"--depth={depth}" if depth < 2147483647 else "--unshallow"
                logger.info("Trying fetch %s for %s", depth_arg, commit_hash)
                try:
                    subprocess.run(
                        ["git", "fetch", "origin", depth_arg],
                        cwd=str(self.repo_path),
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    repo.git.execute(["git", "fetch"])  # refresh GitPython's internal state
                    commit = repo.commit(commit_hash)
                    fetched = True
                    logger.info("Found commit %s after %s fetch", commit_hash, depth_arg)
                    break
                except Exception:
                    continue
            if not fetched:
                raise ValueError(
                    f"Commit {commit_hash} not found even after full fetch. "
                    "Ensure the commit hash is correct and belongs to this repository."
                )
        
        parent = commit.parents[0] if commit.parents else None

        file_diffs         = self._extract_file_diffs(repo, commit, parent)
        changed_functions  = []

        for fd in file_diffs:
            ext = Path(fd.new_path or fd.old_path).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            fns = self._extract_changed_functions(repo, fd, commit, parent)
            changed_functions.extend(fns)

        return file_diffs, changed_functions

    #  Diff extraction 

    def _extract_file_diffs(self, repo, commit, parent) -> list[FileDiff]:
        diffs = []
        if parent is None:
            # Initial commit — treat all blobs as "added"
            for item in commit.tree.traverse():
                if hasattr(item, "data_stream"):
                    diffs.append(FileDiff(
                        old_path="",
                        new_path=item.path,
                        patch="",
                        additions=0,
                        deletions=0,
                        status="added",
                    ))
            return diffs

        try:
            diff_list = parent.diff(commit)
        except Exception as e:
            import subprocess
            logger.info("Failed to diff parent %s. Shallow clone boundary hit. Fetching parent...", parent.hexsha)
            subprocess.run(
                ["git", "fetch", "origin", commit.hexsha, "--depth=2"],
                cwd=str(self.repo_path),
                check=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            diff_list = parent.diff(commit)

        for diff in diff_list:
            additions = deletions = 0
            patch     = ""
            try:
                patch_text = repo.git.diff(
                    parent.hexsha, commit.hexsha, "--", diff.b_path or diff.a_path,
                    unified=3
                )
                patch = patch_text
                for line in patch_text.splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        additions += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        deletions += 1
            except Exception:
                pass

            status = "modified"
            if diff.new_file:       status = "added"
            elif diff.deleted_file: status = "deleted"
            elif diff.renamed_file: status = "renamed"

            diffs.append(FileDiff(
                old_path  = diff.a_path or "",
                new_path  = diff.b_path or diff.a_path or "",
                patch     = patch,
                additions = additions,
                deletions = deletions,
                status    = status,
            ))
        return diffs

    #  Function extraction 

    @staticmethod
    def _return_changed(old_src: str, new_src: str) -> bool:
        """
        Fix 9 — Smarter return-value heuristic.

        Only fires when the *structure* of the return changes, not just
        the text.  Specifically we flag it when:
          - The number of return statements changes (function gains/loses an
            early return, which implies a logic branch change).
          - The return expression changes from a scalar/identifier to a
            compound expression (object literal, array, ternary, logical OR/AND
            added at the top level) — i.e. the *type shape* changes.

        We deliberately ignore purely textual differences like whitespace or
        renaming a variable inside the return expression, which were causing
        nearly every minified function to be tagged LOGIC_CHANGE.
        """
        old_returns = re.findall(r'\breturn\b([^;\n]*)', old_src or "")
        new_returns = re.findall(r'\breturn\b([^;\n]*)', new_src or "")

        # 1. Count change: different number of return paths → structural change
        if len(old_returns) != len(new_returns):
            return True

        if not old_returns:
            return False

        # 2. For each paired return expression, check if the *shape* changed
        #    rather than just the identifier names.
        def _shape(expr: str) -> str:
            """
            Reduce an expression to its structural shape:
              - Strip all identifiers / string literals / numbers.
              - Keep operators and brackets so we can compare structure.
            e.g. 'a > b ? a : b'  →  '_ > _ ? _ : _'
                 'a || b'         →  '_ || _'
                 'foo()'          →  '_()'   
                 'a'              →  '_'
            """
            e = expr.strip()
            # Collapse string literals
            e = re.sub(r'(["\']).*?\1', '_', e)
            # Collapse numbers
            e = re.sub(r'\b\d+\.?\d*\b', '_', e)
            # Collapse identifiers (keep operators and brackets)
            e = re.sub(r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b', '_', e)
            # Normalise whitespace
            e = re.sub(r'\s+', ' ', e).strip()
            return e

        for old_r, new_r in zip(old_returns, new_returns):
            if _shape(old_r) != _shape(new_r):
                return True

        return False

    def _extract_changed_functions(
        self,
        repo,
        fd: FileDiff,
        commit,
        parent,
    ) -> list[ChangedFunction]:
        """
        Compare old and new file contents, find which functions changed,
        and return ChangedFunction records with clean source for both sides.
        """
        try:
            old_content = self._blob_content(repo, parent,  fd.old_path) if parent  else ""
            new_content = self._blob_content(repo, commit,  fd.new_path)
        except Exception as exc:
            logger.debug("Could not read blob for %s: %s", fd.new_path, exc)
            return []

        # Fix 7 — detect minified file up front so all functions from it
        # carry the is_minified flag and get the right thresholds later.
        file_is_minified = _is_minified(new_content or old_content)
        if file_is_minified:
            logger.debug(
                "Minified file detected: %s — JS normalizer will run before embedding",
                fd.new_path or fd.old_path,
            )
            old_content = normalize_js_source(old_content)
            new_content = normalize_js_source(new_content)

        old_fns = _extract_functions(old_content, fd.old_path or fd.new_path)
        new_fns = _extract_functions(new_content, fd.new_path)

        # Which lines changed?
        changed_lines = _changed_line_numbers(fd.patch)

        results: list[ChangedFunction] = []
        all_names = set(old_fns) | set(new_fns)

        for name in all_names:
            old_src = old_fns.get(name, "")
            new_src = new_fns.get(name, "")

            if old_src == new_src:
                continue   # identical — skip

            # Fix 10 — hash-based skip: if source hashes match after stripping
            # whitespace, the functions are semantically identical even if the
            # raw string differs (e.g. trailing newline, indentation change).
            old_src_str = old_src if isinstance(old_src, str) else old_src.get("source", "")
            new_src_str = new_src if isinstance(new_src, str) else new_src.get("source", "")
            old_hash = hashlib.sha256(re.sub(r'\s+', '', old_src_str).encode()).hexdigest()
            new_hash = hashlib.sha256(re.sub(r'\s+', '', new_src_str).encode()).hexdigest()
            if old_hash == new_hash:
                continue   # whitespace-only diff — skip BERT entirely

            # Check if this function's lines overlap with the diff
            fn_info = new_fns.get(name) or old_fns.get(name)
            if isinstance(fn_info, dict):
                fn_lines = set(range(fn_info["start"], fn_info["end"] + 1))
                if not fn_lines.intersection(changed_lines) and old_src and new_src:
                    continue   # function unchanged by this diff

            added   = sum(1 for l in (new_src_str or "").splitlines() if l.strip())
            removed = sum(1 for l in (old_src_str or "").splitlines() if l.strip())

            ret_changed = self._return_changed(old_src_str, new_src_str)

            # Fix: calculate logic signal on function-specific diff to avoid poisoning
            # from other changes in the same file. Skip difflib if function is massive
            # to avoid O(N^2) hanging.
            old_lines_len = len((old_src_str or "").splitlines())
            new_lines_len = len((new_src_str or "").splitlines())
            if old_lines_len + new_lines_len < 3000:
                fn_diff = "".join(difflib.unified_diff(
                    (old_src_str or "").splitlines(keepends=True),
                    (new_src_str or "").splitlines(keepends=True),
                    n=3
                ))
            else:
                fn_diff = fd.patch
            logic_signal = detect_patch_logic_signals(fn_diff)

            results.append(ChangedFunction(
                file_path             = fd.new_path or fd.old_path,
                function_name         = name,
                old_source            = old_src_str,
                new_source            = new_src_str,
                added_lines           = added,
                removed_lines         = removed,
                patch_fragment        = fn_diff or fd.patch, # prefer specific diff
                return_changed        = ret_changed,
                is_minified           = file_is_minified,
                patch_has_logic_signal= logic_signal,
            ))

        return results

    @staticmethod
    def _blob_content(repo, commit_or_tree, path: str) -> str:
        if not path:
            return ""
        try:
            blob = commit_or_tree.tree[path]
            return blob.data_stream.read().decode("utf-8", errors="replace")
        except Exception:
            return ""


#  Patch-level logic signal detector 

def detect_patch_logic_signals(patch: str) -> bool:
    """
    Analyzes the raw unified diff for high-confidence signals of a
    semantic/logic change that BERT misses when overall file similarity
    is high (e.g. changing one operator in a 5-line script).

    Returns True if ANY of these signals are found on the changed lines:

    1. Compound assignment operator added or removed
       (-=, +=, *=, /=, //=, **=, %=, &=, |=, ^=)
       → catches:  total += i   →   total = i if ... else total * i

    2. Ternary / conditional expression introduced
       Python: `x if cond else y`
       JS/C:   `cond ? x : y`
       → catches any new branch added inline

    3. New comparison operator on added lines that wasn’t on removed lines
       (==, !=, <=, >=, plain < or >)
       → catches: a new equality/inequality guard being added

    4. New control-flow keyword on added lines absent from removed lines
       (if, elif, else, while, for, try, except, catch, throw, raise,
        break, continue, return)
       → catches: a simple assignment gaining a conditional branch

    5. Logical operator change / addition
       (and/or in Python, &&/|| in C/JS)
       → catches: condition logic being composed differently

    6. Arithmetic operator swap on changed lines
       (+, -, *, /, %, **) when the set of operators changes
       → catches: sum ↔ product, subtraction ↔ addition etc.
    """
    if not patch:
        return False

    added_lines   = [l[1:] for l in patch.splitlines()
                     if l.startswith("+") and not l.startswith("+"+"+")]
    removed_lines = [l[1:] for l in patch.splitlines()
                     if l.startswith("-") and not l.startswith("-"+"-")]

    if not added_lines and not removed_lines:
        return False

    added_text   = " ".join(added_lines)
    removed_text = " ".join(removed_lines)

    # —— Signal 1: compound assignment operator TYPE changed ————————————
    # Only fires when BOTH sides have compound operators but DIFFERENT ones,
    # e.g. += on removed and *= on added (sum logic → product logic).
    # A compound operator simply disappearing (e.g. += replaced by sum())
    # is NOT conclusive — it's a common functional-style refactor pattern.
    _compound_re = re.compile(r'\+=|-=|\*=|/=|//=|%=|\*\*=|&=|\|=|\^=')
    added_ops_compound   = set(_compound_re.findall(added_text))
    removed_ops_compound = set(_compound_re.findall(removed_text))
    if added_ops_compound and removed_ops_compound and added_ops_compound != removed_ops_compound:
        # Both sides use compound assignment but with different operators
        return True

    # —— Signal 2: ternary expression introduced on an added line —————————
    _py_ternary  = re.compile(r'\bif\b.+\belse\b')       # Python: x if c else y
    _c_ternary   = re.compile(r'\?[^:\n]+:')              # JS/C: c ? x : y
    for line in added_lines:
        if _py_ternary.search(line) or _c_ternary.search(line):
            return True

    # —— Signal 3: new comparison on added lines, absent from removed lines ——
    _compare_re  = re.compile(r'==|!=|<=|>=|(?<![<>!])<(?![<=])|(?<![<>!])>(?![>=])')
    added_has_compare   = bool(_compare_re.search(added_text))
    removed_has_compare = bool(_compare_re.search(removed_text))
    if added_has_compare and not removed_has_compare:
        return True

    # —— Signal 4: new control-flow keyword on added lines ——————————————
    _flow_re = re.compile(
        r'\b(if|elif|else|while|try|except|catch|throw|raise|break|continue|return)\b'
    )
    added_flow   = set(_flow_re.findall(added_text))
    removed_flow = set(_flow_re.findall(removed_text))
    new_flow = added_flow - removed_flow
    if new_flow:
        return True

    # —— Signal 5: logical operator composition changed —————————————————
    # 'not' is intentionally excluded: `not expr` and `expr == 0` are
    # equivalent refactors (e.g. `if not n % 2` vs `if n % 2 == 0`).
    # Only and/or composition changes are meaningful signals.
    _logical_re  = re.compile(r'\b(and|or)\b|\&\&|\|\|')
    added_logical   = set(_logical_re.findall(added_text))
    removed_logical = set(_logical_re.findall(removed_text))
    if added_logical != removed_logical:
        return True

    # —— Signal 6: arithmetic operator set changed on changed lines ———————
    # Only check if BOTH sides have arithmetic operators but they are DIFFERENT,
    # e.g. + swapped for *.
    # Adding or removing an operator entirely (without a swap) is NOT a conclusive
    # logic signal in high-similarity files — it often indicates a refactor
    # (like total += i -> sum()).
    _arith_re = re.compile(r'(?<![+\-*/%])([+\-*/%])(?![=+\-*/%])')
    added_ops   = set(_arith_re.findall(added_text))
    removed_ops = set(_arith_re.findall(removed_text))
    if added_ops and removed_ops and added_ops != removed_ops:
        return True

    return False


#  Minification detection & normalization (Fix 7) 

def _is_minified(source: str) -> bool:
    """
    Returns True when a source file looks minified / bundled.
    Heuristics (any one is sufficient):
      - File has fewer than 5 newlines but >200 characters
      - Median line length across non-empty lines is >150 chars
      - More than 40 % of lines are longer than 200 chars
    """
    if not source:
        return False
    lines = [l for l in source.splitlines() if l.strip()]
    if not lines:
        return False

    # Very few newlines but lots of content
    if len(lines) < 5 and len(source) > 200:
        return True

    lengths = [len(l) for l in lines]
    median  = sorted(lengths)[len(lengths) // 2]
    long_pct = sum(1 for l in lengths if l > 200) / len(lengths)

    return median > 150 or long_pct > 0.40


def normalize_js_source(source: str) -> str:
    """
    Fix 7 — Lightweight pure-Python JS normalizer.

    Expands minified JS into a readable, multi-line form so GraphCodeBERT
    gets consistent, human-readable input rather than a single compressed
    line.  This is intentionally simple (no full AST parse) but effective
    for the common minification patterns seen in lodash/Firebase bundles.

    Strategy:
      - Insert newlines after { } ; and before }
      - Normalise all whitespace runs to single spaces
      - Expand common shorthand patterns
    """
    if not source or not _is_minified(source):
        return source

    # Try jsbeautifier if available (pip install jsbeautifier)
    try:
        import jsbeautifier
        opts = jsbeautifier.default_options()
        opts.indent_size = 2
        return jsbeautifier.beautify(source, opts)
    except ImportError:
        pass

    # Pure-Python fallback: rule-based expansion
    s = source

    # Ensure spaces around operators for readability
    s = re.sub(r'([{};,])', r'\1\n', s)       # newline after { } ; ,
    s = re.sub(r'(?<!\n)\s*\{\s*', ' {\n', s) # space before {
    s = re.sub(r'\s*\}\s*', '\n}\n', s)        # newline around }
    s = re.sub(r'[ \t]+', ' ', s)             # collapse inline spaces
    s = re.sub(r'\n{3,}', '\n\n', s)          # collapse blank lines
    return s.strip()


#  Language-agnostic function extractor (regex-based fallback) 

_PYTHON_FN_RE  = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
_JS_FN_RE      = re.compile(r"^(\s*)(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\()", re.MULTILINE)
_CLASS_METHOD_RE = re.compile(r"^(\s{4,})(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)


def _extract_functions(source: str, file_path: str) -> dict[str, str]:
    """
    Extract {qualified_name: source_code} from a source file.
    Tries tree-sitter first, falls back to regex.
    """
    if not source.strip():
        return {}
    try:
        return _extract_with_treesitter(source, file_path)
    except Exception:
        pass
    return _extract_with_regex(source, file_path)


# Module-level cache so Language / Parser objects are created once.
_TREESITTER_PARSERS: dict[str, "Parser"] = {}

def _get_treesitter_parser(ext: str):
    """Return a cached tree-sitter Parser for the given file extension."""
    if ext in _TREESITTER_PARSERS:
        return _TREESITTER_PARSERS[ext]
    from tree_sitter import Language, Parser
    if ext == ".py":
        import tree_sitter_python as _ts_py
        lang = Language(_ts_py.language())
    elif ext in {".js", ".jsx", ".ts", ".tsx"}:
        import tree_sitter_javascript as _ts_js
        lang = Language(_ts_js.language())
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    parser = Parser(lang)
    _TREESITTER_PARSERS[ext] = parser
    return parser


def _extract_with_treesitter(source: str, file_path: str) -> dict[str, str]:
    """Use tree-sitter for accurate AST-based function extraction."""
    ext = Path(file_path).suffix.lower()
    parser = _get_treesitter_parser(ext)
    tree   = parser.parse(source.encode())
    lines  = source.splitlines()
    result = {}

    def visit(node, class_name: str = ""):
        if node.type in ("function_definition", "function_declaration",
                         "method_definition", "arrow_function"):
            name_node = node.child_by_field_name("name")
            fn_name   = name_node.text.decode() if name_node else "<lambda>"
            qname     = f"{class_name}.{fn_name}" if class_name else fn_name
            start, end = node.start_point[0], node.end_point[0]
            result[qname] = "\n".join(lines[start : end + 1])

        elif node.type == "class_definition":
            cn_node = node.child_by_field_name("name")
            cn      = cn_node.text.decode() if cn_node else ""
            for child in node.children:
                visit(child, class_name=cn)
            return

        for child in node.children:
            visit(child, class_name)

    visit(tree.root_node)
    return result


def _extract_with_regex(source: str, file_path: str) -> dict[str, str]:
    """Regex fallback — Python and JS/TS."""
    ext    = Path(file_path).suffix.lower()
    lines  = source.splitlines()
    result = {}

    if ext == ".py":
        pattern = _PYTHON_FN_RE
    else:
        pattern = _JS_FN_RE

    for m in pattern.finditer(source):
        name  = m.group(2) or m.group(3) or "<anon>"
        start = source[:m.start()].count("\n")
        # Grab the next 50 lines as an approximation
        snippet = "\n".join(lines[start: start + 50])
        result[name] = snippet

    return result


def _changed_line_numbers(patch: str) -> set[int]:
    """Parse a unified diff patch → set of NEW-file line numbers that changed."""
    changed: set[int] = set()
    current_line = 0
    for line in patch.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m:
            current_line = int(m.group(1)) - 1
            continue
        if line.startswith("+") and not line.startswith("+++"):
            changed.add(current_line)
            current_line += 1
        elif not line.startswith("-"):
            current_line += 1
    return changed
