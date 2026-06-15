#!/usr/bin/env python3
"""Generate a machine- and LLM-readable map of this repository.

Walks every tracked Python file, parses it with the ``ast`` module (no code is
executed), and extracts:

* the module docstring / a synthesized one-line purpose,
* top-level classes (with their methods) and functions,
* imports, resolved into "internal" (another module in this repo) vs external.

From that it emits three artifacts into ``docs/repo_map/``:

* ``REPO_MAP.md``  -- human/LLM-friendly: file tree, per-file purpose, and an
  embedded Mermaid dependency diagram.
* ``repo_map.mmd`` -- the Mermaid diagram on its own (render with mermaid-cli:
  ``mmdc -i repo_map.mmd -o repo_map.png`` if you want a PNG).
* ``repo_map.json`` -- structured data: every file, its symbols, and the full
  file-level + package-level import edges, for other programs to consume.

Run from anywhere:  ``python tools/repo_map.py``
Stdlib only -- no third-party dependencies.
"""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Repo root = parent of the directory holding this script (tools/..).
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "repo_map"

# Directories we never want in the map even if they contain .py files.
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules",
             ".idea", ".vscode", ".mypy_cache", ".pytest_cache", "build", "dist"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    methods: list[str]
    doc: str


@dataclass
class FuncInfo:
    name: str
    args: list[str]
    doc: str


@dataclass
class FileInfo:
    path: str                       # posix path relative to repo root
    package: str                    # top-level grouping node
    purpose: str                    # one-line description
    doc: str                        # full module docstring
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FuncInfo] = field(default_factory=list)
    imports_internal: list[str] = field(default_factory=list)   # target modules
    imports_external: list[str] = field(default_factory=list)
    imports_missing: list[str] = field(default_factory=list)    # look internal, no file
    loc: int = 0


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def list_python_files() -> list[Path]:
    """Prefer git's view of the repo; fall back to a filtered walk."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        files = [REPO_ROOT / line for line in out.stdout.splitlines() if line.strip()]
        if files:
            return sorted(files)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    files = []
    for p in REPO_ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.relative_to(REPO_ROOT).parts):
            continue
        files.append(p)
    return sorted(files)


def module_name(rel_path: Path) -> str:
    """client/local_trainer.py -> client.local_trainer (drop __init__)."""
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def package_of(rel_path: Path) -> str:
    """Top-level grouping node: first dir, or '(root)' for top-level files."""
    parts = rel_path.parts
    return parts[0] if len(parts) > 1 else "(root)"


# --------------------------------------------------------------------------- #
# Per-file parsing
# --------------------------------------------------------------------------- #
def first_line(text: str | None) -> str:
    if not text:
        return ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def synthesize_purpose(classes: list[ClassInfo], funcs: list[FuncInfo]) -> str:
    bits = []
    if classes:
        names = ", ".join(c.name for c in classes[:4])
        bits.append(f"Defines class{'es' if len(classes) > 1 else ''} {names}")
    if funcs:
        names = ", ".join(f.name for f in funcs[:4])
        bits.append(f"function{'s' if len(funcs) > 1 else ''} {names}")
    return "; ".join(bits) or "(no top-level docstring or symbols)"


def resolve_import(node: ast.AST, current_module: str) -> list[str]:
    """Return the set of fully-qualified module names this import refers to."""
    targets: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            targets.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        if node.level and node.level > 0:
            # Relative import: resolve against the current module's package.
            base_parts = current_module.split(".")
            # one level up == drop the module name itself
            up = node.level
            base = base_parts[: len(base_parts) - up] if up <= len(base_parts) else []
            mod = ".".join(base + ([node.module] if node.module else []))
            targets.append(mod)
        elif node.module:
            targets.append(node.module)
    return [t for t in targets if t]


def parse_file(path: Path, internal_roots: set[str],
               internal_modules: set[str]) -> FileInfo:
    rel = path.relative_to(REPO_ROOT)
    rel_posix = rel.as_posix()
    mod = module_name(rel)
    src = path.read_text(encoding="utf-8", errors="replace")
    loc = src.count("\n") + 1

    info = FileInfo(path=rel_posix, package=package_of(rel), purpose="", doc="", loc=loc)

    try:
        tree = ast.parse(src, filename=rel_posix)
    except SyntaxError as exc:
        info.purpose = f"(could not parse: {exc.msg})"
        return info

    info.doc = ast.get_docstring(tree) or ""

    seen_internal: set[str] = set()
    seen_external: set[str] = set()
    seen_missing: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            info.classes.append(ClassInfo(
                name=node.name,
                bases=[ast.unparse(b) for b in node.bases] if hasattr(ast, "unparse") else [],
                methods=[n.name for n in node.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
                doc=first_line(ast.get_docstring(node)),
            ))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.functions.append(FuncInfo(
                name=node.name,
                args=[a.arg for a in node.args.args],
                doc=first_line(ast.get_docstring(node)),
            ))

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for target in resolve_import(node, mod):
                root = target.split(".")[0]
                if root in internal_roots:
                    # Find the best matching internal module for the edge.
                    match = _best_internal_match(target, internal_modules)
                    if match:
                        seen_internal.add(match)
                    else:
                        seen_missing.add(target)
                else:
                    seen_external.add(root)

    info.imports_internal = sorted(seen_internal)
    info.imports_external = sorted(seen_external)
    info.imports_missing = sorted(seen_missing)

    info.purpose = first_line(info.doc) or synthesize_purpose(info.classes, info.functions)
    return info


def _best_internal_match(target: str, internal_modules: set[str]) -> str | None:
    """`server.aggregator.federated_averaging` -> `server.aggregator`.

    Try the longest dotted prefix that is a real module in the repo.
    """
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in internal_modules:
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Output builders
# --------------------------------------------------------------------------- #
def build_tree(files: list[FileInfo]) -> str:
    """ASCII tree of the python files, grouped by directory."""
    paths = sorted(f.path for f in files)
    purpose_by_path = {f.path: f.purpose for f in files}
    lines: list[str] = []
    last_dirs: list[str] = []

    for path in paths:
        parts = path.split("/")
        dirs, name = parts[:-1], parts[-1]
        # print directory headers that changed
        for depth, d in enumerate(dirs):
            if depth >= len(last_dirs) or last_dirs[depth] != d:
                lines.append(f"{'  ' * depth}{d}/")
        last_dirs = dirs
        indent = "  " * len(dirs)
        purpose = purpose_by_path.get(path, "")
        lines.append(f"{indent}{name}  -- {purpose}")
    return "\n".join(lines)


def package_edges(files: list[FileInfo],
                  modules_by_name: dict[str, FileInfo]) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {}
    for f in files:
        src_pkg = f.package
        for tgt_mod in f.imports_internal:
            tgt_file = modules_by_name.get(tgt_mod)
            if not tgt_file:
                continue
            tgt_pkg = tgt_file.package
            if tgt_pkg == src_pkg:
                continue
            edges.setdefault(src_pkg, set()).add(tgt_pkg)
    return edges


def _mid(name: str) -> str:
    return name.replace("(", "").replace(")", "").replace(".", "_").replace("-", "_")


def build_mermaid(files: list[FileInfo],
                  modules_by_name: dict[str, FileInfo]) -> str:
    edges = package_edges(files, modules_by_name)
    packages = sorted({f.package for f in files})
    lines = ["graph LR"]
    for pkg in packages:
        lines.append(f'    {_mid(pkg)}["{pkg}"]')
    for src in sorted(edges):
        for tgt in sorted(edges[src]):
            lines.append(f"    {_mid(src)} --> {_mid(tgt)}")
    return "\n".join(lines)


def build_markdown(files: list[FileInfo],
                   modules_by_name: dict[str, FileInfo],
                   mermaid: str) -> str:
    from datetime import datetime, timezone

    total_loc = sum(f.loc for f in files)
    packages: dict[str, list[FileInfo]] = {}
    for f in files:
        packages.setdefault(f.package, []).append(f)

    md: list[str] = []
    md.append("# Repository Map\n")
    md.append("> Auto-generated by `tools/repo_map.py`. Do not edit by hand — "
              "re-run the script to refresh.\n")
    md.append(f"- **Files mapped:** {len(files)}")
    md.append(f"- **Total lines of Python:** {total_loc}")
    md.append(f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    # ----- Architecture diagram -----
    md.append("## Architecture (module dependency graph)\n")
    md.append("Arrows point from a package to the packages it imports from.\n")
    md.append("```mermaid")
    md.append(mermaid)
    md.append("```\n")

    # ----- File tree -----
    md.append("## File tree\n")
    md.append("```text")
    md.append(build_tree(files))
    md.append("```\n")

    # ----- Per-package detail -----
    md.append("## Files in detail\n")
    for pkg in sorted(packages):
        md.append(f"### `{pkg}`\n")
        for f in sorted(packages[pkg], key=lambda x: x.path):
            md.append(f"#### `{f.path}`  ({f.loc} LoC)\n")
            md.append(f"{f.purpose}\n")
            if f.classes:
                md.append("**Classes:**")
                for c in f.classes:
                    base = f"({', '.join(c.bases)})" if c.bases else ""
                    meth = f" — methods: {', '.join(c.methods)}" if c.methods else ""
                    doc = f" — {c.doc}" if c.doc else ""
                    md.append(f"- `{c.name}{base}`{doc}{meth}")
                md.append("")
            if f.functions:
                md.append("**Functions:**")
                for fn in f.functions:
                    doc = f" — {fn.doc}" if fn.doc else ""
                    md.append(f"- `{fn.name}({', '.join(fn.args)})`{doc}")
                md.append("")
            if f.imports_internal:
                md.append(f"**Depends on (internal):** {', '.join(f'`{m}`' for m in f.imports_internal)}")
            if f.imports_missing:
                md.append(f"**⚠ Imports not found in repo:** {', '.join(f'`{m}`' for m in f.imports_missing)}")
            if f.imports_external:
                ext = ", ".join(f"`{m}`" for m in f.imports_external)
                md.append(f"**External libs:** {ext}")
            md.append("")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    py_files = list_python_files()
    if not py_files:
        print("No Python files found.")
        return

    rels = [p.relative_to(REPO_ROOT) for p in py_files]
    internal_modules = {module_name(r) for r in rels}
    internal_roots = {r.parts[0] if len(r.parts) > 1 else r.with_suffix("").parts[0]
                      for r in rels}

    files = [parse_file(p, internal_roots, internal_modules) for p in py_files]
    modules_by_name = {module_name(Path(f.path)): f for f in files}

    mermaid = build_mermaid(files, modules_by_name)
    markdown = build_markdown(files, modules_by_name, mermaid)

    payload = {
        "repo_root": REPO_ROOT.name,
        "file_count": len(files),
        "total_loc": sum(f.loc for f in files),
        "files": [asdict(f) for f in files],
        "package_edges": {k: sorted(v)
                          for k, v in package_edges(files, modules_by_name).items()},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPO_MAP.md").write_text(markdown, encoding="utf-8")
    (OUT_DIR / "repo_map.mmd").write_text(mermaid, encoding="utf-8")
    (OUT_DIR / "repo_map.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Mapped {len(files)} files -> {OUT_DIR}")
    print("  - REPO_MAP.md   (tree + per-file purpose + Mermaid diagram)")
    print("  - repo_map.mmd  (diagram source; render with: mmdc -i repo_map.mmd -o repo_map.png)")
    print("  - repo_map.json (structured data)")


if __name__ == "__main__":
    main()
