# tools/callgraph_ast.py
from __future__ import annotations
import ast, json, sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# -------- helpers --------
def module_name(root: Path, file: Path) -> str:
    rel = file.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

def resolve_relative(pkg: str, target: str, level: int) -> str:
    # rudimentary relative import resolver (e.g., from .sub import x)
    if level == 0:
        return target
    base_parts = pkg.split(".")
    if level > len(base_parts):
        return target
    base = ".".join(base_parts[:-level])
    return f"{base}.{target}" if target else base

# -------- AST visitor --------
class CGVisitor(ast.NodeVisitor):
    def __init__(self, modname: str, all_defs: Set[str]) -> None:
        self.modname = modname
        self.all_defs = all_defs  # fully-qualified defs across project
        self.alias_to_module: Dict[str, str] = {}   # name -> module fq
        self.alias_to_object: Dict[str, str] = {}   # name -> module.f
        self.scope_stack: List[str] = []            # ["module.func", "module.Class.method", ...]
        self.edges: Set[Tuple[str, str]] = set()

    # --- defs & imports ---
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.alias_to_module[alias.asname] = alias.name
            else:
                head = alias.name.split(".", 1)[0]
                self.alias_to_module[head] = head
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        fqmod = resolve_relative(self.modname, mod, node.level or 0)
        for alias in node.names:
            name = alias.asname or alias.name
            # from m import f   -> alias_to_object[name] = m.f
            # from m import sub -> may be module or object; assume module first
            self.alias_to_object[name] = f"{fqmod}.{alias.name}"
            # also allow using just 'sub' as a module base
            base = alias.name.split(".", 1)[0]
            self.alias_to_module[name] = f"{fqmod}.{base}"
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        fq = f"{self.modname}.{node.name}"
        self.scope_stack.append(fq)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # record methods as mod.Class.method
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fq = f"{self.modname}.{node.name}.{item.name}"
                self.scope_stack.append(fq)
                self.generic_visit(item)
                self.scope_stack.pop()
        # visit nested classes / other content
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)

    # --- calls ---
    def visit_Call(self, node: ast.Call) -> None:
        if not self.scope_stack:
            # calls at module import time: ignore for now
            self.generic_visit(node)
            return
        caller = self.scope_stack[-1]
        callee_fq = self._resolve_callee(node.func)
        if callee_fq and callee_fq in self.all_defs:
            self.edges.add((caller, callee_fq))
        self.generic_visit(node)

    def _resolve_callee(self, func: ast.AST) -> Optional[str]:
        # f(...)
        if isinstance(func, ast.Name):
            name = func.id
            # local function in same module
            cand = f"{self.modname}.{name}"
            if cand in self.all_defs:
                return cand
            # imported symbol
            if name in self.alias_to_object:
                return self.alias_to_object[name]
            # name refers to a module alias; can't know member without Attribute
            return None

        # mod.f(...), pkg.sub.f(...)
        if isinstance(func, ast.Attribute):
            base = func.value
            attr = func.attr
            # base is a name (alias of module)
            if isinstance(base, ast.Name):
                base_name = base.id
                # from m import f as g -> g()
                if base_name in self.alias_to_object:
                    # it's an object; then attr is likely a method; we can’t resolve further
                    return f"{self.alias_to_object[base_name]}"
                # import m as mm -> mm.f
                if base_name in self.alias_to_module:
                    return f"{self.alias_to_module[base_name]}.{attr}"
                # local class or module-level name — not resolved
                return None
            # base is attribute chain (e.g., pkg.sub.mod.f) — attempt best-effort
            chain = self._attr_chain(func)
            if chain:
                head, *rest = chain.split(".")
                if head in self.alias_to_module:
                    return ".".join([self.alias_to_module[head]] + rest)
                return chain
        return None

    def _attr_chain(self, node: ast.Attribute) -> Optional[str]:
        parts: List[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None

# -------- driver --------
def collect_defs(root: Path, pkg_root: Path) -> Set[str]:
    """Collect all function defs under root as fully-qualified names."""
    defs: Set[str] = set()
    for file in root.rglob("*.py"):
        if any(seg.startswith(".") for seg in file.parts):
            continue
        mod = module_name(pkg_root, file)
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except UnicodeDecodeError:
            tree = ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.add(f"{mod}.{node.name}")
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defs.add(f"{mod}.{node.name}.{item.name}")
    return defs

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/callgraph_ast.py <package_dir> [--out callgraph.json] [--prefix app]", file=sys.stderr)
        return 2
    src_dir = Path(sys.argv[1]).resolve()
    out_path = Path("callgraph.json").resolve()
    prefix = "app"
    args = sys.argv[2:]
    if "--out" in args:
        i = args.index("--out")
        out_path = Path(args[i+1]).resolve()
    if "--prefix" in args:
        i = args.index("--prefix")
        prefix = args[i+1]

    # package root is the repo root that contains the package dir
    pkg_root = src_dir.parents[0]
    all_defs = collect_defs(src_dir, pkg_root)

    edges: Set[Tuple[str, str]] = set()
    for file in src_dir.rglob("*.py"):
        mod = module_name(pkg_root, file)
        try:
            txt = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = file.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(txt, filename=str(file))
        except SyntaxError:
            continue
        vis = CGVisitor(mod, all_defs)
        vis.visit(tree)
        # keep only project-local edges by prefix (default 'app')
        for a, b in vis.edges:
            if a.startswith(prefix + ".") and b.startswith(prefix + "."):
                edges.add((a, b))

    payload = {"edges": [{"caller": a, "callee": b} for (a, b) in sorted(edges)]}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[callgraph_ast] Wrote {len(edges)} edges to {out_path}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
