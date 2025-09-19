# tools/mermaid.py
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
routes_path = ROOT / "routes.json"
callgraph_path = ROOT / "callgraph.json"   # optional (see notes below)

def load_json_bytes(p: Path):
    if not p.exists() or p.stat().st_size == 0:
        sys.stderr.write(f"[mermaid] missing or empty: {p}\n")
        sys.exit(1)
    data = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return json.loads(data.decode(enc))
        except Exception:
            pass
    sys.stderr.write(f"[mermaid] could not decode JSON: {p}\n")
    sys.exit(1)

routes = load_json_bytes(routes_path)

# Optional call graph (edges: list[{"caller": "...", "callee": "..."}])
edges: set[tuple[str, str]] = set()
if callgraph_path.exists() and callgraph_path.stat().st_size > 0:
    try:
        cg = load_json_bytes(callgraph_path)
        # accept either {"edges":[...]} or a plain list
        edge_list = cg["edges"] if isinstance(cg, dict) and "edges" in cg else cg
        for e in edge_list:
            a = e.get("caller") or e.get("from") or e.get("src")
            b = e.get("callee") or e.get("to")   or e.get("dst")
            if a and b:
                edges.add((a, b))
    except Exception as e:
        print(f"%% [mermaid] ignoring callgraph.json: {e}", file=sys.stderr)

def node_id_from_path_and_method(path: str, method: str) -> str:
    base = path.strip("/").replace("/", "_").replace("{","").replace("}","")
    if not base: base = "_"
    return f"EP_{base}_{method}"

def fn_id(fn: str) -> str:
    # safe mermaid id for "pkg.mod.func"
    return "FN_" + fn.replace(":", ".").replace(" ", "_").replace("/", "_")

def data_id(name: str) -> str:
    return "DT_" + name

# Build a quick index of handlers so we can attach function edges later
handlers: set[str] = set()
for r in routes:
    h = r.get("endpoint")
    if h:
        handlers.add(h)

# Collect tag nodes (dedupe)
all_tags: set[str] = set()
for r in routes:
    for t in (r.get("tags") or []):
        all_tags.add(t)

print("flowchart TD")

# Classes for styling
print("""
classDef endpoint fill:#eef,stroke:#88a,stroke-width:1px;
classDef handler  fill:#efe,stroke:#6a6,stroke-width:1px;
classDef data     fill:#fee,stroke:#c88,stroke-width:1px;
classDef tag      fill:#eee,stroke:#bbb,stroke-dasharray: 3 3;
classDef dep      fill:#fff4cc,stroke:#c7a84f,stroke-width:1px;

%% Tag nodes (one per tag)
""".strip())

for t in sorted(all_tags):
    print(f'TAG_{t}["tag: {t}"]:::tag')

# Emit endpoints, handlers, data
for r in routes:
    methods = r.get("methods", [])
    method = methods[0] if methods else ""
    path = r.get("path", "")
    ep_id = node_id_from_path_and_method(path, method)
    handler = r.get("endpoint") or ""
    req = r.get("request_model")
    resp = r.get("response_model")
    tags = r.get("tags") or []

    # Endpoint
    print(f'{ep_id}["{method} {path}"]:::endpoint')

    # Link to handler (and style handler)
    if handler:
        hid = fn_id(handler)
        print(f'{hid}["{handler}"]:::handler')
        print(f"{ep_id} --> {hid}")

        src = r.get("endpoint_file")
        line = r.get("endpoint_line")
        if src:
            href = Path(src).resolve().as_uri() + (f"#L{line}" if line else "")
            # Many Mermaid previewers make these links clickable
            print(f'click {hid} "{href}" "Open source"')
        
        for dep in (r.get("dependencies") or []):
            did = "DEP_" + dep.replace(":", ".").replace(" ", "_").replace("/", "_")
            print(f'{did}["dep: {dep}"]:::dep')
            print(f"{did} --> {hid}")

    # Data models
    if req:
        rid = data_id(req)
        print(f'{rid}["{req}"]:::data')
        print(f"{rid} -->|request| {ep_id}")
    if resp:
        sid = data_id(resp)
        print(f'{sid}["{resp}"]:::data')
        print(f"{ep_id} -->|response| {sid}")

    # Tag links (deduped nodes)
    for t in tags:
        print(f"{ep_id} --- TAG_{t}")

# Optional: attach shallow call graph from handlers outward
# Limit to 1 hop by default to avoid huge graphs; adjust as needed.
if edges:
    print("\n%% Call graph from handlers (1 hop)")
    for (a, b) in sorted(edges):
        if a in handlers:
            aid, bid = fn_id(a), fn_id(b)
            print(f"{aid} --> {bid}")

