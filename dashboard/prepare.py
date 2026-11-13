"""Check the data files and write the store documents for the dashboard into build/.

Usage:  python prepare.py            (checks data/*.json, writes build/ds_<id>.json and build/file_index.json)
Then follow README.md: upload each changed data file, put its new blob id in store.json, run this again, write the documents.
Reads and writes only this folder. No network.
"""
import datetime, json, pathlib, re, sys

D = pathlib.Path(__file__).parent
store = json.load(open(D / "store.json", encoding="utf-8"))
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

REQUIRED = {
    "tracks": ["id", "track", "kind", "method", "node", "state", "state_kind", "running", "checkpoint", "blocker", "blocked", "watch", "next", "updated_at"],
    "targets": ["id", "label", "value_pct", "note"],
    "progress": ["id", "run", "robot", "track", "iteration", "total", "reward", "episode_len", "terrain_level", "note"],
    "apparatus_results": ["id", "date", "track", "checkpoint", "apparatus", "simulator", "trials", "success_pct"],
    "g1_results": ["id", "terrain", "checkpoint", "success_pct", "paper_success_pct", "time_s", "paper_time_s", "robots"],
    "details": ["id", "track", "area", "item", "origin", "origin_kind", "status", "status_kind"],
    "robot": ["id", "area", "item", "amount", "status", "status_kind"],
    "renders": ["id", "group", "title", "caption", "label", "credit", "featured", "url"],
    "open_items": ["id", "item", "who", "who_kind", "status", "status_kind"],
    "worklog": ["id", "date", "area", "what", "where"],
    "eval_rules": ["id", "rule", "text", "status", "status_kind"],
    "paper_diffs": ["id", "track", "item", "paper", "ours", "decision", "status", "status_kind"],
}
ALLOWED = {
    ("tracks", "id"): {"acp", "ssr", "hiking", "eval"},
    ("tracks", "kind"): {"track", "eval"},
    ("tracks", "state_kind"): {"setup", "training", "evaluating", "blocked", "paused", "done"},
    ("progress", "run"): {"acp", "ssr", "hiking", "g1"},
    ("progress", "robot"): {"X2", "G1"},
    ("apparatus_results", "track"): {"ACP", "SSR", "Hiking"},
    ("apparatus_results", "apparatus"): {"ours", "nist"},
    ("details", "track"): {"ACP", "SSR", "Hiking"},
    ("details", "origin_kind"): {"paper", "papers", "ours"},
    ("details", "status_kind"): {"done", "partial", "planned", "risk"},
    ("robot", "area"): {"Robot", "Retargeted motion", "Props", "Camera"},
    ("robot", "status_kind"): {"ready", "partial", "todo", "missing"},
    ("renders", "group"): {"map", "robot", "motion"},
    ("open_items", "who_kind"): {"you", "agent"},
    ("open_items", "status_kind"): {"waiting", "progress", "todo", "paused"},
    ("eval_rules", "status_kind"): {"done", "partial", "waiting"},
    ("paper_diffs", "track"): {"SSR", "ACP"},
    ("paper_diffs", "status_kind"): {"done", "pending", "kept"},
}
errors = []
for name, fields in REQUIRED.items():
    if name not in store["datasets"]:
        errors.append(f"store.json has no entry for {name}"); continue
    try:
        rows = json.load(open(D / "data" / f"{name}.json", encoding="utf-8"))
    except Exception as e:
        errors.append(f"data/{name}.json: {e}"); continue
    if not isinstance(rows, list):
        errors.append(f"data/{name}.json must be a list of rows"); continue
    seen = set()
    for i, r in enumerate(rows):
        for f in fields:
            if f not in r:
                errors.append(f"data/{name}.json row {i}: missing '{f}'")
        if r.get("id") in seen:
            errors.append(f"data/{name}.json row {i}: duplicate id {r.get('id')!r}")
        seen.add(r.get("id"))
        for (n, f), ok in ALLOWED.items():
            if n == name and f in r and r[f] not in ok:
                errors.append(f"data/{name}.json row {i}: {f}={r[f]!r} is not one of {sorted(ok)}")
        if name == "renders":
            if not re.fullmatch(r"/_blob/[0-9a-f]{32}", str(r.get("url", ""))):
                errors.append(f"data/renders.json row {i}: url must be the /_blob/<id> the image upload returned")
            if r.get("group") in ("robot", "motion") and r.get("credit") != "Motion data: Bones Studio":
                errors.append(f"data/renders.json row {i}: robot and motion renders need credit 'Motion data: Bones Studio'")
if errors:
    print("\n".join(errors)); sys.exit(1)

B = D / "build"; B.mkdir(exist_ok=True)
edit = {"via": "claude", "at": now}
for name, m in store["datasets"].items():
    doc = {"title": m["title"], "description": m["description"],
           "source": {"kind": "file", "url": "/_blob/" + m["blob"], "name": name + ".json", "format": "json"},
           "updated": {"at": now, "by": "Claude (coordinator session)"},
           "schedule": "Rewritten by the coordinator session after each track or eval report.",
           "staleAfterHours": 72, "edit": edit}
    json.dump(doc, open(B / f"ds_{name}.json", "w", encoding="utf-8"), ensure_ascii=False)
page = (D / "index.html").read_text(encoding="utf-8")
assert page.startswith('<meta name="dash-one-writer">')
json.dump({"text": page, "edit": edit}, open(B / "file_index.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump({"title": "Perception Agents", "lang": "en"}, open(B / "meta.json", "w", encoding="utf-8"))
print(f"ok: {len(store['datasets'])} dataset documents and the page ({len(page):,} characters) written to build/ at {now}")
