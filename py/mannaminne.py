#!/usr/bin/env python3
"""mannaminne v2 — full-corpus search over the personal life-corpus, backed by
Postgres + pgvector on Darwin.

Sources (fully chunked, no truncation): CC session transcripts (noise-filtered),
project/infra docs, Facebook Messenger, AI-chat archives (ChatGPT + Claude),
Simplenote notes. Postgres FTS (tsvector + trigram) is the guaranteed
any-exact-needle keyword layer; pgvector (Qwen3-Embedding-4B via the Darwin
embedder) is the semantic layer.

Subcommands:
  ingest   discover + chunk + upsert all sources (incremental, hash-based)
  embed    fill NULL embeddings via the Darwin embedder (concurrent)
  search   hybrid keyword + semantic query
  stats    per-source counts + embedding coverage

Aliases (argv0): `ccsearch` scopes to CC sources (session+doc); `minne` /
`mannaminne` search everything. Conn read from ~/.config/mannaminne/db.env.
Design: ~/dotfiles/docs/personal_archives_semantic_search_2026_06_10.md § v2.
"""
from __future__ import annotations
import os, sys, json, glob, hashlib, argparse, concurrent.futures, time, urllib.request
import re, email, html as _htmllib
from email import policy as _emailpolicy
import email.utils as _emailutils
from pathlib import Path

HOME = os.path.expanduser("~")
EMBED_URL = os.environ.get("MANNAMINNE_EMBED_URL", "http://192.168.4.1:8080/v1/embeddings")
EMBED_MODEL = os.environ.get("MANNAMINNE_EMBED_MODEL", "qwen3-embedding-4b")
EMBED_DIM = int(os.environ.get("MANNAMINNE_EMBED_DIM", "1024"))  # 8B native=4096; MRL-truncatable to 1024
CHUNK_SIZE = 750          # chars (~250 tokens, safely under the embedder's 512 budget)
CHUNK_OVERLAP = 80        # so a needle on a boundary isn't orphaned
MAX_CHUNKS = 400          # per source object (bounds pathological mega-objects)

# --- DB ---------------------------------------------------------------------

def load_conn():
    env = {}
    p = Path(HOME) / ".config/mannaminne/db.env"
    for line in p.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    import psycopg
    return psycopg.connect(
        host=env["MANNAMINNE_PG_HOST"], port=env["MANNAMINNE_PG_PORT"],
        dbname=env["MANNAMINNE_PG_DB"], user=env["MANNAMINNE_PG_USER"],
        password=env["MANNAMINNE_PG_PASSWORD"], connect_timeout=10,
    )

# --- helpers ----------------------------------------------------------------

def fix_mojibake(s: str) -> str:
    """Reverse FB's double-encoded UTF-8 (latin1→utf8 reinterpret)."""
    try:
        if all(ord(c) < 256 for c in s):
            return s.encode("latin1").decode("utf8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return s

def chunk(text: str):
    text = text.strip()
    if not text:
        return
    n = len(text)
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    i = idx = 0
    while i < n and idx < MAX_CHUNKS:
        yield idx, text[i:i + CHUNK_SIZE]
        i += step
        idx += 1

def h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:16]

# --- source discovery (yields chunk rows) -----------------------------------
# Each row: (id, source_kind, source_id, chunk_idx, project, title, text, created, content_hash)

def _rows(source_kind, source_id, project, title, full, created):
    title = (title or "").replace("\x00", "")          # Postgres text rejects NUL (0x00)
    for idx, ch in chunk(full):
        ch = ch.replace("\x00", "")
        yield (f"{source_id}#{idx}", source_kind, source_id, idx, project,
               title, ch, created, h(ch))

def discover_messenger():
    base = Path(HOME) / "Projects/messenger-archive/your_activity_across_facebook/messages/inbox"
    for d in sorted(glob.glob(str(base / "*"))):
        if not os.path.isdir(d):
            continue
        tid = os.path.basename(d)
        title, parts, newest = "", [], 0
        for f in sorted(glob.glob(os.path.join(d, "message_*.json"))):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            if not title:
                title = fix_mojibake(j.get("title") or "") or ", ".join(
                    fix_mojibake(p.get("name", "")) for p in j.get("participants", []))
            for m in j.get("messages", []):
                newest = max(newest, m.get("timestamp_ms", 0) or 0)
                c = m.get("content")
                if c:
                    parts.append(f"{fix_mojibake(m.get('sender_name',''))}: {fix_mojibake(c)}")
        full = (title + "\n" + "\n".join(parts)).strip()
        if not full:
            continue
        created = time.strftime("%Y-%m-%d", time.gmtime(newest / 1000)) if newest else ""
        yield from _rows("messenger", f"msg:{tid}", "messenger", title or tid[:60], full, created)

def discover_aichat():
    base = Path(HOME) / "Projects/ai-chat-archives"
    for f in glob.glob(str(base / "chatgpt_*/conversations/*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        title = j.get("title") or ""
        msgs = []
        for node in (j.get("mapping") or {}).values():
            msg = (node or {}).get("message") or {}
            ct = (msg.get("create_time") or 0)
            for part in ((msg.get("content") or {}).get("parts") or []):
                if isinstance(part, str) and part.strip():
                    msgs.append((ct, part))
        msgs.sort(key=lambda x: x[0] or 0)
        full = (title + "\n" + "\n".join(p for _, p in msgs)).strip()
        if not full:
            continue
        sid = j.get("conversation_id") or Path(f).stem
        yield from _rows("aichat", f"aichat:cg:{sid}", "chatgpt", title or sid[:60], full, "")
    for f in glob.glob(str(base / "claude_*/conversations/*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        name = j.get("name") or ""
        created = (j.get("created_at") or "")[:10]
        parts = [f"{m.get('sender','')}: {m.get('text','')}"
                 for m in (j.get("chat_messages") or []) if (m.get("text") or "").strip()]
        full = (name + "\n" + "\n".join(parts)).strip()
        if not full:
            continue
        uuid = j.get("uuid") or Path(f).stem
        yield from _rows("aichat", f"aichat:cl:{uuid}", "claude", name or uuid[:60], full, created)

def discover_notes():
    d = Path(HOME) / "Documents/Simplenote Support Notes"
    for f in glob.glob(str(d / "*.txt")):
        try:
            content = open(f, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if not content.strip():
            continue
        name = Path(f).stem
        yield from _rows("note", f"note:{name}", "simplenote", name, content, "")

def discover_docs():
    scans = [(Path(HOME) / "Projects", "*/docs/**/*.md"), (Path(HOME) / "dotfiles", "docs/**/*.md")]
    for base, pat in scans:
        for f in glob.glob(str(base / pat), recursive=True):
            if any(x in f for x in ("/archive/", "/vendor/", "/node_modules/", "/.")):
                continue
            try:
                if os.path.getsize(f) > 600_000:
                    continue
                content = open(f, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            if not content.strip():
                continue
            project = "dotfiles" if str(base).endswith("dotfiles") else Path(f).relative_to(base).parts[0]
            rel = os.path.relpath(f, base)
            yield from _rows("doc", f"doc:{project}:{rel}", project, Path(f).stem, content, "")

_NOISE = ("<system-reminder>", "This session is being continued", "Caveat:",
          "# CLAUDE.md", "Codebase and user instructions are shown below",
          "<command-name>", "<local-command-stdout>", "DO NOT respond to these")

def discover_sessions():
    """CC transcripts, noise-filtered: keep human + assistant natural-language
    text; drop tool calls, injected CLAUDE.md, system reminders, huge boilerplate."""
    for f in glob.glob(os.path.join(HOME, ".claude/projects/*/*.jsonl")):
        if "subagent" in f:
            continue
        sid = Path(f).stem
        proj = Path(f).parent.name.rsplit("-", 1)[-1]
        parts, created = [], ""
        try:
            fh = open(f, encoding="utf-8", errors="replace")
        except Exception:
            continue
        with fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("isCompactSummary"):
                    continue
                typ = o.get("type")
                if typ not in ("user", "assistant"):
                    continue
                if not created:
                    created = (o.get("timestamp") or "")[:10]
                msg = o.get("message") or {}
                cont = msg.get("content")
                texts = []
                if isinstance(cont, str):
                    texts = [cont]
                elif isinstance(cont, list):
                    texts = [b.get("text", "") for b in cont
                             if isinstance(b, dict) and b.get("type") == "text"]
                for t in texts:
                    if not t or len(t) > 12000:        # skip giant boilerplate dumps
                        continue
                    if any(mark in t for mark in _NOISE):
                        continue
                    parts.append(f"{typ}: {t}")
        full = "\n".join(parts).strip()
        if not full:
            continue
        title = (parts[0][:80] if parts else sid)
        yield from _rows("session", f"session:{sid}", proj, title, full, created)

# --- email (mbox: Gmail Takeout + curated subsets) --------------------------
# Streaming parser — never loads the whole file (the Gmail Takeout is 4.7 GB).
# Dedups by Message-ID across all mboxes; reports unique-new per file.

MBOX_SOURCES = [
    ("takeout2014",
     "/Volumes/FERMI/MacMini-archives additions/demeter_2017_drive/"
     "emails_documents_2014/All mail Including Spam and Trash-2.mbox"),
    ("deliberus",
     os.path.join(HOME, "Projects/deliberus/archive/demeter_2017_dropbox/"
                        "excavated_emails/deliberus_relevant.mbox")),
]

def _strip_html(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = _htmllib.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()

_ENVELOPE = re.compile(rb"^From \S+ (Mon|Tue|Wed|Thu|Fri|Sat|Sun) [A-Z][a-z][a-z] +\d")

def _iter_mbox(path):
    """Yield raw message bytes one at a time. Splits on a `From <id> <Weekday>
    <Mon> <DD> …` envelope line (Gmail Takeout / Apple Mail format) — these are
    NOT reliably blank-preceded, so match the envelope shape directly. The strict
    regex avoids false splits on body lines that merely start with 'From '.
    Memory-safe for multi-GB mboxes (streams one message at a time)."""
    buf = bytearray()
    with open(path, "rb") as fh:
        for line in fh:
            if _ENVELOPE.match(line):
                if buf:
                    yield bytes(buf)
                    buf = bytearray()
            buf += line
    if buf:
        yield bytes(buf)

def _email_body(msg) -> str:
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is not None:
            content = part.get_content()
            return _strip_html(content) if part.get_content_subtype() == "html" else content
    except Exception:
        pass
    out = []                                   # fallback: walk parts
    try:
        for p in msg.walk():
            ct = p.get_content_type()
            try:
                if ct == "text/plain":
                    out.append(p.get_content())
                elif ct == "text/html":
                    out.append(_strip_html(p.get_content()))
            except Exception:
                pass
    except Exception:
        pass
    return "\n".join(out)

def discover_email():
    seen = set()
    for label, path in MBOX_SOURCES:
        if not os.path.exists(path):
            print(f"  (email/{label}: missing at {path})", flush=True)
            continue
        nmsg = nuniq = nskip = 0
        for raw in _iter_mbox(path):
            nmsg += 1
            if raw.startswith(b"From "):           # strip mbox envelope separator
                nl = raw.find(b"\n")
                raw = raw[nl + 1:] if nl != -1 else raw
            try:
                msg = email.message_from_bytes(raw, policy=_emailpolicy.default)
            except Exception:
                nskip += 1; continue
            mid = (str(msg.get("Message-ID") or msg.get("Message-Id") or "")).strip().strip("<>")
            subj = str(msg.get("Subject") or "").strip()
            frm = str(msg.get("From") or "").strip()
            to = str(msg.get("To") or "").strip()
            datehdr = str(msg.get("Date") or "").strip()
            body = _email_body(msg) or ""
            if not subj and not body:
                nskip += 1; continue
            key = mid or h(f"{datehdr}|{frm}|{subj}|{len(body)}")
            if key in seen:
                continue
            seen.add(key); nuniq += 1
            created = ""
            try:
                dt = _emailutils.parsedate_to_datetime(datehdr)
                if dt:
                    created = dt.strftime("%Y-%m-%d")
            except Exception:
                pass
            title = subj or (frm[:60] if frm else "(no subject)")
            full = f"{subj}\nFrom: {frm}\nTo: {to}\nDate: {datehdr}\n\n{body}"
            yield from _rows("email", f"email:{h(key)}", "gmail", title, full, created)
        print(f"  email/{label}: {nmsg} msgs → {nuniq} unique-new, {nskip} skipped", flush=True)

ALL = {"messenger": discover_messenger, "aichat": discover_aichat,
       "note": discover_notes, "doc": discover_docs, "session": discover_sessions,
       "email": discover_email}

# --- ingest -----------------------------------------------------------------

def cmd_ingest(args):
    conn = load_conn()
    cur = conn.cursor()
    kinds = args.sources or list(ALL)
    seen, total = [], 0
    for kind in kinds:
        n, batch = 0, []
        for row in ALL[kind]():
            seen.append(row[0]); batch.append(row)
            if len(batch) >= 500:
                _upsert(cur, batch); conn.commit(); n += len(batch); batch = []
        if batch:
            _upsert(cur, batch); conn.commit(); n += len(batch)
        total += n
        print(f"  {kind}: {n} chunks upserted", flush=True)
    # orphan cleanup: drop chunks of the processed kinds NOT produced this run
    # (source object deleted, or shrank below a chunk_idx). Temp-table anti-join.
    if seen:
        cur.execute("CREATE TEMP TABLE _seen (id text)")
        with cur.copy("COPY _seen (id) FROM STDIN") as cp:
            for x in seen:
                cp.write_row((x,))
        cur.execute("CREATE INDEX ON _seen (id)")
        cur.execute("DELETE FROM chunks WHERE source_kind = ANY(%s) "
                    "AND NOT EXISTS (SELECT 1 FROM _seen s WHERE s.id = chunks.id)", (kinds,))
        pruned = cur.rowcount
        cur.execute("DROP TABLE _seen"); conn.commit()
        print(f"  pruned {pruned} orphaned chunks", flush=True)
    print(f"ingest done: {total} chunks. (full-text search is live now.)", flush=True)

def _upsert(cur, rows):
    # Upsert; if the chunk text changed (hash differs) reset its embedding so it re-embeds.
    cur.executemany(
        """INSERT INTO chunks (id,source_kind,source_id,chunk_idx,project,title,text,created,content_hash)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (id) DO UPDATE SET
             text=EXCLUDED.text, title=EXCLUDED.title, project=EXCLUDED.project,
             created=EXCLUDED.created,
             embedding = CASE WHEN chunks.content_hash <> EXCLUDED.content_hash
                              THEN NULL ELSE chunks.embedding END,
             content_hash=EXCLUDED.content_hash""",
        rows)

# --- embed ------------------------------------------------------------------

def _embed_batch(texts):
    body = json.dumps({"input": texts, "model": EMBED_MODEL}).encode()
    req = urllib.request.Request(EMBED_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return [d["embedding"][:EMBED_DIM] for d in data["data"]]

def _vec(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

def cmd_embed(args):
    conn = load_conn()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NULL")
    pending = cur.fetchone()[0]
    print(f"embedding: {pending} chunks pending", flush=True)
    done = 0
    while True:
        cur.execute("SELECT id,text FROM chunks WHERE embedding IS NULL LIMIT 2000")
        rows = cur.fetchall()
        if not rows:
            break
        # batches of 2 (fits the embedder's 512-token budget), filled concurrently
        pairs = [rows[i:i+2] for i in range(0, len(rows), 2)]
        def work(pair):
            try:
                embs = _embed_batch([t[:CHUNK_SIZE] for _, t in pair])
                return [(pair[i][0], embs[i]) for i in range(len(pair))]
            except Exception:
                out = []
                for cid, t in pair:                       # one-by-one fallback
                    try:
                        out.append((cid, _embed_batch([t[:CHUNK_SIZE]])[0]))
                    except Exception:
                        pass
                return out
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for res in ex.map(work, pairs):
                for cid, emb in res:
                    cur.execute("UPDATE chunks SET embedding=%s::vector WHERE id=%s", (_vec(emb), cid))
        conn.commit()
        done += len(rows)
        print(f"  embedded ~{done}/{pending}", flush=True)
    # build HNSW once vectors exist (idempotent)
    cur.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
    if cur.fetchone()[0] > 0:
        print("building HNSW index (cosine)...", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_emb_hnsw ON chunks "
                    "USING hnsw (embedding vector_cosine_ops)")
        conn.commit()
    print("embed done.", flush=True)

# --- search -----------------------------------------------------------------

def cmd_search(args):
    q = " ".join(args.query)
    scope = _scope(args)
    conn = load_conn()
    cur = conn.cursor()
    where_scope = ""
    params_scope = []
    if scope:
        where_scope = " AND source_kind = ANY(%s)"
        params_scope = [scope]
    results = {}
    # 1) keyword layer (guaranteed any-needle: FTS + trigram substring), no GPU.
    #    Rank exact-phrase (ILIKE) matches first, then ts_rank — so a specific
    #    needle outranks incidental token matches even before embeddings exist.
    cur.execute(
        f"""SELECT id,source_kind,project,title,left(text,200),created,
                   (text ILIKE %s) AS exact,
                   ts_rank(tsv, plainto_tsquery('simple',%s)) AS rank
            FROM chunks
            WHERE (tsv @@ plainto_tsquery('simple',%s) OR text ILIKE %s){where_scope}
            ORDER BY (text ILIKE %s) DESC, ts_rank(tsv, plainto_tsquery('simple',%s)) DESC
            LIMIT 60""",
        [f"%{q}%", q, q, f"%{q}%", *params_scope, f"%{q}%", q])
    for r in cur.fetchall():
        kwscore = (0.5 if r[6] else 0.3) + min(0.2, float(r[7] or 0))
        results[r[0]] = {"r": r[:6], "kw": True, "sem": 0.0, "kwscore": kwscore}
    # 2) semantic layer (if embeddings exist + embedder reachable)
    if not args.keyword:
        try:
            qe = _vec(_embed_batch([q])[0])
            cur.execute(
                f"""SELECT id,source_kind,project,title,left(text,200),created,
                           1-(embedding<=>%s::vector) AS sem FROM chunks
                    WHERE embedding IS NOT NULL{where_scope}
                    ORDER BY embedding<=>%s::vector LIMIT 40""",
                [qe, *params_scope, qe])
            for r in cur.fetchall():
                rid = r[0]
                if rid in results:
                    results[rid]["sem"] = float(r[6])
                else:
                    results[rid] = {"r": r[:6], "kw": False, "sem": float(r[6]), "kwscore": 0.0}
        except Exception as e:
            print(f"(semantic layer skipped: {e})", file=sys.stderr)
    ranked = sorted(results.values(),
                    key=lambda x: x.get("kwscore", 0.0) + x["sem"], reverse=True)[:args.limit]
    if not ranked:
        print("No results."); return
    tag = {"session": "\033[36m[session]", "doc": "\033[33m[doc]", "messenger": "\033[35m[msgr]",
           "aichat": "\033[34m[aichat]", "note": "\033[32m[note]", "email": "\033[90m[email]"}
    for i, x in enumerate(ranked, 1):
        r = x["r"]
        kw = " \033[32m[kw]\033[0m" if x["kw"] else ""
        print(f"{i}. {tag.get(r[1],'[?]')}\033[0m \033[1m{r[3]}\033[0m  "
              f"\033[2m({r[2]}, sem={x['sem']:.2f}{', '+r[5] if r[5] else ''})\033[0m{kw}")
        print(f"   \033[2m{(r[4] or '').replace(chr(10),' ')[:180]}\033[0m")

def cmd_stats(args):
    conn = load_conn(); cur = conn.cursor()
    cur.execute("SELECT source_kind,count(*),count(embedding) FROM chunks GROUP BY source_kind ORDER BY 2 DESC")
    print("mannaminne (Postgres+pgvector on Darwin) — chunk counts:")
    tot = emb = 0
    for k, c, e in cur.fetchall():
        print(f"  {k:10} {c:>8} chunks  ({e} embedded)"); tot += c; emb += e
    print(f"  {'TOTAL':10} {tot:>8} chunks  ({emb} embedded, {tot-emb} pending)")

# --- scope / cli ------------------------------------------------------------

def _scope(args):
    flags = []
    for k in ("session", "doc", "messenger", "aichat", "note", "email"):
        if getattr(args, k, False):
            flags.append(k)
    if flags:
        return flags
    invoked = os.environ.get("MANNAMINNE_INVOKED_AS") or os.path.basename(sys.argv[0])
    if invoked == "ccsearch":
        return ["session", "doc"]   # legacy ccsearch alias → CC sources only
    return None                     # mannaminne / minne → all sources

def _add_search_args(sp):
    sp.add_argument("query", nargs="*")
    sp.add_argument("-k", "--keyword", action="store_true")
    sp.add_argument("-n", "--limit", type=int, default=12)
    sp.add_argument("-s", "--session", action="store_true")
    sp.add_argument("-d", "--doc", action="store_true")
    sp.add_argument("-m", "--messenger", action="store_true")
    sp.add_argument("-a", "--aichat", action="store_true")
    sp.add_argument("--note", action="store_true")
    sp.add_argument("-e", "--email", action="store_true")

def main():
    argv = sys.argv[1:]
    cmds = {"ingest", "embed", "stats", "search"}
    if not argv or argv[0] not in cmds:
        sp = argparse.ArgumentParser(prog="mannaminne")
        _add_search_args(sp)
        a = sp.parse_args(argv)
        if not a.query:
            print("usage: mannaminne <query> | ingest [--sources ...] | embed | search [-smdan] | stats")
            return
        cmd_search(a)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "ingest":
        sp = argparse.ArgumentParser(); sp.add_argument("--sources", nargs="*")
        cmd_ingest(sp.parse_args(rest))
    elif cmd == "embed":
        cmd_embed(None)
    elif cmd == "stats":
        cmd_stats(None)
    elif cmd == "search":
        sp = argparse.ArgumentParser(); _add_search_args(sp)
        cmd_search(sp.parse_args(rest))

if __name__ == "__main__":
    main()
