#!/usr/bin/env python3
# bench_embed.py — decompose embed-throughput: per-pass EMBED phase vs DB-WRITE phase, across
# worker/batch configs. Embeds + writes REAL pending chunks (net progress, not wasted).
# Run with the client PAUSED (they compete for the server + Postgres).
import os, time, json, urllib.request, concurrent.futures, psycopg
from pathlib import Path

EMBED_URL = os.environ.get("MANNAMINNE_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
DIM = 1024; CHUNK = 750
env = {}
for l in Path(os.path.expanduser("~/.config/mannaminne/db.env")).read_text().splitlines():
    if "=" in l and not l.startswith("#"):
        k, v = l.split("=", 1); env[k.strip()] = v.strip()
conn = psycopg.connect(host=env["MANNAMINNE_PG_HOST"], port=env["MANNAMINNE_PG_PORT"],
                       dbname=env["MANNAMINNE_PG_DB"], user=env["MANNAMINNE_PG_USER"],
                       password=env["MANNAMINNE_PG_PASSWORD"], connect_timeout=10)
cur = conn.cursor()

def embed_batch(texts):
    body = json.dumps({"input": texts, "model": "q"}).encode()
    req = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return [d["embedding"][:DIM] for d in json.loads(r.read())["data"]]

def vec(v): return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

def work(pair):
    try:
        embs = embed_batch([t[:CHUNK] for _, t in pair])
        return [(pair[i][0], embs[i]) for i in range(len(pair))]
    except Exception as e:
        return []

print(f"endpoint={EMBED_URL}")
for WORKERS, BATCH, NROWS in [(8,8,2000),(16,8,2000),(8,16,2000),(24,8,2000),(16,16,2000)]:
    cur.execute("SELECT id,text FROM chunks WHERE embedding IS NULL LIMIT %s", (NROWS,))
    rows = cur.fetchall()
    if not rows:
        print("no more pending"); break
    pairs = [rows[i:i+BATCH] for i in range(0, len(rows), BATCH)]
    t0 = time.time(); results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(work, pairs):
            results.extend(r)
    t_embed = time.time() - t0
    n = len(results)
    t1 = time.time()
    if results:
        cur.executemany("UPDATE chunks SET embedding=%s::vector WHERE id=%s",
                        [(vec(e), c) for c, e in results])
        conn.commit()
    t_write = time.time() - t1
    er = n / max(t_embed, 0.01); wr = n / max(t_write, 0.01); tot = n / max(t_embed + t_write, 0.01)
    print(f"W={WORKERS:<2} B={BATCH:<2} n={n}: EMBED {t_embed:5.1f}s={er:4.0f}/s | WRITE {t_write:5.1f}s={wr:5.0f}/s | end-to-end={tot:4.0f}/s")
