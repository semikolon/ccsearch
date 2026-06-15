# mannaminne

Hybrid semantic + keyword search over Fredrik's personal life-corpus: Claude
Code sessions, local docs, Facebook Messenger, AI-chat archives, Simplenote,
email, Things3, Fyr tasks, screenshots, and photo labels/OCR.

`mannaminne` means "in living memory". The old Rust binary remains in the repo
for history, but the active implementation is:

- Python CLI: `py/mannaminne.py`
- Store: dedicated Postgres + pgvector on Darwin
- Keyword: Postgres FTS + trigram substring
- Semantic: Qwen3-Embedding-4B vectors, HNSW index

## Usage

```bash
mannaminne "kausal inferens Kaus"
mannaminne search -d "Z4 embedder backlog HNSW"
mannaminne search --keyword "exact phrase"
mannaminne stats
mannaminne ingest --sources doc
mannaminne embed
mannaminne eval --show-top
```

Aliases:

| Invoke as | Scope |
|---|---|
| `mannaminne` / `minne` | all sources |
| `ccsearch` | legacy CC scope: sessions + docs |

Source flags override the invocation default:

```bash
-s / --session
-d / --doc
-m / --messenger
-a / --aichat
--note
-e / --email
-t / --things3
-f / --fyr
-p / --photos
```

## Embedding Endpoint Selection

If `MANNAMINNE_EMBED_URL` is set, it is used exactly.

Otherwise runtime default order is:

1. Z4 tunnel: `MANNAMINNE_Z4_EMBED_URL`, default `http://127.0.0.1:8081/v1/embeddings`
2. Darwin fallback: `MANNAMINNE_DARWIN_EMBED_URL`, default `http://192.168.4.1:8080/v1/embeddings`

Darwin-safe defaults are intentionally conservative:

```bash
MANNAMINNE_EMBED_BATCH_SIZE=4
MANNAMINNE_EMBED_WORKERS=2
MANNAMINNE_EMBED_SELECT_LIMIT=500
MANNAMINNE_EMBED_TIMEOUT=45
MANNAMINNE_EMBED_PROBE_TIMEOUT=5
```

For a Z4 batch run, override these upward after the Z4 server/tunnel is live.

## Indexing

`ingest` discovers source content, chunks it, and upserts rows. If text changes,
the chunk's embedding is reset to `NULL` so `embed` can refill it.

Docs use heading-aware markdown chunks. The global
`~/.claude/CLAUDE.md` file is indexed as a special doc source because it contains
high-value operating context outside the usual docs roots.

`embed` fills `NULL` embeddings. Failed batches are split recursively, so one bad
batch does not poison the whole pending set.

## Search Ranking

Search is hybrid:

1. Keyword candidates from layered Postgres FTS:
   exact phrase for short queries, strict all-term FTS, then small per-term
   recall probes for useful non-filler terms.
2. Semantic candidates from pgvector.
3. Reciprocal-rank fusion combines the layers, with a small exact-match boost.
4. Final results are de-duplicated by source object so one long conversation or
   doc cannot fill the whole first page with adjacent chunks.

The query embedding uses a Qwen3 instruction prefix; stored document chunks stay
plain text.

## Eval

Golden queries live in `eval/golden_queries.json`.

```bash
mannaminne eval
mannaminne eval --keyword
mannaminne eval -k 20 --json
```

The eval command reports recall@k, MRR, average latency, and p95 latency against
the live DB. It is meant as a small regression harness before changing chunking,
fusion, model, or storage settings.

## Tests

Tests use Python's standard `unittest`, no pytest dependency:

```bash
cd ~/Projects/mannaminne/py
.venv/bin/python -m unittest discover -s tests -v
```

Current tests cover:

- char and markdown chunking
- NUL stripping and chunk hashes
- Z4-first endpoint preference with Darwin fallback
- recursive embedding batch split
- rank fusion
- query term pruning
- source de-duplication
- eval expectation matching

## Setup

Fresh provisioning:

```bash
cd ~/Projects/mannaminne
./setup.sh
```

This creates the venv, wrappers in `~/.local/bin`, and the Darwin pgvector
Postgres container/schema. DB credentials live only in
`~/.config/mannaminne/db.env`.

Design and operational notes:

- `~/dotfiles/docs/personal_archives_semantic_search_2026_06_10.md`
- `~/Projects/mannaminne/z4/README.md`
- `~/dotfiles/docs/local_codebase_semantic_search_research_plan_2026_06_14.md`
