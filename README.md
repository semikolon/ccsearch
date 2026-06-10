# mannaminne

Hybrid semantic + keyword search over Fredrik's **personal life-corpus**:
Claude Code sessions, project/infra docs, Facebook Messenger, AI-chat archives
(ChatGPT + Claude), and Simplenote notes. *"i mannaminne"* = in living memory.

Built in Rust. Uses the local Darwin embedding server (Qwen3-Embedding-4B on the
GTX 1650) for semantic search; plain substring matching for keyword search. Was
`ccsearch` (CC-sessions-only) through 2026-06; renamed + extended 2026-06-10.

## Names / scope

| Invoke as | Scope |
|---|---|
| `mannaminne` / `minne` | **all sources** |
| `ccsearch` (alias) | **CC sources only** (sessions + docs) — preserves the original tool's behavior |

Scope is resolved from the invocation basename (argv0). Explicit source flags
override it. `minne` and `ccsearch` are symlinks to the `mannaminne` binary.

## Usage

```bash
mannaminne index                    # build/update the index (incremental, cached by id)
mannaminne index --force            # full re-embed
mannaminne "kausal inferens Kaus"   # search all sources
mannaminne stats                    # per-source counts

# scope flags (override the invocation default)
mannaminne search -s "<q>"       # CC sessions only
mannaminne search -d "<q>"       # docs only
mannaminne search -m "<q>"       # Messenger only
mannaminne search -a "<q>"       # AI-chat only (ChatGPT + Claude)
mannaminne search --notes "<q>"  # Simplenote only
mannaminne search -k "<q>"       # keyword-only (no embedding call)

ccsearch "<q>"                   # = CC sources only (sessions + docs)
```

## Sources indexed

| Kind | Source | Entry granularity |
|---|---|---|
| `session` | `~/.claude/projects/*/sessions-index.json` (+ raw JSONL) | one per CC session (summary + first prompt) |
| `doc` | `~/Projects/*/docs/**/*.md`, `~/dotfiles/docs/**/*.md` | one per `.md` (first 800 chars) |
| `messenger` | `~/Projects/messenger-archive/.../inbox/<id>/message_*.json` | one per thread (title + de-mojibaked content sample) |
| `aichat` | `~/Projects/ai-chat-archives/{chatgpt_*,claude_*}/conversations/*.json` | one per conversation |
| `note` | `~/Documents/Simplenote Support Notes/*.txt` | one per note |

Each entry stores ~1600 chars of text (for substring) and embeds the first 800
(the embedder's ~512-token window). FB Messenger JSON is double-encoded UTF-8;
`fix_mojibake()` reverses it (latin1→utf8 reinterpret).

## Index storage

Binary JSON at `~/.local/share/ccsearch/index.bin` (dir name kept from the
`ccsearch` era so the existing CC embeddings stay cached across the rename;
internal detail only). Incremental: `index` re-embeds only new ids; `--force`
re-embeds all.

## Build / install

```bash
cd ~/Projects/mannaminne && cargo build --release
cp target/release/mannaminne ~/.local/bin/
ln -sf mannaminne ~/.local/bin/minne
ln -sf mannaminne ~/.local/bin/ccsearch
```

Fleet install is automated by `dotfiles/scripts/darwin/install-extra-binaries.sh`
(builds from this dir, creates the `minne` + `ccsearch` aliases). The GitHub
remote is still named `ccsearch` (`semikolon/ccsearch`); the Cargo package +
binary are `mannaminne`.

## Dependencies

`clap`, `serde`/`serde_json`, `ureq` (sync HTTP), `glob`. Runtime: the Darwin
embedding server reachable at `192.168.4.1:8080` (configurable via
`CCSEARCH_EMBEDDING_URL`).

## Design

Architecture + the three archive sources + the fresh-FB-download question:
`~/dotfiles/docs/personal_archives_semantic_search_2026_06_10.md`.
Messenger archive itself: `~/Projects/messenger-archive/README.md`.
