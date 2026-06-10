#!/usr/bin/env bash
# mannaminne v2 fresh-provision — idempotent.
# Brings up: the Python venv, the ~/.local/bin wrappers (mannaminne/minne/ccsearch),
# and the dedicated pgvector Postgres on Darwin (+ schema). Re-runnable safely.
#
# The DB password is GENERATED here and stored only in ~/.config/mannaminne/db.env
# (0600, gitignored) — NOT in any tracked secret store, because the corpus is fully
# reconstructible by re-ingest, so the password is regenerable. If db.env already
# exists its password is reused (so the existing volume keeps working).
set -euo pipefail

REPO="$HOME/Projects/mannaminne"
PYDIR="$REPO/py"
BIN="$HOME/.local/bin"
CONF="$HOME/.config/mannaminne"
ENVF="$CONF/db.env"
PGPORT=5440
PGHOST_BIND=192.168.4.1   # Darwin LAN

echo "[mannaminne setup] venv + psycopg"
python3 -m venv "$PYDIR/.venv" 2>/dev/null || true
"$PYDIR/.venv/bin/pip" -q install 'psycopg[binary]'

echo "[mannaminne setup] wrappers → $BIN"
mkdir -p "$BIN"
PY="$PYDIR/.venv/bin/python"; SCRIPT="$PYDIR/mannaminne.py"
for name in mannaminne minne ccsearch; do
  pre=""; [ "$name" = ccsearch ] && pre="MANNAMINNE_INVOKED_AS=ccsearch "
  printf '#!/bin/sh\n%sexec "%s" "%s" "$@"\n' "$pre" "$PY" "$SCRIPT" > "$BIN/$name"
  chmod +x "$BIN/$name"
done

echo "[mannaminne setup] Darwin pgvector Postgres"
mkdir -p "$CONF"
if [ -f "$ENVF" ]; then
  PW=$(grep '^MANNAMINNE_PG_PASSWORD=' "$ENVF" | cut -d= -f2-)
else
  PW=$(ssh darwin 'openssl rand -hex 20')
fi
if ! ssh darwin 'docker ps -a --format "{{.Names}}" | grep -qx mannaminne-postgres'; then
  ssh darwin "docker run -d --name mannaminne-postgres --restart unless-stopped \
    -e POSTGRES_PASSWORD='$PW' -e POSTGRES_DB=mannaminne \
    -p $PGHOST_BIND:$PGPORT:5432 -v mannaminne-pgdata:/var/lib/postgresql/data \
    pgvector/pgvector:pg16"
  sleep 5
fi
printf 'MANNAMINNE_PG_HOST=darwin.home\nMANNAMINNE_PG_PORT=%s\nMANNAMINNE_PG_DB=mannaminne\nMANNAMINNE_PG_USER=postgres\nMANNAMINNE_PG_PASSWORD=%s\n' "$PGPORT" "$PW" > "$ENVF"
chmod 600 "$ENVF"

echo "[mannaminne setup] schema (idempotent)"
ssh darwin 'until docker exec mannaminne-postgres pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
docker exec -i mannaminne-postgres psql -U postgres -d mannaminne' <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, source_kind TEXT NOT NULL, source_id TEXT NOT NULL,
  chunk_idx INT NOT NULL, project TEXT, title TEXT, text TEXT NOT NULL,
  created TEXT, content_hash TEXT NOT NULL, embedding vector(1024),
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(title,'') || ' ' || text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin  ON chunks USING gin(tsv);
CREATE INDEX IF NOT EXISTS chunks_trgm_gin ON chunks USING gin(text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS chunks_kind     ON chunks(source_kind);
CREATE INDEX IF NOT EXISTS chunks_src      ON chunks(source_id);
SQL

echo "[mannaminne setup] done. Next: 'mannaminne ingest' then 'mannaminne embed'."
