use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const EMBEDDING_DIM: usize = 1024;
const DEFAULT_EMBEDDING_URL: &str = "http://192.168.4.1:8080/v1/embeddings";
const BATCH_SIZE: usize = 8;

#[derive(Parser)]
#[command(name = "mannaminne", about = "Hybrid semantic + keyword search over the personal life-corpus (CC sessions, docs, Messenger, AI-chat, Simplenote). Invoke as `ccsearch` to scope to CC sources only.")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query (shorthand for `search <query>`)
    #[arg(trailing_var_arg = true)]
    query: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Build or update the search index
    Index {
        /// Force full reindex (ignore cache)
        #[arg(long)]
        force: bool,
    },
    /// Search the indexed corpus (scope defaults to all; `ccsearch` → CC-only)
    Search {
        /// The search query
        query: Vec<String>,
        /// CC sessions only
        #[arg(short = 's', long)]
        sessions: bool,
        /// Project/infra docs only
        #[arg(short = 'd', long)]
        docs: bool,
        /// Facebook Messenger threads only
        #[arg(short = 'm', long)]
        messenger: bool,
        /// AI-chat archives only (ChatGPT + Claude)
        #[arg(short = 'a', long)]
        aichat: bool,
        /// Simplenote notes only
        #[arg(long)]
        notes: bool,
        /// Keyword-only (no semantic search)
        #[arg(short = 'k', long)]
        keyword: bool,
        /// Number of results
        #[arg(short = 'n', long, default_value = "10")]
        limit: usize,
    },
    /// Show index statistics
    Stats,
}

// --- Data types ---

#[derive(Serialize, Deserialize, Clone)]
struct IndexEntry {
    id: String,
    kind: EntryKind,
    project: String,
    title: String,
    text: String, // searchable text (summary + first prompt for sessions, content for docs)
    path: String,
    created: String,
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone, PartialEq)]
enum EntryKind {
    Session,
    Doc,
    Messenger,
    AiChat,
    Note,
}

impl EntryKind {
    /// Sources that the legacy `ccsearch` invocation searches (CC-relevant only).
    fn cc_scope() -> Vec<EntryKind> {
        vec![EntryKind::Session, EntryKind::Doc]
    }
}

/// Facebook "Download Your Information" exports store UTF-8 that has been
/// double-encoded (each original byte became a latin1 char). Reverse it:
/// reinterpret the chars as latin1 bytes and decode as UTF-8. If any char is
/// outside the latin1 range the string wasn't mojibaked — return it unchanged.
fn fix_mojibake(s: &str) -> String {
    if s.chars().all(|c| (c as u32) < 256) {
        let bytes: Vec<u8> = s.chars().map(|c| c as u8).collect();
        if let Ok(fixed) = String::from_utf8(bytes) {
            return fixed;
        }
    }
    s.to_string()
}

#[derive(Serialize, Deserialize)]
struct SearchIndex {
    version: u32,
    entries: Vec<IndexEntry>,
}

// CC sessions-index.json format
#[derive(Deserialize)]
struct SessionsIndex {
    entries: Vec<SessionEntry>,
}

// Sometimes sessions-index.json is just {version: N, entries: [...]}
// Sometimes it's a flat dict. Handle both.
#[derive(Deserialize)]
#[serde(untagged)]
enum SessionsIndexFormat {
    Versioned(SessionsIndex),
    Flat(HashMap<String, serde_json::Value>),
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionEntry {
    session_id: String,
    #[serde(default)]
    first_prompt: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    message_count: u32,
    #[serde(default)]
    created: String,
    #[serde(default)]
    project_path: String,
    #[serde(default)]
    full_path: String,
}

// OpenAI embeddings API response
#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
}

struct SearchResult {
    entry: IndexEntry,
    score: f64,
    keyword_match: bool,
}

// --- Index path ---

fn index_path() -> PathBuf {
    let dir = dirs_path().join("ccsearch");
    std::fs::create_dir_all(&dir).ok();
    dir.join("index.bin")
}

fn dirs_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".local").join("share")
}

// --- Embedding ---

fn embedding_url() -> String {
    std::env::var("CCSEARCH_EMBEDDING_URL").unwrap_or_else(|_| DEFAULT_EMBEDDING_URL.to_string())
}

fn embed_batch(texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
    let url = embedding_url();
    let body = serde_json::json!({
        "input": texts,
        "model": "qwen3-embedding-4b"
    });

    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("Embedding API error: {e}"))?;

    let result: EmbeddingResponse = resp
        .into_json()
        .map_err(|e| format!("Failed to parse embedding response: {e}"))?;

    Ok(result
        .data
        .into_iter()
        .map(|d| {
            d.embedding
                .into_iter()
                .take(EMBEDDING_DIM)
                .map(|v| v as f32)
                .collect()
        })
        .collect())
}

fn embed_single(text: &str) -> Result<Vec<f32>, String> {
    let results = embed_batch(&[text])?;
    results
        .into_iter()
        .next()
        .ok_or_else(|| "Empty embedding response".to_string())
}

// --- Vector math ---

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

// --- Session discovery ---

fn discover_sessions() -> Vec<SessionEntry> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let projects_dir = PathBuf::from(&home).join(".claude").join("projects");

    let mut sessions = Vec::new();

    let pattern = projects_dir.join("*").join("sessions-index.json");
    let pattern_str = pattern.to_string_lossy().to_string();

    for path in glob::glob(&pattern_str).into_iter().flatten().flatten() {
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let project_dir = path
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // Extract project name from directory name (e.g., "-Users-fred-Projects-brf-auto" -> "brf-auto")
        let project_name = project_dir
            .rsplit('-')
            .next()
            .unwrap_or(&project_dir)
            .to_string();

        let parsed: Result<SessionsIndexFormat, _> = serde_json::from_str(&content);
        let entries = match parsed {
            Ok(SessionsIndexFormat::Versioned(idx)) => idx.entries,
            _ => continue,
        };

        for mut entry in entries {
            if entry.project_path.is_empty() {
                entry.project_path = project_name.clone();
            }
            sessions.push(entry);
        }
    }

    // Also scan for raw JSONL files in project dirs that have NO sessions-index.json
    let all_project_dirs: Vec<_> = glob::glob(&projects_dir.join("*").to_string_lossy())
        .into_iter()
        .flatten()
        .flatten()
        .filter(|p| p.is_dir())
        .collect();

    let indexed_dirs: std::collections::HashSet<String> = sessions
        .iter()
        .map(|s| {
            PathBuf::from(&s.full_path)
                .parent()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default()
        })
        .collect();

    for dir in &all_project_dirs {
        let dir_name = dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // Skip if we already have sessions from this dir via sessions-index.json
        if indexed_dirs.contains(&dir_name) {
            continue;
        }

        // Skip if sessions-index.json exists (already processed above, even if empty)
        if dir.join("sessions-index.json").exists() {
            continue;
        }

        // Scan for raw JSONL files (UUID-named, not subagent files)
        let jsonl_pattern = dir.join("*.jsonl");
        for jsonl_path in glob::glob(&jsonl_pattern.to_string_lossy())
            .into_iter()
            .flatten()
            .flatten()
        {
            // Skip subagent dirs
            if jsonl_path
                .to_string_lossy()
                .contains("subagents")
            {
                continue;
            }

            let session_id = jsonl_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            // Extract first user message and timestamp from JSONL
            let mut first_prompt = String::new();
            let mut created = String::new();

            if let Ok(content) = std::fs::read_to_string(&jsonl_path) {
                for line in content.lines().take(20) {
                    if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                        if obj.get("type").and_then(|v| v.as_str()) == Some("user") {
                            if let Some(msg) = obj.get("message") {
                                if let Some(content_arr) = msg.get("content").and_then(|c| c.as_array()) {
                                    for item in content_arr {
                                        if item.get("type").and_then(|v| v.as_str()) == Some("text") {
                                            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                                                first_prompt = text.chars().take(200).collect();
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some(ts) = obj.get("timestamp").and_then(|v| v.as_str()) {
                                created = ts.to_string();
                            }
                            break;
                        }
                    }
                }
            }

            if first_prompt.is_empty() {
                continue;
            }

            // Extract project name from dir name
            let project_name = dir_name
                .rsplit('-')
                .next()
                .unwrap_or(&dir_name)
                .to_string();

            sessions.push(SessionEntry {
                session_id,
                first_prompt: first_prompt.clone(),
                summary: String::new(), // No summary available for raw JSONL
                message_count: 0,
                created,
                project_path: project_name,
                full_path: jsonl_path.to_string_lossy().to_string(),
            });
        }
    }

    sessions
}

// --- Doc discovery ---

fn discover_docs() -> Vec<(String, String, PathBuf)> {
    // (project_name, relative_path, full_path)
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let mut docs = Vec::new();

    // Scan ~/Projects/*/docs/**/*.md and ~/dotfiles/docs/**/*.md
    let scan_dirs = vec![
        (PathBuf::from(&home).join("Projects"), "*/docs/**/*.md"),
        (PathBuf::from(&home).join("dotfiles"), "docs/**/*.md"),
    ];

    for (base, pattern) in &scan_dirs {
        let full_pattern = base.join(pattern);
        let full_pattern_str = full_pattern.to_string_lossy().to_string();

        for path in glob::glob(&full_pattern_str).into_iter().flatten().flatten() {
            // Skip very large files (>100KB) and hidden dirs
            if let Ok(meta) = path.metadata() {
                if meta.len() > 100_000 {
                    continue;
                }
            }
            if path.to_string_lossy().contains("/.") {
                continue;
            }
            // Skip archive/vendor dirs
            let path_str = path.to_string_lossy();
            if path_str.contains("/archive/")
                || path_str.contains("/vendor/")
                || path_str.contains("/node_modules/")
            {
                continue;
            }

            let project = if base.ends_with("dotfiles") {
                "dotfiles".to_string()
            } else {
                path.strip_prefix(base)
                    .ok()
                    .and_then(|p| p.components().next())
                    .map(|c| c.as_os_str().to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            };

            let rel_path = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            docs.push((project, rel_path, path));
        }
    }

    docs
}

// --- Messenger / AI-chat / Simplenote discovery ---

fn home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
}

fn cap_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Facebook Messenger archive: one entry per thread.
/// `~/Projects/messenger-archive/your_activity_across_facebook/messages/inbox/<id>/message_*.json`
fn discover_messenger() -> Vec<IndexEntry> {
    let base = PathBuf::from(home())
        .join("Projects/messenger-archive/your_activity_across_facebook/messages/inbox");
    let mut out = Vec::new();
    let dir_glob = base.join("*");
    for dir in glob::glob(&dir_glob.to_string_lossy())
        .into_iter()
        .flatten()
        .flatten()
        .filter(|p| p.is_dir())
    {
        let thread_id = dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let mut title = String::new();
        let mut body = String::new();
        let mut newest: i64 = 0;
        // message_1.json, message_2.json, ...
        let msg_glob = dir.join("message_*.json");
        let mut files: Vec<PathBuf> = glob::glob(&msg_glob.to_string_lossy())
            .into_iter()
            .flatten()
            .flatten()
            .collect();
        files.sort();
        for f in &files {
            let content = match std::fs::read_to_string(f) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let v: serde_json::Value = match serde_json::from_str(&content) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if title.is_empty() {
                if let Some(t) = v.get("title").and_then(|x| x.as_str()) {
                    title = fix_mojibake(t);
                }
                if title.is_empty() {
                    let names: Vec<String> = v
                        .get("participants")
                        .and_then(|p| p.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|p| p.get("name").and_then(|n| n.as_str()))
                                .map(fix_mojibake)
                                .collect()
                        })
                        .unwrap_or_default();
                    title = names.join(", ");
                }
            }
            if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
                for m in msgs {
                    if let Some(ts) = m.get("timestamp_ms").and_then(|t| t.as_i64()) {
                        if ts > newest {
                            newest = ts;
                        }
                    }
                    if let Some(c) = m.get("content").and_then(|c| c.as_str()) {
                        if body.chars().count() < 1600 {
                            let who = m
                                .get("sender_name")
                                .and_then(|s| s.as_str())
                                .map(fix_mojibake)
                                .unwrap_or_default();
                            body.push_str(&who);
                            body.push_str(": ");
                            body.push_str(&fix_mojibake(c));
                            body.push('\n');
                        }
                    }
                }
            }
        }
        if title.is_empty() && body.trim().is_empty() {
            continue;
        }
        let created = if newest > 0 {
            // ms epoch → YYYY-MM-DD (UTC), cheap manual conversion
            let secs = newest / 1000;
            chrono_like_date(secs)
        } else {
            String::new()
        };
        let text = cap_chars(&format!("{title}\n{body}"), 1600);
        out.push(IndexEntry {
            id: format!("msg:{thread_id}"),
            kind: EntryKind::Messenger,
            project: "messenger".to_string(),
            title: if title.is_empty() {
                cap_chars(&body, 60)
            } else {
                title
            },
            text,
            path: dir.to_string_lossy().to_string(),
            created,
            embedding: vec![],
        });
    }
    out
}

/// Minimal epoch-seconds → "YYYY-MM-DD" (UTC). Avoids a chrono dependency.
fn chrono_like_date(secs: i64) -> String {
    let days = secs.div_euclid(86400);
    let mut y = 1970i64;
    let mut d = days;
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let dy = if leap { 366 } else { 365 };
        if d >= dy {
            d -= dy;
            y += 1;
        } else {
            break;
        }
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let mlen = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0usize;
    while m < 12 && d >= mlen[m] {
        d -= mlen[m];
        m += 1;
    }
    format!("{:04}-{:02}-{:02}", y, m + 1, d + 1)
}

/// AI-chat archives: ChatGPT (mapping format) + Claude (chat_messages format).
fn discover_aichat() -> Vec<IndexEntry> {
    let mut out = Vec::new();
    let base = PathBuf::from(home()).join("Projects/ai-chat-archives");

    // ChatGPT: */conversations/*.json with a `mapping` of message nodes.
    let cg_glob = base.join("chatgpt_*/conversations/*.json");
    for f in glob::glob(&cg_glob.to_string_lossy())
        .into_iter()
        .flatten()
        .flatten()
    {
        let content = match std::fs::read_to_string(&f) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let v: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let title = v.get("title").and_then(|t| t.as_str()).unwrap_or("").to_string();
        let mut msgs: Vec<(f64, String)> = Vec::new();
        if let Some(map) = v.get("mapping").and_then(|m| m.as_object()) {
            for node in map.values() {
                if let Some(msg) = node.get("message") {
                    let ct = msg.get("create_time").and_then(|c| c.as_f64()).unwrap_or(0.0);
                    if let Some(parts) = msg
                        .get("content")
                        .and_then(|c| c.get("parts"))
                        .and_then(|p| p.as_array())
                    {
                        for part in parts {
                            if let Some(s) = part.as_str() {
                                if !s.trim().is_empty() {
                                    msgs.push((ct, s.to_string()));
                                }
                            }
                        }
                    }
                }
            }
        }
        msgs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let body: String = msgs.iter().map(|(_, s)| s.as_str()).collect::<Vec<_>>().join("\n");
        let id_part = v
            .get("conversation_id")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
        if title.is_empty() && body.trim().is_empty() {
            continue;
        }
        let text = cap_chars(&format!("{title}\n{body}"), 1600);
        out.push(IndexEntry {
            id: format!("aichat:cg:{id_part}"),
            kind: EntryKind::AiChat,
            project: "chatgpt".to_string(),
            title: if title.is_empty() { cap_chars(&body, 60) } else { title },
            text,
            path: f.to_string_lossy().to_string(),
            created: String::new(),
            embedding: vec![],
        });
    }

    // Claude: */conversations/*.json with `name` + `chat_messages[].text`.
    let cl_glob = base.join("claude_*/conversations/*.json");
    for f in glob::glob(&cl_glob.to_string_lossy())
        .into_iter()
        .flatten()
        .flatten()
    {
        let content = match std::fs::read_to_string(&f) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let v: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
        let created = v
            .get("created_at")
            .and_then(|c| c.as_str())
            .map(|s| cap_chars(s, 10))
            .unwrap_or_default();
        let mut body = String::new();
        if let Some(msgs) = v.get("chat_messages").and_then(|m| m.as_array()) {
            for m in msgs {
                if let Some(t) = m.get("text").and_then(|t| t.as_str()) {
                    if !t.trim().is_empty() {
                        let who = m.get("sender").and_then(|s| s.as_str()).unwrap_or("");
                        body.push_str(who);
                        body.push_str(": ");
                        body.push_str(t);
                        body.push('\n');
                    }
                }
            }
        }
        let uuid = v
            .get("uuid")
            .and_then(|u| u.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
        if name.is_empty() && body.trim().is_empty() {
            continue;
        }
        let text = cap_chars(&format!("{name}\n{body}"), 1600);
        out.push(IndexEntry {
            id: format!("aichat:cl:{uuid}"),
            kind: EntryKind::AiChat,
            project: "claude".to_string(),
            title: if name.is_empty() { cap_chars(&body, 60) } else { name },
            text,
            path: f.to_string_lossy().to_string(),
            created,
            embedding: vec![],
        });
    }

    out
}

/// Simplenote archive: one entry per .txt note.
fn discover_notes() -> Vec<IndexEntry> {
    let mut out = Vec::new();
    let dir = PathBuf::from(home()).join("Documents/Simplenote Support Notes");
    let pat = dir.join("*.txt");
    for f in glob::glob(&pat.to_string_lossy())
        .into_iter()
        .flatten()
        .flatten()
    {
        let content = match std::fs::read_to_string(&f) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if content.trim().is_empty() {
            continue;
        }
        let name = f
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        out.push(IndexEntry {
            id: format!("note:{name}"),
            kind: EntryKind::Note,
            project: "simplenote".to_string(),
            title: name,
            text: cap_chars(&content, 1600),
            path: f.to_string_lossy().to_string(),
            created: String::new(),
            embedding: vec![],
        });
    }
    out
}

// --- Indexing ---

/// Cache-or-queue an already-built batch of entries (embedding empty) into the
/// index build. Mirrors the sessions/docs inline logic. Embeds the first 800
/// chars of `text` (the embedder's ~512-token window); the full `text` is kept
/// for substring/keyword matching.
fn add_entries(
    entries: &mut Vec<IndexEntry>,
    texts_to_embed: &mut Vec<(usize, String)>,
    existing: &HashMap<String, Vec<f32>>,
    new_entries: Vec<IndexEntry>,
) {
    for mut e in new_entries {
        if let Some(cached) = existing.get(&e.id) {
            e.embedding = cached.clone();
        } else {
            texts_to_embed.push((entries.len(), cap_chars(&e.text, 800)));
        }
        entries.push(e);
    }
}

fn build_index(force: bool) -> Result<SearchIndex, String> {
    let idx_path = index_path();

    // Load existing index for incremental updates
    let existing: HashMap<String, Vec<f32>> = if !force {
        load_index()
            .map(|idx| {
                idx.entries
                    .into_iter()
                    .map(|e| (e.id.clone(), e.embedding))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        HashMap::new()
    };

    let mut entries = Vec::new();
    let mut texts_to_embed: Vec<(usize, String)> = Vec::new(); // (entry_index, text)

    // --- Sessions ---
    let sessions = discover_sessions();
    eprint!("Sessions: {} found", sessions.len());

    for session in &sessions {
        let id = format!("session:{}", session.session_id);
        let text = if !session.summary.is_empty() {
            format!("{}\n{}", session.summary, session.first_prompt)
        } else {
            session.first_prompt.clone()
        };

        if text.trim().is_empty() {
            continue;
        }

        let title = if !session.summary.is_empty() {
            session.summary.clone()
        } else {
            session.first_prompt.chars().take(80).collect()
        };

        let embedding = if let Some(cached) = existing.get(&id) {
            cached.clone()
        } else {
            texts_to_embed.push((entries.len(), text.clone()));
            vec![] // placeholder, filled after batch embedding
        };

        entries.push(IndexEntry {
            id,
            kind: EntryKind::Session,
            project: session.project_path.clone(),
            title,
            text,
            path: session.full_path.clone(),
            created: session.created.clone(),
            embedding,
        });
    }

    // --- Docs ---
    let docs = discover_docs();
    eprint!(", Docs: {} found", docs.len());

    for (project, rel_path, full_path) in &docs {
        let id = format!("doc:{}", rel_path);
        let content = match std::fs::read_to_string(full_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Use first 800 chars for embedding (stays within 512 token ctx)
        let title = full_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let text = content.chars().take(800).collect::<String>();

        let embedding = if let Some(cached) = existing.get(&id) {
            cached.clone()
        } else {
            texts_to_embed.push((entries.len(), text.clone()));
            vec![]
        };

        entries.push(IndexEntry {
            id,
            kind: EntryKind::Doc,
            project: project.clone(),
            title,
            text,
            path: full_path.to_string_lossy().to_string(),
            created: String::new(),
            embedding,
        });
    }

    // --- Messenger ---
    let messenger = discover_messenger();
    eprint!(", Messenger: {} threads", messenger.len());
    add_entries(&mut entries, &mut texts_to_embed, &existing, messenger);

    // --- AI-chat (ChatGPT + Claude) ---
    let aichat = discover_aichat();
    eprint!(", AI-chat: {} conversations", aichat.len());
    add_entries(&mut entries, &mut texts_to_embed, &existing, aichat);

    // --- Simplenote ---
    let notes = discover_notes();
    eprint!(", Notes: {} found", notes.len());
    add_entries(&mut entries, &mut texts_to_embed, &existing, notes);

    // --- Batch embed new entries ---
    let new_count = texts_to_embed.len();
    if new_count > 0 {
        eprint!(", Embedding: {} new items", new_count);

        for chunk in texts_to_embed.chunks(BATCH_SIZE) {
            let texts: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
            match embed_batch(&texts) {
                Ok(embeddings) => {
                    for ((entry_idx, _), emb) in chunk.iter().zip(embeddings.into_iter()) {
                        entries[*entry_idx].embedding = emb;
                    }
                    eprint!(".");
                }
                Err(_) => {
                    // Batch failed (likely context overflow) — try one-by-one
                    for (entry_idx, text) in chunk {
                        // Truncate more aggressively for retry
                        let short: String = text.chars().take(400).collect();
                        match embed_single(&short) {
                            Ok(emb) => entries[*entry_idx].embedding = emb,
                            Err(_) => eprint!("x"), // skip this entry
                        }
                    }
                    eprint!("r"); // retried
                }
            }
        }
    } else {
        eprint!(", all cached");
    }

    let index = SearchIndex {
        version: 1,
        entries,
    };

    // Write index
    let serialized =
        serde_json::to_vec(&index).map_err(|e| format!("Failed to serialize index: {e}"))?;
    std::fs::write(&idx_path, &serialized)
        .map_err(|e| format!("Failed to write index to {}: {e}", idx_path.display()))?;

    let size_kb = serialized.len() / 1024;
    eprintln!(" Done ({size_kb} KB)");

    Ok(index)
}

fn load_index() -> Result<SearchIndex, String> {
    let path = index_path();
    let data =
        std::fs::read(&path).map_err(|_| "No index found. Run `mannaminne index` first.".to_string())?;
    serde_json::from_slice(&data).map_err(|e| format!("Corrupt index: {e}"))
}

// --- Search ---

fn search(
    index: &SearchIndex,
    query: &str,
    scope: &Option<Vec<EntryKind>>,
    keyword_only: bool,
    limit: usize,
) -> Result<Vec<SearchResult>, String> {
    // Semantic search: embed query
    let query_embedding = if !keyword_only {
        Some(embed_single(query)?)
    } else {
        None
    };

    let query_lower = query.to_lowercase();
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

    let mut results: Vec<SearchResult> = Vec::new();

    for entry in &index.entries {
        // Filter by scope (None = all sources)
        if let Some(allowed) = scope {
            if !allowed.contains(&entry.kind) {
                continue;
            }
        }

        // Keyword matching: all query terms must appear in text or title
        let text_lower = entry.text.to_lowercase();
        let title_lower = entry.title.to_lowercase();
        let keyword_match = query_terms
            .iter()
            .all(|term| text_lower.contains(term) || title_lower.contains(term));

        // Semantic similarity
        let semantic_score = if let Some(ref qe) = query_embedding {
            if entry.embedding.is_empty() {
                0.0
            } else {
                cosine_similarity(qe, &entry.embedding)
            }
        } else {
            0.0
        };

        // Combined score
        let keyword_boost = if keyword_match { 0.3 } else { 0.0 };
        let score = if keyword_only {
            if keyword_match {
                1.0
            } else {
                0.0
            }
        } else {
            semantic_score + keyword_boost
        };

        if score > 0.15 || keyword_match {
            results.push(SearchResult {
                entry: entry.clone(),
                score,
                keyword_match,
            });
        }
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    Ok(results)
}

// --- Display ---

fn display_results(results: &[SearchResult], elapsed: std::time::Duration) {
    if results.is_empty() {
        println!("No results found.");
        return;
    }

    let elapsed_ms = elapsed.as_millis();
    println!(
        "\n{} results ({elapsed_ms}ms)\n",
        results.len()
    );

    for (i, result) in results.iter().enumerate() {
        let kind_tag = match result.entry.kind {
            EntryKind::Session => "\x1b[36m[session]\x1b[0m",
            EntryKind::Doc => "\x1b[33m[doc]\x1b[0m",
            EntryKind::Messenger => "\x1b[35m[msgr]\x1b[0m",
            EntryKind::AiChat => "\x1b[34m[aichat]\x1b[0m",
            EntryKind::Note => "\x1b[32m[note]\x1b[0m",
        };

        let keyword_tag = if result.keyword_match {
            " \x1b[32m[keyword]\x1b[0m"
        } else {
            ""
        };

        let score = format!("{:.3}", result.score);

        let project = &result.entry.project;
        let title = &result.entry.title;

        println!(
            "  \x1b[1m{}.\x1b[0m {kind_tag} \x1b[1m{title}\x1b[0m{keyword_tag}",
            i + 1
        );
        println!(
            "     {project} | score: {score} | {}",
            if !result.entry.created.is_empty() {
                &result.entry.created[..result.entry.created.len().min(10)]
            } else {
                &result.entry.path
            }
        );

        // Show text preview (first 120 chars)
        let preview: String = result
            .entry
            .text
            .chars()
            .take(120)
            .map(|c| if c == '\n' { ' ' } else { c })
            .collect();
        println!("     \x1b[2m{preview}\x1b[0m");
        println!();
    }
}

fn kind_label(k: &EntryKind) -> &'static str {
    match k {
        EntryKind::Session => "session",
        EntryKind::Doc => "doc",
        EntryKind::Messenger => "messenger",
        EntryKind::AiChat => "aichat",
        EntryKind::Note => "note",
    }
}

fn display_stats(index: &SearchIndex) {
    let count_kind =
        |k: EntryKind| index.entries.iter().filter(|e| e.kind == k).count();
    let embedded = index
        .entries
        .iter()
        .filter(|e| !e.embedding.is_empty())
        .count();

    let mut projects: HashMap<&str, usize> = HashMap::new();
    for entry in &index.entries {
        *projects.entry(entry.project.as_str()).or_insert(0) += 1;
    }

    let index_size = std::fs::metadata(index_path())
        .map(|m| m.len() / 1024)
        .unwrap_or(0);

    println!("mannaminne index stats:");
    println!("  Total entries:  {}", index.entries.len());
    println!("  Sessions:       {}", count_kind(EntryKind::Session));
    println!("  Docs:           {}", count_kind(EntryKind::Doc));
    println!("  Messenger:      {}", count_kind(EntryKind::Messenger));
    println!("  AI-chat:        {}", count_kind(EntryKind::AiChat));
    println!("  Notes:          {}", count_kind(EntryKind::Note));
    println!("  Embedded:       {embedded}");
    println!("  Index size:     {index_size} KB");
    println!("  Embedding URL:  {}", embedding_url());
    println!();
    println!("  By project (top 20):");

    let mut sorted_projects: Vec<_> = projects.iter().collect();
    sorted_projects.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (project, n) in sorted_projects.iter().take(20) {
        println!("    {project}: {n}");
    }
}

// --- Main ---

/// True when the binary was invoked via the `ccsearch` alias (basename of argv0).
fn invoked_as_ccsearch() -> bool {
    std::env::args()
        .next()
        .and_then(|p| {
            Path::new(&p)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
        })
        .map(|n| n == "ccsearch")
        .unwrap_or(false)
}

/// Resolve the search scope. Explicit source flags win; otherwise the `ccsearch`
/// alias restricts to CC sources (sessions + docs) while `mannaminne`/`minne`
/// search everything (None = all sources).
fn resolve_scope(
    sessions: bool,
    docs: bool,
    messenger: bool,
    aichat: bool,
    notes: bool,
) -> Option<Vec<EntryKind>> {
    let mut v = Vec::new();
    if sessions {
        v.push(EntryKind::Session);
    }
    if docs {
        v.push(EntryKind::Doc);
    }
    if messenger {
        v.push(EntryKind::Messenger);
    }
    if aichat {
        v.push(EntryKind::AiChat);
    }
    if notes {
        v.push(EntryKind::Note);
    }
    if !v.is_empty() {
        return Some(v); // explicit flags take precedence
    }
    if invoked_as_ccsearch() {
        return Some(EntryKind::cc_scope()); // legacy ccsearch → CC sources only
    }
    None // mannaminne / minne → all sources
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Index { force }) => match build_index(force) {
            Ok(index) => display_stats(&index),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },

        Some(Commands::Search {
            query,
            sessions,
            docs,
            messenger,
            aichat,
            notes,
            keyword,
            limit,
        }) => {
            let query_str = query.join(" ");
            if query_str.is_empty() {
                eprintln!("Error: no query provided");
                std::process::exit(1);
            }
            let scope = resolve_scope(sessions, docs, messenger, aichat, notes);
            run_search(&query_str, scope, keyword, limit);
        }

        Some(Commands::Stats) => match load_index() {
            Ok(index) => display_stats(&index),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },

        None => {
            // Bare `mannaminne <query>` (or `minne` / `ccsearch`) — shorthand for search.
            let query_str = cli.query.join(" ");
            if query_str.is_empty() {
                eprintln!("mannaminne — semantic + keyword search over your life-corpus");
                eprintln!("  (aliases: `minne` = all sources; `ccsearch` = CC sources only)\n");
                eprintln!("Usage: mannaminne <query>          Search (all sources)");
                eprintln!("       mannaminne index            Build/update index");
                eprintln!("       mannaminne stats            Show index statistics");
                eprintln!("       mannaminne search -s <q>    CC sessions only");
                eprintln!("       mannaminne search -d <q>    Docs only");
                eprintln!("       mannaminne search -m <q>    Messenger only");
                eprintln!("       mannaminne search -a <q>    AI-chat only");
                eprintln!("       mannaminne search --notes <q>  Simplenote only");
                eprintln!("       mannaminne search -k <q>    Keyword-only (no embeddings)");
                std::process::exit(0);
            }
            // ccsearch alias scopes to CC sources; mannaminne/minne search all.
            let scope = resolve_scope(false, false, false, false, false);
            run_search(&query_str, scope, false, 10);
        }
    }
}

fn run_search(query: &str, scope: Option<Vec<EntryKind>>, keyword_only: bool, limit: usize) {
    let index = match load_index() {
        Ok(idx) => idx,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    let start = Instant::now();
    match search(&index, query, &scope, keyword_only, limit) {
        Ok(results) => display_results(&results, start.elapsed()),
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
