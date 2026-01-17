# Plan: Migrate to libSQL

## Changes

Replace Qdrant with libSQL vector search.

### Files to Modify

- `src/codebase_mcp/server.py`:
  - Remove Qdrant client
  - Add libSQL with `F32_BLOB(1536)` for embeddings
  - Keep Tree-sitter parsing logic unchanged

### Dependencies

```toml
[project.dependencies]
libsql-experimental = ">=0.0.50"
openai = ">=1.0.0"  # For text-embedding-3-small
```

### Database Location

`~/.codeagent/codeagent.db` (shared with other MCPs)

### Schema

```sql
CREATE TABLE IF NOT EXISTS code_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    language TEXT,
    chunk_type TEXT,
    name TEXT,
    content TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    file_hash TEXT,
    project TEXT,
    embedding F32_BLOB(1536),
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS libsql_vector_idx_chunks ON code_chunks(embedding);
CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON code_chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON code_chunks(file_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON code_chunks(project);
CREATE INDEX IF NOT EXISTS idx_chunks_language ON code_chunks(language);
```

### Embedding Model

Use OpenAI `text-embedding-3-small` (1536 dimensions) for code embeddings.
Requires `OPENAI_API_KEY` environment variable.

### Key Implementation Notes

1. **Tree-sitter parsing**: Keep existing logic for chunking code into functions, classes, etc.
2. **Incremental indexing**: Use file_hash to detect changes and only re-index modified files
3. **Hybrid search**: Combine vector similarity with keyword matching for better results

### Verification

- `uv run pytest tests/ -v`
- Manual: index a repo, search for code patterns
