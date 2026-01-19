# codebase-mcp

Codebase RAG MCP for Claude Code - Tree-sitter parsing + libSQL hybrid search.

## Philosophy

Semantic code understanding with hybrid search:

- **Tree-sitter** - AST-aware chunking (functions, classes, not arbitrary splits)
- **Hybrid search** - Vector (1536-dim) + keyword matching via libSQL
- **Shared storage** - Uses `~/.codeagent/codeagent.db` with other CodeAgent MCPs

## Features

| Tool | Description |
|------|-------------|
| `search` | Hybrid semantic + keyword search |
| `index_file` | Index or re-index a single file |
| `delete_file` | Remove a file from the index |
| `clear_project` | Clear all chunks for a project |
| `status` | Get index status and health |

## Supported Languages

- Python (`.py`)
- JavaScript (`.js`, `.jsx`)
- TypeScript (`.ts`, `.tsx`)
- C (`.c`, `.h`)
- C++ (`.cpp`, `.hpp`)
- Rust (`.rs`)
- Go (`.go`)
- C# (`.cs`)
- Lua (`.lua`)
- Bash (`.sh`, `.bash`)

## Architecture

```
                    MCP Server
                        |
            +-----------+-----------+
            |           |           |
       index_file    search      status
            |           |           |
            v           v           v
    +---------------+   |   +---------------+
    | Tree-sitter   |   |   | Health Check  |
    | AST Parser    |   |   +---------------+
    +---------------+   |
            |           |
            v           v
    +---------------------------+
    |         libSQL            |
    | (Vector + Keyword Hybrid) |
    +---------------------------+
            |
            v
    +---------------------------+
    |    OpenAI Embeddings      |
    |  (text-embedding-3-small) |
    +---------------------------+
```

### Components

| Component | Purpose |
|-----------|---------|
| **MCP Server** | Exposes tools to Claude Code via stdio |
| **Tree-sitter** | Parses source code into AST for semantic chunking |
| **libSQL** | SQLite-compatible database with vector search |
| **OpenAI** | Generates 1536-dim embeddings for semantic search |

### Data Flow

1. **Indexing**: File -> Tree-sitter AST -> Semantic chunks -> OpenAI embeddings -> libSQL
2. **Search**: Query -> OpenAI embedding -> libSQL hybrid search -> Results

## Requirements

- **OpenAI API key** for embeddings
- **Python 3.10+**

## Installation

```bash
pip install git+https://github.com/Questi0nM4rk/codebase-mcp.git
```

## Usage with Claude Code

```bash
claude mcp add codebase -- python -m codebase_rag.server
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Required for embeddings |
| `CODEAGENT_HOME` | `~/.codeagent` | Storage location |

## Storage

Database: `~/.codeagent/codeagent.db` (libSQL with vector search)

Schema:
- `code_chunks` table with 1536-dimension F32_BLOB embeddings
- Automatic vector index for similarity search
- Indexes on project, file_path, and chunk_type

## Search Example

```python
# Search for authentication code
search(query="user authentication login", k=10, language="python")

# Filter by chunk type
search(query="repository pattern", chunk_type="class", language="csharp")
```

## Troubleshooting

### OpenAI API Key Missing

```
get_embed returns zero vectors
```

**Solution**: Set the environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

### Tree-sitter Parser Not Loading

```
Failed to load parser for python: ...
```

**Solution**: Reinstall with all dependencies:
```bash
pip install --force-reinstall git+https://github.com/Questi0nM4rk/codebase-mcp.git
```

### Index Not Finding Files

```
search returns empty results
```

**Solution**: Check index status and re-index:
```python
# Check status
status()

# Re-index specific file
index_file(path="/path/to/file.py", project="myproject")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
pyright src/ tests/
```

## License

MIT
