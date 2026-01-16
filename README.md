# codebase-mcp

Codebase RAG MCP for Claude Code - Tree-sitter parsing + Qdrant hybrid search.

## Philosophy

Incremental indexing with semantic understanding:

- **Merkle tree** - Only re-index changed files
- **Tree-sitter** - AST-aware chunking (functions, classes, not arbitrary splits)
- **Hybrid search** - Vector + BM25 + RRF fusion via Qdrant

## Features

| Tool | Description |
|------|-------------|
| `search` | Hybrid semantic + keyword search |
| `index_file` | Index or re-index a single file |
| `status` | Get index status and Qdrant health |

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
    |         Qdrant            |
    | (Vector + BM25 Hybrid)    |
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
| **Qdrant** | Vector database for hybrid search |
| **OpenAI** | Generates embeddings for semantic search |
| **Merkle Tree** | Tracks file changes for incremental indexing |

### Data Flow

1. **Indexing**: File -> Tree-sitter AST -> Semantic chunks -> OpenAI embeddings -> Qdrant
2. **Search**: Query -> OpenAI embedding -> Qdrant hybrid search -> Results

## Requirements

- **Qdrant** running at localhost:6333
- **OpenAI API key** for embeddings

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
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `OPENAI_API_KEY` | - | Required for embeddings |

## Starting Qdrant

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v ~/.qdrant/storage:/qdrant/storage \
  qdrant/qdrant
```

## Search Example

```python
# Search for authentication code
search(query="user authentication login", k=10, language="python")

# Filter by chunk type
search(query="repository pattern", type="class", language="csharp")
```

## Troubleshooting

### Qdrant Connection Failed

```
Qdrant connection failed: Connection refused
```

**Solution**: Ensure Qdrant is running:
```bash
docker ps | grep qdrant
# If not running:
docker start qdrant
# Or start fresh:
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

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

### Memory Issues with Large Files

Files over 2000 characters are truncated in chunks. For very large files:
- Ensure Tree-sitter is available for proper semantic chunking
- Consider splitting large files into smaller modules

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/

# Type check
mypy src/ --ignore-missing-imports --check-untyped-defs
```

## License

MIT
