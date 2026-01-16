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

## License

MIT
