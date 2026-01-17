# codebase-mcp

MCP server for codebase indexing with Tree-sitter parsing and Qdrant hybrid search.

## Structure

```
src/codebase_rag/
  server.py     # Main MCP server, CodebaseRAG class, tools
  __init__.py   # Package exports
tests/
  test_server.py  # Unit tests
```

## Commands

```bash
# Run tests
uv run pytest tests/ -v

# Lint and format
uv run ruff check .
uv run ruff format .

# Install dependencies
uv sync --all-extras
```

## Tools Provided

- `search` - Hybrid BM25 + vector search over indexed code
- `index_file` - Index or re-index a single file
- `status` - Get Qdrant health and chunk count
