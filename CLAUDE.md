# codebase-mcp

MCP server for codebase indexing with Tree-sitter parsing and libSQL hybrid search.

## Structure

```text
src/codebase_rag/
  server.py     # Main MCP server, CodebaseRAG class, tools
  __init__.py   # Package exports
tests/
  test_server.py  # Unit tests
~/.codeagent/
  codeagent.db  # libSQL database (auto-created)
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

- `search` - Hybrid keyword + vector search over indexed code
- `index_file` - Index or re-index a single file
- `delete_file` - Remove a file from the index
- `clear_project` - Remove all indexed chunks for a project
- `status` - Get libSQL health and chunk count
