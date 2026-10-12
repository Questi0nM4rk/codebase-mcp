# Reference

## Tools

| Tool | Arguments | What it does |
| --- | --- | --- |
| `search` | `query` (required), `k` (default 10), `language`, `type`, `project` | Vector search over indexed chunks. Keyword search when no embedding is available. |
| `index_file` | `path` (required), `project` | Parses one file with Tree-sitter, deletes its old chunks, stores the new ones. |
| `delete_file` | `path` (required) | Removes the chunks of one file. |
| `clear_project` | `project` (required) | Removes all chunks of one project. |
| `status` | none | Reports libSQL and parser availability, chunk count, chunks with embeddings, projects, languages. |

`type` filters by Tree-sitter node name, for example `function_definition` or `class_definition`.

Example calls:

```python
search(query="user authentication login", k=10, language="python")
search(query="repository pattern", type="class_definition", project="myproject")
```

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Enables embeddings (`text-embedding-3-small`, 1536 dimensions). Without it chunks are stored with no embedding and search uses keywords. |

There are no other settings. The database path is fixed: `~/.codeagent/codeagent.db`.
The same file is used by the author's sibling `codeagent` project, which is why the directory has that name.

## Search modes

`search` puts a `_search_mode` entry first in its result when it falls back.

| Mode | When | How it ranks |
| --- | --- | --- |
| vector | A query embedding was created and chunks have embeddings | Cosine distance (`vector_distance_cos`), score is `1 - distance` |
| `keyword_only (embeddings unavailable)` | No `OPENAI_API_KEY`, or the API call failed | `LIKE` match on the first five words of the query, any word matches, every hit scores 0.5, no ranking |
| `keyword_only (vector search failed)` | The vector query raised an error | Same as above |

Chunks indexed without a key have no embeddings.
Setting the key later does not add them. Re-index the files after you set it.

## Indexing

- A file is split into chunks at Tree-sitter function, class and method nodes.
- A file with no parser, or a parse that fails, becomes raw chunks of 2000 characters.
- Each chunk stores one `sha256` (first 16 hex characters) of its file. Nothing reads it back: there is no change detection and no tree of hashes.
- Chunk text for embeddings is cut to 8000 characters.

## Languages

Python, JavaScript, TypeScript, C, C++, Rust, Go, C#, Lua, Bash.
`status` lists the parsers that loaded. TypeScript does not load (see Known gaps in the README), so `.ts` and `.tsx` files become raw chunks.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Crash or segfault on start | `libsql-experimental` on Python 3.14 | Use Python 3.12 (`.python-version` pins it). |
| `search` result starts with `_search_mode: keyword_only` | No embeddings | Set `OPENAI_API_KEY`, then call `index_file` again. |
| `search` returns nothing | Nothing indexed for that filter | Call `status`, then `index_file`. |
