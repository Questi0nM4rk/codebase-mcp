"""
Codebase RAG MCP Server

Provides tools for:
- Tree-sitter semantic code parsing
- libSQL hybrid search (keyword + vector)
"""

import asyncio
import hashlib
import json
import logging
import os
import struct
import threading
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# Tree-sitter imports
try:
    import tree_sitter_bash
    import tree_sitter_c
    import tree_sitter_c_sharp
    import tree_sitter_cpp
    import tree_sitter_go
    import tree_sitter_javascript
    import tree_sitter_lua
    import tree_sitter_python
    import tree_sitter_rust
    import tree_sitter_typescript
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    Language = None  # type: ignore[misc,assignment]
    Parser = None  # type: ignore[misc,assignment]
    TREE_SITTER_AVAILABLE = False

# OpenAI for embeddings
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore[misc,assignment]
    OPENAI_AVAILABLE = False

# libSQL
try:
    import libsql_experimental as libsql

    LIBSQL_AVAILABLE = True
except ImportError:
    libsql = None  # type: ignore[assignment]
    LIBSQL_AVAILABLE = False


# Configuration
DB_PATH = os.path.expanduser("~/.codeagent/codeagent.db")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Extension to language name mapping (always available)
EXTENSION_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".cs": "csharp",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
}

# Language to Tree-sitter grammar mapping
LANGUAGE_MAP = {}
if TREE_SITTER_AVAILABLE:
    LANGUAGE_MAP = {
        ".py": ("python", tree_sitter_python),
        ".js": ("javascript", tree_sitter_javascript),
        ".jsx": ("javascript", tree_sitter_javascript),
        ".ts": ("typescript", tree_sitter_typescript),
        ".tsx": ("typescript", tree_sitter_typescript),
        ".c": ("c", tree_sitter_c),
        ".h": ("c", tree_sitter_c),
        ".cpp": ("cpp", tree_sitter_cpp),
        ".hpp": ("cpp", tree_sitter_cpp),
        ".rs": ("rust", tree_sitter_rust),
        ".go": ("go", tree_sitter_go),
        ".cs": ("csharp", tree_sitter_c_sharp),
        ".lua": ("lua", tree_sitter_lua),
        ".sh": ("bash", tree_sitter_bash),
        ".bash": ("bash", tree_sitter_bash),
    }


class Chunk(BaseModel):
    """A code chunk extracted from a source file."""

    id: str
    file: str
    language: str
    type: str  # function, class, method, etc.
    name: str
    signature: Optional[str] = None
    start_line: int
    end_line: int
    content: str
    dependencies: list[str] = Field(default_factory=list)
    parent: Optional[str] = None


# Database helpers
_db_conn = None
_db_lock = threading.Lock()
_openai_client = None


def _get_db():
    """Get or create database connection.

    Thread-safety note: This uses a singleton connection pattern. MCP stdio
    servers process requests sequentially via asyncio.run(stdio_server(...)),
    which runs a single-threaded event loop. Tool calls are awaited one at a
    time, so concurrent DB access does not occur within the MCP protocol flow.

    The double-check locking protects initialization. DB operations (execute,
    commit) are synchronous and complete atomically without yielding, so no
    coroutine interleaving occurs on the connection.

    For multi-threaded or multi-process usage outside MCP stdio context,
    use per-thread connections or a connection pool instead.
    """
    global _db_conn
    if _db_conn is None:
        with _db_lock:
            # Double-check locking for thread-safe initialization
            if _db_conn is None and libsql is not None:
                if not LIBSQL_AVAILABLE:
                    raise RuntimeError("libsql-experimental not installed")
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                _db_conn = libsql.connect(DB_PATH)
                _init_schema(_db_conn)
    return _db_conn


def _init_schema(conn):
    """Initialize database schema."""
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS code_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE NOT NULL,
            file_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            language TEXT,
            chunk_type TEXT,
            name TEXT,
            content TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            file_hash TEXT,
            project TEXT,
            dependencies TEXT,
            parent_name TEXT,
            embedding F32_BLOB({EMBEDDING_DIM}),
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON code_chunks(file_path);
        CREATE INDEX IF NOT EXISTS idx_chunks_file_hash ON code_chunks(file_hash);
        CREATE INDEX IF NOT EXISTS idx_chunks_project ON code_chunks(project);
        CREATE INDEX IF NOT EXISTS idx_chunks_language ON code_chunks(language);
        CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON code_chunks(chunk_type);
        CREATE INDEX IF NOT EXISTS libsql_vector_idx_chunks ON code_chunks(embedding);
    """)
    conn.commit()


def _get_openai_client() -> Optional[OpenAI]:
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None and OPENAI_AVAILABLE:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_embedding(text: str) -> Optional[bytes]:
    """Get embedding vector for text as packed bytes."""
    client = _get_openai_client()
    if not client:
        return None

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000],  # Truncate to fit context
        )
        embedding = response.data[0].embedding
        return struct.pack(f"<{EMBEDDING_DIM}f", *embedding)
    except Exception:
        logger.warning("Failed to generate embedding", exc_info=True)
        return None


def _compute_file_hash(content: bytes) -> str:
    """Compute SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()[:16]


class CodebaseRAG:
    """Codebase indexing and search with Tree-sitter and libSQL."""

    def __init__(self):
        self.parsers: dict[str, Parser] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize Tree-sitter parsers."""
        if self._initialized:
            return

        # Initialize Tree-sitter parsers
        if TREE_SITTER_AVAILABLE:
            for ext, (lang_name, grammar_module) in LANGUAGE_MAP.items():
                if grammar_module and lang_name not in self.parsers:
                    try:
                        parser = Parser()
                        language = Language(grammar_module.language())
                        parser.language = language
                        self.parsers[lang_name] = parser
                    except Exception:
                        logger.debug(
                            "Failed to load parser for %s", lang_name, exc_info=True
                        )

        self._initialized = True

    def parse_file(self, file_path: Path, content: str) -> list[Chunk]:
        """Parse file with Tree-sitter and extract chunks."""
        ext = file_path.suffix.lower()
        if ext not in LANGUAGE_MAP:
            return [self._create_raw_chunk(file_path, content)]

        lang_name, _ = LANGUAGE_MAP[ext]
        if lang_name not in self.parsers:
            return [self._create_raw_chunk(file_path, content)]

        parser = self.parsers[lang_name]
        try:
            tree = parser.parse(content.encode())
            return self._extract_chunks(file_path, content, tree.root_node, lang_name)
        except Exception:
            logger.debug("Parse error for %s", file_path, exc_info=True)
            return [self._create_raw_chunk(file_path, content)]

    def _create_raw_chunk(
        self, file_path: Path, content: str, language: Optional[str] = None
    ) -> Chunk:
        """Create a raw chunk for unparseable files."""
        lines = content.split("\n")
        # Use provided language, or detect from extension, or fallback to "unknown"
        lang = language or EXTENSION_TO_LANG.get(file_path.suffix.lower(), "unknown")
        # Include file_path in hash to prevent collisions for identical content (e.g., empty files)
        raw_id = _compute_file_hash((str(file_path) + "\n" + content).encode())
        return Chunk(
            id=f"raw_{raw_id}",
            file=str(file_path),
            language=lang,
            type="raw",
            name=file_path.name,
            start_line=1,
            end_line=len(lines),
            content=content[:2000],
        )

    def _extract_chunks(
        self, file_path: Path, content: str, root_node: Any, language: str
    ) -> list[Chunk]:
        """Extract semantic chunks from AST."""
        chunks: list[Chunk] = []
        lines = content.split("\n")

        # Node types to extract by language
        chunk_types = {
            "python": [
                "function_definition",
                "class_definition",
                "async_function_definition",
            ],
            "javascript": [
                "function_declaration",
                "class_declaration",
                "arrow_function",
                "method_definition",
            ],
            "typescript": [
                "function_declaration",
                "class_declaration",
                "interface_declaration",
                "method_definition",
            ],
            "c": ["function_definition", "struct_specifier"],
            "cpp": ["function_definition", "class_specifier", "struct_specifier"],
            "rust": ["function_item", "impl_item", "struct_item"],
            "go": ["function_declaration", "method_declaration", "type_declaration"],
            "csharp": [
                "method_declaration",
                "class_declaration",
                "interface_declaration",
            ],
            "lua": ["function_definition", "local_function"],
            "bash": ["function_definition"],
        }

        target_types = chunk_types.get(language, [])

        def visit(node: Any, parent_name: Optional[str] = None) -> None:
            if node.type in target_types:
                # Extract name
                name = self._extract_node_name(node, language)

                # Get line numbers (1-indexed)
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Extract content
                chunk_content = "\n".join(lines[start_line - 1 : end_line])

                # Use null byte separator for unambiguous hash input
                hash_input = f"{file_path}\x00{name}"
                chunk = Chunk(
                    id=f"{language}_{_compute_file_hash(hash_input.encode())}_{start_line}",
                    file=str(file_path),
                    language=language,
                    type=node.type,
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_content[:2000],
                    parent=parent_name,
                )
                chunks.append(chunk)

                # Visit children with this as parent
                for child in node.children:
                    visit(child, name)
            else:
                for child in node.children:
                    visit(child, parent_name)

        visit(root_node)

        # If no chunks found, create raw chunk with known language
        if not chunks:
            return [self._create_raw_chunk(file_path, content, language)]

        return chunks

    def _extract_node_name(self, node: Any, language: str) -> str:
        """Extract name from AST node."""
        # Look for identifier or name child
        for child in node.children:
            if child.type in [
                "identifier",
                "name",
                "type_identifier",
                "property_identifier",
            ]:
                return (
                    child.text.decode()
                    if hasattr(child.text, "decode")
                    else str(child.text)
                )
        return "anonymous"

    async def search(
        self,
        query: str,
        k: int = 10,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        project: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search over indexed codebase."""
        await self.initialize()

        if not LIBSQL_AVAILABLE:
            return [{"error": "libsql-experimental not available"}]

        conn = _get_db()

        # Get query embedding for vector search (offload to thread to avoid blocking)
        query_embedding = await asyncio.to_thread(_get_embedding, query)

        results = []

        # Vector search if embeddings available
        if query_embedding:
            # Build WHERE clause for filters
            conditions = ["embedding IS NOT NULL"]
            params: list[Any] = []

            if language:
                conditions.append("language = ?")
                params.append(language)
            if chunk_type:
                conditions.append("chunk_type = ?")
                params.append(chunk_type)
            if project:
                conditions.append("project = ?")
                params.append(project)

            where_clause = " AND ".join(conditions)

            # Vector similarity search
            sql = f"""
                SELECT
                    chunk_id,
                    file_path,
                    language,
                    chunk_type,
                    name,
                    start_line,
                    end_line,
                    content,
                    vector_distance_cos(embedding, ?) as distance
                FROM code_chunks
                WHERE {where_clause}
                ORDER BY distance ASC
                LIMIT ?
            """
            params_with_embedding = (query_embedding, *params, k)

            try:
                cursor = conn.execute(sql, params_with_embedding)
                rows = cursor.fetchall()
                columns = [
                    "chunk_id",
                    "file_path",
                    "language",
                    "chunk_type",
                    "name",
                    "start_line",
                    "end_line",
                    "content",
                    "distance",
                ]
                for row in rows:
                    result = dict(zip(columns, row, strict=False))
                    # Convert distance to similarity score (1 - distance)
                    result["score"] = 1.0 - (result.pop("distance") or 0.0)
                    result["content"] = (result.get("content") or "")[:500]
                    results.append(result)
            except Exception:
                # Fall back to keyword search
                logger.debug(
                    "Vector search failed, falling back to keyword search",
                    exc_info=True,
                )

        # Keyword search fallback or supplement
        if not results:
            conditions = ["1=1"]
            params = []

            if language:
                conditions.append("language = ?")
                params.append(language)
            if chunk_type:
                conditions.append("chunk_type = ?")
                params.append(chunk_type)
            if project:
                conditions.append("project = ?")
                params.append(project)

            # Simple keyword matching with escaped wildcards
            def escape_like(s: str) -> str:
                return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

            keywords = query.lower().split()
            if keywords:
                keyword_conditions = []
                for kw in keywords[:5]:  # Limit to 5 keywords
                    keyword_conditions.append(
                        "(LOWER(content) LIKE ? ESCAPE '\\' OR LOWER(name) LIKE ? ESCAPE '\\')"
                    )
                    escaped_kw = escape_like(kw)
                    params.extend([f"%{escaped_kw}%", f"%{escaped_kw}%"])
                if keyword_conditions:
                    conditions.append(f"({' OR '.join(keyword_conditions)})")

            where_clause = " AND ".join(conditions)

            sql = f"""
                SELECT chunk_id, file_path, language, chunk_type, name,
                       start_line, end_line, content
                FROM code_chunks
                WHERE {where_clause}
                LIMIT ?
            """
            params.append(k)

            cursor = conn.execute(sql, tuple(params))
            rows = cursor.fetchall()
            columns = [
                "chunk_id",
                "file_path",
                "language",
                "chunk_type",
                "name",
                "start_line",
                "end_line",
                "content",
            ]
            for row in rows:
                result = dict(zip(columns, row, strict=False))
                result["score"] = 0.5  # Lower score for keyword matches
                result["content"] = (result.get("content") or "")[:500]
                results.append(result)

        return results

    async def index_file(
        self, file_path: str, project: Optional[str] = None
    ) -> dict[str, Any]:
        """Index a single file."""
        await self.initialize()

        if not LIBSQL_AVAILABLE:
            return {"error": "libsql-experimental not available"}

        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        # Read raw bytes first for stable hash, then decode for content
        raw_bytes = path.read_bytes()
        content = raw_bytes.decode(errors="ignore")
        file_hash = _compute_file_hash(raw_bytes)
        chunks = self.parse_file(path, content)

        conn = _get_db()

        try:
            # Delete existing chunks for this file
            conn.execute("DELETE FROM code_chunks WHERE file_path = ?", (str(path),))

            # Index new chunks
            chunk_ids = []
            for idx, chunk in enumerate(chunks):
                # Offload embedding to thread to avoid blocking event loop
                embedding = await asyncio.to_thread(_get_embedding, chunk.content)

                conn.execute(
                    """
                    INSERT INTO code_chunks (
                        chunk_id, file_path, chunk_index, language, chunk_type,
                        name, content, start_line, end_line, file_hash, project,
                        dependencies, parent_name, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        chunk.id,
                        chunk.file,
                        idx,
                        chunk.language,
                        chunk.type,
                        chunk.name,
                        chunk.content,
                        chunk.start_line,
                        chunk.end_line,
                        file_hash,
                        project if project else None,  # Store NULL for no project
                        json.dumps(chunk.dependencies),
                        chunk.parent,
                        embedding,
                    ),
                )
                chunk_ids.append(chunk.id)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return {
            "file": str(path),
            "chunks_created": len(chunks),
            "chunk_ids": chunk_ids,
        }

    async def delete_file(self, file_path: str) -> dict[str, Any]:
        """Delete all chunks for a file."""
        if not LIBSQL_AVAILABLE:
            return {"error": "libsql-experimental not available"}

        conn = _get_db()
        cursor = conn.execute(
            "DELETE FROM code_chunks WHERE file_path = ?", (file_path,)
        )
        conn.commit()

        return {"file": file_path, "chunks_deleted": cursor.rowcount}

    async def get_status(self) -> dict[str, Any]:
        """Get index status."""
        await self.initialize()

        status: dict[str, Any] = {
            "libsql_available": LIBSQL_AVAILABLE,
            "tree_sitter_available": TREE_SITTER_AVAILABLE,
            "openai_available": _get_openai_client() is not None,
            "parsers_loaded": list(self.parsers.keys()),
        }

        if LIBSQL_AVAILABLE:
            try:
                conn = _get_db()
                cursor = conn.execute("SELECT COUNT(*) FROM code_chunks")
                row = cursor.fetchone()
                status["indexed_chunks"] = row[0] if row else 0

                cursor = conn.execute(
                    "SELECT COUNT(*) FROM code_chunks WHERE embedding IS NOT NULL"
                )
                row = cursor.fetchone()
                status["chunks_with_embeddings"] = row[0] if row else 0

                cursor = conn.execute("SELECT DISTINCT project FROM code_chunks")
                status["projects"] = [row[0] for row in cursor.fetchall() if row[0]]

                cursor = conn.execute("SELECT DISTINCT language FROM code_chunks")
                status["languages"] = [row[0] for row in cursor.fetchall() if row[0]]
            except Exception:
                logger.warning("Failed to query database status", exc_info=True)
                status["db_error"] = "Failed to query database"

        return status

    async def clear_project(self, project: str) -> dict[str, Any]:
        """Clear all chunks for a project."""
        if not LIBSQL_AVAILABLE:
            return {"error": "libsql-experimental not available"}

        conn = _get_db()
        cursor = conn.execute("DELETE FROM code_chunks WHERE project = ?", (project,))
        conn.commit()

        return {"project": project, "chunks_deleted": cursor.rowcount}


# Server instance
rag = CodebaseRAG()
server = Server("codebase-rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search",
            description="Search indexed codebase using hybrid keyword + vector search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language or code pattern)",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 10,
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter by language (python, csharp, etc.)",
                    },
                    "type": {
                        "type": "string",
                        "description": "Filter by chunk type (Tree-sitter node names: function_definition, class_definition, method_definition, etc.)",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="index_file",
            description="Index or re-index a single file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to index",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name for filtering",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="delete_file",
            description="Remove a file from the index",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to remove",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="clear_project",
            description="Remove all indexed chunks for a project",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project name to clear",
                    },
                },
                "required": ["project"],
            },
        ),
        Tool(
            name="status",
            description="Get index status including chunk counts and available parsers",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    await rag.initialize()

    if name == "search":
        results = await rag.search(
            query=arguments["query"],
            k=arguments.get("k", 10),
            language=arguments.get("language"),
            chunk_type=arguments.get("type"),
            project=arguments.get("project"),
        )
        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    elif name == "index_file":
        result = await rag.index_file(
            file_path=arguments["path"],
            project=arguments.get("project"),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "delete_file":
        result = await rag.delete_file(file_path=arguments["path"])
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "clear_project":
        result = await rag.clear_project(project=arguments["project"])
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "status":
        status = await rag.get_status()
        return [TextContent(type="text", text=json.dumps(status, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def main() -> None:
    """Run the MCP server."""
    asyncio.run(stdio_server(server))


if __name__ == "__main__":
    main()
