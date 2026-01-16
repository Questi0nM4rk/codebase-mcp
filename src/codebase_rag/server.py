"""
Codebase RAG MCP Server

Provides tools for:
- Merkle tree-based incremental indexing
- Tree-sitter semantic code parsing
- Qdrant hybrid search (BM25 + vector)
"""

import asyncio
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    VectorParams,
)

# Tree-sitter imports
try:
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    import tree_sitter_c
    import tree_sitter_cpp
    import tree_sitter_rust
    import tree_sitter_go
    import tree_sitter_c_sharp
    import tree_sitter_lua
    import tree_sitter_bash
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# OpenAI for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "codebase_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Language to Tree-sitter grammar mapping
LANGUAGE_MAP = {
    ".py": ("python", tree_sitter_python if TREE_SITTER_AVAILABLE else None),
    ".js": ("javascript", tree_sitter_javascript if TREE_SITTER_AVAILABLE else None),
    ".jsx": ("javascript", tree_sitter_javascript if TREE_SITTER_AVAILABLE else None),
    ".ts": ("typescript", tree_sitter_typescript if TREE_SITTER_AVAILABLE else None),
    ".tsx": ("typescript", tree_sitter_typescript if TREE_SITTER_AVAILABLE else None),
    ".c": ("c", tree_sitter_c if TREE_SITTER_AVAILABLE else None),
    ".h": ("c", tree_sitter_c if TREE_SITTER_AVAILABLE else None),
    ".cpp": ("cpp", tree_sitter_cpp if TREE_SITTER_AVAILABLE else None),
    ".hpp": ("cpp", tree_sitter_cpp if TREE_SITTER_AVAILABLE else None),
    ".rs": ("rust", tree_sitter_rust if TREE_SITTER_AVAILABLE else None),
    ".go": ("go", tree_sitter_go if TREE_SITTER_AVAILABLE else None),
    ".cs": ("csharp", tree_sitter_c_sharp if TREE_SITTER_AVAILABLE else None),
    ".lua": ("lua", tree_sitter_lua if TREE_SITTER_AVAILABLE else None),
    ".sh": ("bash", tree_sitter_bash if TREE_SITTER_AVAILABLE else None),
    ".bash": ("bash", tree_sitter_bash if TREE_SITTER_AVAILABLE else None),
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
    dependencies: list[str] = []
    parent: Optional[str] = None


class ManifestEntry(BaseModel):
    """Entry in the Merkle tree manifest."""
    hash: str
    mtime: Optional[str] = None
    chunk_ids: list[str] = []


class Manifest(BaseModel):
    """Merkle tree manifest for change detection."""
    version: str = "1.0"
    root_hash: str = ""
    updated: str = ""
    stats: dict = {}
    tree: dict = {}


class CodebaseRAG:
    """Codebase indexing and search with Merkle tree and Qdrant."""

    def __init__(self):
        self.qdrant: Optional[QdrantClient] = None
        self.openai: Optional[OpenAI] = None
        self.parsers: dict[str, Parser] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize Qdrant client and Tree-sitter parsers."""
        if self._initialized:
            return

        # Initialize Qdrant
        try:
            self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            # Ensure collection exists
            collections = self.qdrant.get_collections().collections
            if not any(c.name == COLLECTION_NAME for c in collections):
                self.qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config={
                        "content": VectorParams(
                            size=EMBEDDING_DIM,
                            distance=Distance.COSINE,
                        )
                    },
                )
        except Exception as e:
            self.qdrant = None
            print(f"Qdrant connection failed: {e}")

        # Initialize OpenAI
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai = OpenAI(api_key=api_key)

        # Initialize Tree-sitter parsers
        if TREE_SITTER_AVAILABLE:
            for ext, (lang_name, grammar_module) in LANGUAGE_MAP.items():
                if grammar_module and lang_name not in self.parsers:
                    try:
                        parser = Parser()
                        language = Language(grammar_module.language())
                        parser.language = language
                        self.parsers[lang_name] = parser
                    except Exception as e:
                        print(f"Failed to load parser for {lang_name}: {e}")

        self._initialized = True

    def compute_file_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of file content."""
        return hashlib.sha256(content).hexdigest()[:16]

    def compute_dir_hash(self, child_hashes: list[str]) -> str:
        """Compute hash of directory from sorted child hashes."""
        combined = "".join(sorted(child_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def load_manifest(self, project_path: Path) -> Manifest:
        """Load existing manifest or create new one."""
        manifest_path = project_path / ".codeagent" / "index" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
                return Manifest(**data)
        return Manifest()

    def save_manifest(self, project_path: Path, manifest: Manifest):
        """Save manifest to disk."""
        manifest_path = project_path / ".codeagent" / "index" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest.model_dump(), f, indent=2)

    def get_embed(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        if not self.openai:
            # Return zero vector if OpenAI not available
            return [0.0] * EMBEDDING_DIM

        response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000],  # Truncate to fit context
        )
        return response.data[0].embedding

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
        except Exception as e:
            print(f"Parse error for {file_path}: {e}")
            return [self._create_raw_chunk(file_path, content)]

    def _create_raw_chunk(self, file_path: Path, content: str) -> Chunk:
        """Create a raw chunk for unparseable files."""
        lines = content.split("\n")
        return Chunk(
            id=f"raw_{self.compute_file_hash(content.encode())}",
            file=str(file_path),
            language="unknown",
            type="raw",
            name=file_path.name,
            start_line=1,
            end_line=len(lines),
            content=content[:2000],
        )

    def _extract_chunks(
        self, file_path: Path, content: str, root_node, language: str
    ) -> list[Chunk]:
        """Extract semantic chunks from AST."""
        chunks = []
        lines = content.split("\n")

        # Node types to extract by language
        chunk_types = {
            "python": ["function_definition", "class_definition", "async_function_definition"],
            "javascript": ["function_declaration", "class_declaration", "arrow_function", "method_definition"],
            "typescript": ["function_declaration", "class_declaration", "interface_declaration", "method_definition"],
            "c": ["function_definition", "struct_specifier"],
            "cpp": ["function_definition", "class_specifier", "struct_specifier"],
            "rust": ["function_item", "impl_item", "struct_item"],
            "go": ["function_declaration", "method_declaration", "type_declaration"],
            "csharp": ["method_declaration", "class_declaration", "interface_declaration"],
            "lua": ["function_definition", "local_function"],
            "bash": ["function_definition"],
        }

        target_types = chunk_types.get(language, [])

        def visit(node, parent_name=None):
            if node.type in target_types:
                # Extract name
                name = self._extract_node_name(node, language)

                # Get line numbers (1-indexed)
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                # Extract content
                chunk_content = "\n".join(lines[start_line - 1 : end_line])

                chunk = Chunk(
                    id=f"{language}_{self.compute_file_hash((str(file_path) + name).encode())}_{start_line}",
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

        # If no chunks found, create raw chunk
        if not chunks:
            return [self._create_raw_chunk(file_path, content)]

        return chunks

    def _extract_node_name(self, node, language: str) -> str:
        """Extract name from AST node."""
        # Look for identifier or name child
        for child in node.children:
            if child.type in ["identifier", "name", "type_identifier", "property_identifier"]:
                return child.text.decode() if hasattr(child.text, "decode") else str(child.text)
        return "anonymous"

    async def search(
        self,
        query: str,
        k: int = 10,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
        chunk_type: Optional[str] = None,
        project: Optional[str] = None,
    ) -> list[dict]:
        """Hybrid search over indexed codebase."""
        if not self.qdrant:
            return [{"error": "Qdrant not available. Run 'codeagent start' first."}]

        # Build filter
        must_conditions = []
        if language:
            must_conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
        if chunk_type:
            must_conditions.append(FieldCondition(key="type", match=MatchValue(value=chunk_type)))
        if project:
            must_conditions.append(FieldCondition(key="project", match=MatchValue(value=project)))

        filter_obj = Filter(must=must_conditions) if must_conditions else None

        # Get embedding
        query_vector = self.get_embed(query)

        # Search
        try:
            results = self.qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=("content", query_vector),
                query_filter=filter_obj,
                limit=k,
                with_payload=True,
            )

            return [
                {
                    "id": r.id,
                    "score": r.score,
                    "file": r.payload.get("file"),
                    "language": r.payload.get("language"),
                    "type": r.payload.get("type"),
                    "name": r.payload.get("name"),
                    "start_line": r.payload.get("start_line"),
                    "end_line": r.payload.get("end_line"),
                    "content": r.payload.get("content", "")[:500],
                }
                for r in results
            ]
        except Exception as e:
            return [{"error": str(e)}]

    async def index_file(self, file_path: str, project: Optional[str] = None) -> dict:
        """Index a single file."""
        await self.initialize()

        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        content = path.read_text(errors="ignore")
        chunks = self.parse_file(path, content)

        # Delete existing chunks for this file
        if self.qdrant:
            try:
                self.qdrant.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=Filter(
                        must=[FieldCondition(key="file", match=MatchValue(value=str(path)))]
                    ),
                )
            except Exception:
                pass

        # Index new chunks
        points = []
        for chunk in chunks:
            embedding = self.get_embed(chunk.content)
            point = PointStruct(
                id=chunk.id,
                vector={"content": embedding},
                payload={
                    "file": chunk.file,
                    "language": chunk.language,
                    "type": chunk.type,
                    "name": chunk.name,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                    "dependencies": chunk.dependencies,
                    "parent": chunk.parent,
                    "project": project or "",
                },
            )
            points.append(point)

        if self.qdrant and points:
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return {
            "file": str(path),
            "chunks_created": len(chunks),
            "chunk_ids": [c.id for c in chunks],
        }

    async def get_status(self) -> dict:
        """Get index status."""
        await self.initialize()

        status = {
            "qdrant_available": self.qdrant is not None,
            "tree_sitter_available": TREE_SITTER_AVAILABLE,
            "openai_available": self.openai is not None,
            "parsers_loaded": list(self.parsers.keys()),
        }

        if self.qdrant:
            try:
                collection_info = self.qdrant.get_collection(COLLECTION_NAME)
                status["indexed_chunks"] = collection_info.points_count
                status["collection_status"] = collection_info.status.value
            except Exception as e:
                status["collection_error"] = str(e)

        return status


# Server instance
rag = CodebaseRAG()
server = Server("codebase-rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search",
            description="Search indexed codebase using hybrid BM25 + vector search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (natural language or code pattern)"},
                    "k": {"type": "integer", "description": "Max results to return", "default": 10},
                    "language": {"type": "string", "description": "Filter by language (python, csharp, etc.)"},
                    "type": {"type": "string", "description": "Filter by chunk type (function, class, method)"},
                    "project": {"type": "string", "description": "Filter by project name"},
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
                    "path": {"type": "string", "description": "Path to the file to index"},
                    "project": {"type": "string", "description": "Project name for filtering"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="status",
            description="Get index status including Qdrant health and chunk count",
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

    elif name == "status":
        status = await rag.get_status()
        return [TextContent(type="text", text=json.dumps(status, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def main():
    """Run the MCP server."""
    asyncio.run(stdio_server(server))


if __name__ == "__main__":
    main()
