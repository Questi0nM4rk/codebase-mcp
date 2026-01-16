"""Tests for codebase_rag.server module."""

import tempfile
from pathlib import Path

import pytest

from codebase_rag.server import (
    Chunk,
    CodebaseRAG,
    Manifest,
    ManifestEntry,
)


class TestChunk:
    """Tests for Chunk model."""

    def test_chunk_creation_with_required_fields(self) -> None:
        """Chunk should be created with required fields."""
        chunk = Chunk(
            id="test_id",
            file="/path/to/file.py",
            language="python",
            type="function_definition",
            name="test_func",
            start_line=1,
            end_line=10,
            content="def test_func(): pass",
        )

        assert chunk.id == "test_id"
        assert chunk.file == "/path/to/file.py"
        assert chunk.language == "python"
        assert chunk.type == "function_definition"
        assert chunk.name == "test_func"

    def test_chunk_optional_fields_default_to_none_or_empty(self) -> None:
        """Chunk optional fields should have correct defaults."""
        chunk = Chunk(
            id="test_id",
            file="/path/to/file.py",
            language="python",
            type="function_definition",
            name="test_func",
            start_line=1,
            end_line=10,
            content="def test_func(): pass",
        )

        assert chunk.signature is None
        assert chunk.dependencies == []
        assert chunk.parent is None


class TestManifestEntry:
    """Tests for ManifestEntry model."""

    def test_manifest_entry_creation(self) -> None:
        """ManifestEntry should be created with required fields."""
        entry = ManifestEntry(hash="abc123")

        assert entry.hash == "abc123"
        assert entry.mtime is None
        assert entry.chunk_ids == []


class TestManifest:
    """Tests for Manifest model."""

    def test_manifest_default_values(self) -> None:
        """Manifest should have correct default values."""
        manifest = Manifest()

        assert manifest.version == "1.0"
        assert manifest.root_hash == ""
        assert manifest.updated == ""
        assert manifest.stats == {}
        assert manifest.tree == {}


class TestCodebaseRAG:
    """Tests for CodebaseRAG class."""

    def test_compute_file_hash_returns_16_char_hex(self) -> None:
        """compute_file_hash should return 16 character hex string."""
        rag = CodebaseRAG()
        content = b"test content"

        hash_result = rag.compute_file_hash(content)

        assert len(hash_result) == 16
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_compute_file_hash_deterministic(self) -> None:
        """Same content should produce same hash."""
        rag = CodebaseRAG()
        content = b"test content"

        hash1 = rag.compute_file_hash(content)
        hash2 = rag.compute_file_hash(content)

        assert hash1 == hash2

    def test_compute_file_hash_different_content_different_hash(self) -> None:
        """Different content should produce different hash."""
        rag = CodebaseRAG()

        hash1 = rag.compute_file_hash(b"content1")
        hash2 = rag.compute_file_hash(b"content2")

        assert hash1 != hash2

    def test_compute_dir_hash_returns_16_char_hex(self) -> None:
        """compute_dir_hash should return 16 character hex string."""
        rag = CodebaseRAG()

        hash_result = rag.compute_dir_hash(["abc", "def"])

        assert len(hash_result) == 16
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_compute_dir_hash_order_independent(self) -> None:
        """compute_dir_hash should produce same result regardless of input order."""
        rag = CodebaseRAG()

        hash1 = rag.compute_dir_hash(["abc", "def"])
        hash2 = rag.compute_dir_hash(["def", "abc"])

        assert hash1 == hash2

    def test_load_manifest_returns_empty_manifest_when_no_file(self) -> None:
        """load_manifest should return empty Manifest when file doesn't exist."""
        rag = CodebaseRAG()

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = rag.load_manifest(Path(tmpdir))

        assert manifest.version == "1.0"
        assert manifest.root_hash == ""

    def test_save_and_load_manifest_roundtrip(self) -> None:
        """Manifest should be saved and loaded correctly."""
        rag = CodebaseRAG()
        manifest = Manifest(
            version="1.0",
            root_hash="abc123",
            updated="2026-01-17",
            stats={"files": 10},
            tree={"src": {"hash": "xyz"}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            rag.save_manifest(project_path, manifest)
            loaded = rag.load_manifest(project_path)

        assert loaded.version == manifest.version
        assert loaded.root_hash == manifest.root_hash
        assert loaded.updated == manifest.updated
        assert loaded.stats == manifest.stats
        assert loaded.tree == manifest.tree

    def test_create_raw_chunk_for_unknown_file(self) -> None:
        """parse_file should create raw chunk for unknown file types."""
        rag = CodebaseRAG()
        content = "some content\nmore content"

        chunks = rag.parse_file(Path("/tmp/test.unknown"), content)

        assert len(chunks) == 1
        assert chunks[0].type == "raw"
        assert chunks[0].language == "unknown"

    def test_get_embed_returns_zero_vector_without_openai(self) -> None:
        """get_embed should return zero vector when OpenAI not available."""
        rag = CodebaseRAG()
        rag.openai = None

        embedding = rag.get_embed("test text")

        assert len(embedding) == 1536  # EMBEDDING_DIM
        assert all(v == 0.0 for v in embedding)


@pytest.mark.asyncio
class TestCodebaseRAGAsync:
    """Async tests for CodebaseRAG class."""

    async def test_get_status_without_qdrant(self) -> None:
        """get_status should work without Qdrant connection."""
        rag = CodebaseRAG()
        rag._initialized = True
        rag.qdrant = None

        status = await rag.get_status()

        assert status["qdrant_available"] is False

    async def test_search_returns_error_without_qdrant(self) -> None:
        """search should return error when Qdrant not available."""
        rag = CodebaseRAG()
        rag._initialized = True
        rag.qdrant = None

        results = await rag.search("test query")

        assert len(results) == 1
        assert "error" in results[0]

    async def test_index_file_returns_error_for_missing_file(self) -> None:
        """index_file should return error for non-existent file."""
        rag = CodebaseRAG()
        rag._initialized = True

        result = await rag.index_file("/nonexistent/file.py")

        assert "error" in result
        assert "not found" in result["error"].lower()
