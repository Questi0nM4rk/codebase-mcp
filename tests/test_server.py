"""Tests for codebase_rag.server module."""

from pathlib import Path

import pytest

from codebase_rag.server import (
    Chunk,
    CodebaseRAG,
    _compute_file_hash,
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


class TestComputeFileHash:
    """Tests for _compute_file_hash function."""

    def test_compute_file_hash_returns_16_char_hex(self) -> None:
        """_compute_file_hash should return 16 character hex string."""
        content = b"test content"

        hash_result = _compute_file_hash(content)

        assert len(hash_result) == 16
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_compute_file_hash_deterministic(self) -> None:
        """Same content should produce same hash."""
        content = b"test content"

        hash1 = _compute_file_hash(content)
        hash2 = _compute_file_hash(content)

        assert hash1 == hash2

    def test_compute_file_hash_different_content_different_hash(self) -> None:
        """Different content should produce different hash."""
        hash1 = _compute_file_hash(b"content1")
        hash2 = _compute_file_hash(b"content2")

        assert hash1 != hash2


class TestCodebaseRAG:
    """Tests for CodebaseRAG class."""

    def test_create_raw_chunk_for_unknown_file(self) -> None:
        """parse_file should create raw chunk for unknown file types."""
        rag = CodebaseRAG()
        content = "some content\nmore content"

        chunks = rag.parse_file(Path("/tmp/test.unknown"), content)

        assert len(chunks) == 1
        assert chunks[0].type == "raw"
        assert chunks[0].language == "unknown"

    def test_parse_python_file_returns_chunks(self) -> None:
        """parse_file should return chunks for Python files."""
        rag = CodebaseRAG()
        content = "def hello():\n    pass\n"

        chunks = rag.parse_file(Path("/tmp/test.py"), content)

        # Should return at least one chunk (may be raw if tree-sitter unavailable)
        assert len(chunks) >= 1
        assert chunks[0].language == "python"

    def test_parse_empty_file(self) -> None:
        """parse_file should handle empty files."""
        rag = CodebaseRAG()

        chunks = rag.parse_file(Path("/tmp/test.py"), "")

        assert isinstance(chunks, list)


@pytest.mark.asyncio
class TestCodebaseRAGAsync:
    """Async tests for CodebaseRAG class."""

    async def test_get_status(self) -> None:
        """get_status should return status dict."""
        rag = CodebaseRAG()
        await rag.initialize()

        status = await rag.get_status()

        assert isinstance(status, dict)

    async def test_index_file_returns_error_for_missing_file(self) -> None:
        """index_file should return error for non-existent file."""
        rag = CodebaseRAG()
        await rag.initialize()

        result = await rag.index_file("/nonexistent/file.py")

        assert "error" in result
        assert "not found" in result["error"].lower()

    async def test_search_returns_list(self) -> None:
        """search should return a list of results."""
        rag = CodebaseRAG()
        await rag.initialize()

        results = await rag.search("test query")

        assert isinstance(results, list)

    async def test_delete_file_nonexistent(self) -> None:
        """delete_file should handle non-existent files gracefully."""
        rag = CodebaseRAG()
        await rag.initialize()

        result = await rag.delete_file("/nonexistent/file.py")

        assert isinstance(result, dict)
