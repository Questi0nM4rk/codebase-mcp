# Plan: Support sentence-transformers (Drop OpenAI Dependency)

From review session (Jan 2026).

## Problem

Current implementation requires OpenAI API key for embeddings.

## Solution

Add sentence-transformers as primary, OpenAI as optional fallback.

```python
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...

class SentenceTransformerProvider: ...  # default
class OpenAIProvider: ...  # optional
```

## Done Criteria

- [ ] EmbeddingProvider protocol defined
- [ ] SentenceTransformerProvider as default
- [ ] OpenAI moved to optional
- [ ] Tests for both providers
