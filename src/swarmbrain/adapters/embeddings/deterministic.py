"""Deterministic, dependency-free embeddings for tests and local demos."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class DeterministicEmbeddingProvider:
    """Hash-seeded bag-of-words embeddings with process-stable output.

    These vectors are not intended to model real semantic similarity.  They
    let the durable enqueue, lease, upsert, and recall path run locally without
    credentials or network access.
    """

    def __init__(self, *, dimensions: int = 1024, model_name: str = "deterministic-v0") -> None:
        if dimensions < 2:
            raise ValueError("dimensions must be at least 2")
        if not model_name or len(model_name) > 255:
            raise ValueError("model_name must contain between 1 and 255 characters")
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed(text)

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed(text) for text in texts)

    def _embed(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._dimensions
        tokens = _TOKEN_PATTERN.findall(text.casefold()) or ["empty"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(8):
                pair = digest[offset * 2 : offset * 2 + 2]
                index = int.from_bytes(pair, "big") % self._dimensions
                sign = 1.0 if digest[16 + offset] % 2 == 0 else -1.0
                values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            values[0] = 1.0
            norm = 1.0
        return tuple(value / norm for value in values)


__all__ = ["DeterministicEmbeddingProvider"]
