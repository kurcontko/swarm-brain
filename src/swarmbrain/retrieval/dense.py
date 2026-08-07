"""Stable dense-projection identity shared by writers and query lanes."""

from __future__ import annotations

DENSE_PROJECTION_ID = "memory-content-v1:current:cosine"
DENSE_PROJECTION_REVISION = "dense-v2"
DENSE_VECTOR_DIMENSIONS = 1024


def dense_projection_signature(model: str, dimensions: int) -> str:
    """Identify every input/model choice that changes vector meaning.

    The deliberately readable signature also has a byte-for-byte SQL
    equivalent for migrating legacy vectors.  A model name may contain the
    separator, but it occupies the final field and therefore remains
    unambiguous.
    """

    normalized_model = model.strip()
    if not normalized_model or len(normalized_model) > 255:
        raise ValueError("model must contain between 1 and 255 characters")
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    return (
        f"{DENSE_PROJECTION_REVISION}|{DENSE_PROJECTION_ID}|"
        "normalization=provider|truncation=provider|"
        f"dimensions={dimensions}|model={normalized_model}"
    )


__all__ = [
    "DENSE_PROJECTION_ID",
    "DENSE_PROJECTION_REVISION",
    "DENSE_VECTOR_DIMENSIONS",
    "dense_projection_signature",
]
