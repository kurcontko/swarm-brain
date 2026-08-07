"""The read-only swarm console page.

The page is static and carries no credentials: every byte of run data it shows
is fetched by the browser from the ordinary bearer-authenticated routes with a
viewer token the operator pastes in.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def console_html() -> str:
    """Return the self-contained console document (no external requests)."""
    return files(__package__).joinpath("index.html").read_text(encoding="utf-8")


__all__ = ["console_html"]
