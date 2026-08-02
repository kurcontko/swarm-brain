"""Deterministic source-code extraction with exact source spans."""

from __future__ import annotations

import re
from collections.abc import Iterator

from swarmbrain.domain.extraction import ExtractionCandidate, ExtractionInput, SourceSpan
from swarmbrain.domain.memory import MemoryKind

_DECLARATION = re.compile(
    r"^\s*(?:(?:export|public|private|protected|static|async)\s+)*"
    r"(?:(?:def|class|function|interface|type|enum|struct|trait|fn)\s+[A-Za-z_]\w*"
    r"|(?:const|let|var)\s+[A-Za-z_]\w*\s*(?::[^=]+)?=)"
)
_SIGNAL = re.compile(r"(?i)(?:#|//|/\*|\*)\s*(TODO|FIXME|WARNING|WARN|NOTE)\b[:\s-]*(.*)")
_NAME = re.compile(
    r"(?:(?:def|class|function|interface|type|enum|struct|trait|fn|const|let|var)\s+)"
    r"([A-Za-z_]\w*)"
)


class CodingRuleExtractor:
    """Extract declarations and explicit engineering notes without inference."""

    name = "coding-rules"
    revision = "1"

    def __init__(self, *, max_candidates: int = 64) -> None:
        if not 1 <= max_candidates <= 1000:
            raise ValueError("max_candidates must be between 1 and 1000")
        self.max_candidates = max_candidates

    async def extract(self, request: ExtractionInput) -> tuple[ExtractionCandidate, ...]:
        candidates: list[ExtractionCandidate] = []
        seen: set[tuple[str, int]] = set()
        for line, start, end in self._lines(request.raw_content):
            stripped = line.strip()
            if not stripped or len(line) > 8192:
                continue
            chunk_index = self._containing_chunk(request, start, end)
            if chunk_index is None:
                continue
            span = SourceSpan(
                chunk_index=chunk_index,
                char_start=start,
                char_end=end,
                excerpt=line,
            )
            signal = _SIGNAL.search(line)
            if signal is not None:
                label = signal.group(1).upper()
                detail = signal.group(2).strip() or stripped
                content = f"{label}: {detail}"
                key = (content.casefold(), start)
                if key not in seen:
                    candidates.append(
                        ExtractionCandidate(
                            kind=(
                                MemoryKind.WARNING
                                if label in {"TODO", "FIXME", "WARNING", "WARN"}
                                else MemoryKind.OBSERVATION
                            ),
                            content=content,
                            title=f"Code {label.lower()}",
                            tags=("code", label.lower()),
                            confidence=0.98,
                            valid_from=request.source.observed_at,
                            spans=(span,),
                        )
                    )
                    seen.add(key)
            elif _DECLARATION.match(line):
                name_match = _NAME.search(line)
                name = name_match.group(1) if name_match else "declaration"
                content = f"Source declares {stripped}"
                key = (content.casefold(), start)
                if key not in seen:
                    candidates.append(
                        ExtractionCandidate(
                            kind=MemoryKind.OBSERVATION,
                            content=content,
                            title=f"Code declaration: {name}",
                            tags=("code", "declaration"),
                            confidence=0.9,
                            valid_from=request.source.observed_at,
                            spans=(span,),
                        )
                    )
                    seen.add(key)
            if len(candidates) >= self.max_candidates:
                break
        return tuple(candidates)

    @staticmethod
    def _lines(content: str) -> Iterator[tuple[str, int, int]]:
        offset = 0
        for raw_line in content.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            end = offset + len(line)
            if line:
                yield line, offset, end
            offset += len(raw_line)
        if offset < len(content):
            line = content[offset:]
            if line:
                yield line, offset, len(content)

    @staticmethod
    def _containing_chunk(request: ExtractionInput, start: int, end: int) -> int | None:
        for chunk in request.chunks:
            if start >= chunk.char_start and end <= chunk.char_end:
                return chunk.chunk_index
        return None


__all__ = ["CodingRuleExtractor"]
