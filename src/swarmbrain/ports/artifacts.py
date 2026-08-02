"""Binary artifact storage ports; metadata stays evidence-addressable."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import ArtifactId
from swarmbrain.domain.evidence import ArtifactRef, StoreArtifactCommand


@runtime_checkable
class ArtifactReader(Protocol):
    async def get_artifact_ref(
        self,
        actor: ActorContext,
        artifact_id: ArtifactId,
    ) -> ArtifactRef | None: ...

    async def read_artifact(
        self,
        actor: ActorContext,
        artifact_id: ArtifactId,
    ) -> bytes | None: ...


@runtime_checkable
class ArtifactWriter(Protocol):
    async def put_artifact(
        self,
        actor: ActorContext,
        command: StoreArtifactCommand,
        content: bytes,
    ) -> ArtifactRef: ...


@runtime_checkable
class ArtifactStore(ArtifactReader, ArtifactWriter, Protocol):
    """Convenience composition for adapters implementing both small ports."""


__all__ = ["ArtifactReader", "ArtifactStore", "ArtifactWriter"]
