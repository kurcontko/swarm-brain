from __future__ import annotations

from swarmbrain.domain.agents import ActorContext, Capability

from .errors import Forbidden


def require_capability(actor: ActorContext, capability: Capability | str) -> None:
    name = str(capability)
    if not actor.has_capability(name):
        raise Forbidden(name)
