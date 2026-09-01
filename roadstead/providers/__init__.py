"""Provider registry — engine name in, adapter out.

``models.yaml`` spells ``backend_engine`` for human readers, so the strings in
the wild are not tidy: ``llama.cpp``, ``vllm``, ``shim``, ``llama.cpp (Vulkan)``.
Resolution is therefore prefix-tolerant and ends at a DEFAULT rather than an
error — which is not laziness, it is the behaviour being preserved. Every
branch this package replaces was written as ``backend_engine == "vllm"``, so an
unrecognised engine has always taken the llama.cpp path, and a config typo has
always degraded to "the common shape" instead of taking an endpoint offline.

``shim`` resolves here too. It marks the non-OpenAI embed/rerank FastAPI shims,
which have no chat route to normalize and no ``/props`` to discover; what
actually keeps the poller off them is ``EndpointConfig.skip_discovery``, a
per-endpoint switch that health checks BEFORE asking a provider anything. Those
two have never been the same flag and this package does not make them one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    CapacityReport,
    Provider,
    ProviderDescriptor,
    ProviderError,
    ProviderMisconfigured,
    UnsupportedRequest,
)
from .llamacpp import LLAMACPP, LlamaCppProvider
from .openrouter import OPENROUTER, OpenRouterProvider
from .vllm import VLLM, VLLMProvider

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import EndpointConfig

__all__ = [
    "CapacityReport",
    "LLAMACPP",
    "LlamaCppProvider",
    "OPENROUTER",
    "OpenRouterProvider",
    "Provider",
    "ProviderDescriptor",
    "ProviderError",
    "ProviderMisconfigured",
    "UnsupportedRequest",
    "VLLM",
    "VLLMProvider",
    "provider_for",
    "provider_for_engine",
    "register_provider",
]

#: The default when nothing matches — see the module docstring.
DEFAULT_PROVIDER: Provider = LLAMACPP

_PROVIDERS: dict[str, Provider] = {}


def register_provider(provider: Provider) -> None:
    """Add a provider under its descriptor name. Stateless singletons only."""
    _PROVIDERS[provider.descriptor.name.lower()] = provider


register_provider(LLAMACPP)
register_provider(VLLM)
register_provider(OPENROUTER)


def provider_for_engine(engine: str | None) -> Provider:
    """Resolve a ``backend_engine`` string. Never raises, never returns None."""
    key = (engine or "").strip().lower()
    if not key:
        return DEFAULT_PROVIDER
    hit = _PROVIDERS.get(key)
    if hit is not None:
        return hit
    # Decorated spellings — "llama.cpp (Vulkan)" is one endpoint's build note,
    # not a different engine.
    for name, provider in _PROVIDERS.items():
        if key.startswith(name):
            return provider
    return DEFAULT_PROVIDER


def provider_for(ep_cfg: "EndpointConfig") -> Provider:
    """The provider for one endpoint. Cheap enough for the dispatch path (a
    dict lookup); deliberately NOT cached on the config object, so an engine
    corrected at runtime takes effect on the next call rather than at the next
    restart."""
    return provider_for_engine(getattr(ep_cfg, "backend_engine", ""))
