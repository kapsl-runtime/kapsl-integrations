"""Out-of-tree SGLang KV participant adapter for Kapsl (opaque tier)."""

from .capacity import (
    PINNED_SGLANG_VERSIONS,
    SchedulerFacts,
    UnsupportedProfileError,
    opaque_capacity_groups,
)
from .client import KapslKvControlClient, KapslKvControlError
from .config import AdapterConfig, ConfigError, load_config
from .contract import ABI_VERSION, ContractValidationError, opaque_registration
from .participant import KvParticipant

ADAPTER_VERSION = "0.1.0"

__all__ = [
    "ABI_VERSION",
    "ADAPTER_VERSION",
    "AdapterConfig",
    "ConfigError",
    "ContractValidationError",
    "KapslKvControlClient",
    "KapslKvControlError",
    "KvParticipant",
    "PINNED_SGLANG_VERSIONS",
    "SchedulerFacts",
    "UnsupportedProfileError",
    "load_config",
    "opaque_capacity_groups",
    "opaque_registration",
]
