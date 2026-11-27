"""Out-of-tree SGLang KV participant adapter for Kapsl.

Two tiers: the opaque tier (default; SGLang owns its KV pool and the adapter
leases capacity per request) and the elastic shared pool (the Kapsl runtime
owns the KV memory and the adapter maps it into SGLang's buffers).
"""

from ._version import ADAPTER_VERSION
from .capacity import (
    PINNED_SGLANG_VERSIONS,
    SchedulerFacts,
    UnsupportedProfileError,
    opaque_capacity_groups,
)
from .client import KapslKvControlClient, KapslKvControlError
from .config import (
    PARTICIPANT_CONFIG_ENV,
    AdapterConfig,
    ConfigError,
    load_config,
)
from .contract import ABI_VERSION, ContractValidationError, opaque_registration
from .participant import KvParticipant
from .pool_contract import STRIPED_POOL_ABI_VERSION as SHARED_POOL_ABI_VERSION
from .pool_participant import SharedPoolParticipant
from .shared_pool import ADAPTER_ID, SHARED_POOL_PROFILE_ID, SharedPoolError

__all__ = [
    "ABI_VERSION",
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "AdapterConfig",
    "ConfigError",
    "ContractValidationError",
    "KapslKvControlClient",
    "KapslKvControlError",
    "KvParticipant",
    "PARTICIPANT_CONFIG_ENV",
    "PINNED_SGLANG_VERSIONS",
    "SHARED_POOL_ABI_VERSION",
    "SHARED_POOL_PROFILE_ID",
    "SchedulerFacts",
    "SharedPoolError",
    "SharedPoolParticipant",
    "UnsupportedProfileError",
    "load_config",
    "opaque_capacity_groups",
    "opaque_registration",
]
