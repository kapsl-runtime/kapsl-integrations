"""Shared fixture: an SGLang stand-in started on a runtime-owned KV pool.

The fake model has 2 layers of 2 KV heads of 8 in bf16, so one buffer row is
32 bytes and the pool has 4 stripes. The fake device's allocation granule is
1,024 bytes: 32 rows. The server reserves 255 tokens (256 rows, 8 granules per
buffer); the fake runtime maps 64 blocks at registration (63 tokens) and
allows the pool to shrink to 32 (31 tokens).
"""

from __future__ import annotations

import logging
import os
import unittest
from typing import Any
from unittest import mock

from fake_sglang import (
    PINNED_VERSION,
    POOL_ENVIRON,
    FakeSglang,
    HookRegistry,
    ServerArgs,
)
from fake_vmm import FakePoolRuntime

from kapsl_sglang_adapter import hooks
from kapsl_sglang_adapter.pool_participant import SharedPoolParticipant

GRANULE = 1024
ROW = 32
STRIPES = 4
MAX_TOKENS = 255
# Offset of each buffer inside the fake arena: 8 granules reserved, 3 of slack.
STRIPE_STRIDE = 11 * GRANULE


def silence_logging(case: unittest.TestCase) -> None:
    logging.disable(logging.CRITICAL)
    case.addCleanup(logging.disable, logging.NOTSET)


class PoolHookTestCase(unittest.TestCase):
    environ = POOL_ENVIRON

    def setUp(self) -> None:
        silence_logging(self)
        hooks.reset_for_tests()
        self.addCleanup(hooks.reset_for_tests)
        self.sglang = FakeSglang().install()
        self.addCleanup(self.sglang.uninstall)
        self.runtime = FakePoolRuntime(self.sglang.events)
        self.addCleanup(self.close_descriptors)
        self.now = 0.0
        self.sleeps: list[float] = []
        for patcher in (
            mock.patch.dict(os.environ, self.environ, clear=True),
            mock.patch.object(
                hooks, "installed_sglang_version", return_value=PINNED_VERSION
            ),
            mock.patch.object(hooks, "build_participant", self.build_participant),
            mock.patch.object(hooks.atexit, "register"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def close_descriptors(self) -> None:
        self.runtime.close()

    def build_participant(self, config: Any) -> SharedPoolParticipant:
        participant = SharedPoolParticipant(
            config,
            self.runtime,
            adapter_version=hooks.ADAPTER_VERSION,
            clock=lambda: self.now,
            heartbeat_interval_seconds=3600,
            sleep=self.sleeps.append,
        )
        self.addCleanup(participant.shutdown)
        return participant

    def install_hooks(self, skip: tuple[str, ...] = ()) -> None:
        hooks.activate()
        HookRegistry.apply_hooks(skip=skip)

    def server_args(self, **changes: Any) -> ServerArgs:
        arguments = {"max_total_tokens": MAX_TOKENS, "vmm_pool": True, **changes}
        return ServerArgs(**arguments)

    def construct(self, **changes: Any) -> Any:
        return self.sglang.scheduler.Scheduler(self.server_args(**changes), None)

    def start(self, **changes: Any) -> Any:
        """A serving scheduler, with the startup traffic and events cleared."""

        self.install_hooks()
        scheduler = self.construct(**changes)
        self.runtime.calls.clear()
        self.sglang.events.clear()
        return scheduler

    def step(self, scheduler: Any, seconds: float = 1.0) -> None:
        """One scheduler step, far enough after the last for a poll to be due."""

        self.now += seconds
        scheduler.get_next_batch_to_run(running_batch=None, last_batch=None)

    @property
    def participant(self) -> SharedPoolParticipant:
        participant = hooks._STATE.participant
        assert isinstance(participant, SharedPoolParticipant)
        return participant

    def capacity(self, scheduler: Any) -> dict[str, int]:
        """The six places SGLang keeps the pool capacity in."""

        runner = scheduler.tp_worker.model_runner
        return {
            "pool.size": int(runner.token_to_kv_pool.size),
            "allocator.size": int(scheduler.token_to_kv_pool_allocator.size),
            "runner": int(runner.max_total_num_tokens),
            "scheduler": int(scheduler.max_total_num_tokens),
            "observer": int(scheduler.pool_stats_observer.max_total_num_tokens),
            "checker": int(scheduler.invariant_checker.max_total_num_tokens),
        }

    def assert_capacity(self, scheduler: Any, tokens: int) -> None:
        self.assertEqual(set(self.capacity(scheduler).values()), {tokens})

    def free_slots(self, scheduler: Any) -> list[int]:
        allocator = scheduler.token_to_kv_pool_allocator
        return sorted(allocator.free_pages.values + allocator.release_pages.values)

    def mappings(self, scheduler: Any) -> list[tuple[int, int, int]]:
        """Every mapping in SGLang's arena as (stripe, offset in stripe, bytes)."""

        owner = scheduler.tp_worker.model_runner.token_to_kv_pool._post_capture_owner
        arena = owner._arena
        return sorted(
            (
                (address - arena.base) // STRIPE_STRIDE,
                (address - arena.base) % STRIPE_STRIDE,
                size,
            )
            for address, size, _ in arena._allocation._mappings
        )
