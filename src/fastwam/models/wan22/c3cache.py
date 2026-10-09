"""Episode-local residual cache for chunked action inference.

The cache owns only Python control state and persistent residual tensors. It is
kept outside compiled denoising functions so torch.compile sees pure tensor
operations and a fresh video prefill can run for every chunk.
"""

from dataclasses import dataclass, field
from typing import Any


C3CACHE_METHODS = frozenset(
    {"hidden", "velocity_delta", "velocity", "prefix", "velocity_virtual", "velocity_probe"}
)


@dataclass
class C3Cache:
    signature: tuple[Any, ...] | None = None
    chunk_index: int = 0
    residuals: dict[int, Any] = field(default_factory=dict)
    anchor: Any = None
    endpoint: Any = None
    cached_from_chunk: int | None = None
    full_steps: int = 0
    reused_steps: int = 0
    scheduler_skipped_steps: int = 0
    probe_steps: int = 0
    probe_blocks: int = 0
    refresh_chunks: int = 0
    reuse_chunks: int = 0
    fallback_chunks: int = 0
    last_chunk_full_steps: int = 0
    last_chunk_reused_steps: int = 0
    last_chunk_scheduler_skipped_steps: int = 0
    last_chunk_probe_steps: int = 0
    last_chunk_probe_blocks: int = 0
    last_chunk_reason: str = ""

    def reset(self) -> None:
        self.signature = None
        self.chunk_index = 0
        self.residuals.clear()
        self.anchor = None
        self.endpoint = None
        self.cached_from_chunk = None
        self.full_steps = 0
        self.reused_steps = 0
        self.scheduler_skipped_steps = 0
        self.probe_steps = 0
        self.probe_blocks = 0
        self.refresh_chunks = 0
        self.reuse_chunks = 0
        self.fallback_chunks = 0
        self.last_chunk_full_steps = 0
        self.last_chunk_reused_steps = 0
        self.last_chunk_scheduler_skipped_steps = 0
        self.last_chunk_probe_steps = 0
        self.last_chunk_probe_blocks = 0
        self.last_chunk_reason = ""

    def begin(self, signature: tuple[Any, ...]) -> int:
        if self.signature != signature:
            self.reset()
            self.signature = signature
        return self.chunk_index

    def should_reuse(
        self,
        step: int,
        start_step: int,
        end_step: int,
        refresh_interval: int,
    ) -> bool:
        if not start_step <= step <= end_step or step not in self.residuals:
            return False
        if self.chunk_index == 0:
            return False
        return refresh_interval == 0 or self.chunk_index % refresh_interval != 0

    def reuse_reason(
        self, method: str, start_step: int, end_step: int,
        refresh_interval: int, fixed_noise: bool,
        deterministic_schedule: bool = True,
    ) -> str:
        """Return a reason for a full refresh, or ``reuse`` for a complete hit."""
        if self.chunk_index == 0:
            return "first_chunk"
        if refresh_interval and self.chunk_index % refresh_interval == 0:
            return "scheduled_refresh"
        if method in {"velocity", "prefix"} and not fixed_noise:
            return "unfixed_noise"
        if method in {"velocity", "prefix"} and not deterministic_schedule:
            return "unsupported_scheduler"
        if method == "prefix":
            return "reuse" if self.endpoint is not None else "cache_miss"
        if method in {"velocity_delta", "velocity_virtual", "velocity_probe"} and self.anchor is None:
            return "cache_miss"
        if any(step not in self.residuals for step in range(start_step, end_step + 1)):
            return "cache_miss"
        return "reuse"

    def commit(
        self, residuals: dict[int, Any], full_steps: int, reused_steps: int,
        *, anchor: Any = None, endpoint: Any = None, reason: str = "",
        scheduler_skipped_steps: int = 0, probe_steps: int = 0,
        probe_blocks: int = 0,
    ) -> None:
        if reason == "":
            # Backwards compatible low-level cache API used by lightweight
            # checks and older callers; the sampler always supplies a reason.
            self.residuals.update(residuals)
        elif reason != "reuse":
            # A refresh replaces the entire trajectory atomically after success.
            self.residuals = dict(residuals)
            self.anchor = anchor
            self.endpoint = endpoint
            self.cached_from_chunk = self.chunk_index
            self.refresh_chunks += 1
            if reason not in {"first_chunk", "scheduled_refresh", ""}:
                self.fallback_chunks += 1
        else:
            self.reuse_chunks += 1
        self.chunk_index += 1
        self.full_steps += full_steps
        self.reused_steps += reused_steps
        self.scheduler_skipped_steps += scheduler_skipped_steps
        self.probe_steps += probe_steps
        self.probe_blocks += probe_blocks
        self.last_chunk_full_steps = full_steps
        self.last_chunk_reused_steps = reused_steps
        self.last_chunk_scheduler_skipped_steps = scheduler_skipped_steps
        self.last_chunk_probe_steps = probe_steps
        self.last_chunk_probe_blocks = probe_blocks
        self.last_chunk_reason = reason

    def stats(self) -> dict[str, Any]:
        return {
            "completed_chunks": self.chunk_index,
            "full_steps": self.full_steps,
            "reused_steps": self.reused_steps,
            "scheduler_skipped_steps": self.scheduler_skipped_steps,
            "probe_steps": self.probe_steps,
            "probe_blocks": self.probe_blocks,
            "refresh_chunks": self.refresh_chunks,
            "reuse_chunks": self.reuse_chunks,
            "fallback_chunks": self.fallback_chunks,
            "last_chunk_full_steps": self.last_chunk_full_steps,
            "last_chunk_reused_steps": self.last_chunk_reused_steps,
            "last_chunk_scheduler_skipped_steps": self.last_chunk_scheduler_skipped_steps,
            "last_chunk_probe_steps": self.last_chunk_probe_steps,
            "last_chunk_probe_blocks": self.last_chunk_probe_blocks,
            "last_chunk_reason": self.last_chunk_reason,
            "cached_steps": tuple(sorted(self.residuals)),
            "has_anchor": self.anchor is not None,
            "has_endpoint": self.endpoint is not None,
            "cached_from_chunk": self.cached_from_chunk,
        }


def validate_c3cache_range(
    num_inference_steps: int,
    start_step: int,
    end_step: int,
    refresh_interval: int,
) -> None:
    if isinstance(start_step, bool) or not isinstance(start_step, int):
        raise ValueError("`c3cache_start_step` must be an integer.")
    if isinstance(end_step, bool) or not isinstance(end_step, int):
        raise ValueError("`c3cache_end_step` must be an integer.")
    if isinstance(refresh_interval, bool) or not isinstance(refresh_interval, int):
        raise ValueError("`c3cache_refresh_interval` must be an integer.")
    if not 0 <= start_step <= end_step < num_inference_steps:
        raise ValueError(
            "C3ache step range must satisfy "
            f"0 <= start <= end < num_inference_steps ({num_inference_steps}); "
            f"got [{start_step}, {end_step}]."
        )
    if refresh_interval < 0:
        raise ValueError("`c3cache_refresh_interval` must be >= 0.")


def validate_c3cache_method(method: str, start_step: int, probe_depth: int, num_layers: int) -> None:
    if method not in C3CACHE_METHODS:
        raise ValueError(f"`c3cache_method` must be one of {sorted(C3CACHE_METHODS)}; got {method!r}.")
    if method == "velocity_delta" and start_step != 1:
        raise ValueError("`velocity_delta` requires `c3cache_start_step == 1` for a continuous anchored prefix.")
    if method in {"velocity", "prefix", "velocity_virtual", "velocity_probe"} and start_step != 0:
        raise ValueError(f"`{method}` requires a continuous prefix starting at step 0.")
    if isinstance(probe_depth, bool) or not isinstance(probe_depth, int) or not 1 <= probe_depth <= num_layers:
        raise ValueError(f"`c3cache_probe_depth` must be an integer in [1, {num_layers}].")
