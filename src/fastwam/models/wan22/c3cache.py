"""Episode-local residual cache for chunked action inference.

The cache owns only Python control state and persistent residual tensors. It is
kept outside compiled denoising functions so torch.compile sees pure tensor
operations and a fresh video prefill can run for every chunk.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class C3Cache:
    signature: tuple[Any, ...] | None = None
    chunk_index: int = 0
    residuals: dict[int, Any] = field(default_factory=dict)
    full_steps: int = 0
    reused_steps: int = 0
    last_chunk_full_steps: int = 0
    last_chunk_reused_steps: int = 0

    def reset(self) -> None:
        self.signature = None
        self.chunk_index = 0
        self.residuals.clear()
        self.full_steps = 0
        self.reused_steps = 0
        self.last_chunk_full_steps = 0
        self.last_chunk_reused_steps = 0

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

    def commit(self, residuals: dict[int, Any], full_steps: int, reused_steps: int) -> None:
        self.residuals.update(residuals)
        self.chunk_index += 1
        self.full_steps += full_steps
        self.reused_steps += reused_steps
        self.last_chunk_full_steps = full_steps
        self.last_chunk_reused_steps = reused_steps

    def stats(self) -> dict[str, Any]:
        return {
            "completed_chunks": self.chunk_index,
            "full_steps": self.full_steps,
            "reused_steps": self.reused_steps,
            "last_chunk_full_steps": self.last_chunk_full_steps,
            "last_chunk_reused_steps": self.last_chunk_reused_steps,
            "cached_steps": tuple(sorted(self.residuals)),
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
