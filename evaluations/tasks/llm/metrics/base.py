from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RequestRecord:
    """Raw measurement data for a single inference request.

    The benchmark loop populates the timestamp and token-count fields.
    Derived metrics (ttft, tpot, throughput) are computed on access from
    the raw timestamps so the caller never needs to calculate them.
    """

    request_id: str
    prompt: str
    t_start: float = 0.0
    t_first_token: float = 0.0
    t_last_token: float = 0.0
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0
    output_text: str = ""

    # -- derived metrics --------------------------------------------------

    @property
    def ttft(self) -> float:
        """Time to first token (seconds)."""
        if self.t_first_token > 0 and self.t_start > 0:
            return self.t_first_token - self.t_start
        return 0.0

    @property
    def tpot(self) -> float:
        """Time per output token (seconds).

        Measured as the decoding duration divided by the number of *inter-token
        gaps* (output_tokens - 1).  Returns 0 when fewer than 2 tokens were
        generated.
        """
        if (
            self.num_output_tokens > 1
            and self.t_last_token > 0
            and self.t_first_token > 0
        ):
            return (self.t_last_token - self.t_first_token) / (
                self.num_output_tokens - 1
            )
        return 0.0

    @property
    def throughput(self) -> float:
        """Output tokens per second (end-to-end)."""
        duration = self.t_last_token - self.t_start
        if duration > 0 and self.num_output_tokens > 0:
            return self.num_output_tokens / duration
        return 0.0

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat dict for the ``raw`` array in the output JSON."""
        return {
            "request_id": self.request_id,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_output_tokens": self.num_output_tokens,
            "ttft": round(self.ttft, 6),
            "tpot": round(self.tpot, 6),
            "throughput": round(self.throughput, 2),
        }


class RequestStore:
    """Thread-safe storage for per-request measurement records.

    The benchmark loop calls :meth:`add` from any thread; collectors read
    via the :attr:`records` property after the run completes.
    """

    def __init__(self) -> None:
        self._records: list[RequestRecord] = []
        self._lock = threading.Lock()

    def add(self, record: RequestRecord) -> None:
        with self._lock:
            self._records.append(record)

    @property
    def records(self) -> list[RequestRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


class BaseCollector(ABC):
    """Interface every metric collector must implement."""

    @abstractmethod
    def start(self) -> None:
        """Called before the benchmark begins."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Called after the benchmark ends."""
        ...

    @abstractmethod
    def results(self) -> dict[str, Any]:
        """Return collected metrics as a dict."""
        ...
