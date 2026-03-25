from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseCollector, RequestStore

PERCENTILES: tuple[int, ...] = (50, 75, 90, 95, 99)


class LatencyCollector(BaseCollector):
    """Computes TTFT, TPOT and Throughput statistics from the request store.

    All values are derived from the raw timestamps recorded in each
    :class:`RequestRecord`.  Percentiles are calculated using
    ``numpy.percentile`` with linear interpolation.
    """

    def __init__(self, store: RequestStore) -> None:
        self._store = store

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def results(self) -> dict[str, Any]:
        records = self._store.records
        if not records:
            return {"latency": {}}

        ttfts = [r.ttft for r in records if r.ttft > 0]
        tpots = [r.tpot for r in records if r.tpot > 0]
        throughputs = [r.throughput for r in records if r.throughput > 0]

        return {
            "latency": {
                "ttft": _compute_stats(ttfts, unit="s"),
                "tpot": _compute_stats(tpots, unit="s"),
                "throughput": _compute_stats(throughputs, unit="tok/s"),
            },
        }


def _compute_stats(values: list[float], *, unit: str) -> dict[str, Any]:
    """Return mean / min / max and percentiles for *values*."""
    if not values:
        return {}

    arr = np.asarray(values, dtype=np.float64)
    stats: dict[str, Any] = {
        "unit": unit,
        "mean": round(float(np.mean(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }
    for p in PERCENTILES:
        stats[f"p{p}"] = round(float(np.percentile(arr, p)), 6)
    return stats
