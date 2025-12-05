"""
Lightweight profiling utilities for the pipeline.

Design goals:
- Fast no-op when profiling is disabled.
- Cheap aggregation (count, total, min, max) with a bounded deque of samples
  for percentile estimation.
- Minimal external dependencies (standard library only).
"""
from __future__ import annotations

import json
import math
import statistics
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional


class StageStats:
    """Tracks basic stats for a single stage."""

    __slots__ = (
        "name",
        "max_samples",
        "enabled",
        "count",
        "total_ms",
        "min_ms",
        "max_ms",
        "samples",
    )

    def __init__(self, name: str, max_samples: int, enabled: bool) -> None:
        self.name = name
        self.max_samples = max_samples
        self.enabled = enabled
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = math.inf
        self.max_ms = 0.0
        self.samples: deque[float] = deque(maxlen=max_samples)

    def record(self, value_ms: float) -> None:
        if not self.enabled:
            return
        self.count += 1
        self.total_ms += value_ms
        if value_ms < self.min_ms:
            self.min_ms = value_ms
        if value_ms > self.max_ms:
            self.max_ms = value_ms
        self.samples.append(value_ms)

    def percentile(self, pct: float) -> Optional[float]:
        if not self.samples:
            return None
        arr = sorted(self.samples)
        idx = int((pct / 100.0) * (len(arr) - 1))
        return arr[idx]

    def summary(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "mean_ms": None,
                "min_ms": None,
                "max_ms": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "std_ms": None,
            }
        mean_ms = self.total_ms / self.count
        std_ms = None
        # Snapshot samples to avoid concurrent mutation while computing stats
        samples_snapshot = list(self.samples)
        if len(samples_snapshot) > 1:
            std_ms = statistics.pstdev(samples_snapshot)
        return {
            "count": self.count,
            "mean_ms": mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "std_ms": std_ms,
        }

    def short_line(self) -> str:
        s = self.summary()
        if s["count"] == 0:
            return f"{self.name}: n=0"
        parts = [
            f"{self.name}: n={s['count']}",
            f"mean={s['mean_ms']:.2f}ms",
            f"p90={s['p90']:.2f}ms" if s["p90"] is not None else "p90=na",
            f"max={s['max_ms']:.2f}ms",
        ]
        return " | ".join(parts)


class ProfilingManager:
    """Registry of StageStats with helper APIs."""

    def __init__(self, enabled: bool = True, max_samples: int = 10000, log_interval: float = 5.0) -> None:
        self.enabled = enabled
        self.max_samples = max_samples
        self.log_interval = log_interval
        self._stages: Dict[str, StageStats] = {}
        self._last_log_time = time.perf_counter()

    def stage(self, name: str) -> StageStats:
        if name not in self._stages:
            self._stages[name] = StageStats(name, self.max_samples, self.enabled)
        return self._stages[name]

    def record(self, name: str, value_ms: float) -> None:
        if not self.enabled:
            return
        self.stage(name).record(value_ms)

    @contextmanager
    def time_block(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.record(name, (t1 - t0) * 1000.0)

    def maybe_log(self, logger, stages: Optional[Iterable[str]] = None, interval_sec: Optional[float] = None, keys: Optional[Iterable[str]] = None) -> None:
        if not self.enabled:
            return
        # Backward compatibility: accept keys= as an alias for stages=
        if stages is None and keys is not None:
            stages = keys
        if stages is None:
            return
        interval = interval_sec if interval_sec is not None else self.log_interval
        now = time.perf_counter()
        if (now - self._last_log_time) < interval:
            return
        self._last_log_time = now
        try:
            lines = [self._stages[k].short_line() for k in stages if k in self._stages]
        except Exception as exc:
            logger.warning("Profiler maybe_log failed: %s", exc)
            return
        if lines:
            logger.info(" | ".join(lines))

    def dump_all_stats(self, logger, output_path: Optional[str] = None) -> Dict[str, Any]:
        data = {name: stage.summary() for name, stage in self._stages.items()}
        logger.info("==== Profiling summary ====")
        for name, stats in data.items():
            logger.info("%s -> %s", name, stats)
        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                logger.info("Profile summary dumped to %s", output_path)
            except Exception as exc:
                logger.warning("Failed to dump profiling data to %s: %s", output_path, exc)
        return data
