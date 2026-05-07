"""Timing and memory tracking utilities."""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import psutil


_PROCESS = psutil.Process()
_BASELINE_RSS_MB: Optional[float] = None


def baseline_rss_mb() -> float:
    global _BASELINE_RSS_MB
    if _BASELINE_RSS_MB is None:
        _BASELINE_RSS_MB = _PROCESS.memory_info().rss / (1024 * 1024)
    return _BASELINE_RSS_MB


@dataclass
class MemorySampler:
    """Polls RSS in a background thread; returns peak marginal RSS (MB) above baseline."""

    interval_sec: float = 0.05
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    _peak_rss_mb: float = 0.0

    def start(self) -> None:
        baseline_rss_mb()  # ensure baseline initialized
        self._stop.clear()
        self._peak_rss_mb = _PROCESS.memory_info().rss / (1024 * 1024)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = _PROCESS.memory_info().rss / (1024 * 1024)
                if rss > self._peak_rss_mb:
                    self._peak_rss_mb = rss
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        peak_marginal = max(0.0, self._peak_rss_mb - baseline_rss_mb())
        return peak_marginal


@contextmanager
def measure_peak_memory():
    """Usage:  with measure_peak_memory() as m: ...; m.peak_mb
    """

    class _Holder:
        peak_mb: float = 0.0

    sampler = MemorySampler()
    holder = _Holder()
    sampler.start()
    try:
        yield holder
    finally:
        holder.peak_mb = sampler.stop()


def now() -> float:
    return time.perf_counter()
