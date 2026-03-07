import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._lock = threading.Lock()
        self._timestamps: deque = deque()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.period
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_calls:
                sleep_for = self.period - (now - self._timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                cutoff = now - self.period
                while self._timestamps and self._timestamps[0] < cutoff:
                    self._timestamps.popleft()
            self._timestamps.append(time.monotonic())
