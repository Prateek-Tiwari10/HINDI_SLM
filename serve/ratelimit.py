"""Per-client sliding-window rate limit (in memory; one uvicorn worker per replica)."""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, max_clients: int = 10_000) -> None:
        self.limit, self.window, self.max_clients = limit, window_s, max_clients
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> float | None:
        """None if the request is allowed (and counted); else seconds until the client may retry."""
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return round(self.window - (now - q[0]), 1)
            q.append(now)
            if len(self._hits) > self.max_clients:
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[k]
            return None


def client_key(headers, peer: str | None) -> str:
    """The client address. Azure Container Apps' ingress (Envoy) appends the address it saw to
    X-Forwarded-For, so the LAST entry is the one a client cannot forge; without the header, the peer."""
    xff = headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip()
    return peer or "unknown"
