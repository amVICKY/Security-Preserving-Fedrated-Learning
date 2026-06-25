"""
Downstream resiliency for inter-node calls (UDS Ch16).

Wraps every HTTP call with:
  - timeout            : a dead peer fails fast instead of hanging forever
  - retry + backoff    : transient failures (connection/timeout/5xx) are retried
  - FULL JITTER        : randomized backoff so lockstep workers don't all retry a
                         recovering peer in the same instant (retry amplification)
  - circuit breaker    : after repeated failures to an endpoint, fail fast for a
                         cooldown instead of hammering a peer that is clearly down

Safe to retry because every write is idempotent (Phase 5 Goal 3): /send_update is
deduped by update_id, /cluster_update / /register are last-write-wins.
"""

import random
import threading
import time
from urllib.parse import urlparse

import requests


class CircuitOpenError(Exception):
    """Raised when a circuit is open — fail fast rather than call a known-bad peer."""


class CircuitBreaker:
    """Per-endpoint breaker. CLOSED normally; after `failure_threshold` consecutive
    failures it OPENs for `reset_timeout` seconds (fail fast), then HALF-OPENs to let a
    trial through. A success closes it; a failure re-opens it."""

    def __init__(self, failure_threshold=5, reset_timeout=10.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._opened_at = None
        self._lock = threading.Lock()

    @property
    def state(self):
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if time.time() - self._opened_at >= self.reset_timeout:
                return "half-open"
            return "open"

    def allow(self):
        if self.state == "open":
            raise CircuitOpenError("circuit open")

    def record_success(self):
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.time()


_breakers = {}
_breakers_lock = threading.Lock()


def _breaker_for(url):
    """One breaker per endpoint (host+path) so a failing /backup_update can't trip
    the main-path /cluster_update on the same host."""
    p = urlparse(url)
    key = f"{p.netloc}{p.path}"
    with _breakers_lock:
        b = _breakers.get(key)
        if b is None:
            b = CircuitBreaker()
            _breakers[key] = b
        return b


def breaker_states():
    """Snapshot of every breaker's state (for health inspection)."""
    with _breakers_lock:
        return {k: b.state for k, b in _breakers.items()}


# (connect, read): unreachable host fails in ~3s; big weight transfers get 30s
DEFAULT_TIMEOUT = (3.05, 30)


def resilient_request(method, url, *, timeout=DEFAULT_TIMEOUT, max_retries=3,
                      backoff_base=0.2, backoff_cap=5.0, **kwargs):
    """HTTP call with timeout + retry(exponential backoff + full jitter) + circuit breaker.

    Retries transient failures (connection errors, timeouts, 5xx). A 4xx is returned
    as-is (retrying won't help, and the server is clearly alive). Raises CircuitOpenError
    immediately if the endpoint's circuit is open.
    """
    breaker = _breaker_for(url)
    attempt = 0
    while True:
        breaker.allow()  # fail fast if circuit is open
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code >= 500:
                raise requests.exceptions.HTTPError(f"server error {resp.status_code}")
            breaker.record_success()
            return resp
        except requests.exceptions.RequestException:
            breaker.record_failure()
            attempt += 1
            if attempt > max_retries:
                raise
            backoff = min(backoff_cap, backoff_base * (2 ** attempt))
            time.sleep(random.uniform(0, backoff))  # full jitter
