"""
Central network configuration, resolved from environment variables.

This is the single seam that lets the SAME code run in three environments:

  * loopback (default)  : no env vars set -> every value falls back to the
                          localhost behavior the swarm_launcher relies on.
  * docker-compose      : each service gets NODE_HOST / GLOBAL_SERVER_URL /
                          DISCOVERY_MODE=registry injected via the compose file.
  * kubernetes          : the same env vars are injected by the manifests
                          (NODE_HOST from the downward API / pod DNS name).

Nothing here is orchestrator-specific, so what works in Compose works in k8s.
Keep all address/port/discovery decisions in this module so no other file has
to read os.environ directly.
"""

import os


def _get(name, default):
    """Env var lookup that treats empty string as 'unset' (compose passes ""s)."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


# --- Global server ---------------------------------------------------------
# The FastAPI coordinator (server.app). In loopback this is localhost:8000; in
# Docker/k8s it is the service DNS name, e.g. http://global-server:8000.
def global_server_url():
    return _get("GLOBAL_SERVER_URL", "http://127.0.0.1:8000")


# --- Node identity / addressing -------------------------------------------
# The address a node ADVERTISES to its peers and to the registry. Must be an
# address other containers can route to. Loopback default keeps localhost runs
# working; in Docker/k8s set NODE_HOST to the pod/service DNS name (or podIP).
def node_host():
    return _get("NODE_HOST", "127.0.0.1")


# Optional explicit API (HTTP LeaderService) port. When unset we keep the
# historical convention of raft_port + 1000 so loopback runs are unchanged.
# In Docker/k8s each container owns its IP, so a fixed port can be set instead.
def api_port(raft_port):
    override = _get("API_PORT", None)
    if override is not None:
        return int(override)
    return raft_port + 1000


# --- Peer discovery --------------------------------------------------------
# "udp"      : original LAN broadcast discovery (works only on a shared L2, i.e.
#              a single host) -> the loopback default.
# "registry" : poll the global server's /registered_nodes to learn peers. Works
#              identically in Docker and k8s (no broadcast/multicast needed).
def discovery_mode():
    return _get("DISCOVERY_MODE", "udp").lower()


def registry_poll_interval():
    return float(_get("REGISTRY_POLL_INTERVAL", "5"))
