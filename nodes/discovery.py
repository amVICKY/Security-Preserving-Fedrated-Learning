import socket
import json
import threading
import time
from datetime import datetime

import requests

from utils.env import discovery_mode, global_server_url, registry_poll_interval
# from nodes.node import Node

class DiscoveryService:

    DISCOVERY_PORT = 9999
    def __init__(self,node,peer_table):
        self.node = node
        self.peer_table = peer_table

    # ------------------------------------------------------------------ UDP
    # Original LAN-broadcast discovery. Works only on a shared L2 segment
    # (i.e. all nodes on one host), so this is the loopback-mode path.

    def advertise(self):
        sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1)

        while True:
            message = {
                "node_id":self.node.node_id,
                "ip":self.node.ip,
                "port":self.node.port,
                "api_port":self.node.api_port,

                "cluster_id":self.node.cluster_id,

                "region":self.node.region,
                "simulated_latency":self.node.simulated_latency,
                "consensus_state":self.node.consensus_state
            }
            data = json.dumps(message).encode()
            sock.sendto(data,("255.255.255.255",self.DISCOVERY_PORT))
            time.sleep(5)

    def browse(self):
        from .node import Node
        sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(("",self.DISCOVERY_PORT))
        while True:
            data, addr = sock.recvfrom(4096)
            message = json.loads(data.decode())

            if message["node_id"] == self.node.node_id:
                continue

            peer = Node(
                node_id=message["node_id"],
                ip=message["ip"],
                port=message["port"],
                api_port=message["api_port"],
                last_seen=datetime.now(),

                cluster_id=message["cluster_id"],

                region=message["region"],
                simulated_latency=message["simulated_latency"],
                consensus_state=message["consensus_state"]
            )

            if peer.node_id in self.peer_table.peers:
                self.peer_table.update_peer(peer.node_id, consensus_state=peer.consensus_state)
            else:
                self.peer_table.add_peer(peer)
                print(f"[DISCOVERY] New peer: {peer.node_id[:8]} | state={peer.consensus_state} | cluster={peer.cluster_id} | addr={peer.ip}:{peer.port}")

    # ------------------------------------------------------------- REGISTRY
    # Docker/k8s-friendly discovery: the global server already receives every
    # node's role changes (node.py calls register_node), so /registered_nodes
    # is a live registry. We (a) heartbeat ourselves into it and (b) poll it to
    # learn peers. No broadcast/multicast needed -> identical in Docker and k8s.

    def registry_heartbeat(self):
        """Periodically re-register self so the server's TTL keeps us 'alive'."""
        from communication.registry import register_node
        interval = registry_poll_interval()
        while True:
            try:
                register_node(self.node)
            except Exception as e:
                print(f"[DISCOVERY] Heartbeat failed: {e}")
            time.sleep(interval)

    def registry_browse(self):
        """Poll the global registry and reconcile the peer table."""
        from .node import Node
        url = f"{global_server_url()}/registered_nodes"
        interval = registry_poll_interval()
        while True:
            try:
                resp = requests.get(url, timeout=3)
                registry = resp.json()
            except Exception as e:
                print(f"[DISCOVERY] Registry poll failed: {e}")
                time.sleep(interval)
                continue

            for node_id, info in registry.items():
                if node_id == self.node.node_id:
                    continue
                if node_id in self.peer_table.peers:
                    self.peer_table.update_peer(node_id, consensus_state=info.get("consensus_state"))
                else:
                    peer = Node(
                        node_id=info["node_id"],
                        ip=info["ip"],
                        port=info["port"],
                        api_port=info["api_port"],
                        last_seen=datetime.now(),
                        cluster_id=info["cluster_id"],
                        simulated_latency=info.get("latency"),
                        consensus_state=info.get("consensus_state"),
                    )
                    self.peer_table.add_peer(peer)
                    print(f"[DISCOVERY] New peer: {peer.node_id[:8]} | state={peer.consensus_state} | cluster={peer.cluster_id} | addr={peer.ip}:{peer.port}")
            time.sleep(interval)

    # ----------------------------------------------------------- SHARED
    def cleanup(self):
        while True:
            now = datetime.now()
            for peer in list(self.peer_table.list_peer()):
                age = (now - peer.last_seen).total_seconds()
                if age > 15:
                    print(f"[CLEANUP] Removing stale peer: {peer.node_id[:8]} (last seen {age:.1f}s ago) | was {peer.consensus_state}")
                    self.peer_table.remove_peer(peer.node_id)
            time.sleep(5)

    def leader_lookup(self):
        while True:
            if self.node.consensus_state == "follower":
                leader = self.peer_table.get_cluster_leader(self.node.cluster_id)
                if leader is not None:
                    print(f"[LOOKUP] Cluster leader: {leader.node_id[:8]} at {leader.ip}:{leader.port}")
                else:
                    print(f"[LOOKUP] No leader found for cluster {self.node.cluster_id}")
            elif self.node.consensus_state == "leader":
                print(f"[LOOKUP] I am the leader of cluster {self.node.cluster_id}")
            time.sleep(5)

    def start(self):
        mode = discovery_mode()
        if mode == "registry":
            print(f"[DISCOVERY] Mode: registry | server={global_server_url()}")
            threading.Thread(target=self.registry_heartbeat,daemon=True).start()
            threading.Thread(target=self.registry_browse,daemon=True).start()
        else:
            print(f"[DISCOVERY] Mode: udp (broadcast :{self.DISCOVERY_PORT})")
            threading.Thread(target=self.advertise,daemon=True).start()
            threading.Thread(target=self.browse,daemon=True).start()
        threading.Thread(target=self.cleanup,daemon=True).start()
        # threading.Thread(target=self.leader_lookup,daemon=True).start()
