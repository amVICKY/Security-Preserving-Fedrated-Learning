import socket
import json
import threading
import time
from datetime import datetime
# from nodes.node import Node

class DiscoveryService:

    DISCOVERY_PORT = 9999
    def __init__(self,node,peer_table):
        self.node = node
        self.peer_table = peer_table

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
        threading.Thread(target=self.advertise,daemon=True).start()
        threading.Thread(target=self.browse,daemon=True).start()
        threading.Thread(target=self.cleanup,daemon=True).start()
        # threading.Thread(target=self.leader_lookup,daemon=True).start()
