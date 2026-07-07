import argparse
from dataclasses import dataclass
from datetime import datetime
import uuid
from typing import Optional
import time
import threading
import uvicorn

from .peer_table import PeerTable
from .discovery import DiscoveryService
from .consensus import ClusterConsensus

from client.client import FederatedClient
from communication.registry import register_node
from server.leader_service import LeaderService
from utils.env import node_host, api_port as resolve_api_port

@dataclass # This help automatically generates the special boilerplate method like __init__(),__repr__()
class Node:
    node_id: str
    ip: str
    port: int       # raft port
    api_port:int    # HTTP port
    last_seen: datetime

    cluster_id:str

    region:Optional[str] = None
    simulated_latency:Optional[int] = None
    consensus_state:Optional[str] = None
    num_workers:Optional[int] = None   # training workers this cluster expects (set per-cluster at launch)


    def update_last_seen(self):
        self.last_seen = datetime.now()

    def __str__(self):
        return (
            f"Node(id={self.node_id}, "
            f"ip={self.ip}, "
            f"port={self.port}, "
            f"last_seen={self.last_seen})"
        )
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_id",type=str,required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--region",type=str,required=True)
    parser.add_argument("--latency",type=int,required=True)
    parser.add_argument("--client_id",type=int,required=True)
    parser.add_argument("--num_workers",type=int,default=None,
                        help="number of training workers in this cluster (falls back to config if omitted)")
    args = parser.parse_args()

    node = Node(
        node_id=str(uuid.uuid4()),
        ip=node_host(),                       # advertised address (loopback default; DNS name in Docker/k8s)
        port=args.port,
        api_port=resolve_api_port(args.port), # raft_port + 1000 by default; API_PORT env overrides
        last_seen=datetime.now(),

        cluster_id=args.cluster_id,

        region=args.region,
        simulated_latency=args.latency,
        consensus_state="follower",
        num_workers=args.num_workers
    )
    register_node(node)
    peer_table = PeerTable()
    discovery = DiscoveryService(
        node = node,
        peer_table=peer_table
    )
    discovery.start()

    print(f"[NODE] Waiting for peer discovery (10s)...")
    time.sleep(10)

    self_address = f"{node.ip}:{node.port}"
    peers = peer_table.list_peer()
    partner_addresses = [
        f"{peer.ip}:{peer.port}"
        for peer in peers
        if peer.cluster_id == node.cluster_id
        and peer.node_id != node.node_id
    ]

    print(f"[NODE] Raft peers found: {partner_addresses}")

    consensus = ClusterConsensus(
        self_address=self_address,
        partner_addresses=partner_addresses,
        node = node
    )

    print(f"[NODE] Waiting for Raft cluster to stabilize...")
    for _ in range(30):
        if consensus._isReady():
            break
        time.sleep(1)
    print(f"[NODE] Raft cluster ready | node={node.node_id[:8]}")

    leader_service_started = False
    client_started = False
    client_thread = None
    client = None
    previous_state = None

    while True:
        consensus.update_role()

        if node.consensus_state != previous_state:
            print(f"[NODE] Role change: {previous_state} -> {node.consensus_state} | node={node.node_id[:8]}")
            register_node(node)
            previous_state = node.consensus_state

        if node.consensus_state == "leader":
            # Stop client if promoted
            if client_started:
                client_started = False
                client_thread = None
                client = None
                print(f"[NODE] Training client stopped (node promoted to leader)")

            # Start leader service if not running
            if not leader_service_started:
                leader_service = LeaderService(
                    node = node,
                    peer_table=peer_table
                )
                threading.Thread(
                    target=lambda:uvicorn.run(
                        leader_service.app,
                        host="0.0.0.0",
                        port=node.api_port
                    ),
                    daemon=True
                ).start()
                leader_service_started=True
                print(f"[NODE] Leader service started on port {node.api_port}")

        elif node.consensus_state == "follower":
            thread_dead = client_thread is not None and not client_thread.is_alive()
            if not client_started or thread_dead:
                if thread_dead:
                    print(f"[NODE] Training thread died unexpectedly, restarting...")
                client = FederatedClient(
                    client_id=args.client_id,
                    node=node,
                    peer_table=peer_table,
                    consensus=consensus
                )
                client_thread = threading.Thread(
                    target=client.run,
                    daemon=True
                )
                client_thread.start()
                client_started=True
                print(f"[NODE] Training client started | node={node.node_id[:8]} | cluster={node.cluster_id}")

        time.sleep(2)