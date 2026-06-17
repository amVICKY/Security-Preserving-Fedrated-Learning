import argparse
from dataclasses import dataclass
from datetime import datetime
import uuid
from typing import Optional
import time

from .peer_table import PeerTable
from .discovery import DiscoveryService
from .consensus import ClusterConsensus

from client.client import FederatedClient

from communication.registry import (
    register_node
)

from server.leader_service import LeaderService

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
    args = parser.parse_args()

    node = Node(
        node_id=str(uuid.uuid4()),
        ip="127.0.0.1",
        port=args.port,
        api_port=args.port + 1000,
        last_seen=datetime.now(),

        cluster_id=args.cluster_id,

        region=args.region,
        simulated_latency=args.latency,
        consensus_state="follower"
    )
    register_node(node)
    peer_table = PeerTable()
    discovery = DiscoveryService(
        node = node,
        peer_table=peer_table
    )
    discovery.start()

    print("Waiting for peer discovery")
    time.sleep(5)

    self_address = f"{node.ip}:{node.port}"
    peers = peer_table.list_peer()
    partner_addresses = [
        f"{peer.ip}:{peer.port}" 
        for peer in peers 
        if peer.cluster_id == node.cluster_id 
        and peer.node_id != node.node_id
    ]

    print("="*20)
    print(f"My Raft peers:{partner_addresses}")
    print("="*20)

    consensus = ClusterConsensus(
        self_address=self_address,
        partner_addresses=partner_addresses,
        node = node
    )

    leader_service_started = False
    client_started = False
    client = None

    while True:
        consensus.update_role()
        print(f"Node state:{node.consensus_state}")

        if node.consensus_state == "leader" and not leader_service_started:
            leader_service = LeaderService(
                node = node,
                peer_table = peer_table
            )

            import uvicorn
            import threading

            threading.Thread(
                target = lambda:uvicorn.run(
                    leader_service.app,
                    host="0.0.0.0",
                    port = node.api_port #+1000
                ),
                daemon=True
            ).start()
            leader_service_started = True
            print("Leader service started")

        elif node.consensus_state == "follower" and not client_started:
            client = FederatedClient(
                client_id=0,
                node=node,
                peer_table=peer_table,
                consensus = consensus
            )
            client_started = True
            print("client Initialized")
        
        time.sleep(2)