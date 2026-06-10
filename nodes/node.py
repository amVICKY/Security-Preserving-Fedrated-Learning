import argparse
from dataclasses import dataclass
from datetime import datetime
import uuid
from typing import Optional

from .peer_table import PeerTable
from .discovery import DiscoveryService
import time

from client.client import FederatedClient

from communication.registry import (
    register_node
)

@dataclass # This help automatically generates the special boilerplate method like __init__(),__repr__()
class Node:
    node_id: str
    ip: str
    port: int
    last_seen: datetime

    cluster_id:str

    region:Optional[str] = None
    simulated_latency:Optional[int] = None
    role:Optional[str] = None


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
    parser.add_argument("--role",type=str,required=True)
    args = parser.parse_args()

    node = Node(
        node_id=str(uuid.uuid4()),
        ip="127.0.0.1",
        port=args.port,
        last_seen=datetime.now(),

        cluster_id=args.cluster_id,

        region=args.region,
        simulated_latency=args.latency,
        role = args.role,
    )
    register_node(node)
    peer_table = PeerTable()
    discovery = DiscoveryService(
        node = node,
        peer_table=peer_table
    )
    discovery.start()
    # print(f"Started {node}")
    from server.leader_service import LeaderService
    import uvicorn

    if node.role == "leader":
        leader_service = LeaderService(node=node,peer_table=peer_table)
        uvicorn.run(leader_service.app,host="0.0.0.0",port=node.port)
    elif node.role == "worker":
        client = FederatedClient(
            client_id=0,
            node=node,
            peer_table=peer_table
        )

    while True:
        leader = peer_table.get_cluster_leader(node.cluster_id)
        if leader:
            client.run()
            break
        print("Waiting for leader")
        time.sleep(5)
    # nodes = [Node(str(uuid.uuid4()),ip="127.0.0.1",port=(5000+i),last_seen=datetime.now()) for i in range(5)]
    # for node in nodes:
    #     print(node)

