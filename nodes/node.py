import argparse
from dataclasses import dataclass
from datetime import datetime
import uuid

from .peer_table import PeerTable
from .discovery import DiscoveryService
import time

@dataclass # This help automatically generates the special boilerplate method like __init__(),__repr__()
class Node:
    node_id: str
    ip: str
    port: int
    last_seen: datetime

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
    parser.add_argument("--port",type=int,required=True)
    args = parser.parse_args()

    node = Node(
        node_id=str(uuid.uuid4()),
        ip="127.0.0.1",
        port=args.port,
        last_seen=datetime.now()
    )
    peer_table = PeerTable()
    discovery = DiscoveryService(
        node = node,
        peer_table=peer_table
    )
    discovery.start()
    print(f"Started {node}")
    while True:
        time.sleep(1)
    

    # nodes = [Node(str(uuid.uuid4()),ip="127.0.0.1",port=(5000+i),last_seen=datetime.now()) for i in range(5)]
    # for node in nodes:
    #     print(node)

