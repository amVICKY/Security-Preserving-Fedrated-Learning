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
            }
            data = json.dumps(message).encode()
            sock.sendto(data,("255.255.255.255",self.DISCOVERY_PORT ))
            
            time.sleep(5)
    
    def browse(self):
        from .node import Node
        sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(
            ("",self.DISCOVERY_PORT)
        )
        while True:
            data, addr = sock.recvfrom(4096)
            message = json.loads(data.decode())
            
            if message["node_id"] == self.node.node_id:
                continue

            peer = Node(
                node_id=message["node_id"],
                ip = message["ip"],
                port=message["port"],
                last_seen=datetime.now()
            )

            # self.peer_table.add_peer(peer)
            # print(f"Discovered {peer.node_id} \n at {peer.ip}:{peer.port}")
            if peer.node_id in self.peer_table.peers:
                self.peer_table.update_peer(peer.node_id)
                print(f"Updated the last_seen time of {peer.node_id} at {peer.ip}:{peer.port}")
            else:
                self.peer_table.add_peer(peer)
                print(f"Discovered {peer.node_id} at {peer.ip}:{peer.port}")

    
    def cleanup(self):
        while True:
            now = datetime.now()
            for peer in list(self.peer_table.list_peer()):
                age = (now - peer.last_seen).total_seconds()
                if age > 15:
                    print(f"!!!!!!!!!Removing stale peer {peer.node_id}!!!!!!!!!!!!!!!")
                    self.peer_table.remove_peer(peer.node_id)
            time.sleep(5)
    
    def start(self):
        threading.Thread(target=self.advertise,daemon=True).start()
        threading.Thread(target=self.browse,daemon=True).start()
        threading.Thread(target=self.cleanup,daemon=True).start()