from datetime import datetime
# from .node import Node

class PeerTable:
    
    def __init__(self):
        self.peers = {}

    def add_peer(self,node):
        self.peers[node.node_id] = node
    
    def update_peer(self,node_id):
        if node_id in self.peers:
            # self.peers[node_id].last_seen = datetime.now()
            self.peers[node_id].update_last_seen()

    def remove_peer(self,node_id):
        if node_id in self.peers:
            del self.peers[node_id]
    
    def list_peer(self,node_identity:bool=False):
        if node_identity:
            return list(self.peers.keys())
        else:
            return list(self.peers.values())
        
    def get_peer(self,node_id):
        return self.peers.get(node_id)