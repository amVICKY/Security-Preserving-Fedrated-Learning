from datetime import datetime
# from .node import Node

class PeerTable:

    def __init__(self):
        self.peers = {}

    def add_peer(self,node):
        self.peers[node.node_id] = node

    def update_peer(self, node_id, consensus_state=None):
        if node_id in self.peers:
            self.peers[node_id].update_last_seen()
            if consensus_state is not None and self.peers[node_id].consensus_state != consensus_state:
                old = self.peers[node_id].consensus_state
                self.peers[node_id].consensus_state = consensus_state
                print(f"[PEER TABLE] {node_id[:8]} state changed: {old} -> {consensus_state}")

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
    
    def get_cluster_leader(self,cluster_id):
        for peer in self.peers.values():
            if (peer.cluster_id == cluster_id and peer.consensus_state == "leader"):
                return peer
        return None
