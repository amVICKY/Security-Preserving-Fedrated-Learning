from pysyncobj import SyncObj,replicated

class ClusterConsensus(SyncObj):

    def __init__(
        self,
        self_address,
        partner_addresses,
        node
    ):
        super().__init__(self_address,partner_addresses)
        self.node = node
        self.current_leader = None

    @replicated
    def set_leader(self,node_id):
        self.current_leader = node_id

    def get_leader(self):
        return self.current_leader
    
    def update_role(self):
        if self._isLeader():
            self.node.consensus_state = "leader"
            if self.current_leader != self.node.node_id:
                self.set_leader(self.node.node_id)
            
        else:
            self.node.consensus_state = "follower"