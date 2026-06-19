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
        print(f"[CONSENSUS] Initialized | self={self_address} | partners={partner_addresses}")

    @replicated
    def set_leader(self,node_id):
        self.current_leader = node_id
        print(f"[CONSENSUS] Leader committed -> {node_id[:8]}")

    def get_leader(self):
        return self.current_leader

    def update_role(self):
        if self._isLeader():
            self.node.consensus_state = "leader"
            if self.current_leader != self.node.node_id:
                print(f"[CONSENSUS] This node elected as leader | node_id={self.node.node_id[:8]}")
                self.set_leader(self.node.node_id)
        else:
            self.node.consensus_state = "follower"