from fastapi import FastAPI
from server.coordinator import (
    FederatedCoordinator
)

class LeaderService:

    def __init__(
        self,
        node,
        peer_table
    ):
        self.node = node
        self.peer_table = peer_table
        self.coordinator = FederatedCoordinator(
            cluster_id=node.cluster_id,
            num_workers=node.num_workers
        )
        print(
            f"[LEADER SERVICE] Initialized | node={node.node_id[:8]} "
            f"| cluster={node.cluster_id} | workers={self.coordinator.num_clients}"
        )

        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        def root():
            return {
                "message":"Leader Running",
                "node_id":self.node.node_id,
                "cluster_id":self.node.cluster_id,
                "role":self.node.consensus_state
            }
        
        @self.app.get("/get_model")
        def get_model():
            return self.coordinator.get_model()

        @self.app.get("/model_status")
        def model_status():
            return self.coordinator.get_status()
        
        @self.app.post("/send_update")
        def send_update(update:dict):
            return self.coordinator.receive_update(update)
        
        @self.app.post("/aggregate")
        def aggregate():
            return self.coordinator.aggregate_updates()
        
