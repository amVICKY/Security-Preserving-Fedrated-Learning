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
        self.coordinator = FederatedCoordinator()
        print(f"[LEADER SERVICE] Initialized with fresh global model | node={node.node_id[:8]}")

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
        
        @self.app.post("/send_update")
        def send_update(update:dict):
            return self.coordinator.receive_update(update)
        
        @self.app.post("/aggregate")
        def aggregate():
            return self.coordinator.aggregate_updates()
        
