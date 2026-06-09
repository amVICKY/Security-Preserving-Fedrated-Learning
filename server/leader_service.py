from fastapi import FastAPI
from communication.model_sync import ModelSync
from server.coordinator import (
    FederatedCoordinator
)

SERVER_URL = ("http://127.0.0.1:8000")
class LeaderService:

    def __init__(
        self,
        node,
        peer_table
    ):
        self.node = node
        self.peer_table = peer_table
        self.global_weights = ModelSync.download_model(SERVER_URL)

        self.coordinator = (FederatedCoordinator())
        self.coordinator.set_global_weights(self.global_weights)

        self.app = FastAPI()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.get("/")
        def root():
            return {
                "message":"Leader Running",
                "node_id":self.node.node_id,
                "cluster_id":self.node.cluster_id,
                "role":self.node.role
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
        
