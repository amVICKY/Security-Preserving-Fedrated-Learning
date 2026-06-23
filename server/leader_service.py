from fastapi import FastAPI
from server.coordinator import (
    FederatedCoordinator,
    GLOBAL_SERVER_URL
)
from communication.model_sync import ModelSync

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

        # Shared initialization: pull the canonical model from the global server so
        # EVERY cluster starts in the same weight-space basin. Cross-cluster FedAvg
        # only produces a meaningful average when the clusters' weights are aligned;
        # independent random inits land in different basins and average to mush.
        try:
            global_weights, global_version = ModelSync.download_model(GLOBAL_SERVER_URL)
            self.coordinator.set_global_weights(global_weights)
            print(
                f"[LEADER SERVICE] Initialized from GLOBAL model (v{global_version}) "
                f"| node={node.node_id[:8]} | cluster={node.cluster_id} "
                f"| workers={self.coordinator.num_clients}"
            )
        except Exception as e:
            print(
                f"[LEADER SERVICE] Could not fetch global init ({e}); using local random init "
                f"| node={node.node_id[:8]} | cluster={node.cluster_id} "
                f"| workers={self.coordinator.num_clients}"
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
        
