import os
import time
import torch
from models.cnn import CNN
from data.dataset import (
    get_dataloaders,
    get_client_dataloader
)
from utils.config import load_config
from utils.partition import create_non_iid_partition
from client.local_trainer import local_train
from communication.delta import compute_delta
from communication.model_sync import ModelSync
from communication.clock import LamportClock, VectorClock


class FederatedClient:

    def __init__(
        self,
        client_id,
        node,
        peer_table,
        consensus
    ):
        self.client_id = client_id
        self.node = node
        self.peer_table = peer_table
        self.consensus = consensus
        self.config = load_config()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.lamport = LamportClock()
        self.vector_clock = VectorClock(node.node_id)
        self.base_version = 0  # global model version this round was trained from

    def get_leader_url(self):
        leader_id = self.consensus.get_leader()
        if leader_id is not None:
            leader = self.peer_table.get_peer(leader_id)
            if leader is not None:
                return f"http://{leader.ip}:{leader.api_port}"

        # Fallback: consensus.get_leader() may be None before set_leader() propagates
        leader = self.peer_table.get_cluster_leader(self.node.cluster_id)
        if leader is None:
            return None

        return f"http://{leader.ip}:{leader.api_port}"

    def resolve_target_url(self):
        if self.node.consensus_state == "follower":
            return self.get_leader_url()
        return None
    
    def train_round(self):
        if self.node.consensus_state != "follower":
            return
        print(f"[CLIENT] Downloading global model from leader...")
        global_weights, self.base_version = ModelSync.download_model(self.resolve_target_url())
        print(f"[CLIENT] Downloaded global model | base_version={self.base_version}")

        model = CNN(num_classes=10).to(self.device)
        model.load_state_dict(global_weights)
        initial_weights = {key:value.clone() for key,value in model.state_dict().items()}

        train_loader, test_loader = (get_dataloaders(self.config))
        train_dataset = train_loader.dataset
        # Per-cluster worker count from the CLI (--num_workers); falls back to config.
        num_clients = self.node.num_workers or self.config["federated"]["num_clients"]

        partitions = create_non_iid_partition(train_dataset, num_clients)

        # Rank this node among all followers in the cluster (sorted for consistency).
        # Using rank instead of client_id means no partition slot is wasted
        # regardless of which node becomes leader.
        cluster_followers = sorted(
            [self.node.node_id] + [
                peer.node_id
                for peer in self.peer_table.list_peer()
                if peer.cluster_id == self.node.cluster_id
                and peer.consensus_state == "follower"
            ]
        )
        worker_rank = cluster_followers.index(self.node.node_id)
        client_loader = get_client_dataloader(self.config, train_dataset, partitions[worker_rank])
        print(f"[CLIENT] Local training started | rank={worker_rank}/{num_clients} | partition_size={len(partitions[worker_rank])} | device={self.device}")

        model,metrics = local_train(model,client_loader,test_loader,self.device,self.config["federated"]["local_epochs"])
        print(f"[CLIENT] Local training complete | metrics={metrics}")

        delta = compute_delta(initial_weights,model.state_dict())

        ts = self.lamport.tick()
        vc = self.vector_clock.tick()
        # Idempotency key for exactly-once: derived from the logical clock so a retried
        # upload of THIS update carries the same id and is deduped at the aggregator.
        update_id = f"{self.node.node_id}:{ts}"
        print(f"[CLIENT] Uploading update to leader | update_id={update_id} | base_version={self.base_version} | vc={vc}")

        response = ModelSync.upload_update(
            self.resolve_target_url(),
            delta,
            node_id=self.node.node_id,
            lamport_ts=ts,
            vector_clock=vc,
            base_version=self.base_version,
            update_id=update_id,
            cluster_id=self.node.cluster_id
        )
        print(f"[CLIENT] Upload response: {response}")

        # Causal lockstep (option 2): don't start the next round until the global
        # model's vector clock confirms THIS node's contribution (lamport_ts=ts) has
        # been aggregated. Guarantees the next download is the latest merged model,
        # so we never train on a stale pre-aggregation base and never get rejected.
        status = response.get("status", "")
        if "rejected" in status or "duplicate" in status:
            print(f"[CLIENT] Update not merged ({status}) — re-syncing on next download")
        else:
            self.wait_for_merge(ts)

    def wait_for_merge(self, my_ts):
        """Poll the leader's model vector clock until it has merged our lamport_ts."""
        timeout = self.config["async_training"]["window_seconds"] * 2 + 5
        deadline = time.time() + timeout
        my_id = self.node.node_id

        while time.time() < deadline:
            target_url = self.resolve_target_url()
            if target_url is None:
                time.sleep(1)
                continue
            try:
                version, model_vc = ModelSync.get_model_status(target_url)
            except Exception:
                time.sleep(1)
                continue

            merged_ts = model_vc.get(my_id, 0)
            if merged_ts >= my_ts:
                print(f"[CLIENT] Contribution merged | my_ts={my_ts} | model_version={version} | vc_self={merged_ts}")
                return
            print(f"[CLIENT] Waiting for merge | my_ts={my_ts} | vc_self={merged_ts} | model_version={version}")
            time.sleep(1)

        print(f"[CLIENT] Merge wait timed out after {timeout}s — proceeding with latest model")

    def run(self):
        num_rounds = self.config["federated"]["num_rounds"]
        local_epochs = self.config["federated"]["local_epochs"]
        print(f"[CLIENT] Started | node={self.node.node_id[:8]} | client_id={self.client_id} | cluster={self.node.cluster_id}")
        print(f"[CLIENT] Training plan: {num_rounds} rounds x {local_epochs} epochs each")

        round_num = 0
        while self.node.consensus_state == "follower" and round_num < num_rounds:
            target_url = self.resolve_target_url()
            if target_url is None:
                print(f"[CLIENT] No leader found for cluster={self.node.cluster_id}, retrying in 3s...")
                time.sleep(3)
                continue
            print(f"[CLIENT] --- Round {round_num + 1}/{num_rounds} | leader={target_url} ---")
            try:
                self.train_round()
                round_num += 1
            except Exception as e:
                print(f"[CLIENT] Round {round_num + 1} failed: {e}, retrying in 3s...")
                time.sleep(3)

        if round_num >= num_rounds:
            print(f"[CLIENT] All {num_rounds} rounds complete | node={self.node.node_id[:8]}")
            print(f"[CLIENT] Shutting down worker node")
            os._exit(0)