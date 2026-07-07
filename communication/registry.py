import requests

from utils.env import global_server_url

def register_node(node):
    SERVER_URL = global_server_url()
    payload = {
        "node_id":node.node_id,
        "ip":node.ip,
        "port":node.port,
        "api_port":node.api_port,
        "consensus_state":node.consensus_state,
        "latency":node.simulated_latency,
        "cluster_id":node.cluster_id
    }

    try:
        response = requests.post(
            f"{SERVER_URL}/register",
            json=payload,
            timeout = 3
        )

        return response.json()
    
    except Exception:
        return None