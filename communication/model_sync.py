import requests
from communication.serialization import (
    deserialize_weights,
    serialize_weights
)

class ModelSync:
    
    @staticmethod
    def download_model(url):
        response = requests.get(f"{url}/get_model")
        weights = response.json()["weights"]
        return deserialize_weights(weights)
    
    @staticmethod
    def upload_update(url,weights):
        response = requests.post(
            f"{url}/send_update",
            json={
                "weights":serialize_weights(weights)
            }
        )
        return response.json()