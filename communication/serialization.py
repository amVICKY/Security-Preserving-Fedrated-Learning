import io
import base64
import torch

def serialize_weights(weights):
    
    buffer = io.BytesIO()
    torch.save(weights,buffer)
    binary_data = (buffer.getvalue())
    encoded = (base64.b64encode(binary_data).decode("utf-8"))
    
    return encoded


def deserialize_weights(encoded_weights):

    binary_data = (base64.b64decode(encoded_weights))
    buffer = io.BytesIO(binary_data)
    weights = torch.load(buffer,weights_only=True)
    
    return weights