import torch

def serialize_weights(weights):
    
    serialized = {}
    for key, value in weights.items():
        serialized[key] = value.tolist()

    return serialized


def deserialize_weights(weights):

    deserialize = {}
    for key,value in weights.items():
        deserialize[key] = torch.tensor(value)
    
    return deserialize