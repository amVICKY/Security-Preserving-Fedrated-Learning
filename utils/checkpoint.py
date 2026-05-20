import torch
import os

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    torch.save(model.state_dict(),path)

def load_checkpoint(model,path):
    model.load_state_dict(torch.load(path))
    return model