import copy

def compute_delta(global_weights, local_weights):
    
    delta = copy.deepcopy(local_weights)
    for key in global_weights.keys():
        delta[key] = (local_weights[key] - global_weights[key])
    return delta

def apply_delta(global_weights,averaged_delta):
    
    updated_weights = copy.deepcopy(global_weights)
    for key in global_weights.keys():
        updated_weights[key] += averaged_delta[key].to(updated_weights[key].device)
    return updated_weights