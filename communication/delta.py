import copy


def compute_delta(global_weights, local_weights):
    
    delta = copy.deepcopy(local_weights)
    for key in global_weights.keys():
        if key not in local_weights:
            raise KeyError(f"Missing parameter {key} in local_weights")

        local_tensor = local_weights[key].to(
            device=global_weights[key].device,
            dtype=global_weights[key].dtype
        )
        delta[key] = local_tensor - global_weights[key]
    return delta


def apply_delta(global_weights, averaged_delta):
    
    updated_weights = copy.deepcopy(global_weights)
    for key in global_weights.keys():
        updated_weights[key] += averaged_delta[key].to(updated_weights[key].device)
    return updated_weights