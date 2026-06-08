# import copy
# import torch

# def federated_averaging(client_weights):

#     global_weights = copy.deepcopy(client_weights[0])
#     for key in global_weights.keys():
        
#         for i in range(1,len(client_weights)):
#             global_weights[key]+=client_weights[i][key]

#         global_weights[key] = torch.div(
#             global_weights[key],
#             len(client_weights)
#         )

#     return global_weights

import copy
import torch
import random


def poly_aggregation(client_weights, sample_ratio=1.0):

    """
    PolyAgg Inspired Aggregation

    Args:
        client_weights : list of state_dicts
        sample_ratio   : percentage of parameters to aggregate
                         1.0 = full aggregation
                         0.2 = 20% sampled aggregation

    Returns:
        global_weights
    """

    if not client_weights:
        raise ValueError("client_weights must contain at least one model update")

    if not all(isinstance(weights, dict) for weights in client_weights):
        raise TypeError("client_weights must be a list of state_dict-like dictionaries")

    if sample_ratio < 0.0 or sample_ratio > 1.0:
        raise ValueError("sample_ratio must be between 0.0 and 1.0")

    base_weights = client_weights[0]
    expected_keys = set(base_weights.keys())
    if any(set(weights.keys()) != expected_keys for weights in client_weights[1:]):
        raise ValueError("All client updates must contain the same parameter keys")

    global_weights = copy.deepcopy(base_weights)

    for key in global_weights.keys():

        shape = global_weights[key].shape
        flat_tensor = global_weights[key].view(-1)
        total_params = flat_tensor.numel()

        if total_params == 0:
            global_weights[key] = flat_tensor.view(shape)
            continue

        if sample_ratio == 0.0:
            global_weights[key] = flat_tensor.view(shape)
            continue

        sampled_size = int(total_params * sample_ratio)
        if sampled_size == 0:
            sampled_size = 1
        if sampled_size > total_params:
            sampled_size = total_params

        sampled_indices = random.sample(
            range(total_params),
            sampled_size
        )

        aggregated = flat_tensor.clone()

        for idx in sampled_indices:

            poly_sum = 0.0

            for client_id, client_weights_i in enumerate(client_weights):

                client_flat = client_weights_i[key].view(-1)
                coefficient = client_id + 1
                poly_sum += coefficient * client_flat[idx]

            aggregated[idx] = poly_sum / len(client_weights)

        global_weights[key] = aggregated.view(shape)

    return global_weights