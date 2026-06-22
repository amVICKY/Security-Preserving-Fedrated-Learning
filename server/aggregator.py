import copy
import torch


def staleness_weighted_averaging(client_weights, staleness_list):
    """
    Weighted FedAvg where stale updates contribute less.
    staleness_list[i] = current_version - base_version for client i.
    Weight formula: w_i = 1 / (1 + staleness_i), then normalized to sum=1.
    """
    # Edge case: nothing to aggregate
    if not client_weights:
        raise ValueError("staleness_weighted_averaging called with no client weights")

    # Edge case: mismatched lengths would silently misweight or IndexError
    if len(client_weights) != len(staleness_list):
        raise ValueError(
            f"length mismatch: {len(client_weights)} weights vs {len(staleness_list)} staleness values"
        )

    # Edge case: negative staleness (clock drift / version bug) -> clamp to 0 so 1/(1+s) never divides by zero
    safe_staleness = [max(0, s) for s in staleness_list]

    raw_weights = [1.0 / (1.0 + s) for s in safe_staleness]
    total = sum(raw_weights)
    norm_weights = [w / total for w in raw_weights]

    global_weights = copy.deepcopy(client_weights[0])
    for key in global_weights.keys():
        global_weights[key] = sum(
            client_weights[i][key] * norm_weights[i]
            for i in range(len(client_weights))
        )

    return global_weights, norm_weights


def federated_averaging(client_weights):

    global_weights = copy.deepcopy(client_weights[0])
    for key in global_weights.keys():
        
        for i in range(1,len(client_weights)):
            global_weights[key]+=client_weights[i][key]

        global_weights[key] = torch.div(
            global_weights[key],
            len(client_weights)
        )

    return global_weights