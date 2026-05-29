import copy
import torch

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