import random
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def multi_predict(model, x_obs, x_model, steps):
    """
    Take a conditioning sequence of indices in x (of shape (b,t)) and predict the next value in
    the sequence, feeding the predictions back into the model each time. 
    Note that we only want to predict certain inputs, i.e. the inputs corresponding to the observations.
    The NWP model data, we already have good predictions for the next step from the NWP model,
    so we want to use that
    
    Clearly the sampling has quadratic complexity unlike an RNN that is only linear, and has a finite
    context window of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    x = torch.cat((x_obs[:,:block_size,:],x_model[:,:block_size,:]), dim=2)

    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop the sequence if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step
        logits = logits[:, -1, :]
        # Concatenate the predicted value(s) and the forecast for the next step,
        # since we don't want to predict the model data, but use the current NWP forecast
        x_new = torch.cat((logits,x_model[:,block_size+k]), dim=1)
        x_new = x_new.unsqueeze(1)
        # Append to the sequence and continue to predict the next step using the newest prediction and the next NWP forecast step
        x = torch.cat((x, x_new), dim=1)

    return x



# ORIGINAL CODE FOR PREDICTING ONE FORECAST AT A TIME
def predict(model, x_obs, x_model, steps):
    """
    Take a conditioning sequence of indices in x (of shape (b,t)) and predict the next value in
    the sequence, feeding the predictions back into the model each time. 
    Note that we only want to predict certain inputs, i.e. the inputs corresponding to the observations.
    The NWP model data, we already have good predictions for the next step from the NWP model,
    so we want to use that
    
    Clearly the sampling has quadratic complexity unlike an RNN that is only linear, and has a finite
    context window of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    x = torch.cat((x_obs[:block_size,:],x_model[:block_size,:]), dim=1)
    x = x.unsqueeze(0)
    x_model = x_model.unsqueeze(0)

    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop the sequence if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step
        logits = logits[:, -1, :]
        # Concatenate the predicted value(s) and the forecast for the next step,
        # since we don't want to predict the model data, but use the current NWP forecast
        x_new = torch.cat((logits,x_model[:,block_size+k]), dim=1)
        x_new = x_new.unsqueeze(0)
        # Append to the sequence and continue to predict the next step using the newest prediction and the next NWP forecast step
        x = torch.cat((x, x_new), dim=1)

    return x

