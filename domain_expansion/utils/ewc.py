import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def collect_params(sub_dict):
    Ws = {}
    for k, v in sub_dict.items():
        if isinstance(v, list):
            Ws[k] = []
            for i , vv in enumerate(v):
                Ws[k].append(nn.Parameter(torch.Tensor(*vv.shape), requires_grad=True))
                Ws[k][i].data = v[i]
        elif isinstance(v, dict):
            Ws[k] = collect_params(v)
    return Ws

def init_fishers(sub_dict):
    Fs = {}
    for k, v in sub_dict.items():
        if isinstance(v, list):
            Fs[k] = [torch.zeros_like(vv) for vv in v]
        elif isinstance(v, dict):
            Fs[k] = init_fishers(v)
    return Fs

def update_fishers(Fs, sub_dict):
    for k, v in sub_dict.items():
        if isinstance(v, list):
            for i, vv in enumerate(v):
                Fs[k][i] += torch.pow(vv.grad.detach(), 2)
        elif isinstance(v, dict):
            update_fishers(Fs[k], v)

def collect_fishers(sub_dict):
    Ws = []
    for k, v in sub_dict.items():
        if isinstance(v, list):
            Ws.extend(v)
        elif isinstance(v, dict):
            Ws.extend(collect_fishers(v))
    return Ws

def zero_grads(sub_dict):
    for k, v in sub_dict.items():
        if isinstance(v, list):
            for vv in v:
                if vv.grad is not None:
                    vv.grad.data.zero_()
        elif isinstance(v, dict):
            zero_grads(v)

def compute_fisher(model, dataloader, domain_id, n_max=-1):

    batch_size = dataloader.batch_size
    total_samples = len(dataloader.dataset)

    if n_max != -1:
        total_samples = min(n_max, total_samples)
    
    num_batches = total_samples // batch_size

    model.cuda().eval()

    weights = model.hmotion_decoder(domain_id)
    params = collect_params(weights)
    fisher = init_fishers(params)
    extnorms = model.hmotion_decoder.get_domain_norm(domain_id)

    iter_dataloader = iter(dataloader)
    for _ in range(num_batches):
        batch = next(iter_dataloader)
        batch = model.transfer_batch_to_device(batch, model.device, dataloader_idx=0)
        enc_dict = model.context_encoder(batch)
        out_dict = model.mmotion_decoder(enc_dict, params, extnorms, single=True)
        loss, tb_dict, disp_dict = model.mmotion_decoder.get_loss()
        zero_grads(params)
        torch.autograd.backward(loss, retain_graph=False, create_graph=False)
        update_fishers(fisher, params)

    fisher = collect_fishers(fisher)  
    
    for i in range(len(fisher)):
        fisher[i] /= total_samples
    
    return fisher
    
    

