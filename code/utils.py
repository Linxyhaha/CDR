import torch
import os
import numpy as np

def size_of_model(net: torch.nn.Module):
    torch.save(net.state_dict(), "temp.p")
    size =  os.path.getsize("temp.p")/1e6 
    print('Size (MB):', size)
    os.remove('temp.p')
    return size

def count_params(net: torch.nn.Module):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def print_model_parameters(net: torch.nn.Module, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in net.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)

def print_nonzeros(net: torch.nn.Module):
    nonzero = total = 0
    for name, p in net.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')