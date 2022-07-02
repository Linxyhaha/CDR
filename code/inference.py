import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import ipdb
import copy

# from tensorboardX import SummaryWriter
from scipy import sparse
import models
import random
import data_utils
import utils
import evaluate_util
import os

parser = argparse.ArgumentParser(description='PyTorch CDR')

parser.add_argument('--dataset', type=str, default='electronics', help='dataset name, choose from electronics, book, and yelp')
parser.add_argument('--data_path', type=str, default='../data/electronics', help='directory of all datasets')

parser.add_argument('--batch_size', type=int, default=500,help='batch size')
parser.add_argument('--T', type=int, default=1, help='the number of training environments')
parser.add_argument('--add_T', type=int, default=2, help='the number of additional T when infering')

parser.add_argument("--topN", default='[10, 20, 50, 100]',   help="the recommended item num")
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='1', help='GPU id')

parser.add_argument('--ckpt', type=str, help='pre-trained model path')

parser
args = parser.parse_args()
print(args)

random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")
print(f'using device {device}')

###############################################################################
# Load data
###############################################################################
train_dict_path = '../data/' + args.dataset + '/training_dict.npy'
timestamp_path  = '../data/' + args.dataset + '/interaction_timestamp.npy' 

train_path = '../data/' + args.dataset + '/training_list.npy'
valid_path = '../data/' + args.dataset + '/validation_dict.npy'
test_path = '../data/' + args.dataset + '/testing_dict.npy'

train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items = \
                                            data_utils.data_load(args.dataset, train_path, valid_path, test_path)

N = train_data.shape[0]
idxlist = list(range(N))

time_tr = np.load(train_dict_path, allow_pickle=True).item()
timestamp = np.load(timestamp_path, allow_pickle=True).item()

train_data_env = data_utils.gen_env_tr(args.dataset, train_dict_path, train_data, args.T)

print(train_data_env.shape)
train_data_env = torch.FloatTensor(train_data_env)
train_dataset = data_utils.DataVAE(train_data_env)


valid_data = data_utils.gen_env_tr(args.dataset, train_dict_path, train_data, args.T+args.add_T)
test_data = np.concatenate((valid_data,valid_y_data.A[np.newaxis,...]),0) # 
test_data = torch.FloatTensor(test_data)
test_dataset = data_utils.DataVAE(test_data)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

valid_data = torch.FloatTensor(valid_data)
val_dataset = data_utils.DataVAE(valid_data)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

mask_tv = train_data + valid_y_data
###############################################################################
# Load the model
###############################################################################
model = torch.load('./models/' + args.ckpt).cuda()

###############################################################################
# Inference code
###############################################################################
def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def evaluate(data_tr, data_te, his_mask, topN, istest=0, tst_w_val=0):
    if istest and tst_w_val:
        data_loader = test_loader
    else:
        data_loader = val_loader
    assert data_tr.shape[1] == data_te.shape[0]
    # Turn on evaluation mode
    model.eval()
    
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[1]))
    e_N = data_tr.shape[1]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i,:].nonzero()[1].tolist())
    #ipdb.set_trace()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data_tensor = torch.transpose(data.to(device),0,1)
            his_data = his_mask[e_idxlist[batch_idx*args.batch_size:batch_idx * args.batch_size+len(data)]]

            recon_batch, _, _, _, _ = model(data_tensor)

            # Exclude examples from training set
            recon_batch[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(recon_batch, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    total_loss /= len(range(0, e_N, args.batch_size))
    test_results = evaluate_util.computeTopNAccuracy(target_items, predict_items, topN)
    return total_loss, test_results  

valid_loss, valid_results = evaluate(train_data_env, valid_y_data, valid_x_data, eval(args.topN))
test_loss, test_results = evaluate(train_data_env, test_y_data, mask_tv, eval(args.topN),1,1)     
print('==='*18)
evaluate_util.print_results(None, valid_results, test_results)        
print('==='*18)







