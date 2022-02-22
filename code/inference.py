import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ipdb
from torch.utils.data import DataLoader

# from tensorboardX import SummaryWriter
from scipy import sparse
import models
import random
import data_utils
import utils
import evaluate_util
import os
import pickle

parser = argparse.ArgumentParser(description='Inference with validation')
parser.add_argument('--model_name', type=str, default='COR',
                    help='model name')
parser.add_argument('--dataset', type=str, default='time',
                    help='dataset name')
parser.add_argument('--data_path', type=str, default='/storage/shjing/recommendation/causal_discovery/data/amazon_book/data_relative_20/',
                    help='directory of all datasets')
parser.add_argument('--log_name', type=str, default='',
                    help='log/model special name')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')
parser.add_argument('--T', type=int, default=1,
                    help='the number of training environments')
parser.add_argument("--topN", default='[10, 20, 50, 100]',  
                    help="the recommended item num")
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU id')
parser.add_argument('--split_global', type=int, default=0,
                    help='whether or not use global time for splitting training environments')
parser.add_argument('--ckpt', type=str, default=None,
                    help='checkpoint path')
parser.add_argument('--infer_mode', type=int, default=1,
                    help='inference mode, e.g. 0 is split data; 1 is no updating for users who have no training data; \
                        2 is reuse zt and feed all history interactions to encoder; \
                            3 is using z0 and feed all history interactions ')

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

train_dict_path = args.data_path + 'split_' + args.dataset + '/training_dict.npy'
timestamp_path  = args.data_path + 'split_' + args.dataset + '/interaction_timestamp.npy' 

train_path = args.data_path + 'split_' + args.dataset + '/training_list.npy'
valid_path = args.data_path + 'split_' + args.dataset + '/validation_dict.npy'
test_path = args.data_path + 'split_' + args.dataset + '/testing_dict.npy'

train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items = \
                                            data_utils.data_load(train_path, valid_path, test_path)

N = train_data.shape[0]
idxlist = list(range(N))


# train_data = train_data + valid_y_data  # try1

if args.dataset == 'time':
    time_tr = np.load(train_dict_path, allow_pickle=True).item()
    timestamp = np.load(timestamp_path, allow_pickle=True).item()
    if args.split_global:
        train_data_env = data_utils.gen_env_tr_global(train_data, time_tr, timestamp, args.T)
    else:
        train_data_env = data_utils.gen_env_tr(train_dict_path, train_data, args.T)

train_data_env = np.concatenate((train_data_env,valid_y_data.A[np.newaxis,...]),0) # try2

# train_data_env[-1] = train_data_env[-1] + valid_y_data.A # try3

print(train_data_env.shape)
train_data_env = torch.FloatTensor(train_data_env)
train_dataset = data_utils.DataVAE(train_data_env)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data + valid_y_data

if args.group:
    with open(args.data_path + 'user_multi_group.pkl','rb') as f:
        user_group = pickle.load(f)
###############################################################################
# Load the model
###############################################################################
args.checkpoint = 'models/MultiVAE_Z_W_time_[800]q_[]p1_[]p2_0.001lr_0.0wd_500bs_0anneal_0.75cap_0.5drop_300a_300z_2c_0.1tau_1bn_1freq_0.0reg_0.01lam1_1.0lam2_0z0_tanh_4env_0global_0.1sigma_log.pth'
model = torch.load(args.checkpoint)
model.infer_mode = args.infer_mode
print(f'total parameters:{utils.count_params(model)}')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = models.loss_variance

print('model loaded.')
#
model.T = args.T+1
###############################################################################
# Training code
###############################################################################

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def evaluate(data_tr, data_te, his_mask, topN):
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
        for batch_idx, data in enumerate(test_loader):
            data_tensor = torch.transpose(data.to(device),0,1)
            his_data = his_mask[e_idxlist[batch_idx*args.batch_size:batch_idx * args.batch_size+len(data)]]
     
            recon_batch, rec_xT, mu_T, logvar_T, _ = model(data_tensor)
            
            # Exclude examples from training set
            recon_batch[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(recon_batch, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)
    
    total_loss /= len(range(0, e_N, args.batch_size))
    test_results = evaluate_util.computeTopNAccuracy(target_items, predict_items, topN)
    return total_loss, test_results


def evaluate_group(data_tr, data_te, his_mask, topN):
    assert data_tr.shape[1] == data_te.shape[0]
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[1]))
    e_N = data_tr.shape[1]

    predict_items = []
    target_items = []

    predict_items_20 = []
    target_items_20 = []
    predict_items_40 = []
    target_items_40 = []
    predict_items_60 = []
    target_items_60 = []
    predict_items_80 = []
    target_items_80 = []
    predict_items_100 = []
    target_items_100 = []
    predict_items_else = []
    target_items_else = []

    for i in range(e_N):
        target_items.append(data_te[i,:].nonzero()[1].tolist())
    #ipdb.set_trace()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data_tensor = torch.transpose(data.to(device),0,1)
            his_data = his_mask[e_idxlist[batch_idx*args.batch_size:batch_idx * args.batch_size+len(data)]]

     
            recon_batch, rec_xT, mu_T, logvar_T, _ = model(data_tensor)
            # if args.infer_mode==1:
            #     rec_loss = models.e_recon_loss(rec_xT, data_tensor, mu_T,logvar_T, 1)

            #     penalty_w = torch.sum(models.loss_penalty(model.W_CK/model.sigma))
            #     penalty_w += torch.sum(models.loss_penalty(model.W_CD/model.sigma))
            #     total_loss += rec_loss.item() + args.lam2 * penalty_w
            
            # Exclude examples from training set
            recon_batch[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(recon_batch, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)
    
    
    total_loss /= len(range(0, e_N, args.batch_size))

    for u_id in user_group[20]:
        target_items_20.append(target_items[u_id])
        predict_items_20.append(predict_items[u_id])
    for u_id in user_group[40]:
        target_items_40.append(target_items[u_id])
        predict_items_40.append(predict_items[u_id])
    for u_id in user_group[60]:
        target_items_60.append(target_items[u_id])
        predict_items_60.append(predict_items[u_id])
    for u_id in user_group[80]:
        target_items_80.append(target_items[u_id])
        predict_items_80.append(predict_items[u_id])
    for u_id in user_group[100]:
        target_items_100.append(target_items[u_id])
        predict_items_100.append(predict_items[u_id])
    for u_id in user_group['else']:
        target_items_else.append(target_items[u_id])
        predict_items_else.append(predict_items[u_id])

    assert len(target_items_20)+len(target_items_40)+len(target_items_60)+len(target_items_80)+len(target_items_100)+len(target_items_else) == len(predict_items_20)+len(predict_items_40)+len(predict_items_60)+len(predict_items_80)+len(predict_items_100)+len(predict_items_else) == len(target_items)== len(predict_items)

    test_results_20 = evaluate_util.computeTopNAccuracy(target_items_20, predict_items_20, topN)
    test_results_40 = evaluate_util.computeTopNAccuracy(target_items_40, predict_items_40, topN)
    test_results_60 = evaluate_util.computeTopNAccuracy(target_items_60, predict_items_60, topN)
    test_results_80 = evaluate_util.computeTopNAccuracy(target_items_80, predict_items_80, topN)
    test_results_100 = evaluate_util.computeTopNAccuracy(target_items_100, predict_items_100, topN)
    test_results_else = evaluate_util.computeTopNAccuracy(target_items_else, predict_items_else, topN)
    return total_loss, test_results_20, test_results_40,test_results_60,test_results_80,test_results_100,test_results_else


update_count = 0
# At any point you can hit Ctrl + C to break out of training early.
print('start inferencing')
_,valid_results = evaluate(train_data_env, valid_y_data, valid_x_data, eval(args.topN))
_,test_results = evaluate(train_data_env, test_y_data, mask_tv, eval(args.topN))

print('==='*18)
evaluate_util.print_results(None, valid_results, test_results)
print('==='*18)











