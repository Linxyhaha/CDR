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
parser.add_argument('--log_name', type=str, default='', help='log/model special name')

parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500,help='batch size')

parser.add_argument("--mlp_dims", default='[100, 20]', help="the dims of the mlp encoder")
parser.add_argument("--mlp_p1_dims", default='[100, 200]', help="the dims of the mlp p1")
parser.add_argument("--mlp_p2_dims", default='[]',  help="the dims of the mlp p2")
parser.add_argument("--a_dims", type=int, default=2,help="hidden size of alpha")
parser.add_argument('--z_dims', type=int, default=2,help='hidden size of z')
parser.add_argument('--c_dims', type=int, default=2,help='the number of categories for factorization of W')

parser.add_argument('--tau',type=float, default=0.2, help='temperature of softmax')
parser.add_argument('--T', type=int, default=1, help='the number of training environments')
parser.add_argument('--sigma', type=float, default=0.1, help='sigma of CDF for N0 norm penalty for W')

parser.add_argument('--total_anneal_steps', type=int, default=0, help='the total number of gradient updates for annealing')
parser.add_argument('--lam1', type=float, default=0.2,help='largest annealing parameter')
parser.add_argument('--lam2', type=float, default=0.1, help='weight of w penalty')
parser.add_argument('--lam3', type=float, default=0.1, help='weight of variance loss')

parser.add_argument('--regs', type=float, default=0, help='regs')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--bn', type=int, default=1, help='batch norm')


parser.add_argument('--epochs', type=int, default=500, help='upper epoch limit')
parser.add_argument("--topN", default='[10, 20, 50, 100]',   help="the recommended item num")
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='1', help='GPU id')
parser.add_argument('--save_path', type=str, default='./models/', help='path to save the final model')

parser.add_argument('--std', type=float, default=0.1, help='sampling std for reparameterize')
parser.add_argument('--w_sigma', type=float, default=0.5, help='sigma of normal initialization of W_CK and W_CD')

parser.add_argument('--add_T', type=int, default=2, help='the number of additional T when infering')
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
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

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
# Build the model
###############################################################################
mlp_q_dims = [n_items] + eval(args.mlp_dims) + [args.a_dims]
mlp_p1_dims = [args.a_dims+args.z_dims] + eval(args.mlp_p1_dims) + [args.z_dims]
mlp_p2_dims = [args.z_dims] + eval(args.mlp_p2_dims) + [n_items]

model = models.CDR(mlp_q_dims, mlp_p1_dims, mlp_p2_dims, args.c_dims,args.dropout, args.bn, \
                         args.regs, args.T, args.sigma, args.tau, args.std, w_sigma=args.w_sigma).to(device)

print(f'total parameters:{utils.count_params(model)}')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
###############################################################################
# Training code
###############################################################################
def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def train():
    # Turn on training mode
    model.train()
    global update_count
    np.random.shuffle(idxlist)
    for batch_idx, data in enumerate(train_loader):
        data = torch.transpose(data.to(device),0,1) 

        if args.total_anneal_steps > 0:
            anneal = min(args.lam1, 
                            1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.lam1
        
        optimizer.zero_grad()

        _, rec_xT, mu_T, logvar_T, reg_loss = model(data)

        rec_loss, env_variance = models.loss_variance(model, rec_xT, data,mu_T, logvar_T, anneal)
        var_loss = torch.sum(env_variance)
        penalty_w = torch.sum(models.loss_penalty(model.W_CK/model.sigma))
        penalty_w += torch.sum(models.loss_penalty(model.W_CD/model.sigma))

        total_loss = rec_loss + args.lam2 * penalty_w + args.lam3 * var_loss + args.regs * reg_loss 
        assert torch.isnan(total_loss).sum()==0, 'training loss value is nan!'

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=100, norm_type=2)
        optimizer.step()
        update_count += 1

def evaluate(data_tr, data_te, his_mask, topN, istest=0, tst_w_val=0):
    if istest:
        data_loader = test_loader
    else:
        data_loader = val_loader
        
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

best_recall = -np.inf
best_ood_recall = -np.inf
best_epoch = 0
best_ood_epoch = 0
best_valid_results = None
best_test_results = None
best_ood_test_results = None
update_count = 0

evaluate_interval=10

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        if epoch % evaluate_interval == 0:
            valid_loss, valid_results = evaluate(train_data_env, valid_y_data, valid_x_data, eval(args.topN))
            test_loss, test_results = evaluate(train_data_env, test_y_data, mask_tv, eval(args.topN),1)             

            print('---'*18)
            print("Runing Epoch {:03d} ".format(epoch) + 'valid loss {:.4f}'.format(valid_loss) + " costs " + time.strftime(
                                "%H: %M: %S", time.gmtime(time.time()-epoch_start_time)))
            evaluate_util.print_results(None, valid_results, test_results)
            print('---'*18)
            
            # Save the model if recall is the best we've seen so far.
            if valid_results[1][0] > best_recall: # recall@10 for selection
                best_recall, best_epoch = valid_results[1][0], epoch
                best_test_results = test_results
                best_valid_results = valid_results

                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                torch.save(model, '{}{}_{}lr_{}bs_{}q_{}p1_{}p2_{}a_{}z_{}c_{}tau_{}T_{}lam1_{}lam2_{}lam3_{}regs_{}bn_{}dropout_{}std_{}addT_{}.pth'.format(
                        args.save_path, args.dataset, args.lr, args.batch_size, args.mlp_dims, \
                        args.mlp_p1_dims, args.mlp_p2_dims, args.a_dims, args.z_dims, args.c_dims, \
                        args.tau, args.T, args.lam1, args.lam2, args.lam3, args.regs, args.bn, args.dropout,\
                        args.std, args.add_T, args.log_name))
except KeyboardInterrupt:
    print('-'*18)
    print('Exiting from training early')

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_util.print_results(None, best_valid_results, best_test_results)
print('==='*18)




