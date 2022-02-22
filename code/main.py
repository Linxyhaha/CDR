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

parser = argparse.ArgumentParser(description='PyTorch COR')
parser.add_argument('--model_name', type=str, default='COR',
                    help='model name')
parser.add_argument('--dataset', type=str, default='time',
                    help='dataset name')
parser.add_argument('--data_path', type=str, default='/storage/shjing/recommendation/causal_discovery/data/amazon_book/data_1225/',
                    help='directory of all datasets')
parser.add_argument('--log_name', type=str, default='',
                    help='log/model special name')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00,
                    help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=500,
                    help='batch size')
parser.add_argument("--mlp_dims", default='[100, 20]',  
                    help="the dims of the mlp encoder")
parser.add_argument("--mlp_p1_dims", default='[100, 200]',  
                    help="the dims of the mlp p1")
parser.add_argument("--mlp_p2_dims", default='[]',  
                    help="the dims of the mlp p2")
parser.add_argument("--a_dims", type=int, default=2,
                    help="hidden size of alpha")
parser.add_argument('--z_dims', type=int, default=2,
                    help='hidden size of z')
parser.add_argument('--c_dims', type=int, default=2,
                    help='the number of categories for factorization of W')
parser.add_argument('--tau',type=float, default=0.2,
                    help='temperature of softmax')
parser.add_argument('--T', type=int, default=1,
                    help='the number of training environments')
parser.add_argument('--sigma', type=float, default=1,
                    help='sigma of CDF for N0 norm penalty for W')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument('--sample_freq', type=int, default=1,
                    help='sample frequency for Z1/Z2')
parser.add_argument('--bn', type=int, default=1,
                    help='batch norm')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout')
parser.add_argument('--regs', type=float, default=0,
                    help='regs')
parser.add_argument('--lam1', type=float, default=0.1,
                    help='weight of variance loss')
parser.add_argument('--lam2', type=float, default=0.1,
                    help='weight of w penalty')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument("--topN", default='[10, 20, 50, 100]',  
                    help="the recommended item num")
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=str, default='1',
                    help='GPU id')
parser.add_argument('--save_path', type=str, default='./models/',
                    help='path to save the final model')
parser.add_argument('--act_function', type=str, default='tanh',
                    help='activation function')
parser.add_argument('--z0_mode', type=int, default=1,
                    help='initialization mode of z0, e.g. all users the same, all users different, leanable.')
parser.add_argument('--split_global', type=int, default=1,
                    help='whether or not use global time for splitting training environments')
parser.add_argument('--infer_mode', type=int, default=1,
                    help='inference mode, 1 is add T when infer')
parser.add_argument('--mask', type=int, default=0, 
                    help='whether or not using updating mask for zt_last')
parser.add_argument('--clip', action='store_true',
                    help='whether or not use gradient clipping')
parser.add_argument('--grad_clip', type=int, default=10,
                    help='L2 norm threshold for gradient clipping')
parser.add_argument('--std', type=float, default=1,
                    help='sampling std for reparameterize trick')
parser.add_argument('--w_sigma', type=float, default=0.5,
                    help='sigma of normal initialization of W_CK and W_CD')
parser.add_argument('--add_T', type=int, default=2,
                    help='the number of additional T when infering')
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

if args.dataset == 'time':
    time_tr = np.load(train_dict_path, allow_pickle=True).item()
    timestamp = np.load(timestamp_path, allow_pickle=True).item()
    if args.split_global:
        train_data_env = data_utils.gen_env_tr_global(train_data, time_tr, timestamp, args.T+args.add_T)
    else:
        train_data_env = data_utils.gen_env_tr(train_dict_path, train_data, args.T)
    print(train_data_env.shape)
    train_data_env = torch.FloatTensor(train_data_env)
    train_dataset = data_utils.DataVAE(train_data_env)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

    valid_data = data_utils.gen_env_tr(train_dict_path, train_data, args.T+args.add_T)
    test_data = np.concatenate((valid_data,valid_y_data.A[np.newaxis,...]),0) # 
    test_data = torch.FloatTensor(test_data)
    test_dataset = data_utils.DataVAE(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


#ipdb.set_trace()
valid_data = torch.FloatTensor(valid_data)
val_dataset = data_utils.DataVAE(valid_data)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
#test_loader = val_loader

mask_tv = train_data + valid_y_data

###############################################################################
# Build the model
###############################################################################

mlp_q_dims = [n_items] + eval(args.mlp_dims) + [args.a_dims]
mlp_p1_dims = [args.a_dims+args.z_dims] + eval(args.mlp_p1_dims) + [args.z_dims]
mlp_p2_dims = [args.z_dims] + eval(args.mlp_p2_dims) + [n_items]


if args.model_name == 'MultiVAE_Z':
    model = models.MultiVAE_Z(mlp_q_dims, mlp_p1_dims, mlp_p2_dims,\
         args.c_dims,args.dropout, args.bn, args.sample_freq, args.regs, args.act_function, args.T, args.sigma, args.tau, args.std, n_users=N, z0_mode=args.z0_mode, mask=args.mask, inference=args.infer_mode).to(device)
elif args.model_name == 'MultiVAE_Z_W':
    model = models.MultiVAE_Z_W(mlp_q_dims, mlp_p1_dims, mlp_p2_dims,\
         args.c_dims,args.dropout, args.bn, args.sample_freq, args.regs, args.act_function, args.T, args.sigma, args.tau, args.std, n_users=N, z0_mode=args.z0_mode, mask=args.mask, inference=args.infer_mode, w_sigma=args.w_sigma).to(device)


print(f'total parameters:{utils.count_params(model)}')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = models.loss_variance

###############################################################################
# Training code
###############################################################################

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

def adjust_lr(e):
    if args.dataset=='time':
        if e>150:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * args.lr      
    elif args.dataset=='yelp':
        if e>60: # Decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * args.lr
    else:
        pass

def train():
    # Turn on training mode
    model.train()
    global update_count
    np.random.shuffle(idxlist)
    for batch_idx, data in enumerate(train_loader):
        data = torch.transpose(data.to(device),0,1) 

        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 
                            1. * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap
        
        optimizer.zero_grad()

        _, rec_xT, mu_T, logvar_T, reg_loss = model(data)

        rec_loss, env_variance = models.loss_variance(model, rec_xT, data,mu_T, logvar_T, anneal)
        var_loss = torch.sum(env_variance)
        penalty_w = torch.sum(models.loss_penalty(model.W_CK/model.sigma))
        penalty_w += torch.sum(models.loss_penalty(model.W_CD/model.sigma))

        total_loss = rec_loss + args.lam1 * var_loss + args.lam2 * penalty_w + args.regs * reg_loss 

        #print(f'total_loss:{total_loss}, rec_loss{rec_loss}, var_loss:{args.lam1*var_loss}, w_penalty:{args.lam2*penalty_w}')
        # if torch.isnan(total_loss).sum()!=0:
        #     ipdb.set_trace()
        assert torch.isnan(total_loss).sum()==0, 'training loss value is nan!'

        total_loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip, norm_type=2)
        optimizer.step()
        update_count += 1
        #print(total_loss)

        #print(f'batch cost {time.strftime("%H: %M: %S", time.gmtime(time.time()-st_time))}')
        #print(f'epoch training time approximation:{time.strftime("%H: %M: %S", time.gmtime((time.time()-st_time)*round(N/args.batch_size)))}')


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

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                               1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap
     
            recon_batch, rec_xT, mu_T, logvar_T, _ = model(data_tensor)

            if args.infer_mode==1:
                rec_loss = models.e_recon_loss(rec_xT, data_tensor, mu_T,logvar_T, anneal)

                penalty_w = torch.sum(models.loss_penalty(model.W_CK/model.sigma))
                penalty_w += torch.sum(models.loss_penalty(model.W_CD/model.sigma))
                total_loss += rec_loss.item() + args.lam2 * penalty_w
            
            # Exclude examples from training set
            recon_batch[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(recon_batch, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    test_results = evaluate_util.computeTopNAccuracy(target_items, predict_items, topN)
    return total_loss, test_results


def evaluate_moreT(data_tr, data_te, his_mask, topN, istest=0, tst_w_val=0):
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

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                            1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap
    
            recon_batch, rec_xT, mu_T, logvar_T, _ = model(data_tensor)

            if args.infer_mode==1:
                rec_loss = models.e_recon_loss(rec_xT, data_tensor, mu_T,logvar_T, anneal)

                penalty_w = torch.sum(models.loss_penalty(model.W_CK/model.sigma))
                penalty_w += torch.sum(models.loss_penalty(model.W_CD/model.sigma))
                total_loss += rec_loss.item() + args.lam2 * penalty_w
            
            # Exclude examples from training set
            recon_batch[his_data.nonzero()] = -np.inf

            _, indices = torch.topk(recon_batch, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    total_loss /= len(range(0, e_N, args.batch_size))
    test_results = evaluate_util.computeTopNAccuracy(target_items, predict_items, topN)
    # model.T = args.T
    return total_loss, test_results  



best_recall = -np.inf
best_ood_recall = -np.inf
best_epoch = 0
best_ood_epoch = 0
best_valid_results = None
best_test_results = None
best_ood_test_results = None
update_count = 0

# recall@10 for best model selection when K=0, recall@50 when K=2
evaluate_interval=10

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        #adjust_lr(epoch)
        train()
        if epoch % evaluate_interval == 0:
            #ipdb.set_trace()
            if args.infer_mode == 1:
                valid_loss, valid_results = evaluate(train_data_env, valid_y_data, valid_x_data, eval(args.topN))
                test_loss, test_results = evaluate(train_data_env, test_y_data, mask_tv, eval(args.topN))
            elif args.infer_mode == 2:
                valid_loss, valid_results = evaluate_moreT(train_data_env, valid_y_data, valid_x_data, eval(args.topN))
                test_loss, test_results = evaluate_moreT(train_data_env, test_y_data, mask_tv, eval(args.topN),1) 
                test_loss_twv, test_results_twv = evaluate_moreT(train_data_env, test_y_data, mask_tv, eval(args.topN),1,1)             
            #ipdb.set_trace()
            print('---'*18)
            print("Runing Epoch {:03d} ".format(epoch) + 'valid loss {:.4f}'.format(valid_loss) + " costs " + time.strftime(
                                "%H: %M: %S", time.gmtime(time.time()-epoch_start_time)))
            evaluate_util.print_results(None, valid_results, test_results)
            print('---'*18)
            
            print('Test With Validation')
            evaluate_util.print_results(None, None, test_results_twv)
            print('---'*18)

            # Save the model if recall is the best we've seen so far.
            if valid_results[1][0] > best_recall: # recall@10 for selection
                best_recall, best_epoch = valid_results[1][0], epoch
                best_test_results = test_results
                best_test_results_twv = test_results_twv
                best_valid_results = valid_results

                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                torch.save(model, '{}{}_{}_{}q_{}p1_{}p2_{}lr_{}wd_{}bs_{}anneal_{}cap_{}drop_{}a_{}z_{}c_{}tau_{}T_{}bn_{}freq_{}reg_{}lam1_{}lam2_{}z0_{}_{}global_{}infer_{}sigma_{}std_{}wsgma_{}.pth'.format(
                        args.save_path, args.model_name, args.dataset, args.mlp_dims, \
                        args.mlp_p1_dims, args.mlp_p2_dims, args.lr, args.wd,\
                        args.batch_size, args.total_anneal_steps, args.anneal_cap, \
                        args.dropout, args.a_dims, args.z_dims, args.c_dims, args.tau, args.T, \
                        args.bn, args.sample_freq, args.regs, args.lam1, args.lam2, args.z0_mode, \
                        args.act_function, args.split_global, args.infer_mode, args.sigma, args.std, args.w_sigma, args.log_name))
except KeyboardInterrupt:
    print('-'*18)
    print('Exiting from training early')

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_util.print_results(None, best_valid_results, best_test_results)
print('==='*18)
print('Test With Validation')
evaluate_util.print_results(None, None, best_test_results_twv)
print('==='*18)




