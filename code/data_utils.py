import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
from torch.utils.data import Dataset
import torch

class DataVAE(Dataset):
    def __init__(self, data):
        self.data = torch.transpose(data,0,1)
    def __getitem__(self, idx):
        data = self.data[idx]
        return data
    def __len__(self):
        return len(self.data)

""" Construct the VAE training dataset."""
def data_load(dataset, train_path, valid_path, test_path):

    train_list = np.load(train_path, allow_pickle=True)
    valid_dict = np.load(valid_path, allow_pickle=True).item()
    test_dict = np.load(test_path, allow_pickle=True).item()
    
    # get train_dict
    uid_max = 0
    iid_max = 0
    train_dict = {}
    for entry in train_list:
        user, item = entry
        if user not in train_dict:
            train_dict[user] = []
        train_dict[user].append(item)
        if user > uid_max:
            uid_max = user
        if item > iid_max:
            iid_max = item
    
    # get valid_list & test_list
    valid_list = []
    test_list = []
    for u in valid_dict:
        if u > uid_max:
            uid_max = u
        for i in valid_dict[u]:
            valid_list.append([u, i])
            if i > iid_max:
                iid_max = i
                
    for u in test_dict:
        if u > uid_max:
            uid_max = u
        for i in test_dict[u]:
            test_list.append([u, i])
            if i > iid_max:
                iid_max = i

    if dataset == 'book':
        n_users = max(uid_max + 1, 21923) 
        n_items = max(iid_max + 1, 23773) 
    if dataset == 'electronics':
        n_users = max(uid_max + 1, 9279) 
        n_items = max(iid_max + 1, 6065)
    if dataset == 'yelp':
        n_users = max(uid_max + 1, 11622)
        n_items = max(iid_max + 1, 9095)

    print(f'n_users: {n_users}')
    print(f'n_items: {n_items}')
    
    valid_list = np.array(valid_list)
    test_list = np.array(test_list)

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    valid_x_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    test_x_list = train_list
    test_x_data = sp.csr_matrix((np.ones_like(test_x_list[:, 0]),
                 (test_x_list[:, 0], test_x_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
        
    return train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items


def data_load_infer(dataset, train_path, valid_path, test_path):

    train_list = np.load(train_path, allow_pickle=True)
    valid_dict = np.load(valid_path, allow_pickle=True).item()
    test_dict = np.load(test_path, allow_pickle=True).item()
    
    # get train_dict
    uid_max = 0
    iid_max = 0
    train_dict = {}
    for entry in train_list:
        user, item = entry
        if user not in train_dict:
            train_dict[user] = []
        train_dict[user].append(item)
        if user > uid_max:
            uid_max = user
        if item > iid_max:
            iid_max = item
    
    # get valid_list & test_list
    valid_list = []
    test_list = []
    for u in valid_dict:
        if u > uid_max:
            uid_max = u
        for i in valid_dict[u]:
            valid_list.append([u, i])
            if i > iid_max:
                iid_max = i
                
    for u in test_dict:
        if u > uid_max:
            uid_max = u
        for i in test_dict[u]:
            test_list.append([u, i])
            if i > iid_max:
                iid_max = i

    if dataset == 'book':
        n_users = max(uid_max + 1, 21923) # book
        n_items = max(iid_max + 1, 23773)

    if dataset == 'electronics':
        n_users = max(uid_max + 1, 9279) # electronics
        n_items = max(iid_max + 1, 6065)

    if dataset == 'yelp':
        n_users = max(uid_max + 1, 11622) # yelp
        n_items = max(iid_max + 1, 9095)

    print(f'n_users: {n_users}')
    print(f'n_items: {n_items}')
    
    valid_list = np.array(valid_list)
    test_list = np.array(test_list)

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    valid_x_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))

    #test_x_list = train_list 
    test_x_list = np.concatenate([train_list, valid_list], 0)
    
    test_x_data = sp.csr_matrix((np.ones_like(test_x_list[:, 0]),
                 (test_x_list[:, 0], test_x_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
        
    return train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items


def gen_env_tr(dataset, train_path,train_data_sp,n_env=1):
    '''
    given train_data (n_user x n_items), split into N training environment (N x n_user x n_items)
    '''
    print(f'generating {n_env} training environments (relative)')
    train_dict = np.load(train_path, allow_pickle=True).item()
    train_data = []
    # for env in range(n_env):
    #     train_data.append({u_id:None for u_id in train_dict})
    train_data = [{u_id:None for u_id in train_dict} for _ in range(n_env)]

    if dataset == 'book':
        n_users = 21923  
        n_items = 23773  
    elif dataset == 'electronics': 
        n_users = 9279 
        n_items = 6065
    elif dataset == 'yelp':
        n_users = 11622 
        n_items = 9095

    if n_env==1:
        train_data_arr = train_data_sp.A
        return train_data_arr[np.newaxis]
    else:
        for u_id in train_dict:
            env_interaction_num = int((1/n_env) * len(train_dict[u_id]))
            for env in range(n_env):
                if env == n_env-1:
                    train_data[env][u_id] = train_dict[u_id][env_interaction_num*env:]
                else:
                    train_data[env][u_id] = train_dict[u_id][env_interaction_num * env:env_interaction_num*(env+1)]
        
        train_list = [[] for _ in range(n_env)]
        for env in range(n_env):
            for u,items in train_data[env].items():
                for i in items:
                    train_list[env].append([u,i])

        for env in range(n_env):
            train_env_list = np.array(train_list[env])
            
            train_data_env = sp.csr_matrix((np.ones_like(train_env_list[:, 0]),
                    (train_env_list[:, 0], train_env_list[:, 1])), dtype='float64',
                    shape=(n_users, n_items))
            train_data_env = train_data_env.A[np.newaxis]

            if env==0:
                train_data_T = train_data_env
            else:
                train_data_T = np.concatenate((train_data_T,train_data_env),0)
    
    return train_data_T



        

        
