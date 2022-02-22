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
def data_load(train_path, valid_path, test_path):

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
    n_users = max(uid_max + 1, 21923) # ml-1m
    n_items = max(iid_max + 1, 23773)

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
#     test_x_list = np.concatenate([train_list, valid_list], 0)
    test_x_data = sp.csr_matrix((np.ones_like(test_x_list[:, 0]),
                 (test_x_list[:, 0], test_x_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_users, n_items))
        
    return train_data, valid_x_data, valid_y_data, test_x_data, test_y_data, n_users, n_items


def data_load_infer(train_path, valid_path, test_path):

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
    n_users = max(uid_max + 1, 21923) # ml-1m
    n_items = max(iid_max + 1, 23773)

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



def gen_env_tr(train_path,train_data_sp,n_env=1):
    '''
    given train_data (n_user x n_items), split into N training environment (N x n_user x n_items)
    '''
    print(f'generating {n_env} training environments (relative)')
    train_dict = np.load(train_path, allow_pickle=True).item()
    train_data = []
    # for env in range(n_env):
    #     train_data.append({u_id:None for u_id in train_dict})
    train_data = [{u_id:None for u_id in train_dict} for _ in range(n_env)]
    n_users = 21923  # ml-1m
    n_items = 23773
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


def gen_env_tr_global(train_data_sp, time_tr, interaction_time, n_env):
    '''
    corresponding to number of split training env, compute the time point for spliting,
    split each user's interaction into different environment by global time.
    '''
    if n_env==1:
        train_data_arr = train_data_sp.A
        return train_data_arr[np.newaxis]
    
    n_users = 21923 #ml-1m
    n_items = 23773
    print(f'generating {n_env} training environments (global)')
    time_list = []
    for u_id in time_tr:
        for i_id in time_tr[u_id]:
            time_list.append(interaction_time[u_id][i_id])
    time_list = sorted(time_list)

    user_time = {u_id:[] for u_id in time_tr}
    for u_id in time_tr:
        for i_id in time_tr[u_id]:
            user_time[u_id].append(interaction_time[u_id][i_id])

    time_point = [0]
    split_num = int(len(time_list)*1/n_env) # 均分global time
    for i in range(n_env):
        split_idx = min((i+1)*split_num,len(time_list))-1
        time_point.append(time_list[split_idx])

    # spliting interaction into n training env
    train_list = [[] for _ in range(n_env)]
    for i in range(n_env):
        for u_id in time_tr:
            for i_id in interaction_time[u_id]:
                if interaction_time[u_id][i_id]>time_point[i] and interaction_time[u_id][i_id]<time_point[i+1]:
                    train_list[i].append([u_id,i_id])

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



        

        
