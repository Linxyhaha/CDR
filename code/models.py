import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import math
from torch.autograd import grad
import ipdb



class MultiVAE_Z(nn.Module):
    '''
    Version 2-a: basic framework of multivae + z (q + concat(a,z) + p1 + p2)
    '''
    def __init__(self, mlp_q_dims, mlp_p1_dims, mlp_p2_dims,\
                 c_dim=2, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh', T=1, sigma=1, tau=0.1, std=1, n_users=None, z0_mode=1, mask=0, inference=1):
        super(MultiVAE_Z, self).__init__()
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_dims = mlp_p1_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.drop = nn.Dropout(dropout)

        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs

        if act_function == 'tanh':
            self.act_function = torch.tanh
        elif act_function == 'sigmoid':
            self.act_function = torch.sigmoid
        elif act_function == 'relu':
            self.act_function = torch.relu

            
        self.T = T # n_time
        self.sigma = sigma

        self.tau = tau
        self.std = std
        self.z0_mode = z0_mode
        self.mask = mask
        self.infer_mode = inference

        if self.z0_mode ==0:
            self.z0 = torch.zeros((1,mlp_p1_dims[-1]))
        elif self.z0_mode == 1:    
            self.z0 = torch.randn(1,mlp_p1_dims[-1])
        
        self.zt_last = None

        # Last dimension of q- network is for mean and variance
        temp_q_dims = mlp_q_dims[:-1] + [mlp_q_dims[-1] * 2]
        temp_p1_dims = mlp_p1_dims[:-1] + [mlp_p1_dims[-1] * 2]
        temp_p2_dims = mlp_p2_dims
        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for 
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_p1_dims[:-1],temp_p1_dims[1:])]) 
        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_p2_dims[:-1],temp_p2_dims[1:])]) 
        if self.bn:
            self.batchnorm_a = nn.BatchNorm1d(mlp_q_dims[-1]) # a_dims
            self.batchnorm_z = nn.BatchNorm1d(mlp_p1_dims[-1]) #z_dims

        self.init_weights()

    def encode(self, encoder_input):
        encoder_input = F.normalize(encoder_input)
        h = self.drop(encoder_input)
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h) 
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:,self.mlp_q_dims[-1]:]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros_like(std).normal_(mean=0, std=self.std)
            return mu + eps * std
        else:
            return mu

    def decode(self, a_t, z_t_last):
        #z_t_last = F.normalize(z_t_last,dim=1)
        h_p1 = torch.cat((a_t,z_t_last),1)
        for i, layer in enumerate(self.mlp_p1_layers):
            h_p1 = layer(h_p1)
            if i != len(self.mlp_p1_layers) - 1:
                h_p1 = self.act_function(h_p1)
            else:
                z_t_mu = h_p1[:, :self.mlp_p1_dims[-1]]
                z_t_logvar = h_p1[:,self.mlp_p1_dims[-1]:]
   
        for i in range(self.sample_freq):
            if i == 0:
                z_t = self.reparameterize(z_t_mu, z_t_logvar)
                z_t = torch.unsqueeze(z_t, 0)
            else:
                z_t_ = self.reparameterize(z_t_mu, z_t_logvar)
                z_t_ = torch.unsqueeze(z_t_, 0)
                z_t = torch.cat([z_t,z_t_], 0)
        z_t = torch.mean(z_t,0)

        h_p2 = z_t
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)   
        return  z_t, h_p2
        
    def forward(self,x_T):

        if self.z0_mode == 0 or self.z0_mode == 1: # 1.全部user共享
            self.zt_last = self.z0.expand(x_T.shape[1],self.mlp_p1_dims[-1]).cuda()

        for t in range(1,len(x_T)+1): # t=1:T
            #print(x_T[t-1].shape)  #(bs,n_item)
            mu,logvar = self.encode(x_T[t-1])
            a_t = self.reparameterize(mu,logvar) 
            #print(a_t.shape) #(bs,a_dim)
            if self.bn:
                a_t = self.batchnorm_a(a_t)

            z_t, rec_CD = self.decode(a_t, self.zt_last) 
            z_t = z_t.reshape(len(z_t),-1) # b x k

            # update self.zt_last of users who have training interactions in environment t
            if self.mask:
                mask = torch.sum(x_T[t-1],1).reshape(-1,1)
                mask[mask>0] = 1
                self.zt_last = self.zt_last * (torch.ones_like(mask) - mask) + z_t * mask
            else:
                self.zt_last = z_t
            self.zt_last = self.zt_last.detach()

            # matrix 
            # scale rec_CD to the same scale of W_CD
            rec_xt = rec_CD
            #rec_xt = torch.sum(rec_xt,1) 
            assert x_T[t-1].shape==rec_xt.shape

            mu = torch.unsqueeze(mu,0)
            logvar = torch.unsqueeze(logvar,0)
            rec_xt = torch.unsqueeze(rec_xt,0)
            if t==1:
                mu_T = mu
                logvar_T = logvar
                rec_xT = rec_xt
            else:
                mu_T = torch.cat((mu_T,mu),0)             # T,bs,a_dims
                logvar_T = torch.cat((logvar_T,logvar),0) # T,bs,a_dims
                rec_xT = torch.cat((rec_xT,rec_xt),0)     # T,bs,n_items
        reg_loss = self.reg_loss()
        return rec_xt[0], rec_xT, mu_T, logvar_T, reg_loss

    def init_weights(self):
        for layer in self.mlp_q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.mlp_p1_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

            
        for layer in self.mlp_p2_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            
    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        return reg_loss

class MultiVAE_Z_W(nn.Module):
    '''
    Version 2-b: basic framework of multivae + z (q + concat(a,z) + p1 + p2) + W
    '''
    def __init__(self, mlp_q_dims, mlp_p1_dims, mlp_p2_dims,\
                 c_dim=2, dropout=0.5, bn=0, sample_freq=1, regs=0, act_function='tanh', T=1, sigma=1, tau=0.1, std=1, n_users=None, z0_mode=1, mask=0, inference=1,w_sigma=0.5):
        super(MultiVAE_Z_W, self).__init__()
        self.mlp_q_dims = mlp_q_dims
        self.mlp_p1_dims = mlp_p1_dims
        self.mlp_p2_dims = mlp_p2_dims
        self.drop = nn.Dropout(dropout)

        self.bn = bn
        self.sample_freq = sample_freq
        self.regs = regs

        if act_function == 'tanh':
            self.act_function = torch.tanh
        elif act_function == 'sigmoid':
            self.act_function = torch.sigmoid
        elif act_function == 'relu':
            self.act_function = torch.relu

            
        self.T = T # n_time
        self.sigma = sigma
        self.w_sigma = w_sigma

        self.tau = tau
        self.std = std
        self.z0_mode = z0_mode
        self.mask = mask
        self.infer_mode = inference

        if self.z0_mode ==0:
            self.z0 = torch.zeros((1,mlp_p1_dims[-1]))
        elif self.z0_mode == 1:    
            self.z0 = torch.randn(1,mlp_p1_dims[-1])
        
        self.zt_last = None

        # 初始化randn,方差变小 
        W_CK = torch.zeros(c_dim, mlp_p1_dims[-1]).normal_(0.5, self.w_sigma)
        W_CK = torch.clamp(W_CK,0,1).cuda()
        self.W_CK = torch.nn.Parameter(W_CK,requires_grad=True)

        W_CD = torch.zeros(c_dim, mlp_p2_dims[-1]).normal_(0.5, self.w_sigma)
        W_CD = torch.clamp(W_CD,0,1).cuda()
        self.W_CD = torch.nn.Parameter(W_CD,requires_grad=True)

        self.noise_W_CK = torch.zeros_like(self.W_CK).normal_(0,1)
        self.noise_W_CD = torch.zeros_like(self.W_CD).normal_(0,1)

        # Last dimension of q- network is for mean and variance
        temp_q_dims = mlp_q_dims[:-1] + [mlp_q_dims[-1] * 2]
        temp_p1_dims = mlp_p1_dims[:-1] + [mlp_p1_dims[-1] * 2]
        temp_p2_dims = mlp_p2_dims
        self.mlp_q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for 
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.mlp_p1_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_p1_dims[:-1],temp_p1_dims[1:])]) 
        self.mlp_p2_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_p2_dims[:-1],temp_p2_dims[1:])]) 
        if self.bn:
            self.batchnorm_a = nn.BatchNorm1d(mlp_q_dims[-1]) # a_dims
            self.batchnorm_z = nn.BatchNorm1d(mlp_p1_dims[-1]) #z_dims

        self.init_weights()

    def encode(self, encoder_input):
        encoder_input = F.normalize(encoder_input)
        h = self.drop(encoder_input)
        for i, layer in enumerate(self.mlp_q_layers):
            h = layer(h)
            if i != len(self.mlp_q_layers) - 1:
                h = self.act_function(h) 
            else:
                mu = h[:, :self.mlp_q_dims[-1]]
                logvar = h[:,self.mlp_q_dims[-1]:]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros_like(std).normal_(mean=0, std=self.std)
            return mu + eps * std
        else:
            return mu

    def decode(self, a_t, z_t_last):
        #z_t_last = F.normalize(z_t_last,dim=1)
        h_p1 = torch.cat((a_t,z_t_last),1)
        for i, layer in enumerate(self.mlp_p1_layers):
            h_p1 = layer(h_p1)
            if i != len(self.mlp_p1_layers) - 1:
                h_p1 = self.act_function(h_p1)
            else:
                z_t_mu = h_p1[:, :self.mlp_p1_dims[-1]]
                z_t_logvar = h_p1[:,self.mlp_p1_dims[-1]:]
   
        for i in range(self.sample_freq):
            if i == 0:
                z_t = self.reparameterize(z_t_mu, z_t_logvar)
                z_t = torch.unsqueeze(z_t, 0)
            else:
                z_t_ = self.reparameterize(z_t_mu, z_t_logvar)
                z_t_ = torch.unsqueeze(z_t_, 0)
                z_t = torch.cat([z_t,z_t_], 0)
        z_t = torch.mean(z_t,0)

        z_t = torch.unsqueeze(z_t,1)
        W_CK  = self.W_CK + self.sigma * self.noise_W_CK.normal_(0,1).cuda() * self.training
        W_CK = torch.clamp(W_CK, 0, 1) # clipped to (0,1)
        W_CK = torch.softmax(W_CK/self.tau, 0) # softmax on C dim

        h_p2 = z_t * W_CK
        for i, layer in enumerate(self.mlp_p2_layers):
            h_p2 = layer(h_p2)
            if i != len(self.mlp_p2_layers) - 1:
                h_p2 = self.act_function(h_p2)   
        return  z_t, h_p2
        

    def forward(self,x_T):

        if self.z0_mode == 0 or self.z0_mode == 1: # 1.全部user共享
            self.zt_last = self.z0.expand(x_T.shape[1],self.mlp_p1_dims[-1]).cuda()

        
        if self.training or self.infer_mode==1 or self.infer_mode==2:
            for t in range(1,len(x_T)+1): # t=1:T
                #print(x_T[t-1].shape)  #(bs,n_item)
                mu,logvar = self.encode(x_T[t-1])
                a_t = self.reparameterize(mu,logvar) 
                #print(a_t.shape) #(bs,a_dim)
                if self.bn:
                    a_t = self.batchnorm_a(a_t)

                z_t, rec_CD = self.decode(a_t, self.zt_last) 
                
                z_t = z_t.reshape(len(z_t),-1) # b x k

                # update self.zt_last of users who have training interactions in environment t
                if self.mask:
                    mask = torch.sum(x_T[t-1],1).reshape(-1,1)
                    mask[mask>0] = 1
                    self.zt_last = self.zt_last * (torch.ones_like(mask) - mask) + z_t * mask
                else:
                    self.zt_last = z_t
                self.zt_last = self.zt_last.detach()

                # matrix 
                W_CD  = self.W_CD + self.sigma * self.noise_W_CD.normal_(0,1).cuda() * self.training
                W_CD = torch.clamp(W_CD, 0, 1)
                W_CD = torch.softmax(W_CD/self.tau, 0)
                # scale rec_CD to the same scale of W_CD
                rec_xt = rec_CD * W_CD / self.tau
                
                rec_xt = torch.sum(rec_xt,1) 
                assert x_T[t-1].shape==rec_xt.shape
                
                mu = torch.unsqueeze(mu,0)
                logvar = torch.unsqueeze(logvar,0)
                rec_xt = torch.unsqueeze(rec_xt,0)
                if t==1:
                    mu_T = mu
                    logvar_T = logvar
                    rec_xT = rec_xt
                else:
                    mu_T = torch.cat((mu_T,mu),0)             # T,bs,a_dims
                    logvar_T = torch.cat((logvar_T,logvar),0) # T,bs,a_dims
                    rec_xT = torch.cat((rec_xT,rec_xt),0)     # T,bs,n_items
            reg_loss = self.reg_loss()
            return rec_xt[0], rec_xT, mu_T, logvar_T, reg_loss

    def init_weights(self):
        for layer in self.mlp_q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.mlp_p1_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

            
        for layer in self.mlp_p2_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
            

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in self.mlp_q_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p1_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        for name, parm in self.mlp_p2_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.regs * (1/2) * parm.norm(2).pow(2)
        return reg_loss


def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1)) # 对item sum
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD

def loss_variance(model, rec_xT, xT, mu_T, logvar_T, anneal):
    assert len(rec_xT)==len(xT)==len(mu_T)==len(logvar_T)
    grad_avg = 0
    grad_list = []
    loss_T = 0
    penalty = 0
    for t in range(len(rec_xT)):
        # 没有历史交互的user不算入loss
        loss_mask = torch.sum(xT[t],1).reshape(-1,1)
        loss_mask[loss_mask>0] = 1
        rec_xt = rec_xT[t] * loss_mask

        loss_t = loss_function(rec_xt, xT[t], mu_T[t], logvar_T[t], anneal)
        loss_T += loss_t/len(rec_xT)

        for name, param in model.named_parameters():
            if name == 'mlp_p2_layers.0.weight':
                grad_single = grad(loss_t, param, create_graph=True)[-1].reshape(-1)
        #grad_single = grad(loss_t, model.parameters(), create_graph=True)[-6].reshape(-1) # 对谁求梯度？shape:20(Dxk) p2,p1,p1p2
        grad_avg += grad_single / len(rec_xT)
        grad_list.append(grad_single)

    # penalty = torch.tensor(np.zeros(self.input_dim, dtype=np.float32))
    for gradient in grad_list:
        penalty += (gradient - grad_avg)**2
        # print(f'penalty shape:{penalty.shape}')

    return loss_T , penalty

def e_recon_loss(rec_xT,xT,mu_T,logvar_T,anneal=1):
    loss_T = 0
    for t in range(len(rec_xT)):
        loss_t = loss_function(rec_xT[t], xT[t], mu_T[t], logvar_T[t], anneal)
        loss_T += loss_t
    avg_loss_T = loss_T/len(rec_xT)
    return avg_loss_T

def loss_penalty(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

