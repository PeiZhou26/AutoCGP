import sys
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .module_util import MLP, CondResnetBlockFC


class HNEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim, key_dim, n_layer=2, n_wgroup=2):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.key_dim = key_dim
        self.n_layer = n_layer
        
        # input_encoder
        self.linear_in = nn.Linear(in_dim, emb_dim, bias=False)
        self.linear_out = nn.Linear(emb_dim, emb_dim, bias=False)
        
        # HN part
        ## First, we need [n_layer] encoder
        self.hn_0 = nn.Parameter(torch.randn(size=[1, n_layer, key_dim, n_wgroup]))
        ## Then, a shared decoder that output parameters
        self.hn_1 = nn.Parameter(torch.randn(size=[1, 1, n_wgroup, emb_dim*(emb_dim+1)]))
        ## whether to use skip or not
        # self.use_skip = use_skip  # whether to use skip connection in output network
        self.activation = nn.ReLU()
        
    def forward(self, input, key):
        # input [..., in_dim]
        # key [..., key_dim * n_layer]
        # return [..., emb_dim]
        
        shape_ = torch.tensor(input.shape[:-1])   # [...]
        shape_out = torch.cat([shape_, torch.tensor([self.emb_dim], device=shape_.device)])
        shape_out = tuple([item.item() for item in shape_out])
        input_f = input.view(-1, self.in_dim).contiguous()  # [???, in_dim]
        key_f = key.view(-1, self.n_layer, 1, self.key_dim).contiguous()  # [???, n_layer, 1, key_dim]
        L = input_f.shape[0]
        
        x_f = self.linear_in(input_f)  # [L, emb_dim]
        
        x_hidden = x_f.unsqueeze(-2).contiguous()  # [L, 1, emb_dim]
        
        # first prepare paramter from HN output
        whn = torch.matmul(key_f, self.hn_0)  # [L, n_layer, 1, n_wgroup]
        params = torch.matmul(whn, self.hn_1).squeeze(-2).contiguous()  # [L, n_layer, emb_dim*(emb_dim+1)]
        
        # split parameters
        w = params[..., :(-self.emb_dim)].view(L, self.n_layer, self.emb_dim, self.emb_dim)  # [L, n_layer, emb_dim, emb_dim]
        w = w.permute(1, 0, 2, 3).contiguous()  # [n_layer, L, emb_dim, emb_dim]
        
        b = params[..., (-self.emb_dim):].view(L, self.n_layer, self.emb_dim)  # [L, n_layer, emb_dim]
        b = b.permute(1, 0, 2).unsqueeze(-2).contiguous()  # [n_layer, L, 1, emb_dim]
        
        # now we carry out forward process:
        for i in range(self.n_layer):
            wi = w[i]  # [L, emb_dim, emb_dim]
            bi = b[i]  # [L, 1, emb_dim]
            x_identity = x_hidden
            # linear
            x_hidden = torch.matmul(x_hidden, wi)  # [L, 1, emb_dim]
            x_hidden = torch.add(x_hidden, bi)  # [L, 1, emb_dim]
            
            # residual
            x_hidden = x_hidden + x_identity
            
            # activation
            x_hidden = self.activation(x_hidden)
        
        # then give a output
        x_out = x_hidden.squeeze(-2).view(shape_out).contiguous()  # [L, emb_dim]
        x_out = self.linear_out(x_out)
        return x_out
                
        
        
if __name__ == '__main__':
    hn = HNEncoder(in_dim=8, emb_dim=128, key_dim=128, n_layer=2, n_wgroup=4).to(device=0)
    input = torch.randn(size=[2, 10, 8], device=0)
    key = torch.randn(size=[2, 10, 128*2], device=0)
    out = hn(input, key)
    print(out.shape)
