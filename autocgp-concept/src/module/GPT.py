"""
Code for the module architecture of CoTPC, based on the GPT implementation.
Some of the key hyper-parameters are explained in GPTConfig.

References:
(1) https://github.com/karpathy/minGPT
(2) https://github.com/kzl/decision-transformer
"""

import sys
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .module_util import MLP
from .HNnet import HNEncoder


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class SelfAttention(nn.Module):
    """
    Self Attn Layer, without causal, can choose to be of 3 * blocksize or 2 * blocksize
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        
    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalSelfAttention(nn.Module):
    """
    Self Attn Layer, can choose to be of 3 * blocksize or 2 * blocksize
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        if config.attn_type == 'w_key':
            block_size = config.block_size * 3
        elif config.attn_type == 'wo_key':
            block_size = config.block_size * 2
        else:
            block_size = config.block_size
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # Masked attention

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    A Transformer block,
        select it to be:
            BasicCausalSelfAttention or ActCausalSelfAttention
        with config.block_type {'basic', 'act'}
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if config.use_causal:
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = SelfAttention(config)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BlockLayers(nn.Module):
    """
    A wrapper class for a sequence of Transformer blocks
        select it to be:
            BasicCausalSelfAttention or ActCausalSelfAttention
        with config.block_type {'basic', 'act'}
    """

    def __init__(self, config):
        super().__init__()
        # Register all the individual blocks.
        self.block_list = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.n_head = config.n_head

    def forward(self, x, skip_feature=None):
        # B, T, _ = x.shape
        output = []  # Also keep the intermediate results.

        if skip_feature is None:
            for block in self.block_list:
                x = block(x)
                output.append(x)
            return x, output

        else:
            for block in self.block_list:
                x = block(x) + skip_feature.pop()
                output.append(x)
            return x, output


class KeyNet(nn.Module):
    """
    KeyNet
    We try to recognize k_s[0:(t-1)] from s[0:(t-1)], a[0:(t-2)].
    """
    def __init__(self, config, state_dim=-1, action_dim=-1, key_dim=-1, mid_dim=256, use_global=True):
        super().__init__()

        assert state_dim > 0 and action_dim > 0 and key_dim > 0
        assert config.attn_type == 'wo_key'
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim
        self.block_size = config.block_size * 2  # state + action

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.use_global = use_global
        if use_global:
            self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings
        self.state_encoder = MLP(state_dim, config.n_embd, hidden_dims=[mid_dim])
        self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[mid_dim])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)

        self.ln = nn.LayerNorm(config.n_embd)

        # Key soft predictor & state predictor
        self.key_predictor = MLP(config.n_embd, key_dim, hidden_dims=[mid_dim, mid_dim])
        self.state_predictor = MLP(config.n_embd, state_dim, hidden_dims=[mid_dim, mid_dim])

        # print('init module in BasicNet')
        self.apply(self._init_weights)
        # print('init module in BasicNet done')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, timesteps, actions=None):
        B, T = states.shape[0], states.shape[1]
        state_embeddings = self.state_encoder(states)

        # Embeddings for state (action, and key state query) tokens.
        token_embeddings = torch.zeros([B, T*2, self.config.n_embd],
                                       dtype=torch.float32,
                                       device=states.device)

        # states embeddings in token
        token_embeddings[:, :(T*2):2, :] = state_embeddings

        if actions is not None:
            # action embeddings in token
            assert actions.shape[1] >= (T - 1)
            action_embeddings = self.action_encoder(actions[:, :(T-1)])
            token_embeddings[:, 1:(T*2-1):2, :] = action_embeddings

        # Set up position embeddings similar to that in Decision Transformer.
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 2, dim=1)
        if self.use_global:
            global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
            timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
            global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
            pos_emb = global_pos_emb + local_pos_emb
        else:
            pos_emb = local_pos_emb

        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, _ = self.blocks(x)
        x = self.ln(x)

        key_soft = self.key_predictor(x[:, 0:(2*T):2, :])
        # only reconstruct s1, s2, ..., s(T-1)
        state_trans = self.state_predictor(x[:, 1:(2*T-2):2, :])

        return key_soft, state_trans


class ExplicitSAHNGPTDelta(nn.Module):
    def __init__(self, config, state_dim=-1, action_dim=-1, key_dim=-1, KT=0.1, mid_dim=256, use_global=True):
        super().__init__()

        assert state_dim > 0 and action_dim > 0 and key_dim > 0
        assert config.attn_type == 'w_key'

        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.key_dim = key_dim
        self.KT = KT
        self.n_embd = config.n_embd

        self.block_size = config.block_size * 3

        self.n_state_layer = config.n_state_layer  # numbers of layers for reward generation

        # layers to get state grad
        self.reward_hn = HNEncoder(state_dim, config.n_embd, key_dim, config.n_state_layer, 4 * config.n_state_layer)

        # Below tries to use the gradient info to get policy (action prediction)
        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.use_global = use_global
        if use_global:
            self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings & Key embeddings
        self.state_encoder = MLP(state_dim, config.n_embd, hidden_dims=[mid_dim])
        self.use_future_state = config.use_future_state
        if config.use_future_state:
            self.state_grad_encoder = MLP(state_dim * 2, config.n_embd, hidden_dims=[mid_dim])
        else:
            self.state_grad_encoder = MLP(state_dim, config.n_embd, hidden_dims=[mid_dim])
        self.action_encoder = MLP(action_dim, config.n_embd, hidden_dims=[mid_dim])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)

        # Action predictor
        self.action_predictor = MLP(config.n_embd, action_dim, hidden_dims=[mid_dim, mid_dim])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # print(module)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Try to use HyperNet to generate a network
    def get_reward(self, states, keys=None, states_future=None):
        # states (B, T, state_dim)
        # keys (B, T, key_dim * n_state_layer)
        assert keys is not None and states_future is not None
        B, T, _ = states.shape
        # two embedding
        states_emb = self.reward_hn(states, keys)
        states_future_emb = self.reward_hn(states_future, keys)
        
        s_emb_norm = F.normalize(states_emb, p=2.0, dim=-1)
        s_f_emb_norm = F.normalize(states_future_emb, p=2.0, dim=-1)
        cos_score = F.cosine_similarity(s_emb_norm, s_f_emb_norm, dim=-1)  # (B, T)
        r = F.sigmoid(torch.div(cos_score, self.KT))

        return r

    def forward(self, states, timesteps, actions=None, keys=None, states_future=None):
        B, T = states.shape[0], states.shape[1]
        states.requires_grad = True

        # Get state gradient
        r = self.get_reward(states, keys, states_future)
        r_sum = r.sum()
        states_grad = torch.autograd.grad(r_sum, states, retain_graph=True, create_graph=True)[0]

        state_embeddings = self.state_encoder(states)
        token_embeddings = torch.zeros([B, T * 3, self.config.n_embd], dtype=torch.float32, device=states.device)

        token_embeddings[:, 0:(T * 3):3, :] = state_embeddings
        if keys is not None:
            # keys should not be None (???????)
            if self.use_future_state:
                assert states_future is not None
                goal_vecs = torch.cat([states_grad, states_future], dim=-1)
            else:
                goal_vecs = states_grad
            grad_embeddings = self.state_grad_encoder(goal_vecs)
            token_embeddings[:, 1:(T * 3):3, :] = grad_embeddings

        if actions is not None:
            # actions is None when at s0
            # the last action is not used as inputs during ActNet training.
            token_embeddings[:, 2:(T * 3 - 1):3, :] = self.action_encoder(actions[:, :(T - 1)])

        # Set up position embeddings similar to that in Decision Transformer.
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 3, dim=1)
        if self.use_global:
            global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
            timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
            global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
            pos_emb = global_pos_emb + local_pos_emb
        else:
            pos_emb = local_pos_emb
        
        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, _ = self.blocks(x)

        # Always predict next action
        x_action = self.ln(x)
        action_preds = self.action_predictor(x_action[:, 1:(T * 3):3, :])
        # Use it as ActNet, do action prediction
        return action_preds, r


class RecNet(nn.Module):
    def __init__(self, config, state_dim=-1, key_dim=-1, mid_dim=256):
        super().__init__()

        assert state_dim > 0 and key_dim > 0
        assert config.attn_type == '-'  # only try to reconstruct key state back to state
        self.config = config
        self.state_dim = state_dim
        self.key_dim = key_dim
        self.block_size = config.block_size

        # key embeddings
        self.key_encoder = MLP(key_dim, config.n_embd, hidden_dims=[mid_dim])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)
        # State predictor
        self.state_predictor = MLP(config.n_embd, state_dim, hidden_dims=[mid_dim, mid_dim])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, keys, skip_feature=None):
        B, T = keys.shape[0], keys.shape[1]
        key_embeddings = self.key_encoder(keys)

        token_embeddings = key_embeddings

        x = token_embeddings
        x = self.drop(x)
        x, _ = self.blocks(x, skip_feature)
        x = self.ln(x)
        state_preds = self.state_predictor(x)

        return state_preds
    
    
class FutureNetAll(nn.Module):
    def __init__(self, config, state_dim=-1, key_dim=-1, mid_dim=256, use_global=True):
        super().__init__()

        assert state_dim > 0 and key_dim > 0
        assert config.attn_type == '-'

        self.config = config
        self.state_dim = state_dim
        self.key_dim = key_dim

        self.block_size = config.block_size

        self.local_pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.use_global = use_global
        if use_global:
            self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))

        # State embeddings & Action embeddings & Key embeddings
        self.state_key_encoder = MLP(state_dim+key_dim, config.n_embd, hidden_dims=[mid_dim])

        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)

        # state predictor
        self.state_predictor = MLP(config.n_embd, state_dim, hidden_dims=[mid_dim, mid_dim])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # print(module)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, states, timesteps, keys=None):
        B, T = states.shape[0], states.shape[1]
        states_keys = torch.cat([states, keys], dim=-1)
        token_embeddings = self.state_key_encoder(states_keys)
        
        # Set up position embeddings similar to that in Decision Transformer.
        local_pos_emb = torch.repeat_interleave(self.local_pos_emb[:, :T, :], 1, dim=1)
        if self.use_global:
            global_pos_emb = torch.repeat_interleave(self.global_pos_emb, B, dim=0)
            timesteps_rp = torch.repeat_interleave(timesteps[:, None], self.config.n_embd, dim=-1)
            global_pos_emb = torch.gather(global_pos_emb, 1, timesteps_rp.long())  # BS x 1 x D
            pos_emb = global_pos_emb + local_pos_emb
        else:
            pos_emb = local_pos_emb
        
        x = token_embeddings + pos_emb

        x = self.drop(x)
        x, _ = self.blocks(x)
        x = self.ln(x)

        state_preds = self.state_predictor(x)
        return state_preds

