import os
import numpy as np
import h5py

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

from path import DATA_PATH


def stepfunctionlist(milestones):
    """
    return a step function like list with minestones.
    Example:
    minestones = np.ndArray([4, 9, 14])
    we generate a np.ndArray of len 15:
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
                 4              9             14
    """
    lens = milestones[-1] + 1
    res = np.zeros(shape=(lens,))
    for ms in milestones:
        res[ms+1:] += 1
    return res
            

class MultiSameDemos(Dataset):
    def __init__(
        self,
        train_split=0.5,
        multiplier=20,
        seed=None,
        distribution=[2, 2, 1, 1, 1, 2, 0],
        obs='state'
    ):
        super().__init__()
        self.train_split=train_split
        self.seed = seed
        self.multiplier=multiplier
        self.mode = "train"
        
        assert obs in ['state']
        self.traj_paths = [
            f'{DATA_PATH}coffee_d{distribution[0]}',
            f'{DATA_PATH}threading_d{distribution[1]}',
            f'{DATA_PATH}stack_three_d{distribution[2]}',
            f'{DATA_PATH}hammer_cleanup_d{distribution[3]}',
            f'{DATA_PATH}mug_cleanup_d{distribution[4]}',
            f'{DATA_PATH}three_piece_assembly_d{distribution[5]}',
            f'{DATA_PATH}nut_assembly_d{distribution[6]}'
        ]
        
        self.seq_length = 440
        self.train_base_idxs = []
        self.train_lengths = []
        self.train_splits = []
        self.train_traj_len = []
        self.train_data = []
        
        self.eval_base_idxs = []
        self.eval_lengths = []
        self.eval_splits = []
        self.eval_traj_len = []
        self.eval_data = []
        
        np.random.seed(self.seed)
        # set base_idx, lengths, train_splits (org idx), eval_splits (org idx)
        train_base_idx = 0
        eval_base_idx = 0
        for path in self.traj_paths:
            idxs = os.listdir(path)
            idxs = np.array([int(idx) for idx in idxs])
            idxs = np.random.permutation(idxs)
            idxs_train = np.repeat(idxs[:int(len(idxs)*train_split)], repeats=multiplier, axis=0)
            idxs_eval = idxs[int(len(idxs)*train_split):]
            
            # training set
            self.train_base_idxs.append(train_base_idx)
            self.train_lengths.append(int(len(idxs_train)))
            self.train_splits.append(idxs_train)
            print(f'path: {path}')
            print(f'train base idx: {self.train_base_idxs[-1]}')
            print(f'train length: {self.train_lengths[-1]}')
            print(f'train idxs:\n{self.train_splits[-1]}\n')
            ## read real data in
            for idx in idxs_train:
                npy_data = np.load(file=f'{path}/{idx}/state.npy')
                self.train_traj_len.append(npy_data.shape[0])
                if npy_data.shape[0] < self.seq_length:
                    npy_data = np.concatenate([npy_data, np.repeat(npy_data[-1:, :], repeats=self.seq_length-npy_data.shape[0], axis=0)], axis=0)
                self.train_data.append(npy_data)
            train_base_idx += len(idxs_train)
            
            # evaluation set
            self.eval_base_idxs.append(eval_base_idx)
            self.eval_lengths.append(int(len(idxs_eval)))
            self.eval_splits.append(idxs_eval)
            print(f'path: {path}')
            print(f'eval base idx: {self.eval_base_idxs[-1]}')
            print(f'eval length: {self.eval_lengths[-1]}')
            print(f'eval idxs:\n{self.eval_splits[-1]}\n')
            ## read real data in
            for idx in idxs_eval:
                npy_data = np.load(file=f'{path}/{idx}/state.npy')
                self.eval_traj_len.append(npy_data.shape[0])
                if npy_data.shape[0] < self.seq_length:
                    npy_data = np.concatenate([npy_data, np.repeat(npy_data[-1:, :], repeats=self.seq_length-npy_data.shape[0], axis=0)], axis=0)
                self.eval_data.append(npy_data)
            eval_base_idx += len(idxs_eval)
    
    def custom_shuffle(self):
        # actually reset idx at every epoch
        # but you have too control it outside... sorry about that
        if self.mode == 'eval':
            return
        # only set when training
        self.dataloader_idxs = []
        for i in range(len(self.traj_paths)):
            dataloader_idx = np.random.permutation(self.train_lengths[i])
            dataloader_idx += self.train_base_idxs[i]
            self.dataloader_idxs.append(dataloader_idx)
        # then gather them, mix then per task
        self.dataloader_idx = []
        group_idx = 0
        len_all = len(self.train_data)
        while len_all > 0:
            if len(self.dataloader_idxs[group_idx]) > 0:
                self.dataloader_idx.append(self.dataloader_idxs[group_idx][-1])
                self.dataloader_idxs[group_idx] = self.dataloader_idxs[group_idx][:-1]
                group_idx = (group_idx + 1) % len(self.traj_paths)
                len_all -= 1
            else:
                group_idx = (group_idx + 1) % len(self.traj_paths)
    
    def set_mode(self, mode="train"):
        self.mode = mode
        
    def info(self):  # Get observation and action shapes.
        return self.train_data[0].shape[-1], self.train_data[0].shape[-1]
        
    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "eval":
            return len(self.eval_data)
    
    def __getitem__(self, index):
        if self.mode == "train":
            idx = self.dataloader_idx[index]
            data_dict = {
                's': self.train_data[idx][:-1, :].astype(np.float32),
                'a': self.train_data[idx][1:, :].astype(np.float32),
                't': np.array([0]).astype(np.float32),
                'unified_t': np.arange(start=0, stop=self.seq_length-1, step=1.0, dtype=np.float32) / (self.seq_length-1),
                'index': idx
            }
            
        elif self.mode == "eval":
            idx = index
            data_dict = {
                's': self.eval_data[idx][:-1, :].astype(np.float32),
                'a': self.eval_data[idx][1:, :].astype(np.float32),
                't': np.array([0]).astype(np.float32),
                'unified_t': np.arange(start=0, stop=self.seq_length-1, step=1.0, dtype=np.float32) / (self.seq_length-1),
                'index': idx
            }
        else:
            print("Unknown Mode")
            assert False
            
        return data_dict


def get_padding_fn(data_names):
    assert 's' in data_names, 'Should at least include `s` in data_names.'

    def pad_collate(*args):
        assert len(args) == 1
        output = {k: [] for k in data_names}
        for b in args[0]:  # Batches
            for k in data_names:
                output[k].append(torch.from_numpy(b[k]))

        # Include the actual length of each sequence sampled from a trajectory.
        # If we set max_seq_length=min_seq_length, this is a constant across samples.
        output['lengths'] = torch.tensor([len(s) for s in output['s']])

        # Padding all the sequences.
        for k in data_names:
            output[k] = pad_sequence(output[k], batch_first=True, padding_value=0)

        return output

    return pad_collate