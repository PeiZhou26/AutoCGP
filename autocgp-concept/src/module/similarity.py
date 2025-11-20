import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSimilarity:
    def __init__(self, n_v, th=0.2, sim_e=None, device=0):
        self.n_v = n_v
        self.th = th
        self.device = device
        if sim_e is not None:
            sim_e = torch.tensor(sim_e, device=device)
            assert sim_e.shape == (n_v, n_v)
            self.sim_e = sim_e
        else:
            self.sim_e = 1.0 - torch.eye(n_v, device=device)
    
    def reset_sim_e(self, sim_e=None):
        sim_e = torch.tensor(sim_e, device=self.device)
        assert sim_e.shape == (self.n_v, self.n_v)
        self.sim_e = sim_e
        
    def reset_th(self, th):
        self.th = th
    
    def get_cluster(self, heuristic_more=False):
        # first find out the effective edge
        e = torch.less(self.sim_e ,self.th)
        # cnt the number, for heuristic methods
        cnt = torch.sum(e, dim=1)
        if heuristic_more:
            # from the points with less effective edges to more.
            _, v_index = cnt.sort(descending=False)
        else:
            # from the points with more effective edges to less.
            _, v_index = cnt.sort(descending=True)
            
        clusters = [] 
        for i in v_index:
            # considering v_i now
            for i_c in range(len(clusters)):
                # cluster should be a int torch tensor of one dimension
                # if v_i is 'close' to every v_? in this cluster, we should put it in too
                if not False in e[i][clusters[i_c]]:
                    clusters[i_c] = torch.cat([clusters[i_c], torch.tensor([i], device=self.device)])
            # otherwise, it becomes a new cluster
            clusters.append(torch.tensor([i], device=self.device))
        
        return clusters
