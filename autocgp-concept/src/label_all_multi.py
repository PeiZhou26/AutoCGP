"""
Use all checkpoints to label multi
"""

import os, json, shutil, paramiko, time, portalocker
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
import h5py
import cv2

import torch
import torchvision

from autocot import (
    RecNetConfig,
    KeyNetConfig,
    FutureNetConfig,
    ExplicitSAHNGPTDeltaConfig,
    AutoCoT
)

from path import MODEL_PATH, DATA_PATH


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters regarding the demo dataset (used to gather eval_ids)
    parser.add_argument('--task', type=str, help="Task for experiments")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for data spliting.")
    parser.add_argument("--n_traj", default=-1, type=int, help="num of validation trajectory.")
    parser.add_argument("--train_split", default='0.5', type=str, help='Part for training')
    parser.add_argument("--distribution", type=int, nargs='+', help='list of integers indicating mimicgen distribution variance level')

    # Hyper-parameters regarding the module.
    parser.add_argument("--model_name", default='', type=str, help="Model name to be loaded.")
    parser.add_argument("--from_ckpt", default=-1, type=int, help="Ckpt of the module to be loaded.")

    return parser.parse_args()



def mymain(args, ckpt, task_name):
    print(ckpt)
    assert ckpt[-4:] == '.pth' and ckpt[:5] == 'epoch'
    ckpt_name = ckpt[:-4]
    ckpt_idx = ckpt_name[5:]
    
    # Load the module.
    path = os.path.join(MODEL_PATH, f'{args.model_name}/{ckpt}')
    # Load to cpu first to avoid cuda related errors from ManiSkill2.
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    state_dict_from_ckpt, params = ckpt['module'], ckpt['metadata']
    
    
    if params['future_all']:
        json_global = 'glob'
    else:
        json_global = 'woglob'
    
    if 'delta' in params['sa_type']:
        json_delta = '_delta'
    else:
        json_delta = '_wodelta'
    json_name = json_global + json_delta
    json_date = str(args.model_name).split(sep='-')[1]
    print(f'name: {json_name}/{json_date}-{ckpt_name}.zip')

    state_dim = state_dict_from_ckpt['key_net.state_encoder.net.0.weight'].shape[1]
    action_dim = state_dict_from_ckpt['key_net.action_encoder.net.0.weight'].shape[1]
    key_dim = params['dim_key']
    e_dim = params['dim_e']
    print('Loaded ckpt from:', path)
    
    

    from data import MultiSameDemos
    
    print(f"args.distribution {args.distribution}")
    dataset_demo = MultiSameDemos(
        train_split=float(args.train_split),
        multiplier=1,
        seed=args.seed,
        distribution=args.distribution
    )

    key_config = KeyNetConfig(
        n_embd=params['n_embd'],
        n_head=params['n_head'],
        attn_pdrop=float(params['dropout']),
        resid_pdrop=float(params['dropout']),
        embd_pdrop=float(params['dropout']),
        block_size=dataset_demo.seq_length-1,
        n_layer=params['n_key_layer'],
        max_timestep=dataset_demo.seq_length,
        use_causal=params['use_causal']
    )
    rec_config = RecNetConfig(
        n_embd=params['n_embd'],
        n_head=params['n_head'],
        attn_pdrop=float(params['dropout']),
        resid_pdrop=float(params['dropout']),
        embd_pdrop=float(params['dropout']),
        block_size=dataset_demo.seq_length-1,
        n_layer=params['n_rec_layer'],
        max_timestep=dataset_demo.seq_length,
        use_causal=params['use_causal']
    )
    if 'n_future_layer' in params.keys() and params['n_future_layer'] != 0:
        future_config = FutureNetConfig(
            n_embd=params['n_embd'],
            n_head=params['n_head'],
            attn_pdrop=float(params['dropout']),
            resid_pdrop=float(params['dropout']),
            embd_pdrop=float(params['dropout']),
            block_size=dataset_demo.seq_length-1,
            n_layer=params['n_future_layer'],
            max_timestep=dataset_demo.seq_length,
            future_type='all' if params['future_all'] else 'causal'
        )
    else:
        future_config = None
    
    assert params['sa_type'] == 'egpthndelta'
    sa_config = ExplicitSAHNGPTDeltaConfig(
        n_embd=params['n_embd'],
        n_head=params['n_head'],
        attn_pdrop=float(params['dropout']),
        resid_pdrop=float(params['dropout']),
        embd_pdrop=float(params['dropout']),
        block_size=dataset_demo.seq_length-1,
        n_layer=params['n_action_layer'],
        n_state_layer=params['n_state_layer'],
        max_timestep=dataset_demo.seq_length,
        use_skip=params['use_skip'],
        use_future_state=False if 'use_future_state' not in params.keys() else params['use_future_state']
    )

    print(params['vq_n_e'])
    print('stat_dim', state_dim)
    autocot_model = AutoCoT(
        key_config=key_config,
        sa_config=sa_config,
        rec_config=rec_config,
        future_config=future_config,
        vq_n_e=params['vq_n_e'],
        vq_use_r=params['vq_use_r'],
        vq_coe_ema=float(params['vq_coe_ema']),
        KT=float(params['KT']),
        optimizers_config=None,
        scheduler_config=None,
        state_dim=state_dim,
        action_dim=action_dim,
        key_dim=key_dim,
        e_dim=e_dim,
        mid_dim=params['dim_mid'],
        vq_use_ft_emb=params['vq_use_ft_emb'],
        vq_use_st_emb=params['vq_use_st_emb'],
        vq_st_emb_rate=float(params['vq_st_emb_rate']),
        task='multi'
    )

    autocot_model = autocot_model.cuda()
    autocot_model.load_state_dict(state_dict_from_ckpt, strict=False)
    autocot_model.eval()

    bias_sum = 0.0
    
    print(f"ckpt_idx {ckpt_idx}")
    
    task_id_map = {
        'coffee': 0,
        'threading': 1,
        'stack_three': 2,
        'hammer_cleanup': 3,
        'mug_cleanup': 4,
        'three_piece_assembly': 5,
        'nut_assembly': 6,
    }
    
    task_id = task_id_map[task_name]
    task_name = dataset_demo.traj_paths[task_id].split("/")[-1]
    print(task_name)
    
    os.makedirs(f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}", exist_ok=True)
    os.makedirs(f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/train", exist_ok=True)
    os.makedirs(f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/eval", exist_ok=True)
    
    
    
    # len_traj_train_idx = len(dataset_demo.traj_train_idx)
    with open(f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/train.txt", 'w') as fk:
        print('TRAINING SPLIT')
        len_traj_train_idx = len(dataset_demo.train_splits[task_id])
        base_traj_train = dataset_demo.train_base_idxs[task_id]
        
        th_ave = 0.0
        th_max = 0.0
        th_len_ave =0.0
        th_len_max = 0.0
        cnt = 0
        
        for i in range(len_traj_train_idx):
        # for i in range(2):
            i_org = dataset_demo.train_splits[task_id][i]
            i_len = dataset_demo.train_traj_len[base_traj_train+i]
            s = torch.tensor(dataset_demo.train_data[base_traj_train+i][:-1, :].astype(np.float32), device=0)
            a = torch.tensor(dataset_demo.train_data[base_traj_train+i][1:, :].astype(np.float32), device=0)
            t = torch.tensor(np.array([0]).astype(np.float32), device=0)
            unified_t = torch.tensor(np.arange(start=0, stop=dataset_demo.seq_length-1, step=1.0, dtype=np.float32) / (dataset_demo.seq_length-1), device=0)
            s, a, t, unified_t = s.unsqueeze(0), a.unsqueeze(0), t.unsqueeze(0), unified_t.unsqueeze(0)
            
            label_all, label_all_emb = autocot_model.label_one_forward(s, t, unified_t, a)
            label_all = label_all[0, :i_len]
            label_all_emb = label_all_emb[0, :i_len, :]
            print('label_all_emb.shape', label_all_emb.shape)
            # save as numpy
            np.save(file=f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/train/{i_org}_emb.npy", arr=np.array(label_all_emb.to(device='cpu')))
            np.save(file=f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/train/{i_org}_label.npy", arr=np.array(label_all.to(device='cpu')))
            print(f'original idx: {i_org}\n{label_all}')
            fk.write(f'original idx: {i_org}\n')
            
            # we need to get the shape of .png first
            frame_path_example = f"{dataset_demo.traj_paths[task_id]}/{i_org}/0.png"
            assert os.path.exists(frame_path_example)
            frame_example = cv2.imread(frame_path_example)
            height, width, layers = frame_example.shape
            
            # log segmentation
            label = -1
            t_begin = -1
            s = s[0, :i_len, :]
            # th_ave = 0.0
            # th_max = 0.0
            # th_len_ave =0.0
            # th_len_max = 0.0
            # cnt = 0
            
            t = 0
            while t < label_all.shape[0]:
                if label_all[t] != label:
                    if label != -1:
                        
                        print(f"concept {label} [{t_begin},{t-1}]")
                        fk.write(f"concept {label} [{t_begin},{t-1}]\n")
                        
                        # do measure
                        cnt += 1
                        s_be = s[t-1:t, :] - s[t_begin:t_begin+1, :]
                        s_bm = s[t_begin:t, :] - s[t_begin:t_begin+1, :]
                        s_vcos = torch.cosine_similarity(s_be, s_bm, dim=-1)
                        s_vcos = torch.clip(s_vcos, min=-0.999, max=0.999)
                        s_vsin = torch.sin(torch.arccos(s_vcos))
                        s_d = torch.norm(s_bm, dim=-1) * s_vsin
                        s_dmax = torch.max(s_d)
                        print(f"s_dmax: {s_dmax}")
                        th_ave = th_ave * ((cnt - 1) / cnt) + s_dmax * (1 / cnt)
                        th_max = th_ave if th_ave > th_max else th_max
                        th_len_ave = th_len_ave * ((cnt - 1) / cnt) + (t - t_begin) * (1 / cnt)
                        th_len_max = (t - t_begin) if (t - t_begin) > th_len_max else th_len_max 
                        
                        # save video
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_path = f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/train/{label}"
                        os.makedirs(video_path, exist_ok=True)
                        video = cv2.VideoWriter(f"{video_path}/{i_org}.mp4", fourcc, 20.0, (width, height))
                        for iframe in range(t_begin, t, 1):
                            frame_path = f"{dataset_demo.traj_paths[task_id]}/{i_org}/{iframe}.png"
                            if os.path.exists(frame_path):
                                frame = cv2.imread(frame_path)
                                video.write(frame)
                            else:
                                print("Cannot find frame")
                                assert False
                                
                        video.release()
                        cv2.destroyAllWindows()
                        
                    label = label_all[t]
                    t_begin = t
                t += 1
            print(f"concept {label} [{t_begin},{t-1}]")
            fk.write(f"concept {label} [{t_begin},{t-1}]\n")
            
            # do measure
            cnt += 1
            s_be = s[t-1:t, :] - s[t_begin:t_begin+1, :]
            s_bm = s[t_begin:t, :] - s[t_begin:t_begin+1, :]
            s_vcos = torch.cosine_similarity(s_be, s_bm, dim=-1)
            s_vcos = torch.clip(s_vcos, min=-0.999, max=0.999)
            s_vsin = torch.sin(torch.arccos(s_vcos))
            s_d = torch.norm(s_bm, dim=-1) * s_vsin
            s_dmax = torch.max(s_d)
            print(f"s_dmax: {s_dmax}")
            th_ave = th_ave * ((cnt - 1) / cnt) + s_dmax * (1 / cnt)
            th_max = th_ave if th_ave > th_max else th_max
            th_len_ave = th_len_ave * ((cnt - 1) / cnt) + (t - t_begin) * (1 / cnt)
            th_len_max = (t - t_begin) if (t - t_begin) > th_len_max else th_len_max
            
            # save video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/train/{label}"
            os.makedirs(video_path, exist_ok=True)
            video = cv2.VideoWriter(f"{video_path}/{i_org}.mp4", fourcc, 20.0, (width, height))
            for iframe in range(t_begin, t, 1):
                frame_path = f"{dataset_demo.traj_paths[task_id]}/{i_org}/{iframe}.png"
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    video.write(frame)
                else:
                    print("Cannot find frame")
                    assert False
                                
            video.release()
            cv2.destroyAllWindows()
    
    # th_ave = 0.0
    # th_max = 0.0
    # th_len_ave =0.0
    # th_len_max = 0.0
    
        print(f"th_ave: {th_ave}")
        fk.write(f"th_ave: {th_ave}\n")
        print(f"th_max: {th_max}")
        fk.write(f"th_max: {th_max}\n")
        print(f"th_len_ave: {th_len_ave}")
        fk.write(f"th_len_ave: {th_len_ave}\n")
        print(f"th_len_max: {th_len_max}")
        fk.write(f"th_len_max: {th_len_max}\n")
    
    # len_traj_eval_idx = len(dataset_demo.traj_eval_idx)
    with open(f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/eval.txt", 'w') as fk:
        print('EVALUATION SPLIT')
        len_traj_eval_idx = len(dataset_demo.eval_splits[task_id])
        base_traj_eval = dataset_demo.eval_base_idxs[task_id]
        for i in range(len_traj_eval_idx):
        # for i in range(2):
            i_org = dataset_demo.eval_splits[task_id][i]
            i_len = dataset_demo.eval_traj_len[base_traj_eval+i]
            s = torch.tensor(dataset_demo.eval_data[base_traj_eval+i][:-1, :].astype(np.float32), device=0)
            a = torch.tensor(dataset_demo.eval_data[base_traj_eval+i][1:, :].astype(np.float32), device=0)
            t = torch.tensor(np.array([0]).astype(np.float32), device=0)
            unified_t = torch.tensor(np.arange(start=0, stop=dataset_demo.seq_length-1, step=1.0, dtype=np.float32) / (dataset_demo.seq_length-1), device=0)
            s, a, t, unified_t = s.unsqueeze(0), a.unsqueeze(0), t.unsqueeze(0), unified_t.unsqueeze(0)
            label_all, label_all_emb = autocot_model.label_one_forward(s, t, unified_t, a)
            label_all = label_all[0, :i_len]
            label_all_emb = label_all_emb[0, :i_len, :]
            print('label_all_emb.shape', label_all_emb.shape)
            np.save(file=f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/eval/{i_org}_emb.npy", arr=np.array(label_all_emb.to(device='cpu')))
            np.save(file=f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/eval/{i_org}_label.npy", arr=np.array(label_all.to(device='cpu')))
            print(f'original idx: {i_org}\n{label_all}')
            fk.write(f'original idx: {i_org}\n')
            
            # we need to get the shape of .png first
            frame_path_example = f"{dataset_demo.traj_paths[task_id]}/{i_org}/0.png"
            assert os.path.exists(frame_path_example)
            frame_example = cv2.imread(frame_path_example)
            height, width, layers = frame_example.shape
            
            # log segmentation
            label = -1
            t_begin = -1
            t = 0
            while t < label_all.shape[0]:
                if label_all[t] != label:
                    if label != -1:
                        print(f"concept {label} [{t_begin},{t-1}]")
                        fk.write(f"concept {label} [{t_begin},{t-1}]\n")
                        # save video
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_path = f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/eval/{label}"
                        os.makedirs(video_path, exist_ok=True)
                        video = cv2.VideoWriter(f"{video_path}/{i_org}.mp4", fourcc, 20.0, (width, height))
                        for iframe in range(t_begin, t, 1):
                            frame_path = f"{dataset_demo.traj_paths[task_id]}/{i_org}/{iframe}.png"
                            if os.path.exists(frame_path):
                                frame = cv2.imread(frame_path)
                                video.write(frame)
                            else:
                                print("Cannot find frame")
                                assert False
                        
                        video.release()
                        cv2.destroyAllWindows()
                        
                    label = label_all[t]
                    t_begin = t
                t += 1
            print(f"concept {label} [{t_begin},{t-1}]")
            fk.write(f"concept {label} [{t_begin},{t-1}]\n")
            
            # save video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}/eval/{label}"
            os.makedirs(video_path, exist_ok=True)
            video = cv2.VideoWriter(f"{video_path}/{i_org}.mp4", fourcc, 20.0, (width, height))
            for iframe in range(t_begin, t, 1):
                frame_path = f"{dataset_demo.traj_paths[task_id]}/{i_org}/{iframe}.png"
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    video.write(frame)
                else:
                    print("Cannot find frame")
                    assert False
                                
            video.release()
            cv2.destroyAllWindows()
    
    ## first zip it...
    # os.makedirs(f'{task_name}', exist_ok=True)
    # shutil.make_archive(f'{task_name}/multi-{json_date}-{ckpt_name}', 'zip', f"{MODEL_PATH}{args.model_name}/{task_name}/key_{ckpt_idx}")
    # print(f'name: {json_name}/{task_name}/multi-{json_date}-{ckpt_name}.zip, ready to send')
    # ## then send it...
    # succ = send_file(
    #     file_path=f'{task_name}/multi-{json_date}-{ckpt_name}.zip',
    #     destination_path=f'/mnt/lv1/lrz/mimicgen/core/{json_name}/{task_name}/multi-{json_date}-{ckpt_name}.zip',
    #     hostname='147.8.183.108',
    #     port='22000',
    #     username='lrz',
    #     password='Ilrzprzlhlzrorlznrlzerrl4zzlGllrs',
    # )
    # ## then delete it...
    # if os.path.exists(f'{task_name}/multi-{json_date}-{ckpt_name}.zip'):
    #     os.remove(f'{task_name}/multi-{json_date}-{ckpt_name}.zip')
    #     print(f"The file {task_name}/multi-{json_date}-{ckpt_name}.zip has been deleted.")
    # else:
    #     print(f"The file {task_name}/multi-{json_date}-{ckpt_name}.zip does not exist.")



if __name__ == "__main__":

    args = parse_args()
    assert args.model_name, 'Should specify --model_name'
    assert args.from_ckpt > 0, 'Should specify --from_ckpt'
    
    task_name = args.task

    # with open(args.task + '_label_output.txt', 'a') as flabel:
    #     flabel.write(args.model_name + '_' + str(args.from_ckpt) + '\n')
    
    # get all checkpoints in the directory
    for ckpt_path in os.listdir(f'{MODEL_PATH}/{args.model_name}'):
        if ckpt_path[-4:] == '.pth' and ckpt_path[:5] == 'epoch':
            mymain(args, ckpt_path, task_name)
    exit(0)
