import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import utilsxs
import argparse
import robosuite as suite
import pickle
import uuid
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import DictConfig, OmegaConf, open_dict
import omegaconf
from utilsxs.model.diffusion_model import get_resnet, replace_bn_with_gn
from utilsxs.model.encoder import ResnetConv
import random
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DDIMScheduler

import cv2
import collections
import os.path as osp
import hydra
import imageio
import numpy as np
import torch
from PIL import Image
import random
from collections import deque
import queue
from utilsxs.dataset.robomimic_dp_dataset import (
    normalize_data,
    unnormalize_data,
)
import plotly.graph_objects as go
from utilsxs.utility.transform import get_transform_pipeline

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import mimicgen_envs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_number", default=99, type=int,help="ckpt_number")
    parser.add_argument("--action_horizon", default=4, type=int,help="action_horizon")
    parser.add_argument('--pretrain_path', type=str, default=' ', help="pretrain_path")
    
    return parser.parse_args()

def load_checkpoint(ema, model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    print(checkpoint.keys())
    if 'ema_state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['ema_state_dict'])
        ema.averaged_model.load_state_dict(checkpoint['ema_state_dict'])
        return None
        
    model.load_state_dict(checkpoint['model_state_dict'])
    if len(checkpoint.keys()) > 2:
        ema.averaged_model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        ema.averaged_model.load_state_dict(checkpoint['model_state_dict'])

def convert_images_to_tensors(images_arr, pipeline=None):
    images_tensor = np.transpose(images_arr, (0, 3, 1, 2))  # (T,dim,h,w)
    images_tensor = torch.tensor(images_tensor, dtype=torch.float32) / 255
    if pipeline is not None:
        images_tensor = pipeline(images_tensor)
    return images_tensor

def configure_environment(cfg, options):
    camera_names = ["agentview"]
    if cfg.use_wrist:
        camera_names.append("robot0_eye_in_hand")

    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_heights=84,  # set camera height
        camera_widths=84,  # set camera width
        control_freq=20,
        camera_names=camera_names,
    )
    return env

def process_images(obs, camera_name, eval_cfg, obs_horizon):
    image = obs[camera_name]
    image = image[::-1, :, :]  # Flip the image vertically
    resized_image = cv2.resize(image, eval_cfg.bc_resize)
    return collections.deque([resized_image] * obs_horizon, maxlen=obs_horizon)

def generate_key_from_value_corrected(value):
    words = value.split('_')
    suffix = words[-1]  
    transformed = ''.join(word.capitalize() for word in words[:-1])
    key_name = f"{transformed}_{suffix.capitalize()}"
    return key_name

def eval_diffusion_bc():
    print("start evaluation")
    args = parse_args()
    
    ckpt_number = args.ckpt_number
    action_horizon = args.action_horizon
    model_path = args.pretrain_path

    save_path = f'{model_path}/ckpt_number{ckpt_number}'
    cfg = OmegaConf.load(f"{model_path}/hydra_config.yaml")
    
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    proto_horizon = cfg.proto_horizon

    pickle_path = os.path.join(model_path, "stats.pickle")

    # Open the pickle file in binary read mode and load the object.
    with open(pickle_path, "rb") as f:
        all_stats = pickle.load(f)

    if cfg.vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        vision_encoder = ResnetConv(embedding_size=cfg.vision_feature_dim)
        vision_encoder = replace_bn_with_gn(vision_encoder)
        if cfg.use_wrist:
            wrist_vision_encoder = ResnetConv(embedding_size=cfg.vision_feature_dim)
            wrist_vision_encoder = replace_bn_with_gn(wrist_vision_encoder)
    
    vision_feature_dim = cfg.vision_feature_dim * 2 if cfg.use_wrist else cfg.vision_feature_dim
    
    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim
    num_protos = cfg.num_protos
    proto_dim = cfg.num_protos
    tasks = [generate_key_from_value_corrected(value) for value in cfg.task_names]
    task_stats = {cap: orig for cap, orig in zip(tasks, cfg.task_names)}

    noise_pred_net = hydra.utils.instantiate(
        cfg.noise_pred_net,
        global_cond_dim=vision_feature_dim * obs_horizon +
        obs_dim * obs_horizon + proto_horizon * proto_dim,
    )
        
    nets = {
        "vision_encoder": vision_encoder,
        "noise_pred_net": noise_pred_net,
        }
    
    proto_pred_net = hydra.utils.instantiate(
            cfg.proto_pred_net,
            input_dim=vision_feature_dim * obs_horizon + obs_dim * obs_horizon,
            proto_dim = num_protos,
        )
    nets["proto_pred_net"] = proto_pred_net

    if cfg.use_wrist:
        nets["wrist_vision_encoder"] = wrist_vision_encoder
        
    nets = nn.ModuleDict(nets)
    
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    
    device = torch.device("cuda")
    _ = nets.to(device)

    ema = EMAModel(model=nets, power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    checkpoint_path = os.path.join(model_path, f"ckpt_{ckpt_number}.pt")
    load_checkpoint(ema, nets, optimizer, checkpoint_path)
    del nets, optimizer
    
    for task_name in tasks:
        suc = 0
        tasks_list = []
        for seed in range(1000, 1050):
            print('task_name', task_name, 'seed', seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            eval_cfg = cfg.eval_cfg
            imgs = []
            imgs_wrist = []
            
            # Create dict to hold options that will be passed to env creation call
            options = {}
            options["env_name"] = task_name
            options["robots"] = 'Panda'
            options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")
            
            env = configure_environment(cfg, options)
            obs = env.reset()
            env.viewer.set_camera(camera_id=0)

            # Process images
            img_obs_deque = process_images(obs, 'agentview_image', eval_cfg, obs_horizon)
            if cfg.use_wrist:
                wrist_img_obs_deque = process_images(obs, 'robot0_eye_in_hand_image', eval_cfg, obs_horizon)
            
            # get first observation
            max_steps = min(eval_cfg.max_steps,600)
            # keep a queue of last 2 steps of observations
            obs_horizon = eval_cfg.obs_horizon
                   
            state_t = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos']])
            obs_deque = collections.deque([state_t] * obs_horizon, maxlen=obs_horizon)
            
            done = False
            step_idx = 0
            rewards = list()
            B = 1

            while not done:
                with torch.no_grad():
                    # stack the last obs_horizon (2) number of observations
                    obs_seq = np.stack(obs_deque)
                    visual_seq = np.stack(img_obs_deque)
                    visual_seq = convert_images_to_tensors(visual_seq, None).cuda()
                    visual_feature = ema.averaged_model["vision_encoder"](visual_seq)  # (T,visual_feature)
                    nobs = normalize_data(obs_seq, stats=all_stats[task_stats[task_name]]["obs"])  # (T,obs)
                    nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
                    obs_feature_components = [visual_feature, nobs]
                    
                    if cfg.use_wrist:
                        wrist_seq = np.stack(wrist_img_obs_deque)
                        wrist_seq = convert_images_to_tensors(wrist_seq, None).cuda()
                        wrist_visual_feature = ema.averaged_model["wrist_vision_encoder"](wrist_seq)  # (T,visual_feature)
                        obs_feature_components.insert(1, wrist_visual_feature)
                    obs_feature = torch.cat(obs_feature_components, dim=-1).unsqueeze(0).float()
                    obs_feature = obs_feature.flatten(start_dim=1)
                    predict_proto = ema.averaged_model["proto_pred_net"](obs_feature)
                    
                    if cfg.proto_classifier_softmax:
                        predict_proto_softmaxemb = nn.functional.softmax(predict_proto, dim=-1)
                        obs_cond = torch.cat([obs_feature.flatten(start_dim=1), predict_proto_softmaxemb.flatten(start_dim=1)], dim=1,)
                    else:
                        obs_cond = torch.cat([obs_feature.flatten(start_dim=1), predict_proto.flatten(start_dim=1)], dim=1,)
                   
                    # initialize action from Guassian noise
                    naction = torch.randn((B, eval_cfg.pred_horizon, eval_cfg.action_dim), device=device)
                    noise_scheduler.set_timesteps(eval_cfg.num_diffusion_iters)
                    for k in noise_scheduler.timesteps:
                        noise_pred = ema.averaged_model["noise_pred_net"](sample=naction, timestep=k, global_cond=obs_cond)
                        naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

                    # unnormalize action
                    naction = naction.detach().to("cpu").numpy() # (B, pred_horizon, action_dim)
                    naction = naction[0]
                    action_pred = unnormalize_data(naction, stats=all_stats[task_stats[task_name]]["actions"])

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    # execute action_horizon number of steps
                    # without replanning
                    for i in range(len(action)):
                        # stepping env
                        obs, reward, done, info = env.step(action[i])
                        state_t = np.concatenate([obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos']])
                        obs_deque.append(state_t)
                        
                        agentview_image = obs['agentview_image']
                        agentview_image = agentview_image[::-1, :, :]
                        raw_env_image = cv2.resize(agentview_image, eval_cfg.bc_resize)
                        img_obs_deque.append(raw_env_image.copy())
                        imgs.append(raw_env_image)
                        rewards.append(reward)

                        if cfg.use_wrist:
                            wrist_image = obs['robot0_eye_in_hand_image']
                            wrist_image = wrist_image[::-1, :, :]
                            raw_wrist_image = cv2.resize(wrist_image, eval_cfg.bc_resize)
                            wrist_img_obs_deque.append(raw_wrist_image.copy())
                            imgs_wrist.append(raw_wrist_image)

                        # update progress bar
                        step_idx += 1
                        print('step_idx', step_idx, 'reward', reward)
                        if reward > 0:
                            suc += 1
                            tasks_list.append(seed-1000)
                            done = True
                            break

                        if step_idx > max_steps:
                            done = True
            
            eval_save_path = os.path.join(save_path, f"{task_name}")
            os.makedirs(eval_save_path, exist_ok=True)
            
            video_save_path = osp.join(eval_save_path, f"eval_{seed}_{env.reward()}.gif")
            imageio.mimsave(video_save_path, imgs)
            
            if cfg.use_wrist:
                video_save_path = osp.join(eval_save_path, f"eval_{seed}_wrist_{env.reward()}.gif")
                imageio.mimsave(video_save_path, imgs_wrist)
            
        suc_string = str(suc) 
        with open(os.path.join(eval_save_path, f"one_task_result.txt"), 'w') as file:
            file.write(f"{suc_string}\n")
            list_string = ', '.join(map(str, tasks_list))
            file.write(list_string + '\n')

if __name__ == "__main__":
    eval_diffusion_bc()
