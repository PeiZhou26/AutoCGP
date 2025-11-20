import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import utilsxs

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
from omegaconf import DictConfig, OmegaConf
from utilsxs.model.diffusion_model import get_resnet, replace_bn_with_gn, ConceptConditionalUnet1D
from utilsxs.model.encoder import ResnetConv
import random
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import datetime
from tqdm import tqdm

def save_checkpoint(ema, model, optimizer, epoch, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.averaged_model.state_dict() 
    }
    torch.save(checkpoint, file_path)

def load_checkpoint(ema, model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ema.averaged_model.load_state_dict(checkpoint['ema_state_dict'])
    return checkpoint['epoch']

@hydra.main(
    version_base=None,
    config_path="../config/simulation",
    config_name="dp_concept",
)
def train_diffusion_concept(cfg: DictConfig):
    if isinstance(cfg.use_wrist, str):
        cfg.use_wrist = (cfg.use_wrist == 'True')
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    save_dir = os.path.join(cfg.save_dir, folder_name)
    cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "hydra_config.yaml"))
    print(f"output_dir: {save_dir}")
    wandb.init(project=cfg.project_name, name="test000")
    wandb.config.update(OmegaConf.to_container(cfg))
    #set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    # parameters
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    proto_horizon = cfg.proto_horizon

    task_names = cfg.task_names
    
    datasets = []
    stats_dict = {}
    for task_name in task_names:
        dataset = hydra.utils.instantiate(cfg.dataset)
        datasets.append(dataset)
        stats_dict[task_name] = dataset.stats
    dataset = torch.utils.data.ConcatDataset(datasets)

    with open(os.path.join(save_dir, "stats.pickle"), "wb") as f:
        pickle.dump(stats_dict, f)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
    )

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
    epoch_start = cfg.epoch_start
    proto_dim = cfg.num_protos
    
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
    optimizer = torch.optim.AdamW(params=nets.parameters(),
                                lr=cfg.lr,
                                weight_decay=cfg.weight_decay)

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * cfg.num_epochs,
    )
    
    if cfg.resume:
        checkpoint_path = os.path.join(cfg.resumed_dir, f"ckpt_{cfg.epoch_start}.pt")
        if os.path.isfile(checkpoint_path):        
            epoch_start = load_checkpoint(ema, nets, optimizer, checkpoint_path) + 1
            print(f"Resumed training from epoch {epoch_start}")
        else:
            epoch_start = 0 
            print("No checkpoint found. Starting from scratch.")

    for epoch_idx in tqdm(range(epoch_start, cfg.num_epochs), desc="Epochs"):
        epoch_loss = list()
        epoch_action_loss = list()
        epoch_proto_prediction_loss = list()
        
        for nidx, nbatch in enumerate(dataloader):
            nobs = nbatch["obs"].to(device)
            B = nobs.shape[0]
            nimage = nbatch["images"].to(device)
            if cfg.use_wrist:
                nimage_wrist = nbatch["wrist_images"].to(device)
            nproto_idx = nbatch["protos"].to(device).squeeze(-1)
            naction = nbatch["actions"].to(device)
            
            image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
            image_features = image_features.reshape(*nimage.shape[:2], -1)  # (B,obs_horizon,visual_feature)
                
            obs_feature_components = [image_features, nobs, ]
            if cfg.use_wrist:
                wrist_image_features = nets["wrist_vision_encoder"](nimage_wrist.flatten(end_dim=1))
                wrist_image_features = wrist_image_features.reshape(*nimage_wrist.shape[:2], -1) 
                obs_feature_components.insert(1, wrist_image_features)
            obs_feature = torch.cat(obs_feature_components, dim=-1)

            obs_feature = obs_feature.float()
            obs_feature = obs_feature.flatten(start_dim=1)
            predict_proto = proto_pred_net(obs_feature)    
                
            con_label = nproto_idx.squeeze()
            if cfg.proto_classifier_softmax:
                predict_proto_softmaxemb = nn.functional.softmax(predict_proto, dim=-1)
                obs_cond = torch.cat([obs_feature.flatten(start_dim=1), predict_proto_softmaxemb.flatten(start_dim=1)], dim=1,)
            else:
                obs_cond = torch.cat([obs_feature.flatten(start_dim=1), predict_proto.flatten(start_dim=1)], dim=1,)

            # sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B, ), device=device).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

            # predict the noise residual
            noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            if noise_scheduler.prediction_type=="epsilon":
                action_loss = nn.functional.mse_loss(noise_pred, noise)
            elif noise_scheduler.prediction_type=="sample":
                action_loss = nn.functional.mse_loss(noise_pred, naction)
                
            proto_prediction_loss = nn.functional.cross_entropy(predict_proto, con_label)

            loss = action_loss + proto_prediction_loss * cfg.proto_prediction_weight

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets)

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            epoch_action_loss.append(action_loss.item())
            epoch_proto_prediction_loss.append(proto_prediction_loss.item())
                        
            log_path = os.path.join(save_dir, 'log.txt')
            if nidx % cfg.loss_log_every == 0:
                with open(log_path, 'a' if os.path.exists(log_path) else 'w') as f:
                        print(f'Epoch: {epoch_idx}, Iteration {nidx}: {action_loss.item()}, {proto_prediction_loss.item()}\n')
                        f.write(f'Epoch: {epoch_idx}, Iteration {nidx}: {action_loss.item()}, {proto_prediction_loss.item()}\n')
                        if cfg.use_wandb:
                            wandb.log({"epoch": epoch_idx, "action_loss": action_loss, "proto_prediction_loss": proto_prediction_loss})
        
        if epoch_idx % cfg.ckpt_frequency == 0:
            save_checkpoint(ema, nets, optimizer, epoch_idx, os.path.join(save_dir, f"ckpt_{epoch_idx}.pt"))

if __name__ == "__main__":
    train_diffusion_concept()
