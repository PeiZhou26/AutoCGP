#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_diffusion_concept.py \
    save_dir="your_save_dir" \
    dataset.data_dirs="['your_dataset_path']" \
    dataset.proto_dirs="your_concept_path" \
    dataset.mask="your_generated_mask_path"