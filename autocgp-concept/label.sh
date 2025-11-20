#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
cd src &&

python label_all_multi.py \
    --task coffee \
    --seed 0 \
    --train_split 1.0 \
    --model_name multi-2025_11_20_16_37_30-k4-not_use_causal-r4-f2-c30-KT_0.1-EMA_0.9-ema_ave-st_emb1.2-r_l10.0-use_r-egpthndelta_s2_a1-emb128-key128-e128-mid256-cluster0.001-rec0.1-finetune-state \
    --from_ckpt 1 \
    --distribution 2 2 1 1 1 2 0

# task name | coffee | threading | stack_three | hammer_cleanup | mug_cleanup | three_piece_assembly | nut_assembly
# highest level     2 2 1 1 1 2 0
# medium level      1 1 0 0 0 1 0
# low level         0 0 0 0 0 0 0