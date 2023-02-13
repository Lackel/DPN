#!/usr/bin bash

python DPN.py \
    --dataset stackoverflow \
    --known_cls_ratio 0.75 \
    --cluster_num_factor 1 \
    --seed 0 \
    --freeze_bert_parameters \
    --gpu_id 0 \
    --save_model \
    --pretrain