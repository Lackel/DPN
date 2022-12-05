#!/usr/bin bash
for s in 0 1 2
do 
    python DPN.py \
    --dataset clinc \
    --known_cls_ratio 0.75 \
    --cluster_num_factor 1 \
    --seed $s \
    --freeze_bert_parameters \
    --gpu_id 0 \
    --save_model \
    --pretrain
done
