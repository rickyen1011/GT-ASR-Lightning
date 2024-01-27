#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
config="conf/ctc/ssl/xlsr.json";
dat_config="conf/ctc/ssl/dat_wav2vec2.json"
lp_config="conf/ctc/ssl/lp_wav2vec2.json"
ckpt="Experiments/mswc-en-de-fr-fa-es-it-ru-pl-MP-thresh500-num1000/MP/ctc/wav2vec2/checkpoints/best-v1.ckpt"

# Baseline
# python train.py --config ${config} --gpu 2 --gradient-clip-val 1.0 --strategy ddp_find_unused_parameters_true

# DAT training
# python train.py --config ${dat_config} --gpu 2 --strategy ddp_find_unused_parameters_true


# Linear Probing training
python train.py --config ${lp_config} --gpu 2 --gradient-clip-val 1.0 --checkpoint-path $ckpt --strategy ddp_find_unused_parameters_true