#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
config="conf/ctc/ssl/xlsr.json";
dat_config="conf/ctc/ssl/dat_wav2vec2.json"
uda_config="conf/ctc/ssl/uda_wav2vec2.json"
lp_config="conf/ctc/ssl/lp_wav2vec2.json"
nmr_config="conf/ctc/ssl/nmr_wav2vec2.json"
# ckpt="Experiments/mswc-en-de-fr-fa-es-it-ru-pl-MP-thresh500-num1000/MP/ctc/dat_wav2vec2/checkpoints/epoch=14-step=138600.ckpt"
ckpt="Experiments/mswc-en-de-fr-fa-es-it-ru-pl-MP-thresh500-num1000/phone/ctc/dat_wav2vec2/checkpoints/epoch18-step170940-val_acc0.7946.ckpt"

# Baseline
# python train.py --config ${config} --gpu 2 --gradient-clip-val 1.0 --strategy ddp_find_unused_parameters_true

# DAT training
# python train.py --config ${dat_config} --gpu 2 --strategy ddp_find_unused_parameters_true

# UDA training
python train.py --config ${dat_config} --gpu 2 --strategy ddp_find_unused_parameters_true

# Linear Probing training
# python train.py --config ${lp_config} --gpu 2 --gradient-clip-val 1.0 --checkpoint-path $ckpt --strategy ddp_find_unused_parameters_true

# best-v1.ckpt : MP baseline
# best.ckpt : char, phone baseline
# epoch=14-step=138600.ckpt - MP dat
# epoch18-step170940-val_acc0.7946.ckpt : phone dat
# epoch29-step272580-val_acc0.8061.ckpt : char dat