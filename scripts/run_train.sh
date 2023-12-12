#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
config="conf/ctc/ssl/dat_wav2vec2.json";

python train.py --config ${config} --gpu 2 
