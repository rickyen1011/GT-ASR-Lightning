#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,2
config="conf/ctc/ssl/dat_wav2vec2.json";

python train.py --config ${config} --gpu 2

# for test_config in $(find config/dataset_config/test/ -name "test*.json"); do
#     python train.py --model-config ${model_config} \
#         --dataset-config $dataset_config \
#         --gpus 2
# done
