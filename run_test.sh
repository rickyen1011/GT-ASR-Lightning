#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
root_dir="Experiments/mswc-en-fr-it-MP-thresh1000-num1000/ctc/wav2vec2";
test_config='conf/ctc/ssl/test_wav2vec2.json'

python test.py --config ${root_dir}/config.json \
    --testset-config $test_config \
    --checkpoint-path ${root_dir}/checkpoints/best.ckpt --mode inference

# for test_config in $(find config/dataset_config/test/ -name "test*.json"); do
#     python test.py --config ${root_dir}/config.json \
#         --testset-config $test_config \
#         --checkpoint-path ${root_dir}/checkpoints/best.ckpt --mode evaluate_phase
# done
