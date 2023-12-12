#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
root_dir="Experiments/mswc-en-fr-it-ru-es-de-MP-thresh100-num1000/MP/ctc/wav2vec2";
test_config='conf/ctc/ssl/test_wav2vec2.json'

python test.py --config ${root_dir}/config.json \
    --testset-config $test_config \
    --checkpoint-path ${root_dir}/checkpoints/best.ckpt --mode inference

# for test_config in $(find config/dataset_config/test/ -name "test*.json"); do
#     python test.py --config ${root_dir}/config.json \
#         --testset-config $test_config \
#         --checkpoint-path ${root_dir}/checkpoints/best.ckpt --mode evaluate_phase
# done
