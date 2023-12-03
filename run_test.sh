#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
root_dir="Experiments/VB-DMD/MPSENet/stft-512-400-100/TF-MAE/transform-exp/phase-estimation/GD_data-aug";

for test_config in $(find config/dataset_config/test/ -name "test*.json"); do
    python test.py --config ${root_dir}/config.json \
        --testset-config $test_config \
        --checkpoint-path ${root_dir}/checkpoints/best.ckpt --mode evaluate_phase
done
