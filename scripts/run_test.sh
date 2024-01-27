#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
model=dat_wav2vec2
root_dir="Experiments/mswc-tr-MP-top30/MP/ctc/lp_wav2vec2";
test_config='conf/ctc/ssl/test_wav2vec2.json'

python test.py --config ${root_dir}/config.json \
    --testset-config $test_config \
    --checkpoint-path ${root_dir}/checkpoints/epoch24-step650-val_acc0.8241.ckpt --mode inference
# for test_config in $(find config/dataset_config/test/ -name "test*.json"); do
#     python test.py --config ${root_dir}/config.json \
#         --testset-config $test_config \
#         --checkpoint-path ${root_dir}/checkpoints/best.ckpt --mode evaluate_phase
# done
# epoch=14-step=138600.ckpt
# epoch18-step170940-val_acc0.7946.ckpt
# epoch29-step272580-val_acc0.8061.ckpt
# epoch6-step30030-val_acc0.8594.ckpt
# epoch12-step60060-val_acc0.8794.ckpt