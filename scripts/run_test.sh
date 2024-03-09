#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
model=dat_wav2vec2
root_dir="Experiments/mswc-en-de-fr-fa-es-it-ru-pl-MP-thresh500-num1000/char_li/ctc/dat_wav2vec2";
# root_dir="Experiments/mswc-en-de-fr-fa-es-it-ru-pl-MP-thresh500-num1000/MP/ctc/dat_wav2vec2";
test_config='conf/ctc/ssl/test_dat_wav2vec2.json'

python test.py --config ${root_dir}/config.json \
    --testset-config $test_config \
    --checkpoint-path ${root_dir}/checkpoints/best.ckpt --mode inference_dat
# for test_config in $(find config/dataset_config/test/ -name "test*.json"); do
#     python test.py --config ${root_dir}/config.json \
#         --testset-config $test_config \
#         --checkpoint-path ${root_dir}/checkpoints/best.ckpt --mode evaluate_phase
# done
# best-v1.ckpt : MP baseline
# best.ckpt : char, phone baseline
# epoch=14-step=138600.ckpt - MP dat
# epoch18-step170940-val_acc0.7946.ckpt : MP phone
# epoch29-step272580-val_acc0.8061.ckpt : MP char
# epoch6-step30030-val_acc0.8594.ckpt
# epoch12-step60060-val_acc0.8794.ckpt

# for i in 0 1 2 3 4 5 6 7 8 9; do
#     python test.py --config ${root_dir}/config.json \
#         --testset-config $test_config \
#         --checkpoint-path ${root_dir}/checkpoints/best.ckpt --mode inference
# done