#!/bin/bash

set -vx

CUDA="0,1"
NUM_GPUS=2

{
    CUDA_VISIBLE_DEVICES=${CUDA} bash train_multi.sh ${NUM_GPUS} \
        --task_name='SetPre4DEE' \
        --use_bert=False \
        --skip_train=False \
        --train_nopair_sets=True  \
        --start_epoch=0 \
        --num_train_epochs=100 \
        --train_batch_size=32 \
        --gradient_accumulation_steps=8 \
        --learning_rate=0.0002 \
        --decoder_lr=0.0001 \
        --train_file_name='train.json' \
        --dev_file_name='dev.json' \
        --test_file_name='test.json' \
        --train_on_multi_events=True \
        --train_on_single_event=True \
        --num_event2role_decoder_layer=2 \
        --parallel_decorate
}
