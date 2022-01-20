#! /bin/bash

NUM_GPUS=$1
shift

python -m torch.distributed.launch --master_port=23550 --nproc_per_node ${NUM_GPUS} run_main.py $*
