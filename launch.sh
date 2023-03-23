#!/bin/bash
# export XLA_USE_BF16=1
# export PT_XLA_DEBUG=1
export WANDB_PROJECT=metts
export TPU_NUM_DEVICES=8
/usr/bin/python3 $HOME/transformers/examples/pytorch/xla_spawn.py --num_cores $TPU_NUM_DEVICES main.py
