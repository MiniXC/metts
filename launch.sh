#!/bin/bash
# export XLA_USE_BF16=1
export WANDB_PROJECT=metts
/usr/bin/python3 $HOME/transformers/examples/pytorch/xla_spawn.py --num_cores $TPU_NUM_DEVICES main_vocoder.py
