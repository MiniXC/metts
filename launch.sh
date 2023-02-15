#!/bin/bash
#export PT_XLA_DEBUG=1
export WANDB_PROJECT=metts
/usr/bin/python3 $HOME/transformers/examples/pytorch/xla_spawn.py --num_cores $TPU_NUM_DEVICES main_vocoder.py
