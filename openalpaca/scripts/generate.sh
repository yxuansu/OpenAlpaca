#!/bin/bash

# dude, what the fuck !
export NCCL_IB_DISABLE=1

deepspeed --include localhost:2 --master_port 28444 generate.py --model openllama --test_base_model False
