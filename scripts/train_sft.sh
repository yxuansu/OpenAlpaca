#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model openllama \
    --model_path ./openllama_ckpt/ \
    --data_path ./openalpaca.json \
    --save_path ./ckpt/openalpaca/ \
    --log_path ./rest/
