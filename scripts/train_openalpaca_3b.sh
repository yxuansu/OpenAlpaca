#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model openllama \
    --model_path openlm-research/open_llama_3b_600bt_preview\
    --data_path ./openalpaca.json \
    --max_length 1536\
    --save_path  ./ckpt/openalpaca_3b/6bt_preview/\
    --log_path ./ckpt/openalpaca_3b/6bt_preview/rest


