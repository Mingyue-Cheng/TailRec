#!/bin/bash

python main.py \
--data_path /home/movielens30.csv \
--save_path /home/sasrec-ml30 \
--d_model 64 \
--max_len 30 \
--attn_heads 4 \
--sasrec_layers 16 \
--train_batch_size 256 \
--val_batch_size 256 \
--test_batch_size 256 \
--eval_per_steps 3000 \
--num_epoch 10 \
--device cuda:1 \
--up 674 \
--down 5 \
--train_with_note 1 \
--predict_with_note 1 \
--updata_strategy pooling \
--alpha 0.1 \
--gama 0.7 \
--wind_size 5 \
--enable_sample 0 \
--sampled_evaluation 0 \
--samples_ratio 0.01 \
