import argparse
import time
import os
import json
from numpy import array
import pandas as pd
import re
import torch
parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--data_path', type=str, default='/home/movielens30.csv')
parser.add_argument('--save_path', type=str, default='test')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--max_len', type=int, default=50)
parser.add_argument('--mask_prob', type=float, default=0.3)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--up', default=47, type=int ) 
parser.add_argument('--down', default=0, type=int ) 


# model args
parser.add_argument('--model', type=str, default='sasrec')
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--eval_per_steps', type=int, default=3000)
parser.add_argument('--enable_res_parameter', type=int, default=1)

# sasrec args
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--d_ffn', type=int, default=256)
parser.add_argument('--sasrec_layers', type=int, default=16)

# train args
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_rate', type=float, default=1)
parser.add_argument('--lr_decay_steps', type=int, default=1250)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--num_epoch', type=int, default=4)
parser.add_argument('--metric_ks', type=list, default=[10,20])
# sample
parser.add_argument('--samples_ratio', type=float, default=0.01) 
parser.add_argument('--enable_sample', type=int, default=0) 
parser.add_argument('--sampled_evaluation', type=int, default=0) 



# note dict
parser.add_argument('--alpha', default=0.1, type=float )
parser.add_argument('--gama', default=0.7, type=float )
parser.add_argument('--predict_with_note',type=int, default=1)
parser.add_argument('--train_with_note', type=int,default=1)
parser.add_argument('--updata_strategy', default='pooling', type=str )
parser.add_argument('--wind_size', default=11, type=int)


args = parser.parse_args()

if torch.cuda.is_available():
    pass    
else: 
    args.device = 'cpu'


if args.save_path == 'None':
    loss_str = args.loss_type
    path_str = 'note-predict-' + str(args.predict_with_note)+'_note-train-' + str(args.train_with_note)+'_gama-' + str(args.gama)+'D-' + str(args.d_model) + \
               '_Lr-' + str(args.lr)  + '/'
    args.save_path = path_str
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

