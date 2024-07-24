import torch
import random
import pandas as pd
import numpy as np
from torch._C import TensorType
import torch.utils.data as Data
import os


def neg_sample(seq, labels, num_item,sample_size):
    negs = set()
    seen = set(labels)

    while len(negs) < sample_size:
        candidate = np.random.randint(0, num_item) + 1
        while candidate in seen or candidate in negs:
            candidate = np.random.randint(0, num_item) + 1
        negs.add(candidate)
    return negs

class TrainDataset(Data.Dataset):
    def __init__(self, args):

        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.num_user = self.data.shape[0] 

        self.num_item = self.data.max() 
        if args.eval_per_steps == 0:
            args.eval_per_steps = self.num_user//args.train_batch_size
        self.mask_token = args.num_item + 1 
        self.enable_sample = args.enable_sample
        self.sample_size = (args.samples_ratio * args.num_item) // args.train_batch_size 
        self.mask_prob = args.mask_prob
        self.max_len = args.max_len
        self.model = args.model

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        
        seq = self.data[index, -self.max_len - 3:-3].tolist()
        labels = [self.data[index,-3].tolist()]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq  
        labels = [0] * (self.max_len-1)+labels
        
        if self.enable_sample:
            neg = neg_sample(seq, labels, self.num_item, self.sample_size)
            return torch.LongTensor(seq), torch.LongTensor(labels), torch.LongTensor(list(neg))

        else:
            return torch.LongTensor(seq), torch.LongTensor(labels)

class EvalDataset(Data.Dataset):
    def __init__(self,args,mode):
        

        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.num_item = self.data.max()
        self.num_user = self.data.shape[0]
        self.mask_token = args.num_item + 1
        self.max_len = args.max_len
        self.sample_size = 100
        self.model = args.model
        self.sampled_evaluation = args.sampled_evaluation
        self.mode = mode

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, :-2] if self.mode == 'val' else self.data[index, :-1]
        pos = self.data[index, -2] if self.mode == 'val' else self.data[index, -1]

        seq = list(seq)
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        labels = [0] * (self.max_len-1)+[pos]

        if self.sampled_evaluation:
            neg = neg_sample(seq, labels, self.num_item, self.sample_size)
            return torch.LongTensor(seq), torch.LongTensor(labels), torch.LongTensor(list(neg))

        else:
            return torch.LongTensor(seq), torch.LongTensor(labels)
