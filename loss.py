import math
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from joblib import Parallel, delayed

class CE:
    def __init__(self, model,args):
        self.model = model
        self.enable_sample = args.enable_sample     
        self.num_item = args.num_item
        self.device = args.device
        self.model_name = args.model
        if self.enable_sample:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=0)


    def compute(self, batch):

        if self.enable_sample:
            seq,labels,negs = batch
        else:
            seq,labels = batch

        pred = self.model(batch)  # B * L * N
        if self.enable_sample:
            cl = torch.LongTensor([0]*pred.size(0)).to(self.device)
            loss = self.ce(pred, cl)
        else:
            loss = self.ce(pred.view(-1,pred.size(-1)), labels.view(-1))
        return loss

