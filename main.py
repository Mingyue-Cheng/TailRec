from dataset import TrainDataset, EvalDataset
from train import Trainer
from args import args
import pandas as pd
import torch
import torch.utils.data as Data
from sasrec import sasrec
import numpy as np
import os

def get_index():
    df=pd.read_csv(args.data_path, header=None) 
    df = df.replace(-1,0) 

    args.num_item = df.max().max()
    
    counts = df.apply(pd.value_counts)
    counts =counts.apply(lambda x:x.sum(),axis =1)
    counts = counts.sort_values(ascending=False) 

    counts[counts<args.down] = 0
    counts[counts>args.up] = 0
    counts[counts!=0] = args.gama

    i = counts.to_dict() 
    args.item_set = set(i.keys())-set([0]) 
    tmp = np.arange(args.num_item+2)
    a = pd.DataFrame([0.5]*(args.num_item+2),index=tmp,columns=None)
    a = a.to_dict()[0]
    a[args.num_item+1] = 0.0
    a.update(i)
    index_item = torch.nn.Embedding(args.num_item+2, 1,  _weight=torch.tensor(list(a.values())).float().unsqueeze(1))
    index_item.requires_grad = False  
    index_item.weight.requires_grad = False  
    return index_item

def main():

    print(args)

    index_item = get_index()

    train_dataset = TrainDataset(args)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    
    val_dataset  = EvalDataset(args, mode='val')
    val_loader  = Data.DataLoader(val_dataset, batch_size=args.test_batch_size)

    test_dataset = EvalDataset(args, mode='test')
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    
    print('dataset initial ends')

    model = sasrec(args, index_item)
    
    trainer = Trainer(args, model, train_loader, val_loader, test_loader)
    
    print('train process ready')
    argsDict = args.__dict__
    with open(os.path.join(args.save_path,'setting.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            if eachArg != 'longtail_set':
                f.writelines(eachArg + ' : ' + str(value) + '\n')
    
    trainer.train()


if __name__ == '__main__':

    main()
