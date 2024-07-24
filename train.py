import time
import torch
from tqdm import tqdm
from loss import CE
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import torch.utils.data as Data




class Trainer():
    def __init__(self, args, model, train_loader, val_loader, test_loader):
        self.args = args
        self.device = args.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps
        self.enable_sample = args.enable_sample
        self.num_epoch = args.num_epoch
        self.metric_ks = args.metric_ks
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        self.sampled_evaluation = args.sampled_evaluation
        
        self.step = 0
        self.best_metric = -1e9
        
        self.model=model.to(torch.device(self.device))
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=True)
 

        self.cr = CE(self.model, args)
        

    def train(self):
        # BERT training

        for epoch in range(self.num_epoch):

            loss_epoch, time_cost = self._train_one_epoch()
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            print('train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()    
        # save best metric
        self.result_file = open(self.save_path + '/result.txt', 'a+')
        print('best NDCG@10:{0}'.format(self.best_metric),
                file=self.result_file)
        self.result_file.close()
        
    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            loss = self.cr.compute(batch)

            loss_sum += loss.item()
            loss.backward()

            self.optimizer.step()


            self.step += 1
            if self.step % self.lr_decay_steps == 0:
                self.scheduler.step()
            if self.step % self.eval_per_steps == 0:
                print(self.step)
                self.sample_time = 0
                metric = {}
                for mode in ['val','test']:
                    metric[mode] = self.eval_model(mode)
                print(metric)
                self.result_file = open(self.save_path + '/result.txt', 'a+')
                print('step{0}'.format(self.step), file=self.result_file)
                print(metric, file=self.result_file)
                self.result_file.close()
                if metric['test']['NDCG@10'] > self.best_metric:
                    torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    self.result_file = open(self.save_path + '/result.txt', 'a+')
                    print('saving model of step{0}'.format(self.step), file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric['test']['NDCG@10']
                self.model.train()

        return loss_sum / (idx+1), time.perf_counter() - t0

    def eval_model(self, mode):
        self.model.eval()
        tqdm_data_loader = tqdm(self.val_loader) if mode == 'val' else tqdm(self.test_loader)
        metrics = {}
        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                metrics_batch = self.compute_metrics(batch)

                for k, v in metrics_batch.items(): 
                    if not metrics.__contains__(k):
                        metrics[k] = v
                    else:
                        metrics[k] += v

        for k, v in metrics.items():
            metrics[k] = v / (idx+1)
        return metrics


    def compute_metrics(self, batch):
        if self.sampled_evaluation:
            scores = self.model(batch) 
            labels = torch.LongTensor([1]+[0]*(scores.size(1)-1)).repeat(scores.size(0),1).to(self.device)
        else:
            seqs, answers = batch
            scores = self.model(batch)

            scores = scores[:, -1, :]
            answers = answers[:,-1].view(-1)
            scores = scores[answers>0]
            answers = answers[answers>0]

            labels = torch.zeros(seqs.shape[0], scores.size(-1)).to(self.device)
            row = []
            col = []
            seqs = seqs.tolist()
            answers = answers.tolist()
            for i in range(len(answers)):
                seq = list(set(seqs[i] + [answers[i]]))
                seq.remove(answers[i])
                if self.args.num_item + 1 in seq:
                    seq.remove(self.args.num_item + 1)
                row += [i] * len(seq)
                col += seq
                labels[i][answers[i]] = 1
            scores[row, col] = -1e9

        metrics = self.recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    @staticmethod
    def recalls_and_ndcgs_for_ks(scores, labels, ks):  
        metrics = {}

        answer_count = labels.sum(1)

        labels_float = labels.float()
        rank = (-scores).argsort(dim=1)
        cut = rank
        for k in sorted(ks, reverse=True):
            cut = cut[:, :k]
            hits = labels_float.gather(1, cut)
            metrics['Recall@%d' % k] = \
                (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device),
                                         labels.sum(1).float())).mean().cpu().item()

            position = torch.arange(2, 2 + k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
            ndcg = (dcg / idcg).mean()
            metrics['NDCG@%d' % k] = ndcg.cpu().item()

        return metrics
