import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_




class IndexLinear(nn.Module):

    def __init__(self, embedding_dim, num_classes,enable_sample,sampled_evaluation):
        super(IndexLinear, self).__init__()
        # use Embedding to store the output embedding
        # it's efficient when it comes sparse update of gradients
        self.emb = nn.Embedding(num_classes, embedding_dim)
        # self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.bias = nn.Embedding(num_classes, 1)
        self.enable_sample = enable_sample
        self.sampled_evaluation = sampled_evaluation

    def forward(self, seqs,bias_index=1,index=1):
        # x 64   64 1054
        if self.enable_sample and self.training:  
            return torch.matmul(seqs.unsqueeze(1), self.emb(index).transpose(2,1)).squeeze(1) +  self.bias(bias_index)
        elif self.sampled_evaluation and (not self.training):
            return torch.matmul(seqs.unsqueeze(1), self.emb(index).transpose(2,1)).squeeze(1) +  self.bias(bias_index)
        else:
            return F.linear(seqs, self.emb.weight) +  self.bias.weight[bias_index]





class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, enable_res_parameter, dropout = 0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if not self.enable:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + self.dropout(self.a * sublayer(x)))


class PointWiseFeedForward(nn.Module):
    """
    FFN implement
    """

    def __init__(self,d_model, d_ffn, dropout = 0.1):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    TRM layer
    """
    
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter,dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(attn_heads, d_model, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ffn, dropout)
        self.skipconnect1 = SublayerConnection(d_model, enable_res_parameter,dropout)
        self.skipconnect2 = SublayerConnection(d_model, enable_res_parameter, dropout)

    def forward(self, x, mask):
        x = self.skipconnect1(x, lambda _x: self.attn.forward(_x, _x, _x, mask=mask))
        x = self.skipconnect2(x, self.ffn)
        return x


class sasrec(nn.Module):
    """
    BERT model
    i.e., Embbedding + n * TRM + Output
    """

    def __init__(self, args, index_item):
        super(sasrec, self).__init__()
        self.num = args.num_item
        self.num_item = args.num_item + 2
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = args.d_ffn
        layers = args.sasrec_layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        self.device = args.device
        self.model = args.model
        self.max_len = args.max_len
        # Embedding
        self.token = nn.Embedding(self.num_item, d_model)
        self.position = PositionalEmbedding(self.max_len, d_model)
        # TRMs
        self.TRMs = nn.ModuleList([TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter,dropout) for i in range(layers)])
        self.attention_mask = torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool)).to(self.device)
        self.attention_mask.requires_grad = False  

        self.output = IndexLinear(d_model,self.num_item - 1,args.enable_sample,args.sampled_evaluation)
    
        self.enable_sample = args.enable_sample
        self.sampled_evaluation = args.sampled_evaluation
        
        # note dict
        self.updata_strategy = args.updata_strategy 
        self.predict_note = args.predict_with_note  
        self.train_note = args.train_with_note      
        self.pool = torch.nn.AvgPool1d(args.wind_size, stride=1, padding=int((args.wind_size -1)/2)) #  args.wind_size  (args.wind_size -1)/2   3 1  /  5 2
        self.pool.requires_grad = False  

        self.alpha =  args.alpha 
        self.gama = args.gama 

        self.note_item_emb = torch.nn.Embedding(self.num_item, d_model, padding_idx=0)
        self.note_item_emb.requires_grad = False
        self.note_item_emb.weight.requires_grad = False
  
        self.apply(self._init_weights)
        self.index_item = index_item.to(self.device)
        self.index_item.requires_grad = False
        self.index_item.weight.requires_grad = False
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_item)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)
    
    def update_note(self, log_feats, log_seqs):
        if self.updata_strategy=='pooling':
             item_seq_emb = self.pool(log_feats.transpose(2,1)).transpose(2,1) 
        else:
            item_seq_emb=log_feats

        index = log_seqs.view(-1)
        item_seq_emb = item_seq_emb.contiguous().view(-1, log_feats.size(-1))
        self.note_item_emb.weight.data[index] = (1 - self.alpha) * self.note_item_emb(
            index) + self.alpha * item_seq_emb

    def use_note(self, seqs, log_seqs):
        index = log_seqs.view(-1)

        seqs = (1 - self.index_item(index)) * seqs.view(-1, seqs.size(-1)) + self.note_item_emb(
            index) * self.index_item(index)

        return seqs.view(log_seqs.size(0), log_seqs.size(1), -1)


    def forward(self, batch):
#        print('index_item:'+str(self.index_item.device()))

        if self.enable_sample and self.training:
            x,labels,negs = batch
        elif self.sampled_evaluation and (not self.training):
            x,labels,negs = batch
        else:
            x,labels = batch
        del batch

        log_seqs = x 

        mask = (log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)
        mask *= self.attention_mask
        x = self.token(x) + self.position(x)    # B * L --> B * L * D

        if self.training:
            if self.train_note:
                x = self.use_note(x, log_seqs) 
        else: 
            if self.predict_note:
                x = self.use_note(x, log_seqs) 

        for TRM in self.TRMs:
            x = TRM(x, mask)

        if self.training:
            if self.train_note:
                self.update_note(x, log_seqs)
        

        if self.enable_sample  and self.training:
            
            x = x[labels>0]
            cl = labels[labels>0]
            negs = set(negs.view(-1))
            negs = torch.LongTensor(list(negs - set(cl))) 
            negs = torch.LongTensor(list(negs)).repeat(len(cl),1).to(self.device)
            index = torch.cat((cl.unsqueeze(1),negs),1)
            return self.output(x,cl,index)   # B * L * D --> B * L * N
           
        elif self.sampled_evaluation  and (not self.training):
            x = x[labels>0]
            cl = labels[labels>0]
            index = torch.cat((cl.unsqueeze(1),negs),1)
            return self.output(x,cl,index)   # B * L * D --> B * L * N
        else:
            return self.output(x)   # B * L * D --> B * L * N


