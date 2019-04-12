import os
import csv
import sys
import collections
from tqdm import tqdm

from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert.optimization import BertAdam
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from models import BertTokenizer, MnliProcessor, BinaryMnliProcessor, BertForSequenceClassification

BERT_SIZE = 'base'  # or 'large'
BERT_CASED = False
DATA_DIR = 'glue_data/MNLI'
CACHE_DIR = 'cache'
MODEL = f'bert-{BERT_SIZE}-{"cased" if BERT_CASED else "uncased"}'

def train(model, dataloader, lr=5e-5, warmup=0.1, num_epochs=2, device='cuda', finetune=False):
    #if finetune is False, freeze pretrained weights
    if not finetune:
        for param in model.bert.parameters():
            param.requires_grad = False
            
    loss_fct = CrossEntropyLoss()
    
    batch_size = dataloader.batch_size
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(params, lr=lr, warmup=warmup, t_total=len(dataloader))

    model.to(device)
    
    model.train()
    
    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            optimizer.zero_grad()

            logits, _ = model(input_ids, segment_ids, input_mask, labels=None)

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            loss.backward()

            optimizer.step()

tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=not BERT_CASED)

processor = BinaryMnliProcessor()

num_labels = len(processor.get_labels())

binary_model = BertForSequenceClassification.from_pretrained(MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)

train_dataloader = processor.get_dataloader(DATA_DIR, 'small_binary_train', tokenizer, max_seq_len=70)

print("training...")
train(binary_model, train_dataloader, num_epochs=3)

torch.save(binary_model, "models/small_binary/fine_tune.pt")
