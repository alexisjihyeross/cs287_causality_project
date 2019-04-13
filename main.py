import os
import csv
import sys
import collections
from tqdm import tqdm

from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertConfig
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


def evaluate(model, pos_dataloader, neg_dataloader, output_file_name):
    with open(output_file_name + ".tsv", mode="w") as out_file:
        writer = csv.writer(out_file, delimiter = '\t')
        
        for pos_batch, neg_batch in zip(pos_dataloader, neg_dataloader):
            pos_input_ids, pos_input_mask, pos_segment_ids, pos_label_ids = pos_batch
            neg_input_ids, neg_input_mask, neg_segment_ids, neg_label_ids = neg_batch
        
            pos_logits, pos_pooled_output = model(pos_input_ids, pos_input_mask, pos_segment_ids, pos_label_ids)
            neg_logits, neg_pooled_output = model(neg_input_ids, neg_input_mask, neg_segment_ids, neg_label_ids)
        
            def meta_modify(pooled_output, idx, value):
                def modify(pooled_output):
                    pooled_output[0][idx] = value
                    return pooled_output
                return modify
        
            row = [None]*(pos_pooled_output.shape[1]*2+2)
            row_idx[0] = pos_logits
            row_idx[1] = neg_logits
            row_idx = 2
            
            for i in range(pos_pooled_output.shape[1]):
                pos_i_value = pos_pooled_output[0][i]
                neg_i_value = neg_pooled_output[0][i]
 
                # direct effect: change input to negative, change ith neuron to positive value 
                dir_logits, _ = model(neg_input_ids, neg_input_mask, neg_segment_ids, neg_label_ids, modification = meta_modify(neg_pooled_output, i, pos_i_value))

                # indirect effect: input positive, change ith neuron to negative value
                indir_logits, _ = model(pos_input_ids, pos_input_mask, pos_segment_ids, pos_label_ids, modification = meta_modify(pos_pooled_output, i, neg_i_value))
                row[row_idx] = dir_logits
                row[row_idx+1] = indir_logits
                row_idx += 2

            writer.writerow(row)



tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=not BERT_CASED)

processor = BinaryMnliProcessor()
num_labels = len(processor.get_labels())

binary_model = BertForSequenceClassification.from_pretrained(MODEL, cache_dir = CACHE_DIR, num_labels=num_labels)

train_dataloader = processor.get_dataloader(DATA_DIR, 'small_binary_train', tokenizer, max_seq_len=70)

print("training...")
train(binary_model, train_dataloader, num_epochs=3, finetune=False)
torch.save(binary_model, "models/small_binary/no_finetune.pt")
with open('models/small_binary/bert_config.json', 'w') as f:
    f.write(binary_model.config.to_json_string())

print("loading model...")
config = BertConfig('models/small_binary/bert_config.json')
eval_model = BertForSequenceClassification(config, num_labels = num_labels)
eval_model.load_state_dict(torch.load("models/small_binary/no_finetune.pt"))
eval_model.eval()

pos_processor = BinaryMnliProcessor(5, 6)
neg_processor = BinaryMnliProessor(7, 6)
pos_dataloader = pos_processor.get_dataloader(DATA_DIR, "neg_test_mismatched", tokenizer, batch_size = 1)
neg_dataloader = neg_processor.get_dataloader(DATA_DIR, "neg_test_mismatched", tokenizer, batch_size = 1)

evaluate(eval_model, pos_dataloader, neg_dataloader, "experiments/finetune") 


