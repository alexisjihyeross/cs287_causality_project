import os
import csv
import sys
import collections
from tqdm import tqdm

import numpy as np

from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from models import BertTokenizer, MnliProcessor, BinaryMnliProcessor, BertForSequenceClassification
import copy

BERT_SIZE = 'base'  # or 'large'
BERT_CASED = False
DATA_DIR = 'glue_data/MNLI'
CACHE_DIR = 'cache'
MODEL = f'bert-{BERT_SIZE}-{"cased" if BERT_CASED else "uncased"}'

def train(model, dataloader, lr=5e-5, warmup=0.1, num_epochs=2, device='cuda', finetune=False, evaluate=False):
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

    correct, total, cnt = 0, 0, 0

    for epoch in range(num_epochs):
        for i, batch in tqdm(enumerate(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            optimizer.zero_grad()

            logits, _ = model(input_ids, segment_ids, input_mask)

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            loss.backward()

            optimizer.step()

            preds = logits.argmax(dim=1)

            if evaluate:
                correct_in_batch = logits.argmax(dim=1) == label_ids
                correct += len(correct_in_batch.nonzero())
                total += len(label_ids)
                cnt += len(preds.nonzero())

            if evaluate and (i + 1) % 40 == 0:
                print(f"batch {i+1} acc: {correct / max(total, 1)}, {cnt} ones of {40 * len(label_ids)}")
                correct, total, cnt = 0, 0, 0


def simple_evaluate(model, dataloader, device='cuda'):
    correct, total = 0, 0

    for i, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits, _ = model(input_ids, input_mask, segment_ids)
        print(logits.argmax(dim=1), label_ids)
        correct_in_batch = logits.argmax(dim=1) == label_ids
        correct += len(correct_in_batch.nonzero())
        total += len(label_ids)
        
        if i % 100 == 0:
            print(f"epoch {i} accuracy {correct / total}")

    return correct / total


def evaluate(model, pos_dataloader, neg_dataloader, output_file_name):
    with open(output_file_name + ".tsv", mode="w") as out_file, open(output_file_name + ".txt", mode="w") as out_log, open(output_file_name + "_significant_dim.tsv", mode="w") as significant_dim_file:
        writer = csv.writer(out_file, delimiter = '\t')
        significant_dim_writer = csv.writer(significant_dim_file, delimiter = '\t') 
        significant_dim_writer.writerow(["direct_effect_dimensions", "indirect_effect_dimensions"])
        batch_num = 0
        for pos_batch, neg_batch in (zip(pos_dataloader, neg_dataloader)):
            significant_direct_dims = []
            significant_indirect_dims = []
            pos_input_ids, pos_input_mask, pos_segment_ids, pos_label_ids = pos_batch
            neg_input_ids, neg_input_mask, neg_segment_ids, neg_label_ids = neg_batch
            #print("pos: ") 
            pos_logits, pos_pooled_output = model(pos_input_ids, pos_input_mask, pos_segment_ids)
            
            #print("neg: ")
            neg_logits, neg_pooled_output = model(neg_input_ids, neg_input_mask, neg_segment_ids)
        
            def meta_modify(pooled_output, idx, value):
                def modify(pooled_output):
                    pooled_output[0][idx] = value
                    return pooled_output
                return modify
      
            hidden_dim = pos_pooled_output.shape[1]
            row = [None]*(hidden_dim*2+2)
            row[0] = nn.functional.softmax(pos_logits.flatten(), dim=0).tolist()
            row[1] = nn.functional.softmax(neg_logits.flatten(), dim=0).tolist()
            row_idx = 2

            print(pos_logits)
            print(neg_logits)
            print(row[0])
            print(row[1])

            print("batch: ", str(batch_num), file=out_log, flush=True)
            print("batch: ", str(batch_num))
            for i in range(hidden_dim):
                if i % 100 == 0:
                #if True:
                    print("hidden dim ", i, " of ", hidden_dim, file=out_log, flush=True)
                    print("hidden dim ", i, " of ", hidden_dim)
                pos_i_value = pos_pooled_output[0][i]
                neg_i_value = neg_pooled_output[0][i]
 
                # direct effect: change input to negative, change ith neuron to positive value 
                #print("direct effect: ") 
                dir_logits, _ = model(neg_input_ids, neg_input_mask, neg_segment_ids, modification = meta_modify(neg_pooled_output, i, pos_i_value))

                # indirect effect: input positive, change ith neuron to negative value
                #print("indirect effect: ")
                indir_logits, _ = model(pos_input_ids, pos_input_mask, pos_segment_ids, modification = meta_modify(pos_pooled_output, i, neg_i_value))
                row[row_idx] = nn.functional.softmax(dir_logits.flatten(), dim=0).tolist()
                row[row_idx+1] = nn.functional.softmax(indir_logits.flatten(), dim=0).tolist()
                row_idx += 2

                
                #direct effect - neg logits (effect of changing ith neuron to positive)
                if abs(row[row_idx-2][0] - row[1][0]) >= .1:
                    print("dim: ", i, file=out_log, flush=True)
                    print("\tneg: ", row[1], file=out_log, flush=True)
                    print("\tdir: ", row[row_idx-2], file=out_log, flush=True)
                    significant_direct_dims.append(i)
                    
                #indirect effect - pos logits (effect of changing ith neuron to negative)
                if abs(row[row_idx-1][0] - row[0][0]) >= .1:
                    print("dim: ", i, file=out_log, flush=True)
                    print("\tpos: ", row[0], file=out_log, flush=True)
                    print("\tindir: ", row[row_idx-1], file=out_log, flush=True)
                    significant_indirect_dims.append(i)

            writer.writerow(row)
            significant_dim_writer.writerow([significant_direct_dims, significant_indirect_dims])
            batch_num += 1


tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=not BERT_CASED)

processor = BinaryMnliProcessor()
num_labels = len(processor.get_labels())

binary_model = BertForSequenceClassification.from_pretrained(MODEL, cache_dir = CACHE_DIR, num_labels=num_labels)

train_dataloader = processor.get_dataloader(DATA_DIR, 'binary_train', tokenizer, max_seq_len=70, shuffle=True)

print("training...")
train(binary_model, train_dataloader, num_epochs=3, finetune=False, evaluate=True)
torch.save(binary_model.state_dict(), "models/binary/untrained.pt")
with open('models/binary/bert_config_untrained.json', 'w') as f:
   f.write(binary_model.config.to_json_string())

"""
print("training...")
train(binary_model, train_dataloader, num_epochs=3, finetune=False, evaluate=True)
torch.save(binary_model.state_dict(), "models/binary/finetune.pt")
with open('models/binary/bert_config.json', 'w') as f:
    f.write(binary_model.config.to_json_string())

print("loading model...")
config = BertConfig('models/binary/bert_config_untrained.json')
eval_model = BertForSequenceClassification(config, num_labels = num_labels)
eval_model.load_state_dict(torch.load("models/binary/untrained.pt"))
eval_model.cuda()
eval_model.eval()

pos_dataloader = processor.get_dataloader(DATA_DIR, "dev_mismatched", tokenizer, batch_size = 1, a_idx = 6, b_idx = 7)
neg_dataloader = processor.get_dataloader(DATA_DIR, "neg_dev_mismatched", tokenizer, batch_size = 1, a_idx = 8, b_idx = 7)

evaluate(eval_model, pos_dataloader, neg_dataloader, "experiments/binary_finetune") 

print("loading model...")
config = BertConfig('models/binary/bert_config.json')
eval_model = BertForSequenceClassification(config, num_labels = num_labels)
eval_model.load_state_dict(torch.load("models/binary/finetune.pt"))
eval_model.cuda()
eval_model.eval()

print("loading data...")
dataloader = processor.get_dataloader(DATA_DIR, "binary_train", tokenizer, batch_size = 40)

print("Positive")
simple_evaluate(eval_model, dataloader) 
"""
