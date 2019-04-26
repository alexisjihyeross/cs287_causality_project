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

def train(model, dataloader, val_dataloader, num_labels, lr=5e-5, warmup=0.1, num_epochs=2, device='cuda', finetune=False, evaluate=False):
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

    optimizer = BertAdam(params, lr=lr, warmup=warmup, t_total=len(dataloader)*num_epochs)

    model.to(device)
    
    for epoch in range(num_epochs):
        # for i, batch in tqdm(enumerate(train_dataloader)):
        correct, total, zero_cnt, one_cnt, two_cnt = 0, 0, 0, 0, 0
        model.train()
        for i, batch in tqdm(enumerate(dataloader)):
            total_loss = 0
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            #print(input_ids)
            #print(segment_ids)
            #print(input_mask)
            logits, _ = model(input_ids, segment_ids, input_mask)

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            total_loss += loss.item()
            loss.backward()


            optimizer.step()
            optimizer.zero_grad()
            #print(logits)
            preds = logits.argmax(dim=1)

            if evaluate:
                correct_in_batch = logits.argmax(dim=1) == label_ids
                correct += len(correct_in_batch.nonzero())
                total += len(label_ids)
                zero_cnt += len((preds == 0).nonzero())
                one_cnt += len((preds == 1).nonzero())
                two_cnt += len((preds == 2).nonzero())
            if evaluate and (i + 1) % 40 == 0:
                print(f"batch {i+1} acc: {correct / max(total, 1)}, {zero_cnt} zeros, {one_cnt} ones, {two_cnt} twos of {40 * len(label_ids)} loss: ", total_loss)
                correct, total, zero_cnt, one_cnt, two_cnt, total_loss = 0, 0, 0, 0, 0, 0

        model.eval()
        val_loss = 0
        for i, batch in tqdm(enumerate(val_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits, _ = model(input_ids, segment_ids, input_mask)

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            val_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct_in_batch = logits.argmax(dim=1) == label_ids
            correct += len(correct_in_batch.nonzero())
            total += len(label_ids)
            zero_cnt += len((preds == 0).nonzero())
            one_cnt += len((preds == 1).nonzero())
            two_cnt += len((preds == 2).nonzero())

        print(f"val acc: {correct / max(total, 1)}, {zero_cnt} zeros, {one_cnt} ones, {two_cnt} twos of {total} loss: ", val_loss)

def simple_evaluate(model, dataloader, output_file_name, device='cpu'):
    correct, total = 0, 0

    with open(output_file_name + ".tsv", mode = "w") as out_file:
        writer = csv.writer(out_file, delimiter = '\t')
        for i, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits, _ = model(input_ids, input_mask, segment_ids)
            #print(logits.argmax(dim=1), label_ids)
            correct_in_batch = logits.argmax(dim=1) == label_ids
            correct += len(correct_in_batch.nonzero())
            total += len(label_ids)
            #print("preds: ", logits.argmax(dim=1))
            #print("original labels: ", label_ids)
            '''
            for j in range((correct_in_batch).size(0)):
                if correct_in_batch[j] == 1:
                    print(input_ids[j])
                    print(label_ids[j])
            '''
            #if i % 100 == 0:
            #    print(f"batch{i} accuracy {correct / total}")
           
            if i % 10 == 0:
                print ("batch", i, "out of ", len(dataloader))
            preds = logits.argmax(dim=1)
            for j in range(preds.size(0)):
                writer.writerow([preds[j].item()])
   
    return correct / total


def evaluate(model, pos_dataloader, neg_dataloader, output_file_name, FLUSH_FLAG=True, DEBUG=False):
    with open(output_file_name + ".tsv", mode="w") as out_file, open(output_file_name + ".txt", mode="w") as out_log, open(output_file_name + "_significant_dim.tsv", mode="w") as significant_dim_file:
        writer = csv.writer(out_file, delimiter = '\t')
        significant_dim_writer = csv.writer(significant_dim_file, delimiter = '\t') 
        significant_dim_writer.writerow(["direct_effect_dimensions", "indirect_effect_dimensions"])
        batch_num = 0
        if DEBUG:
            offset = 0
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


            print("pos_logits: ", pos_logits)
            print("neg_logits: ", neg_logits)

            print("batch: ", str(batch_num), file=out_log, flush=FLUSH_FLAG)
            print("batch: ", str(batch_num))
            for i in range(hidden_dim):
                if i % 10 == 0:
                #if True:
                    print("hidden dim ", i, " of ", hidden_dim, file=out_log, flush=FLUSH_FLAG)
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
                    print("dim: ", i, file=out_log, flush=FLUSH_FLAG)
                    print("\tneg: ", row[1], file=out_log, flush=FLUSH_FLAG)
                    print("\tdir: ", row[row_idx-2], file=out_log, flush=FLUSH_FLAG)
                    significant_direct_dims.append(i)
                    print("dim: ", i)
                    print("\tneg: ", row[1])
                    print("\tdir: ", row[row_idx-2])
                    
                #indirect effect - pos logits (effect of changing ith neuron to negative)
                if abs(row[row_idx-1][0] - row[0][0]) >= .1:
                    print("dim: ", i, file=out_log, flush=FLUSH_FLAG)
                    print("\tpos: ", row[0], file=out_log, flush=FLUSH_FLAG)
                    print("\tindir: ", row[row_idx-1], file=out_log, flush=FLUSH_FLAG)
                    significant_indirect_dims.append(i)
                    print("dim: ", i)
                    print("\tpos: ", row[0])
                    print("\tindir: ", row[row_idx-1])


            writer.writerow(row)
            if DEBUG:
                if line_count(output_file_name + ".tsv") >  batch_num + 1 - offset:
                    print("AN ERROR OCCURED IN LINE NUMS HERE", file=out_log)
                    offset += 1
            significant_dim_writer.writerow([significant_direct_dims, significant_indirect_dims])
            batch_num += 1

def line_count(fname):
    with open(fname, "r") as f:
        for i, _ in enumerate(f):
            pass
        return i
