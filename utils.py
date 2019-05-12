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

from models import BertTokenizer, MnliProcessor, BinaryMnliProcessor, BertForSequenceClassification, modified_forward, post_modification
import copy

BERT_SIZE = 'base'  # or 'large'
BERT_CASED = False
DATA_DIR = 'glue_data/MNLI'
CACHE_DIR = 'cache'
MODEL = f'bert-{BERT_SIZE}-{"cased" if BERT_CASED else "uncased"}'


def train(model,
          dataloader,
          val_dataloader,
          num_labels,
          lr=5e-5,
          warmup=0.1,
          num_epochs=2,
          device='cuda',
          finetune=False,
          evaluate=False):
    #if finetune is False, freeze pretrained weights
    if not finetune:
        for param in model.bert.parameters():
            param.requires_grad = False

    loss_fct = CrossEntropyLoss()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    params = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    optimizer = BertAdam(params,
                         lr=lr,
                         warmup=warmup,
                         t_total=len(dataloader) * num_epochs)

    model.to(device)

    for epoch in range(num_epochs):
        correct, total, zero_cnt, one_cnt, two_cnt = 0, 0, 0, 0, 0
        model.train()
        for i, batch in tqdm(enumerate(dataloader)):
            total_loss = 0
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

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
                print(
                    f"batch {i+1} acc: {correct / max(total, 1)}, {zero_cnt} zeros, {one_cnt} ones, {two_cnt} twos of {40 * len(label_ids)} loss: ",
                    total_loss)
                correct, total, zero_cnt, one_cnt, two_cnt, total_loss = 0, 0, 0, 0, 0, 0

        model.eval()
        val_loss = 0
        correct, total, zero_cnt, one_cnt, two_cnt, total_loss = 0, 0, 0, 0, 0, 0
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

        print(
            f"val acc: {correct / max(total, 1)}, {zero_cnt} zeros, {one_cnt} ones, {two_cnt} twos of {total} loss: ",
            val_loss)


def simple_evaluate(model, dataloader, output_file_name, device='cuda:0'):
    correct, total, zero_cnt, one_cnt, two_cnt  = 0, 0, 0, 0, 0

    with open(output_file_name + ".tsv", mode="w") as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        for i, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits, _ = model(input_ids, segment_ids, input_mask)
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

            if i % 10 == 0:
                # print(f"batch{i} accuracy {correct / total}")
                print("batch", i, "out of ", len(dataloader))
            preds = logits.argmax(dim=1)
            for j in range(preds.size(0)):
                writer.writerow([preds[j].item()])

            zero_cnt += len((preds == 0).nonzero())
            one_cnt += len((preds == 1).nonzero())
            two_cnt += len((preds == 2).nonzero())
        print(zero_cnt, " 0s; ", one_cnt, " 1s; ", two_cnt, " 2s")
        print("val acc: ", correct / total)



def evaluate(model,
             pos_dataloader,
             neg_dataloader,
             output_file_name,
             modify_layer=3,
             FLUSH_FLAG=True,
             DEBUG=False):
    batch_size = pos_dataloader.batch_size
    with open(output_file_name + ".tsv", mode="w") as out_file, open(
            output_file_name + ".txt", mode="w") as out_log, open(
                output_file_name + "_significant_dim.tsv",
                mode="w") as significant_dim_file:
        writer = csv.writer(out_file, delimiter='\t')
        significant_dim_writer = csv.writer(significant_dim_file,
                                            delimiter='\t')
        significant_dim_writer.writerow(
            ["direct_effect_dimensions", "indirect_effect_dimensions"])
        batch_num = 0
        if DEBUG:
            offset = 0
        for pos_batch, neg_batch in (zip(pos_dataloader, neg_dataloader)):
            significant_direct_dims = []
            significant_indirect_dims = []

            # print("pos: ")
            pos_logits, pos_attn, pos_modify_output = modified_forward(model, pos_batch, modify_layer=modify_layer)

            # print("neg: "n)
            neg_logits, neg_attn, neg_modify_output = modified_forward(model, neg_batch, modify_layer=modify_layer)

            print(pos_modify_output.shape)
            hidden_dim = pos_modify_output.shape[-1]

            row = torch.zeros(2 * hidden_dim + 2, batch_size, 2)
            row[0] = nn.functional.softmax(pos_logits, dim=1)
            row[1] = nn.functional.softmax(neg_logits, dim=1)
            row_idx = 2

            if modify_layer == 1:
                print('trying fast')
                # store VW = v_i * w_i for all i, pos and neg
                W = model.classifier.weight
                pos_VW = torch.einsum('bh,mh->bmh', pos_modify_output, W)
                neg_VW = torch.einsum('bh,mh->bmh', neg_modify_output, W)

                # difference vi,wi - vi',wi across hidden dimension for all i
                diff = pos_VW - neg_VW

                # compute logits - vi,wi + vi',wi for each i
                pL = nn.functional.softmax(pos_logits.unsqueeze(dim=2) - diff, dim=1)
                nL = nn.functional.softmax(neg_logits.unsqueeze(dim=2) + diff, dim=1)

            print(hidden_dim)

            print("pos_logits: ", pos_logits)
            print("neg_logits: ", neg_logits)

            print("batch: ", str(batch_num), file=out_log, flush=FLUSH_FLAG)
            print("batch: ", str(batch_num))
            for i in range(hidden_dim):
                if modify_layer == 1:
                    row[row_idx] = nL[:, :, i].flatten().tolist()
                    row[row_idx + 1] = pL[:, :, i].flatten().tolist()
                    row_idx += 2
                else:
                    if i % 10 == 0:
                    #if True:
                        print("hidden dim ", i, " of ", hidden_dim, file=out_log, flush=FLUSH_FLAG)
                        print("hidden dim ", i, " of ", hidden_dim)

                    # direct effect: change input to negative, change ith neuron to positive value 
                    #print("direct effect: ") 
                    nso = neg_modify_output.clone().detach()
                    nso[0, ..., i] = pos_modify_output[0, ..., i]
                    dir_logits = post_modification(model, nso, neg_attn, layer=modify_layer)

                    # indirect effect: input positive, change ith neuron to negative value
                    #print("indirect effect: ")
                    pso = pos_modify_output.clone().detach()
                    pso[0, ..., i] = neg_modify_output[0, ..., i]
                    indir_logits = post_modification(model, pso, pos_attn, layer=modify_layer)
                    row[row_idx] = nn.functional.softmax(dir_logits, dim=1)
                    row[row_idx+1] = nn.functional.softmax(indir_logits, dim=1)
                    row_idx += 2

                # #direct effect - neg logits (effect of changing ith neuron to positive)
                #
                # if abs(row[row_idx - 2][0] - row[1][0]) >= .1:
                #     print("dim: ", i, file=out_log, flush=FLUSH_FLAG)
                #     print("\tneg: ", row[1], file=out_log, flush=FLUSH_FLAG)
                #     print("\tdir: ",
                #           row[row_idx - 2],
                #           file=out_log,
                #           flush=FLUSH_FLAG)
                #     significant_direct_dims.append(i)
                #     print("dim: ", i)
                #     print("\tneg: ", row[1])
                #     print("\tdir: ", row[row_idx - 2])
                #
                # #indirect effect - pos logits (effect of changing ith neuron to negative)
                # if abs(row[row_idx - 1][0] - row[0][0]) >= .1:
                #     print("dim: ", i, file=out_log, flush=FLUSH_FLAG)
                #     print("\tpos: ", row[0], file=out_log, flush=FLUSH_FLAG)
                #     print("\tindir: ",
                #           row[row_idx - 1],
                #           file=out_log,
                #           flush=FLUSH_FLAG)
                #     significant_indirect_dims.append(i)
                #     print("dim: ", i)
                #     print("\tpos: ", row[0])
                #     print("\tindir: ", row[row_idx - 1])

            for b in range(batch_size):

                writer.writerow(row[:, b, :].tolist())
                # if DEBUG:
                #     if line_count(output_file_name +
                #                   ".tsv") > batch_num + 1 - offset:
                #         print("AN ERROR OCCURED IN LINE NUMS HERE", file=out_log)
                #         offset += 1
                # significant_dim_writer.writerow(
                #     [significant_direct_dims, significant_indirect_dims])
                batch_num += 1


def line_count(fname):
    with open(fname, "r") as f:
        for i, _ in enumerate(f):
            pass
        return i + 1
