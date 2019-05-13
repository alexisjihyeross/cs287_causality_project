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

from utils import train, evaluate, simple_evaluate
import copy

BERT_SIZE = 'base'  # or 'large'
BERT_CASED = False
DATA_DIR = 'glue_data/MNLI'
CACHE_DIR = 'cache'
MODEL = f'bert-{BERT_SIZE}-{"cased" if BERT_CASED else "uncased"}'



tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=not BERT_CASED)

processor = BinaryMnliProcessor()
num_labels = len(processor.get_labels())

binary_model = BertForSequenceClassification.from_pretrained(MODEL, cache_dir = CACHE_DIR, num_labels=num_labels)


do_train = False
do_finetune = True
do_evaluate = True
other = False

model_name = 'finetune' if do_finetune else 'no_finetune'

if do_train:
    print("training...")
    print("loading data...")
    train_dataloader = processor.get_dataloader(DATA_DIR, 'binary_train', tokenizer, max_seq_len=70, shuffle=True)

    val_dataloader = processor.get_dataloader(DATA_DIR, 'binary_dev_matched', tokenizer, max_seq_len=70, label_idx = -1, shuffle=True)

    train(binary_model, train_dataloader, val_dataloader, num_labels, num_epochs=3, finetune=do_finetune, evaluate=True)
    torch.save(binary_model.state_dict(), "models/binary/" + model_name + ".pt")
    with open('models/binary/' + model_name + '_config.json', 'w') as f:
        f.write(binary_model.config.to_json_string())

if do_evaluate:
    print("loading model...")
    # config = BertConfig('models/binary/' + model_name + '_config.json')
    # eval_model = BertForSequenceClassification(config, num_labels = num_labels)
    # eval_model.load_state_dict(torch.load("models/binary/" + model_name + ".pt"))
    eval_model = binary_model
    eval_model.eval()

    print("loading data...")
    # Note these are different, see indices
    pos_dataloader = processor.get_dataloader(DATA_DIR, "neg_dev_mismatched", tokenizer, batch_size = 1, a_idx = 6, b_idx = 7)
    neg_dataloader = processor.get_dataloader(DATA_DIR, "neg_dev_mismatched", tokenizer, batch_size = 1, a_idx = 8, b_idx = 7)

    evaluate(eval_model, pos_dataloader, neg_dataloader, "experiments/may10/binary_finetune", DEBUG=True, modify_layer=3)

if other:
    config = BertConfig('models/binary/' + model_name + '_config.json')
    eval_model = BertForSequenceClassification(config, num_labels = num_labels)
    eval_model.load_state_dict(torch.load("models/binary/" + model_name + ".pt"))
    eval_model.to('cuda:0')
    eval_model.eval()
    eval_pos_dataloader = processor.get_dataloader(DATA_DIR, "binary_dev_mismatched", tokenizer, batch_size = 10, a_idx = 6, b_idx = 7, label_idx = 5)
    #eval_neg_dataloader = processor.get_dataloader(DATA_DIR, "neg_binary_dev_mismatched", tokenizer, batch_size = 10, a_idx = 8, b_idx = 7, label_idx = 5)
    simple_evaluate(eval_model, eval_pos_dataloader, "experiments/binary_" + model_name + "_pos_preds")
    #simple_evaluate(eval_model, eval_neg_dataloader, "experiments/binary_finetune_neg_preds")


