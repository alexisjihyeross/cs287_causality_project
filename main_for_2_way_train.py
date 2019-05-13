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

from models import BertTokenizer, MnliProcessor, BinaryMnliProcessor, TwoFullMnliProcessor, BertForSequenceClassification

from utils import train, evaluate, simple_evaluate
import copy

BERT_SIZE = 'base'  # or 'large'
BERT_CASED = False
DATA_DIR = 'glue_data/MNLI'
CACHE_DIR = 'cache'
MODEL = f'bert-{BERT_SIZE}-{"cased" if BERT_CASED else "uncased"}'



tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=not BERT_CASED)

processor = TwoFullMnliProcessor()
num_labels = len(processor.get_labels())

full_model = BertForSequenceClassification.from_pretrained(MODEL, cache_dir = CACHE_DIR, num_labels=num_labels)


do_train = False
do_finetune = True
do_evaluate = True
other = False


if do_finetune == True:
    model_name = "finetune"
else:
    model_name = "no_finetune"

if do_train:
    print("training...")
    print("loading data...")
    train_dataloader = processor.get_dataloader(DATA_DIR, 'train', tokenizer, max_seq_len=70, shuffle=True)

    val_dataloader = processor.get_dataloader(DATA_DIR, 'dev_matched', tokenizer, max_seq_len=70, label_idx = -1, shuffle=True)

    full_model = train(full_model, train_dataloader, val_dataloader, num_labels, num_epochs=1, finetune=do_finetune, evaluate=True)
    simple_evaluate(full_model, val_dataloader, "experiments/")
    torch.save(full_model.state_dict(), "models/full_two_way/" + model_name + ".pt")
    with open('models/full_two_way/' + model_name + '_config.json', 'w') as f:
        f.write(full_model.config.to_json_string())

if do_evaluate:
    print("loading model...")
    config = BertConfig('models/full_two_way/' + model_name + '_config.json')
    #eval_model = BertForSequenceClassification(config, num_labels = num_labels)
    eval_model = BertForSequenceClassification(config, num_labels=num_labels)
    eval_model.load_state_dict(torch.load("models/full_two_way/" + model_name + ".pt"))
    #eval_model.to('cuda:0')
    eval_model.eval()

    pos_dataloader = processor.get_dataloader(DATA_DIR, "neg_dev_matched", tokenizer, max_seq_len=70, batch_size = 30, a_idx=6,b_idx=7)
    neg_dataloader = processor.get_dataloader(DATA_DIR, "neg_dev_matched", tokenizer, max_seq_len=70, batch_size = 30, a_idx=8, b_idx=7)

    evaluate(eval_model, pos_dataloader, neg_dataloader, "experiments/may12_2/three_way_finetune", 2, DEBUG=True)


if other:
    config = BertConfig('models/full_two_way/' + model_name + '_config.json')
    eval_model = BertForSequenceClassification(config, num_labels = num_labels)
    eval_model.load_state_dict(torch.load("models/full_two_way/" + model_name + ".pt"))
    eval_model.to('cuda:0')
    eval_model.eval()
    eval_pos_dataloader = processor.get_dataloader(DATA_DIR, "dev_mismatched", tokenizer, batch_size = 10, a_idx = 6, b_idx = 7, label_idx = 5)
    #eval_neg_dataloader = processor.get_dataloader(DATA_DIR, "neg_binary_dev_mismatched", tokenizer, batch_size = 10, a_idx = 8, b_idx = 7, label_idx = 5)
    simple_evaluate(eval_model, eval_pos_dataloader, "experiments/binary_finetune_pos_preds")
    #simple_evaluate(eval_model, eval_neg_dataloader, "experiments/binary_finetune_neg_preds")
