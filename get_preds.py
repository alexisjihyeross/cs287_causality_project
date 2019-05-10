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

from models import BertTokenizer, MnliProcessor, TwoFullMnliProcessor, BinaryMnliProcessor, BertForSequenceClassification

from utils import train, evaluate, simple_evaluate
import copy

BERT_SIZE = 'base'  # or 'large'
BERT_CASED = False
DATA_DIR = 'glue_data/MNLI'
CACHE_DIR = 'cache'
MODEL = f'bert-{BERT_SIZE}-{"cased" if BERT_CASED else "uncased"}'



#options are full_two_way, binary
which_model = "binary"
do_finetune = True
exp_folder = "may9"
preds_file = "neg_dev_matched"


get_preds = False
get_effects = True

model_name = 'finetune' if do_finetune else 'no_finetune'

config = BertConfig('models/' + which_model + '/' + model_name + '_config.json')
tokenizer = BertTokenizer.from_pretrained(MODEL, do_lower_case=not BERT_CASED)

if which_model == "full_two_way":
    processor = TwoFullMnliProcessor()
else:
    processor = BinaryMnliProcessor()
num_labels = len(processor.get_labels())

binary_model = BertForSequenceClassification.from_pretrained(MODEL, cache_dir = CACHE_DIR, num_labels = num_labels)

if get_preds:
    eval_model = BertForSequenceClassification(config, num_labels = num_labels)
    eval_model.load_state_dict(torch.load("models/" + which_model + "/" + model_name + ".pt"))
    eval_model.to('cuda:0')
    eval_model.eval()
    eval_pos_dataloader = processor.get_dataloader(DATA_DIR, preds_file, tokenizer, batch_size = 10, a_idx = 6, b_idx = 7)
    eval_neg_dataloader = processor.get_dataloader(DATA_DIR, preds_file, tokenizer, batch_size = 10, a_idx = 8, b_idx = 7)
    os.makedirs(os.path.dirname("experiments/" + exp_folder + "/" + preds_file), exist_ok = True)
    simple_evaluate(eval_model, eval_pos_dataloader, "experiments/" + exp_folder + "/" + preds_file + "/" + which_model + "_" + model_name + "_pos_preds")
    simple_evaluate(eval_model, eval_neg_dataloader, "experiments/" + exp_folder + "/" + preds_file + "/" + which_model + "_" + model_name + "_neg_preds")

if get_effects:
    eval_model = BertForSequenceClassification(config, num_labels = num_labels)
    eval_model.load_state_dict(torch.load("models/" + which_model + "/" + model_name + ".pt"))
    eval_model.eval()
    eval_pos_dataloader = processor.get_dataloader(DATA_DIR, preds_file, tokenizer, max_seq_len = 70, batch_size = 1, a_idx = 6, b_idx = 7)
    eval_neg_dataloader = processor.get_dataloader(DATA_DIR, preds_file, tokenizer, max_seq_len = 70, batch_size = 1, a_idx = 8, b_idx = 7)
    os.makedirs(os.path.dirname("experiments/" + exp_folder + "/" + preds_file + "/effects/"), exist_ok = True)
    evaluate(eval_model, eval_pos_dataloader, eval_neg_dataloader, "experiments/" + exp_folder + "/" + preds_file + "/effects/" + which_model + "_" + model_name)
