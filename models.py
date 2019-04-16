import os
import csv
import sys
import collections
from tqdm import tqdm
import json
import copy

from pytorch_pretrained_bert import BertConfig, BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert.optimization import BertAdam
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

InputExample = collections.namedtuple('InputExample', ['guid', 'text_a', 'text_b', 'label'])
InputFeatures = collections.namedtuple('InputFeatures', ['input_ids', 'input_mask', 'segment_ids', 'label_id'])

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b) if example.text_b else None

        # truncate sequence pairs, longest one first, accounting for [cls] and [sep]
        if example.text_b:
            while len(tokens_a) + len(tokens_b) > max_seq_length - 3:
                max(tokens_a, tokens_b, key=len).pop()
        elif len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # segment_ids indicate token in first or second seq, w/ embedding vectors
        # learned for them during pre-training, added to position embedding
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids) # Only attend to real tokens

        seqs = [tokens, input_ids, input_mask, segment_ids]

        padding = [0] * (max_seq_length - len(input_ids))
        for seq in seqs[1:]:
            seq += padding
            assert len(seq) == max_seq_length

        label_map = {label: i for i, label in enumerate(label_list)}
        label_id = label_map[example.label]

        if ex_index < 3:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            for n, d in zip(['tokens', 'input_ids', 'input_mask', 'segment_ids'], [tokens, input_ids, input_mask, segment_ids]):
                print(f'{n}: {" ".join([str(x) for x in d])}')
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(InputFeatures(input_ids, input_mask, segment_ids, label_id))
    return features

class MnliProcessor():
    """Processor for the MultiNLI data set (GLUE version)."""
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, a_idx = None, b_idx = None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == "neg_test_matched" or set_type == "neg_test_mismatched" or set_type == "neg_dev_matched" or set_type == "neg_dev_mismatched":
                text_a = line[a_idx]
                text_b = line[b_idx]
                label = "entailment"
            else:
                text_a = line[8]
                text_b = line[9]
                label = line[-1]
            examples.append(InputExample(guid, text_a, text_b, label))
        return examples

    def get_dataloader(self, data_dir, data_file, tokenizer, batch_size=10, max_seq_len=70, a_idx = None, b_idx = None):
        if data_file not in ['dev_mismatched', 'dev_matched', 'test_matched', 'test_mismatched', 'neg_test_matched', 'neg_test_mismatched', 'neg_dev_matched', 'neg_dev_mismatched','train', 'small_train']:
            raise KeyError(f'Invalid data file {data_file}')

        data = self._read_tsv(os.path.join(data_dir, f"{data_file}.tsv"))
        examples = self._create_examples(data, data_file, a_idx = a_idx, b_idx = b_idx)
        labels = self.get_labels()

        train_feats = convert_examples_to_features(examples, labels, max_seq_len, tokenizer)
        dataset = [torch.tensor(d, dtype=torch.long) for d in zip(*train_feats)]

        return DataLoader(TensorDataset(*dataset), batch_size=batch_size)

class BinaryMnliProcessor():
    """Processor for the MultiNLI data set with neutral examples removed."""
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["contradiction", "entailment"]

    def _create_examples(self, lines, set_type, a_idx = None, b_idx = None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == "neg_test_matched" or set_type == "neg_test_mismatched" or set_type == "neg_dev_matched" or set_type == "neg_dev_mismatched":
                text_a = line[a_idx]
                text_b = line[b_idx]
                label = "entailment"
            else:
                text_a = line[8]
                text_b = line[9]
                label = line[-1]
            examples.append(InputExample(guid, text_a, text_b, label))
        return examples

    def get_dataloader(self, data_dir, data_file, tokenizer, batch_size=10, max_seq_len=70, a_idx = None, b_idx = None):
        if data_file not in ['binary_dev_mismatched', 'binary_dev_matched', 'test_matched', 'test_mismatched', 'neg_test_matched', 'neg_test_mismatched', 'neg_dev_matched', 'neg_dev_mismatched', 'binary_train', 'small_binary_train']:
            raise KeyError(f'Invalid data file {data_file}')

        data = self._read_tsv(os.path.join(data_dir, f"{data_file}.tsv"))
        examples = self._create_examples(data, data_file, a_idx = a_idx, b_idx = b_idx)
        labels = self.get_labels()

        train_feats = convert_examples_to_features(examples, labels, max_seq_len, tokenizer)
        dataset = [torch.tensor(d, dtype=torch.long) for d in zip(*train_feats)]

        return DataLoader(TensorDataset(*dataset), batch_size=batch_size)

class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model w/ pooled output to linear layer 
    Params:
        `config`: a BertConfig instance with the config to build a new model
        `num_labels`: number of classes for classifier. Default = 2
    Inputs:
        `input_ids`: [batch_size, seq_length] of word token indices
        `token_type_ids`: optional [batch_size, seq_length] with [0, 1] for sentence [A, B] tokens
        `attention_mask`: optional [batch_size, seq_length] with [0, 1] to mask out attention on padding
        `labels`: optional [batch_size] of class output [0, ..., num_labels]
    Outputs:
        if `labels` is not `None`: CrossEntropy loss of output with labels.
        if `labels` is `None`: Class logits [batch_size, num_labels]
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, modification=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        
        if modification:
            pooled_output = modification(pooled_output)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output

#        if labels is not None:
#            loss_fct = CrossEntropyLoss()
#            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#            return loss, pooled_output
#        else:
#            return logits, pooled_output
