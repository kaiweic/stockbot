'''
Copyright (C) 2020 for project StockBot.

Websites consulted:
https://pytorch.org/docs/stable/nn.html
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
'''

import torch
from torch import nn, optim
import numpy as np
from numpy.random import choice, uniform
from random import shuffle
import re
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence
from torchnlp.nn import WeightDrop
import json


'''
Parameters
'''
batch_size = 250


'''
Data
'''
global test_text

def prepare_data(data_path):
    voc2ind = json.load(open('../model/voc2ind.json'))

    global test_text

    with open(data_path) as f:
        # This reads all the data from the file, but does not do any processing on it.
        raw_data = f.read()

    raw_data = raw_data.replace('\f', ' ').replace('\r', ' ').replace('\t', ' ').replace('\v', ' ').replace('"', ' ').replace('\'', ' ')
    raw_data = re.sub(r'http\S+', r'http', raw_data)
    raw_data = re.sub(r'[.]{2,}', r' etcetcetc ', raw_data)
    raw_data = raw_data.replace(',', ' , ').replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? ').replace('(', ' ( ').replace(')', ' ) ').replace(';', ' ; ').replace('-', ' - ')
    raw_data = re.sub(r'@\S+', r'@', raw_data)
    raw_data = re.sub(r'#\S+', r'#', raw_data)
    raw_data = raw_data.lower()

    data = []

    # Compute voc2ind and transform the data into an integer representation of the tokens.
    batch_word_list = []
    batch_count = 0
    lines = raw_data.splitlines()

    for i in range(len(lines)):
        line = lines[i]

        # if reaches the batch size, add this batch to dataset and create a new empty batch list
        if batch_count == batch_size:
            batch_count = 0
            data.append(pad_sequence(batch_word_list, batch_first=True, padding_value=0))
            batch_word_list = []

        # create new line list and add label
        line_word_list = []

        # add each word to the line list
        for word in line.split():
            curr_word = word if word in voc2ind else 'UNKUNKUNK'
            line_word_list.append(voc2ind[curr_word])

        # append current line of words into current batch
        batch_word_list.append(torch.tensor(line_word_list, dtype=torch.int64))
        batch_count += 1

    data.append(pad_sequence(batch_word_list, batch_first=True, padding_value=0))

    test_text = data



def output_data(data, data_path):
    with open(data_path, 'w') as f:
        for batch in data:
            for label in batch:
                f.write(str(label) + '\n')



class SentimentBiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(SentimentBiGRU, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.projection = nn.Linear(2 * hidden_size, 1)

    def forward(self, sentences):
        embedding = self.embedding(sentences)
        output, _ = self.gru(embedding)
        return self.projection(output)


'''
Train and Test
'''

def main():
    prepare_data('../input/input.txt')

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = torch.load('../model/BestSentimentBiGRU.pt')
    model = model.to(device)
    model.eval()
    outputs = []
    for batch_idx in range(len(test_text)):
        data = test_text[batch_idx]
        data = data.to(device)
        output = model(data)
        output = output.squeeze()
        output = output[:, -1]
        output = torch.sigmoid(output)
        output = output.cpu().detach().numpy()
        outputs.append(output)

    output_data(outputs, '../output/output.txt')

if __name__ == '__main__':
    main()