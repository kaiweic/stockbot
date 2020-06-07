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
import io


'''
Parameters
'''

COMPANY_NAME = 'BAC'
BATCH_SIZE = 250
COMPANY_TWEET_FILE = '../input/' + COMPANY_NAME + '.tsv'
COMPANY_MODEL_LOCATION = '../model/' + COMPANY_NAME + '_weights.pt'
COMPANY_VOC2IND_LOCATION = '../model/' + COMPANY_NAME + '_voc2ind.json'
OUTPUT_PATH = '../output/' + COMPANY_NAME + '_out.tsv'


'''
Data
'''
global test_text
test_dates = []

def prepare_data(data_path):
    voc2ind = json.load(open(COMPANY_VOC2IND_LOCATION))

    global test_text
    global test_dates

    with io.open(data_path, encoding='utf8') as f:
        # This reads all the data from the file, but does not do any processing on it.
        raw_data = f.read()
        lines = raw_data.splitlines()
        # print(lines[210275])
        # print(lines[210276])

    raw_data = raw_data.replace('\f', ' ').replace('\r', ' ').replace('\t', ' ').replace('\v', ' ').replace('"', ' ').replace('\'', ' ')
    raw_data = re.sub(r'http\S+', r'http', raw_data)
    raw_data = re.sub(r'[.]{2,}', r' etcetcetc ', raw_data)
    raw_data = raw_data.replace(',', ' , ').replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? ').replace('(', ' ( ').replace(')', ' ) ').replace(';', ' ; ').replace('-', ' - ')
    raw_data = re.sub(r'@\S+', r'@', raw_data)
    raw_data = re.sub(r'#\S+', r'#', raw_data)
    raw_data = raw_data.replace(chr(133), ' ')
    raw_data = raw_data.replace(chr(30), ' ')
    raw_data = raw_data.lower()

    data = []

    # Compute voc2ind and transform the data into an integer representation of the tokens.
    batch_word_list = []
    batch_count = 0
    lines = raw_data.splitlines()

    for i in range(len(lines)):
        line = lines[i]

        # if reaches the batch size, add this batch to dataset and create a new empty batch list
        if batch_count == BATCH_SIZE:
            batch_count = 0
            data.append(pad_sequence(batch_word_list, batch_first=True, padding_value=0))
            batch_word_list = []

        # create new line list and add label
        try:
            a = int(line[0:4])
            b = int(line[7:9])
            c = int(line[12:14])
        except:
            print('error for dates on line ', str(i + 1))
            print(line)
            exit()
        test_dates.append(line[0:4] + '-' + line[7:9] + '-' + line[12:14])

        line_word_list = []
        words = line[15:].split()

        # add each word to the line list
        for word in words:
            curr_word = word if word in voc2ind else 'UNKUNKUNK'
            line_word_list.append(voc2ind[curr_word])

        # append current line of words into current batch
        batch_word_list.append(torch.tensor(line_word_list, dtype=torch.int64))
        batch_count += 1

    data.append(pad_sequence(batch_word_list, batch_first=True, padding_value=0))

    test_text = data



def output_data(data, data_path):
    with open(data_path, 'w') as f:
        date_index = 0
        curr_date = test_dates[0]
        curr_sum = 0
        curr_count = 0
        for batch in data:
            for label in batch:
                if date_index > 0 and test_dates[date_index] != test_dates[date_index - 1]:
                    f.write(test_dates[date_index - 1] + '\t' + str(curr_sum / curr_count) + '\n')
                    curr_sum = 0
                    curr_count = 0
                curr_sum += label
                curr_count += 1
                date_index += 1
        f.write(test_dates[-1] + '\t' + str(curr_sum / curr_count) + '\n')



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
    prepare_data(COMPANY_TWEET_FILE)

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = torch.load(COMPANY_MODEL_LOCATION)
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

    output_data(outputs, OUTPUT_PATH)

if __name__ == '__main__':
    main()