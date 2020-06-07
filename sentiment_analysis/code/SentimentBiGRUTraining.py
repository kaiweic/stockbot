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
# learning_rate = 0.0025
weight_decay = 0.00001
unk_prob = 0.6
np.random.seed(0)
epochs = 40
embedding_size = 300
hidden_size = 300
num_layers = 2
dropout = 0.025
batch_size = 250


'''
Data
'''
voc2ind = {}
vocab = set()
global train_text
global test_text
global train_label
global test_label
global vocab_size

def prepare_data(data_path):
    global train_text
    global test_text
    global train_label
    global test_label
    global vocab_size
    global dropout

    with open(data_path) as f:
        raw_data = f.read()

    # data processing
    raw_data = raw_data.replace('\f', ' ').replace('\r', ' ').replace('\t', ' ').replace('\v', ' ').replace('"', ' ').replace('\'', ' ')
    raw_data = re.sub(r'http\S+', r' ', raw_data)
    raw_data = re.sub(r'[.]{2,}', r' etcetcetc ', raw_data)
    raw_data = raw_data.replace(',', ' , ').replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? ').replace('(', ' ').replace(')', ' ').replace(';', ' . ').replace('-', ' ')
    raw_data = re.sub(r'@\S+', r' ', raw_data)
    raw_data = re.sub(r'#\S+', r' ', raw_data)
    raw_data = raw_data.lower()

    global voc2ind

    data = []
    labels = []
    index_count = 1

    # Compute voc2ind and transform the data into an integer representation of the tokens.
    batch_word_list = []
    batch_label_list = []
    batch_count = 0
    lines = raw_data.splitlines()

    # make the order random
    shuffle(lines)

    # maps from word to vocab count
    vocab_count = {}

    for i in range(int(len(lines) * 0.9)):
        line = lines[i]
        line = line[4:]
        elements = line.split()
        for word in elements:
            vocab_count[word] = vocab_count.get(word, 0) + 1
    for key in vocab_count:
        if vocab_count[key] == 1 and uniform() < unk_prob:
            continue
        else:
            vocab.add(key)
    vocab.add('UNKUNKUNK')

    for i in range(len(lines)):
        line = lines[i]

        # if reaches the batch size, add this batch to dataset and create a new empty batch list
        if batch_count == batch_size:
            batch_count = 0
            data.append(pad_sequence(batch_word_list, batch_first=True, padding_value=0))
            labels.append(torch.tensor(batch_label_list))
            batch_word_list = []
            batch_label_list = []

        # create new line list and add label
        line_word_list = []
        line_label_list = []
        label = int(line[0]) / 4
        line = line[4:]
        elements = line.split()
        line_label_list.append(label)

        # add each word to the line list
        for word in elements:
            curr_word = word if word in vocab else 'UNKUNKUNK'
            if i < len(lines) * 0.9:
                curr_word = 'UNKUNKUNK' if uniform() < dropout else word
            if curr_word not in voc2ind:
                voc2ind[curr_word] = index_count
                index_count += 1
            line_word_list.append(voc2ind[curr_word])

        # append current line of words into current batch
        batch_word_list.append(torch.tensor(line_word_list, dtype=torch.int64))
        batch_label_list.append(torch.tensor(line_label_list, dtype=torch.float64))
        batch_count += 1

    data.append(pad_sequence(batch_word_list, batch_first=True, padding_value=0))
    labels.append(torch.tensor(batch_label_list))

    train_text = data[:int(len(data) * 0.9)]
    test_text = data[int(len(data) * 0.9):]
    train_label = labels[:int(len(labels) * 0.9)]
    test_label = labels[int(len(labels) * 0.9):]

    vocab_size = index_count



'''
Model
'''

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
    prepare_data('../data/train2.csv')

    j = json.dumps(voc2ind)
    with open('../model/voc2ind.json', 'w') as f:
        f.write(j)
        f.close()

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = SentimentBiGRU(vocab_size, embedding_size, hidden_size, num_layers, dropout).to(device)

    loss_func = nn.BCEWithLogitsLoss()
    # optim = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optim = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

    train_loss_list = []
    test_loss_list = []
    best_test_accuracy = 0
    for epoch in range(1, epochs + 1):
        train_loss = 0
        test_loss = 0

        train_all = 0
        train_correct = 0

        test_all = 0
        test_correct = 0

        model.train()
        for batch_idx in range(len(train_text)):
            data = train_text[batch_idx]
            label = train_label[batch_idx]
            data, label = data.to(device), label.to(device)
            optim.zero_grad()
            output = model(data)
            output = output.squeeze()
            output = output[:, -1]
            loss_batch = loss_func(output, label)
            loss_batch.backward()
            optim.step()
            train_loss += loss_batch.item()

            output = output.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            output_positive = np.where(output > 0.0, 1, 0)
            output_negative = np.where(output < 0.0, 1, 0)
            label_positive = np.where(label == 1.0, 1, 0)
            label_negative = np.where(label == 0.0, 1, 0)

            train_all += np.where(label == 0.5, 0, 1).sum()
            train_correct += np.sum(output_positive & label_positive)
            train_correct += np.sum(output_negative & label_negative)

        model.eval()
        for batch_idx in range(len(test_text)):
            data = test_text[batch_idx]
            label = test_label[batch_idx]
            data, label = data.to(device), label.to(device)
            output = model(data)
            output = output.squeeze()
            output = output[:, -1]
            loss_batch = loss_func(output, label)
            test_loss += loss_batch.item()

            output = output.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            output_positive = np.where(output > 0.0, 1, 0)
            output_negative = np.where(output < 0.0, 1, 0)
            label_positive = np.where(label == 1.0, 1, 0)
            label_negative = np.where(label == 0.0, 1, 0)

            test_all += np.where(label == 0.5, 0, 1).sum()
            test_correct += np.sum(output_positive & label_positive)
            test_correct += np.sum(output_negative & label_negative)

        train_loss = train_loss / batch_size / len(train_text)
        test_loss = test_loss / batch_size / len(test_text)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        test_accuracy = test_correct / test_all
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model, '../model/BestSentimentBiGRU.pt')

        print('epoch', epoch, 'loss', round(train_loss, 5), round(test_loss, 5), 'accuracy', round(train_correct / train_all, 5), round(test_accuracy, 5))


if __name__ == '__main__':
    main()