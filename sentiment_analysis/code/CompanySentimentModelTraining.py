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
from datetime import date, timedelta
import io
from threading import Thread
from math import ceil
from queue import Queue


'''
Parameters
'''
WEIGHT_DECAY = 0#0.000001
UNK_PROB = 0.6
np.random.seed(0)
EPOCHS = 40
EMBEDDING_SIZE = 350
HIDDEN_SIZE = 350
NUM_LAYERS = 2
DROPOUT = 0.025
BATCH_SIZE = 200
THREAD_COUNT = 12


'''
Don't edit the following constants
'''
COMPANY_NAMES = ['AAPL', 'BAC', 'CMG', 'DAL', 'FB', 'GOOG', 'JPM', 'KO', 'LUV', 'MCD', 'MSFT', 'PEP', 'UAL', 'V', 'WFC']
COMPANY_STOCK_FILE = '../input/{}.csv'
COMPANY_TWEET_FILE = '../input/{}.tsv'
COMPANY_MODEL_LOCATION = '../model/weights.pt'
COMPANY_VOC2IND_LOCATION = '../model/voc2ind.json'


'''
Data Processing
'''
global train_text
global test_text
global train_label
global test_label
global vocab_size

def prepare_data():
    global train_text
    global test_text
    global train_label
    global test_label
    global vocab_size

    raw_data = ''
    raw_label = []
    raw_data_test = ''
    raw_label_test = []
    vocab = set()
    voc2ind = {}

    for company_name in COMPANY_NAMES:
        dates_to_change = {}

        with open(COMPANY_STOCK_FILE.format(company_name)) as f:
            line_count = 0
            prev_price = -1
            for line in f:
                line_count += 1
                elements = line.split(',')
                if line_count is 2:
                    prev_price = float(elements[5])
                elif line_count >= 3:
                    date_element = elements[1].split('-')
                    dates_to_change[date(int(date_element[0]), int(date_element[1]), int(date_element[2]))] = (float(elements[5]) - prev_price) / prev_price
                    prev_price = float(elements[5])

        print(company_name)
        with io.open(COMPANY_TWEET_FILE.format(company_name), encoding='utf-8') as f:
            lines = f.readlines()
            raw_data_train_thread = [[''] for _ in range(THREAD_COUNT)]
            raw_label_train_thread = [[] for _ in range(THREAD_COUNT)]
            raw_data_test_thread = [[''] for _ in range(THREAD_COUNT)]
            raw_label_test_thread = [[] for _ in range(THREAD_COUNT)]

            def process_file_(thread_id, lines, raw_data_train_thread, raw_label_train_thread, raw_data_test_thread, raw_label_test_thread):
                # count = 0
                for line in lines:
                    # count += 1
                    # print(thread_id, count, len(lines))
                    curr_date = date(int(line[0:4]), int(line[5:7]), int(line[8:10]))
                    if curr_date.weekday() is 6:
                        curr_date += timedelta(2)
                    elif curr_date.weekday() is 7:
                        curr_date += timedelta(1)
                    tweet = line[11:]
                    if curr_date in dates_to_change:
                        if curr_date < date(2017, 1, 5):
                            raw_data_train_thread[0] += tweet.replace('\n', '').replace('\r', '') + '\n'
                            raw_label_train_thread.append(dates_to_change[curr_date])
                        else:
                            raw_data_test_thread[0] += tweet.replace('\n', '').replace('\r', '') + '\n'
                            raw_label_test_thread.append(dates_to_change[curr_date])

            threads = [Thread(target=process_file_, args=(i, lines[ceil(len(lines) / THREAD_COUNT * i):ceil(len(lines) / THREAD_COUNT * (i + 1))], raw_data_train_thread[i], raw_label_train_thread[i], raw_data_test_thread[i], raw_label_test_thread[i])) for i in range(THREAD_COUNT)]
            for t in threads: t.start()
            for t in threads: t.join()

            threads = [Thread()]
            for i in range(THREAD_COUNT):
                raw_data += raw_data_train_thread[i][0]
                raw_label += raw_label_train_thread[i]
                raw_data_test += raw_data_test_thread[i][0]
                raw_label_test += raw_label_test_thread[i]

    len_training = start_of_test_line_index = len(raw_label)
    raw_data += raw_data_test[:-1]
    raw_label += raw_label_test
    raw_label = [1 if i > 0 else 0 for i in raw_label]

    # data processing
    raw_data = raw_data.replace('\f', ' ').replace('\r', ' ').replace('\t', ' ').replace('\v', ' ').replace('"', ' ').replace('\'', ' ')
    raw_data = re.sub(r'http\S+', r' ', raw_data)
    raw_data = re.sub(r'[.]{2,}', r' etcetcetc ', raw_data)
    raw_data = raw_data.replace(',', ' , ').replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? ').replace('(', ' ').replace(')', ' ').replace(';', ' . ').replace('-', ' ')
    raw_data = re.sub(r'@\S+', r' ', raw_data)
    raw_data = re.sub(r'#\S+', r' ', raw_data)
    raw_data = raw_data.lower()

    # Compute voc2ind and transform the data into an integer representation of the tokens.
    batch_word_list = []
    batch_label_list = []
    batch_count = 0
    lines = raw_data.split('\n')

    # maps from word to vocab count
    vocab_count = {}

    for i in range(len_training):
        line = lines[i]
        elements = line.split()
        for word in elements:
            vocab_count[word] = vocab_count.get(word, 0) + 1
    for key in vocab_count:
        if vocab_count[key] == 1 and uniform() < UNK_PROB:
            continue
        else:
            vocab.add(key)
    vocab.add('UNKUNKUNK')

    data = []
    labels = []

    for i in range(len(lines)):
        line = lines[i]
        label = raw_label[i]

        # if reaches the batch size, add this batch to dataset and create a new empty batch list
        if batch_count == BATCH_SIZE or i == start_of_test_line_index:
            batch_count = 0
            data.append(pad_sequence(batch_word_list, batch_first=True, padding_value=0))
            labels.append(torch.tensor(batch_label_list))
            if i == start_of_test_line_index:
                start_of_test_batch_index = len(data)
            batch_word_list = []
            batch_label_list = []

        # create new line list and add label
        line_word_list = []
        line_label_list = []
        elements = line.split()
        line_label_list.append(label)

        # add each word to the line list
        for word in elements:
            curr_word = word if word in vocab else 'UNKUNKUNK'
            if i < start_of_test_line_index:
                curr_word = 'UNKUNKUNK' if uniform() < DROPOUT else word
            if curr_word not in voc2ind:
                voc2ind[curr_word] = len(voc2ind)
            line_word_list.append(voc2ind[curr_word])

        # append current line of words into current batch
        batch_word_list.append(torch.tensor(line_word_list, dtype=torch.int64))
        batch_label_list.append(torch.tensor(line_label_list, dtype=torch.float64))
        batch_count += 1

    data.append(pad_sequence(batch_word_list, batch_first=True, padding_value=0))
    labels.append(torch.tensor(batch_label_list))

    print('start_of_test_batch_index', start_of_test_batch_index)
    train_text = data[:start_of_test_batch_index]
    test_text = data[start_of_test_batch_index:]
    train_label = labels[:start_of_test_batch_index]
    test_label = labels[start_of_test_batch_index:]

    vocab_size = len(voc2ind)

    print('vocab size', vocab_size)
    print('len of train', len(train_label))
    print('len of test', len(test_label))

    j = json.dumps(voc2ind)
    with open(COMPANY_VOC2IND_LOCATION, 'w') as f:
        f.write(j)
        f.close()

'''
Model
'''

class SentimentBiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(SentimentBiGRU, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
    prepare_data()
    print('finished preparing')

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = SentimentBiGRU(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)

    loss_func = nn.BCEWithLogitsLoss()
    # optim = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optim = torch.optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)

    train_loss_list = []
    test_loss_list = []
    best_test_accuracy = 0

    '''
    here
    '''
    model.eval()
    test_all = 0
    test_correct = 0
    for batch_idx in range(len(test_text)):
        data = test_text[batch_idx]
        label = test_label[batch_idx]
        data, label = data.to(device), label.to(device)
        output = model(data)
        output = output.squeeze()
        output = output[:, -1]
        loss_batch = loss_func(output, label)

        output = output.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        output_positive = np.where(output > 0.0, 1, 0)
        output_negative = np.where(output < 0.0, 1, 0)
        label_positive = np.where(label > 0.5, 1, 0)
        label_negative = np.where(label < 0.5, 1, 0)

        test_all += len(label)
        # print(np.sum(output_positive & label_positive), np.sum(output_negative & label_negative))
        test_correct += np.sum(output_positive & label_positive)
        test_correct += np.sum(output_negative & label_negative)
    print(test_correct / test_all)
    '''
    here
    '''

    for epoch in range(1, EPOCHS + 1):
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
            label_positive = np.where(label > 0.5, 1, 0)
            label_negative = np.where(label < 0.5, 1, 0)

            train_all += len(label)
            # print(np.sum(output_positive & label_positive), np.sum(output_negative & label_negative))
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
            label_positive = np.where(label > 0.5, 1, 0)
            label_negative = np.where(label < 0.5, 1, 0)

            test_all += len(label)
            # print(np.sum(output_positive & label_positive), np.sum(output_negative & label_negative))
            test_correct += np.sum(output_positive & label_positive)
            test_correct += np.sum(output_negative & label_negative)

        train_loss = train_loss / BATCH_SIZE / len(train_text)
        test_loss = test_loss / BATCH_SIZE / len(test_text)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        test_accuracy = test_correct / test_all
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model, COMPANY_MODEL_LOCATION)

        print('epoch', epoch, 'loss', round(train_loss, 5), round(test_loss, 5), 'accuracy', round(train_correct / train_all, 5), round(test_accuracy, 5))


if __name__ == '__main__':
    main()