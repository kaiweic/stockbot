'''
Copyright (C) 2020 for project StockBot.

'''

from datetime import date

file_name = '../output/BAC_new_out.tsv'
start = date(2018, 11, 30)
end = date(2018, 12, 21)

with open(file_name) as f:
    total = 0
    count = 0
    is_recording = False
    for line in f:
        curr = date(int(line[0:4]), int(line[5:7]), int(line[8:10]))
        if start <= curr and curr <= end:
            total += float(line.split()[2])
            count += 1
    print('Average Sentiment Score: ', total / count)