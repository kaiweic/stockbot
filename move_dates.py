import os 
import datetime

company_ticker = 'AAPL'
PATH = './twitter_data/{}.tsv'.format(company_ticker)
TEMP_PATH = './twitter_data/temp.tsv'
with open(PATH, 'r') as f:
    with open(TEMP_PATH, 'w') as outfile:
        for line in f:
            try:
                date, text = line.split('\t', 1)
                correct_date = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                outfile.write("{}\t{}".format(correct_date, text))
            except Exception as e:
                print(e)
                print(line)

with open(TEMP_PATH, 'r') as f:
    with open(PATH, 'w') as outfile:
        for line in f:
            outfile.write(line)


print('cleaning up {}'.format(TEMP_PATH))
os.remove(TEMP_PATH)
