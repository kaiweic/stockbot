from os import listdir
from os.path import isfile, join

company_ticker = 'AAPL' # TODO: Change this to correct ticker
PATH = '/d/stockbot_data/{}/'.format(company_ticker) # TODO: Change this to correct data path
 
files = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
print(files)

with open('./twitter_data/{}.tsv'.format(company_ticker), 'w') as outfile:
    for f in files:
        with open(f) as infile:
            for line in infile:
                outfile.write(line)
