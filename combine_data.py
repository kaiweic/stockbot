from os import listdir
from os.path import isfile, join

company_ticker = 'CMG'
PATH = '/d/stockbot_data/{}/'.format(company_ticker) # Eric change this to './stockbot_data' instead of '/d/stockbot_data'
 
files = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
print(files)

with open('./twitter_data/{}.tsv'.format(company_ticker), 'w') as outfile:
    for f in files:
        with open(f) as infile:
            for line in infile:
                outfile.write(line)