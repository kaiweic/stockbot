import datetime
import os
import io

import selenium_tweets

company_ticker = selenium_tweets.company_ticker

PATH = selenium_tweets.DATA_PATH

tweet_by_year = {}

def main():

    with io.open('recovered_tweets.txt', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                date, tweet = line.split('\t', 1)
            except Exception as e:
                print(line, e)

            year = date.split('-')[0]
            if year not in tweet_by_year:
                tweet_by_year[year] = {}
            if date not in tweet_by_year[year]:
                tweet_by_year[year][date] = []
            tweet_by_year[year][date].append(line)

    missing_dates = set()
    temp_path = PATH.format(company_ticker, company_ticker, "temp")

    for year in tweet_by_year:
        
        start_date = datetime.datetime.strptime("{}-01-01".format(year), '%Y-%m-%d')
        end_date = datetime.datetime.strptime("{}-01-01".format(int(year) + 1), '%Y-%m-%d')
        days_in_year = (end_date - start_date).days
        dates = [(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_in_year)] # accounts for leap years
        
        missed_tweets = tweet_by_year[year]
        
        curr_path = PATH.format(company_ticker, company_ticker, year)

        print('writing to {}'.format(temp_path))
        with io.open(temp_path, 'w', encoding='utf-8') as writefile:

            # NOTE: Assumes that such a file exists
            print('reading from {}'.format(PATH.format(company_ticker, company_ticker, year)))
            with io.open(curr_path, 'r', encoding='utf-8') as readfile:
                line = readfile.readline()

                for date in dates:
                    # print(date, line.strip(), end=' ')
                    if line.startswith(date):
                        # print(1)
                        while (line.startswith(date)):
                            writefile.write(line)
                            line = readfile.readline()

                    elif date in missed_tweets:
                        # print(2)
                        for missed_tweet in missed_tweets[date]:
                            writefile.write(missed_tweet)

                    else:
                        # print(3)
                        print("looking for {}, but is matching with {}".format(date, line[0:10]))
                        missing_dates.add(date)
        # return
        with io.open(curr_path, 'w', encoding='utf-8') as writefile:
            with io.open(temp_path, 'r', encoding='utf-8') as readfile:
                for line in readfile:
                    writefile.write(line)

    print(missing_dates)

    print('cleaning up {}'.format(temp_path))
    os.remove(temp_path)

if __name__ == '__main__':
    main()
