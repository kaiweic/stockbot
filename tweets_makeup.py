import selenium_tweets 
import datetime
import time

ticker = 'CMG'
tag = selenium_tweets.company_tags[ticker]

date_to_tweets = {}
failed_links = {}

with open('makeup.txt', 'r') as f:
    for line in f:
        date, _ = line.split(' ', 1)
        end_date = datetime.datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - datetime.timedelta(days=1)
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")
        articles, success, link = selenium_tweets.get_tweets(ticker, tag, start, end)
        if not success:
            failed_links[end] = link
            print('failed')
            print(link)
        print('got it for {} with {} results'.format(end, len(articles)))
        date_to_tweets[end] = articles
        time.sleep(2)

with open('madeup.txt', 'w') as f:
    for date in date_to_tweets:
        tweets = date_to_tweets[date]
        for tweet in tweets:
            f.write('{}\t{}\n'.format(date, tweet))

if failed_links:
    print('failed dates')
    for date in failed_links:
        link = failed_links[date]
        print(date, link)