import selenium_tweets_attu as selenium_tweets
import datetime
import time

ticker = selenium_tweets.company_ticker
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
        time.sleep(3)

if failed_links:
    print('writing failed dates to makeup.txt')
    with open('makeup.txt', 'w') as f:
        for year in failed_year:
            print(year)
            failed_links = failed_year[year]
            for date in failed_links:
                link = failed_links[date]
                f.write(date + " " + link + "\n")
