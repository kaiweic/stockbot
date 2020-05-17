import selenium_tweets 
import datetime
import time
import io

import rewrite_tweets

date_to_tweets = {}
failed_links = {}

company_ticker = selenium_tweets.company_ticker
company_tag = selenium_tweets.company_tags[company_ticker] if company_ticker in selenium_tweets.company_tags else company_ticker

alt = selenium_tweets.alt

def main(alt=False):
    count = 0

    selenium_tweets.restart_driver()

    with io.open('missing_tweets.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if len(line) == 1:
                print('There are probably no missing tweets, check with check_consecutive_tweets.py')
                continue
            count += 1
            date, _ = line.split(' ', 1)
            start_date = datetime.datetime.strptime(date, '%Y-%m-%d')
            end_date = start_date + datetime.timedelta(days=1)
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            articles, success, link = selenium_tweets.get_tweets(start, end, company_tag, alt)
            if not success:
                failed_links[start] = link
                print('failed')
                print(link)
            print('got it for {} with {} results'.format(start, len(articles)))
            date_to_tweets[start] = articles
            time.sleep(3.5)

    if count == 0:
        print("There weren't any missing tweets, move on")
        return False

    with io.open('recovered_tweets.txt', 'w', encoding='utf-8') as f:
        for date in date_to_tweets:
            tweets = date_to_tweets[date]
            curr_date = datetime.datetime.strptime(date, '%Y-%m-%d')
            for tweet in tweets:
                f.write('{}\t{}\n'.format(date, tweet))
        if not date_to_tweets:
            f.write('')

    print('writing failed dates to missing_tweets.txt')
    with io.open('missing_tweets.txt', 'w', encoding='utf-8') as f:
        for date in failed_links:
            link = failed_links[date]
            f.write(date + " " + str(link) + "\n")
        if not failed_links:
            print("there were no failed dates")
            f.write('')

    return True

if __name__ == '__main__':
    success = main(alt=alt)
    if success:
        rewrite_tweets.main()