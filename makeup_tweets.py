import selenium_tweets 
import datetime
import time

date_to_tweets = {}
failed_links = {}

company_ticker = selenium_tweets.company_ticker
company_tag = selenium_tweets.company_tags[company_ticker] if company_ticker in selenium_tweets.company_tags else company_ticker

def main(alt=False):
    selenium_tweets.restart_driver()

    with open('missing_tweets.txt', 'r') as f:
        for line in f:
            if len(line) == 1:
                print('There are probably no missing tweets, check with check_consecutive_tweets.py')
                continue
            date, _ = line.split(' ', 1)
            start_date = datetime.datetime.strptime(date, '%Y-%m-%d')
            end_date = start_date + datetime.timedelta(days=1)
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            articles, success, link = selenium_tweets.get_tweets(start, end, company_tag, alt)
            if not success:
                failed_links[end] = link
                print('failed')
                print(link)
            print('got it for {} with {} results'.format(end, len(articles)))
            date_to_tweets[end] = articles
            time.sleep(3.5)

    prev_date = None
    with open('recovered_tweets.txt', 'w') as f:
        for date in date_to_tweets:
            tweets = date_to_tweets[date]
            curr_date = datetime.datetime.strptime(date, '%Y-%m-%d')
            # if (prev_date != None and ((prev_date != curr_date - datetime.timedelta(days=1)) or (prev_date.year != curr_date.year))):
            #     f.write('\n')
            for tweet in tweets:
                f.write('{}\t{}\n'.format(date, tweet))
            prev_date = curr_date
        if not date_to_tweets:
            f.write('')

    if failed_links:
        print('writing failed dates to missing_tweets.txt')
        with open('missing_tweets.txt', 'w') as f:
            for date in failed_links:
                link = failed_links[date]
                f.write(date + " " + str(link) + "\n")
            if not failed_links:
                f.write('')
    else:
        print("there were no failed dates")

if __name__ == '__main__':
    main()