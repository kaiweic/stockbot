import selenium_tweets_attu as selenium_tweets
import makeup_tweets
import datetime
import time

company_ticker = selenium_tweets.company_ticker

makeup_tweets.company_ticker = company_ticker
makeup_tweets.company_tag = selenium_tweets.company_tags[company_ticker] if company_ticker in selenium_tweets.company_tags else company_ticker

makeup_tweets.main()