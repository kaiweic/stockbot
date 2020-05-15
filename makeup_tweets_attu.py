import selenium_tweets_attu as selenium_tweets
import tweets_makeup
import datetime
import time

tweets_makeup.company_ticker = selenium_tweets.company_ticker
tweets_makeup.company_tag = selenium_tweets.company_tags[company_ticker] if company_ticker in selenium_tweets.company_tags else company_ticker

tweets_makeup.main()