# Instructions for running tweet scraper by hash tag

1.  You run `selenium_tweets.py`, which currently gets tweets from 2010 to 2013 for Dec 31st for "#Amazon" 
    because you should test it before you actually run it. You can change it from 2010-2019 from Jan 1st to 
    Dec 31st when you actually go to run it. I marked the places to change with `TODO`.
   -  `selenium_tweets_attu.py` now just imports `selenium_tweets` so I don't have to change everything twice to 
      make it work for Eric on attu. The output will write to `DATA_DIR/{company_ticker}/" and it'll create a 
      directory if there wasn't one already. 
2.  It'll write failed dates and links to `missing_tweets.txt`. From there, you can run `makeup_tweets.py` or 
    `makeup_tweets_attu.py` to get the missing tweets, which will write the recovered tweets to 
    `recovered_tweets.txt` and any failed dates back into `missing_tweets.txt`. 
3.  Then you can run `rewrite_tweets.py` to write the recovered tweets to the files they were missing from and 
    you'd rerun `makeup_tweets.py` and `rewrite_tweets.py` until there are no more missing dates.
4.  When you have all the data, you can fill out and run `combine_data.py` to combine all the data, which will 
    write it to `./twitter_data/\[company_ticker\].tsv`, then from there, you can edit and run 
    `check_consecutive_tweets.py` to check whether there were any missing dates and you can manually check to 
    see if those dates actually had tweets with the links provided
